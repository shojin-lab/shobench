"""Prime Intellect's prime-agent, the harness. The flags and stop signals are sourced in
``docs/harness-autonomy.md``, which cites where every one came from.
"""

from __future__ import annotations

import json
from pathlib import Path

from shobench.harness import (
    Harness,
    LaunchSpec,
    StopKind,
    StopVerdict,
    UsageLimitRule,
    jsonl_events,
    stderr_evidence,
    tail,
)
from shobench.harnesses._trace import (
    _first_event_of_type,
    _first_parseable_event,
    _last_event_of_type,
)

# The prime-agent skill that makes the served stream reachable. Declaring the HTTP server in
# settings.json is only half of the wiring: prime-agent hands the model no MCP tools, it reaches
# a server by importing a Python-backed skill in its kernel and calling it, so the cell HOME
# must also carry the `shogym-stream` skill package. It is vendored under the repo's
# `prime_agent/skills/` and seeded into the isolated HOME beside the settings file, which is
# written per leg while this is written once and then belongs to the agent.
SHOGYM_STREAM_SKILL = "shogym-stream"
_SKILL_HOME_PREFIX = ".prime/agent/skills"
# What prime-agent's discovery requires of a Python-backed skill: the frontmatter that makes it
# loadable at all, the pyproject that makes it Python-backed rather than markdown, and the module
# the kernel imports, whose name is the skill's with hyphens turned into underscores. A directory
# missing any one of them is skipped or installed without the client, and either way the leg runs
# with no reachable tools.
_SKILL_IMPORT_NAME = SHOGYM_STREAM_SKILL.replace("-", "_")
_SKILL_REQUIRED_FILES = (
    "SKILL.md",
    "pyproject.toml",
    f"src/{_SKILL_IMPORT_NAME}/__init__.py",
)


def _vendored_skill_dir() -> Path:
    from shobench.config import repo_root

    return repo_root() / "prime_agent" / "skills" / SHOGYM_STREAM_SKILL


def shogym_stream_skill_files() -> dict[str, str]:
    """The vendored ``shogym-stream`` skill as ``{home-relative path: contents}``.

    Every file under the vendored skill package, keyed by where it lands in the cell HOME, so a
    prime-agent kernel installs it editable at session start and exposes it as ``shogym_stream``.
    Vendored rather than fetched from the pinned shogym examples because the package must carry
    the runner's own token variable (``bearerTokenEnvVar``), not the quickstart's, so a verbatim
    copy would be wrong here rather than merely drift-prone; the file is a few dozen lines of
    text either way.

    Absent or incomplete, the skill is an error and not an empty mapping. A leg launched with
    the settings entry alone is the failure this wiring exists to prevent: prime-agent starts,
    the model has nothing to import, and the run looks like an agent that chose to do nothing
    rather than a harness that was never connected. The asset is resolved from the checkout, so
    the way it goes missing is an install that is not one, which must fail here rather than
    twelve serving hours later.
    """
    root = _vendored_skill_dir()
    files: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        files[f"{_SKILL_HOME_PREFIX}/{SHOGYM_STREAM_SKILL}/{rel}"] = path.read_text(
            encoding="utf-8"
        )
    prefix = f"{_SKILL_HOME_PREFIX}/{SHOGYM_STREAM_SKILL}"
    missing = [name for name in _SKILL_REQUIRED_FILES if f"{prefix}/{name}" not in files]
    if missing:
        raise FileNotFoundError(
            f"the vendored {SHOGYM_STREAM_SKILL} skill at {root} is missing "
            f"{', '.join(missing)}; a prime_agent leg without it reaches no tools"
        )
    return files


class PrimeAgent(Harness):
    """Prime Intellect's prime-agent, non-interactive.

    prime-agent installs from its vendor script, never from npm: the ``prime-agent`` name
    returns 404 on the npm registry, and the inherited npm names in its source tree resolve to
    Pi, a different agent. Its own docs say the workspace names are implementation details and
    not the install path.

    There is nothing to bypass. prime-agent has no permission prompt, no approval policy, and
    no sandbox; its docs state that workers are process-isolated for failure containment and
    not security-sandboxed. The container is the only boundary, which is what this runner
    already provides.

    What does need configuring is autonomy's budgets. Autonomous mode starts disabled, and
    once enabled it carries four budgets whose defaults are far below an 8-hour rollout: 3
    continuations, 12 turns, 80,000 tokens, and 30 minutes. Leaving any of them would end the
    rollout on an imposed cutoff, and the docs are explicit that reaching a limit "does not
    imply task success", so recording it as the agent's own stop would be exactly the confound
    the scope forbids. The runner raises all of them and reads a limit that was still reached
    as a cutoff.

    Value-taking autonomous flags take a separate argument; ``--flag=value`` is rejected.
    stdin is closed, because print mode reads piped stdin and merges it into the prompt.
    """

    name = "prime_agent"

    # Where its persisted sessions live: --resume <id> resolves against <sessions>/<id>.jsonl
    # in the HOME it runs with (cwd-matching sessions first, then all of them; every leg runs
    # at /work so the rollout's session is in the local set), errors with "No session found
    # matching" when none does, and appends the resumed turn to the file it found. The
    # artifact tree rides beside the transcript because at 0.7.1 it IS part of the persisted
    # session: session-artifacts/<id>/ holds the kernel snapshot (kernel-state.dill/json,
    # revived into the fresh kernel right after start), the sub-* child session dirs the RLM
    # runtime reads completed children back from, the local harness-state dir, and the
    # session's scheduled jobs. A fork that carried the transcript alone would reopen the
    # conversation while silently dropping the learned kernel state and every child
    # conversation. Leases and daemon state live in sibling subtrees (session-leases/,
    # daemon-workers/) and stay behind with the rest of the noise.
    session_state_dirs = (".prime/agent/sessions", ".prime/agent/session-artifacts")

    # The structured signal, which is the cleanest of the three harnesses: a stream failure is
    # classified into a kind, and rate_limit is one of them.
    usage_limit_rules = (
        UsageLimitRule(
            where="stdout",
            pattern=r'"kind"\s*:\s*"rate_limit"',
            citation="classifyStreamFailure kind, attached as diagnostics[].details.kind",
        ),
        UsageLimitRule(
            where="stdout",
            pattern=r"provider rate limit exceeded",
            citation="the rendered errorMessage for the rate_limit kind",
        ),
        UsageLimitRule(
            where="stderr",
            pattern=r"provider rate limit exceeded",
            citation="stderr fallback",
        ),
    )

    # Reaching one of these ended the run on a host policy, not on the agent's judgment.
    LIMIT_MARKERS = (
        "autonomous run stopped before terminal evidence",
        "maxcontinuations",
        "maxturns",
        "maxtokens",
        "timeoutms",
    )

    MAX_CONTINUATIONS = 100000
    MAX_TURNS = 100000
    MAX_TOKENS = 1000000000

    # Which provider serves each model, passed as an explicit ``--provider`` on every launch
    # and probe. Left to resolve a bare model id on its own, prime-agent 0.7.1 routed
    # ``gpt-5.6-terra`` to azure-openai-responses, a provider nothing here is logged into, and
    # the run died with "No API key found" while the openai-codex login sat unused. The map is
    # exact rather than by prefix: an explicit provider disables 0.7.1's catalog check (an
    # absent id falls back to a custom-model launch), so a prefix rule would silently misroute
    # any GPT id the codex backend does not serve. Every model a cell may name is listed here
    # against the provider whose catalog carries it, and anything else stops the launch.
    PROVIDER_BY_MODEL = {
        "claude-opus-5": "anthropic",
        "gpt-5.6-terra": "openai-codex",
    }

    @classmethod
    def provider_for(cls, model: str) -> str:
        provider = cls.PROVIDER_BY_MODEL.get(model)
        if provider is None:
            raise ValueError(
                f"no provider mapping for model {model!r}; add it to "
                "PrimeAgent.PROVIDER_BY_MODEL against the provider whose catalog serves it"
            )
        return provider

    # prime-agent's MCP client resolves a bearer token before every connection and refuses to
    # open a session without one, even against a server that ignores it. The value is a
    # formality; its absence is a silent no-tools run.
    MCP_TOKEN_VAR = "SHOBENCH_MCP_TOKEN"

    # The one HOME file the runner rewrites every leg, because the endpoint is per leg: the
    # rollout and each concurrent eval task serve on their own port. Excluded from the durable
    # digest for that reason, since a file the runner overwrites on every launch cannot carry
    # anything the agent chose to keep.
    runner_owned_home_files = (".prime/agent/settings.json",)

    # The long prompt cache, which prime-agent asks for only when told to. Verified in the pinned
    # 0.7.1 bundle inside the agent image: `resolveCacheRetention` takes an explicit option, then
    # `PI_CACHE_RETENTION === "long"`, and otherwise returns "short"; "long" is what makes
    # `getCacheControl` attach a 1h ttl on the anthropic path (24h on the openai-responses one)
    # where the model supports it, and nothing in the CLI ever passes the option itself.
    #
    # What it buys is cross-harness parity in cost, not behavior. Claude Code already runs with 1h
    # retention, so a prime cell measured against it was paying to rewrite a cache every five
    # minutes for the same conversation: an asymmetry between two cells of the same matrix with no
    # scientific content, since a cache hit and a cache miss return the same tokens. It changes
    # nothing the agent sees, decides, or can act on.
    CACHE_RETENTION_VAR = "PI_CACHE_RETENTION"

    def base_env(self) -> dict[str, str]:
        return {
            **super().base_env(),
            self.MCP_TOKEN_VAR: "local",
            self.CACHE_RETENTION_VAR: "long",
        }

    def home_seed_files(self) -> dict[str, str]:
        return shogym_stream_skill_files()

    # Its assistant messages carry the provider and the model, so the trace answers "which model
    # answered" directly.
    reports_observed_models = True

    def session_id_from_trace(self, trace_path: Path) -> str | None:
        # prime-agent's first line is its session header, which carries the id.
        header = _first_event_of_type(trace_path, ("session",))
        return None if header is None else str(header.get("id") or "") or None

    # Where every leg runs: the containers module mounts the task workdir at /work and starts
    # the harness there, for the rollout and for every eval fork alike. The cwd is a value
    # domain of the resume, not bookkeeping: the resolver treats a session recorded at any
    # other cwd (or at none) as another project's, answers "Session found in different
    # project", and stalls on an interactive "Fork this session into current directory?"
    # prompt, which under the runner's closed stdin is exit 13 and no resume (observed).
    LEG_CWD = "/work"

    def session_transcript(self, home: Path, session_id: str) -> Path | None:
        """The one session file whose FIRST parseable line is a header naming this id,
        recorded at the cwd the fork resumes in, whatever the file is called.

        The filename is NOT the identity, and requiring it broke real runs. The daemon mints
        a session file under one id and the print run's header carries another, rewritten
        into that same file: observed on a CLI-written session (a failed-auth run persisted a
        file whose header id differed from its filename), and on both real prime rollouts,
        whose recorded terminal id sits inside a file named for a different id. The CLI's
        resolver never looks at the name: it indexes saved sessions by the header id
        (source: ``scanSessionInfo`` walks ``readdir`` of the flat sessions dir), a file
        named for one id whose header names another is "No session found matching" for the
        FILENAME's id (observed), and resuming the HEADER id out of a differently named file
        resolves and appends to that file (observed). So this scans the flat sessions dir for
        header matches, and requires exactly one: the resolver refuses an ambiguous selector
        (source: ``resolveUniqueMatch`` raises on more than one match), so two files carrying
        the same header id are a refusal here too.

        The rest of the floor is unchanged and still the CLI's own: the header must be the
        first parseable line (a message line above it is "No session found matching",
        observed), must name this id, and must be recorded at the resuming cwd; recorded
        elsewhere or nowhere, the run dies on the interactive fork prompt (observed for an
        absent cwd and for a different one). A header alone is the whole floor: such a
        one-line file, timestamp absent and all, resumed to the auth boundary (observed,
        network off). An empty file has no header and is not found either.
        """
        sessions = home / ".prime" / "agent" / "sessions"
        if not sessions.is_dir():
            return None
        matches = []
        for path in sorted(sessions.glob("*.jsonl")):
            if not path.is_file():
                continue
            header = _first_parseable_event(path)
            if (
                header is not None
                and header.get("type") == "session"
                and str(header.get("id") or "") == session_id
                and header.get("cwd") == self.LEG_CWD
            ):
                matches.append(path)
        return matches[0] if len(matches) == 1 else None

    def observed_models(self, trace_path: Path) -> list[str]:
        """Which model actually answered, off prime-agent's own assistant messages.

        Its assistant message carries three related fields: ``provider``, ``model`` (what was
        asked for) and an optional ``responseModel`` (what the provider says replied). The last
        is preferred where present, because a provider that silently substitutes a model is
        exactly the case the manifest is supposed to catch, and the requested model is already
        recorded from the cell file.

        Both event shapes are read. ``message_end`` fires per message and is what a live run
        emits; ``agent_end`` carries the whole message list and is what survives when the stream
        was cut before every message_end landed. A leg that produced neither reports nothing.
        """
        seen: set[str] = set()
        for event in jsonl_events(trace_path, limit=4000):
            messages = (
                event.get("messages")
                if event.get("type") == "agent_end"
                else [event.get("message")]
                if event.get("type") in ("message_end", "message_start")
                else []
            )
            for message in messages or ():
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                model = message.get("responseModel") or message.get("model")
                if model:
                    seen.add(str(model))
        return sorted(seen)

    def version_probe(self) -> list[str]:
        return ["prime-agent", "--version"]

    def model_probe(self) -> list[str] | None:
        # The scope requires the manifest to record which model actually answered. prime-agent
        # has no operational history in this program, so it is asked rather than assumed.
        return ["prime-agent", "model", "list"]

    def launch(
        self,
        *,
        mcp_url: str,
        system_prompt: str,
        user_prompt: str,
        model: str,
        trace_path: Path,
        session_id: str | None = None,
        resume: bool = False,
        leg_timeout_s: int = 3600,
        effort: str = "",
    ) -> LaunchSpec:
        # prime-agent exposes no reasoning-effort control today; the parameter is accepted for
        # interface parity and ignored, and the cell's manifest still records the intent.
        prompt = f"{system_prompt}\n\n{user_prompt}"
        argv = [
            "prime-agent",
            "-p",
            "--mode",
            "json",
            "--provider",
            self.provider_for(model),
            "--model",
            model,
            "--autonomous",
            "--autonomous-max-continuations",
            str(self.MAX_CONTINUATIONS),
            "--autonomous-max-turns",
            str(self.MAX_TURNS),
            "--autonomous-max-tokens",
            str(self.MAX_TOKENS),
            "--autonomous-timeout-ms",
            str(leg_timeout_s * 1000 * 2),
        ]
        if resume and session_id:
            argv += ["--resume", session_id]
        elif resume:
            argv += ["--continue"]
        # In print mode a resume and its prompt are separated by --, so the id is never read
        # as part of the prompt.
        argv += ["--", prompt]
        # Global settings live in the isolated HOME, so the endpoint is configured where every
        # session in this cell sees it and nowhere else. Only http servers are honored; a stdio
        # entry is dropped without an error. This is per-leg because the endpoint is: the
        # rollout and each concurrent eval task serve on their own port, and an eval task's HOME
        # is a copy of the rollout's, so its inherited url points at a server that is gone.
        home_files = {
            self.runner_owned_home_files[0]: json.dumps(
                {
                    "mcpServers": {
                        "shogym": {
                            "type": "http",
                            "url": mcp_url,
                            "bearerTokenEnvVar": self.MCP_TOKEN_VAR,
                        }
                    }
                },
                indent=2,
            )
            + "\n"
        }
        # The skill package rides in the same HOME, because the settings entry alone reaches
        # nothing: prime-agent's client is a kernel-side import, not a host-managed tool bridge.
        # It seeds rather than rewrites. The vendored bytes are only a starting point, the
        # rollout may improve them like any other durable artifact, and the eval that follows
        # has to read what the rollout left rather than what this file shipped.
        return LaunchSpec(
            argv=argv,
            env=self.base_env(),
            home_files=home_files,
            home_seed_files=self.home_seed_files(),
        )

    def classify(
        self, *, returncode: int, stdout_path: Path, stderr_path: Path, timed_out: bool
    ) -> StopVerdict:
        if timed_out:
            return self._timed_out_verdict()
        stop_reason = _prime_stop_reason(stdout_path)
        texts = {"stdout": tail(stdout_path, lines=300), "stderr": tail(stderr_path)}
        # As with Claude Code, only a turn that ended in an error can be a usage limit; a
        # clean turn's text is the agent's own and must not be pattern-matched for one.
        if stop_reason in (None, "error", "aborted"):
            limit = self._match_usage_limit(texts)
            if limit is not None:
                return limit

        stderr_text = texts["stderr"].lower()
        # The markers that fired, by name, out of the tuple above. They are literals in this
        # file, so naming them says exactly what the classification read without quoting a byte
        # of what the harness wrote around them.
        hits = [marker for marker in self.LIMIT_MARKERS if marker in stderr_text]
        if hits:
            return StopVerdict(
                StopKind.LEG_TIMEOUT,
                "an autonomous host limit ended the run, which is a cutoff and not a stop",
                {
                    "returncode": returncode,
                    "stderr": stderr_evidence(stderr_path, matched=hits),
                },
            )

        ended = _last_event_of_type(stdout_path, ("agent_end",))
        evidence = {
            "returncode": returncode,
            "stop_reason": stop_reason,
            "saw_agent_end": ended is not None,
            "stderr": stderr_evidence(stderr_path),
        }
        # The exit code is not usable here: every model-level failure sets exit 1 only in text
        # mode, so a json-mode run that errored still exits 0. The event stream is the record.
        if stop_reason in ("stop", "toolUse"):
            return StopVerdict(StopKind.CHOSEN, "the last message ended the turn", evidence)
        if stop_reason == "length":
            return StopVerdict(StopKind.CHOSEN, "the output cap ended the message", evidence)
        if stop_reason == "error":
            return StopVerdict(
                StopKind.ERROR, "the last message ended in a provider error", evidence
            )
        if stop_reason == "aborted":
            return StopVerdict(StopKind.ERROR, "the run was aborted", evidence)
        if ended is not None:
            return StopVerdict(
                StopKind.UNKNOWN, "agent_end carried no readable stop reason", evidence
            )
        return StopVerdict(StopKind.ERROR, "prime-agent emitted no agent_end event", evidence)


def _prime_stop_reason(path: Path) -> str | None:
    """The stop reason of prime-agent's last assistant message.

    Read off ``agent_end``'s message list when there is one, else off the last ``message_end``,
    because a run killed before agent_end still leaves the messages it did produce.
    """
    ended = _last_event_of_type(path, ("agent_end",))
    if ended is not None:
        for message in reversed(ended.get("messages") or []):
            if isinstance(message, dict) and message.get("role") == "assistant":
                reason = message.get("stopReason")
                if reason:
                    return str(reason)
    last = _last_event_of_type(path, ("message_end",))
    if last is not None:
        reason = (last.get("message") or {}).get("stopReason")
        if reason:
            return str(reason)
    return None
