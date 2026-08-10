"""Prime Intellect's prime-agent, the harness. The flags and stop signals are sourced in
``docs/harness-autonomy.md``, which cites where every one came from.
"""

from __future__ import annotations

import json
from pathlib import Path

from shobench.harness import Harness, LaunchSpec, StopKind, StopVerdict, UsageLimitRule, tail
from shobench.harnesses._trace import _first_event_of_type, _last_event_of_type

# The prime-agent skill that makes the served stream reachable. Declaring the HTTP server in
# settings.json is only half of the wiring: prime-agent hands the model no MCP tools, it reaches
# a server by importing a Python-backed skill in its kernel and calling it, so the cell HOME
# must also carry the `shogym-stream` skill package. It is vendored under the repo's
# `prime_agent/skills/` and installed into the isolated HOME exactly as the settings file is.
SHOGYM_STREAM_SKILL = "shogym-stream"
_SKILL_HOME_PREFIX = ".prime/agent/skills"


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
    """
    root = _vendored_skill_dir()
    files: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        files[f"{_SKILL_HOME_PREFIX}/{SHOGYM_STREAM_SKILL}/{rel}"] = path.read_text(
            encoding="utf-8"
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

    # prime-agent's MCP client resolves a bearer token before every connection and refuses to
    # open a session without one, even against a server that ignores it. The value is a
    # formality; its absence is a silent no-tools run.
    MCP_TOKEN_VAR = "SHOBENCH_MCP_TOKEN"

    def base_env(self) -> dict[str, str]:
        return {**super().base_env(), self.MCP_TOKEN_VAR: "local"}

    def session_id_from_trace(self, trace_path: Path) -> str | None:
        # prime-agent's first line is its session header, which carries the id.
        header = _first_event_of_type(trace_path, ("session",))
        return None if header is None else str(header.get("id") or "") or None

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
        # entry is dropped without an error. The skill package rides in the same HOME: the
        # settings entry alone reaches nothing, because prime-agent's client is a kernel-side
        # import, not a host-managed tool bridge.
        home_files = {
            ".prime/agent/settings.json": json.dumps(
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
        home_files.update(shogym_stream_skill_files())
        return LaunchSpec(argv=argv, env=self.base_env(), home_files=home_files)

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
        if any(marker in stderr_text for marker in self.LIMIT_MARKERS):
            return StopVerdict(
                StopKind.LEG_TIMEOUT,
                "an autonomous host limit ended the run, which is a cutoff and not a stop",
                {"returncode": returncode, "stderr_tail": texts["stderr"][-2000:]},
            )

        ended = _last_event_of_type(stdout_path, ("agent_end",))
        evidence = {
            "returncode": returncode,
            "stop_reason": stop_reason,
            "saw_agent_end": ended is not None,
            "stderr_tail": texts["stderr"][-2000:],
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
