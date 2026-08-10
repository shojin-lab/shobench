"""The three v0 harnesses.

Each class carries two things the runner needs and nothing else: the invocation that makes the
harness autonomous from its first turn, and the rule that says how a leg ended. Both are
sourced in ``docs/harness-autonomy.md``, which cites where every flag and every stop signal
came from. Constants that were established by running the CLI say so; constants taken from
documentation say that instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from shobench.harness import (
    BASE_ENV,
    Harness,
    LaunchSpec,
    StopKind,
    StopVerdict,
    UsageLimitRule,
    jsonl_events,
    tail,
)


class ClaudeCode(Harness):
    """Claude Code, non-interactive.

    Autonomy comes from ``-p`` (print mode, no TTY, no trust dialog) plus
    ``--permission-mode bypassPermissions``, which auto-approves every tool including writes
    to the agent's own ``~/.claude``. The container is the integrity boundary, so there is no
    allow-list, and per the scope there is no web denylist either: leakage is observed rather
    than gated.

    ``IS_SANDBOX=1`` is required because the container runs as root, where the CLI otherwise
    refuses bypassPermissions outright.

    The standing instruction goes in ``--append-system-prompt`` rather than the user turn: a
    long rollout auto-compacts, and anything in the user turn can be summarized away, silently
    dropping the objective mid-run. The system prompt survives compaction.
    """

    name = "claude_code"
    # Claude Code accepts --session-id, so the runner pins it before launch and an interrupted
    # leg is resumable even if it died before writing anything.
    pins_session_id = True

    # Established by running the CLI: a clean finish is is_error=false,
    # terminal_reason="completed", api_error_status=null; a bad token gives is_error=true,
    # terminal_reason="api_error", api_error_status=401. Note subtype stays "success" in BOTH,
    # so subtype is never an error discriminator here.
    usage_limit_rules = (
        UsageLimitRule(
            where="result_json",
            pattern=r'"api_error_status"\s*:\s*429',
            citation="observed: api_error_status carries the HTTP status (401 on a bad token)",
        ),
        UsageLimitRule(
            where="result_text",
            pattern=r"you'?ve hit your .{0,40}limit",
            citation="the CLI's own limit message family: session, weekly, Opus, usage credit",
        ),
    )

    # terminal_reason values that mean a bounded stop rather than a failure: the turn ended
    # because something the runner or the harness set said so, and resuming is correct.
    BOUNDED_STOPS = frozenset(
        {"max_turns", "budget_exhausted", "tool_deferred", "background_requested"}
    )
    # Values that are a context problem, not a quota problem. Treating blocking_limit as a
    # usage limit would be wrong: it is the prompt token limit, and resuming would loop.
    CONTEXT_ERRORS = frozenset({"blocking_limit", "prompt_too_long"})

    def base_env(self) -> dict[str, str]:
        # IS_SANDBOX=1 is what lets bypassPermissions run as root, which is what the container
        # is. Without it the CLI refuses before it ever reaches a credential.
        return {**super().base_env(), "IS_SANDBOX": "1"}

    def version_probe(self) -> list[str]:
        return ["claude", "--version"]

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
    ) -> LaunchSpec:
        argv = [
            "claude",
            "-p",
            user_prompt,
            "--model",
            model,
            "--mcp-config",
            "/cfg/claude.mcp.json",
            "--strict-mcp-config",
            # Only the flags this run sets may configure it. Without this, a settings file the
            # image or the home happens to carry changes the initial conditions invisibly.
            "--setting-sources",
            "",
            "--permission-mode",
            "bypassPermissions",
            "--append-system-prompt",
            system_prompt,
            "--forward-subagent-text",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
        ]
        if resume and session_id:
            argv += ["--resume", session_id]
        elif resume:
            argv += ["--continue"]
        elif session_id:
            argv += ["--session-id", session_id]
        return LaunchSpec(
            argv=argv,
            env=self.base_env(),
            config_files={
                "claude.mcp.json": json.dumps(
                    {"mcpServers": {"shogym": {"type": "http", "url": mcp_url}}}, indent=2
                )
                + "\n"
            },
        )

    def session_id_from_trace(self, trace_path: Path) -> str | None:
        for event in jsonl_events(trace_path, limit=50):
            if event.get("type") == "system" and event.get("session_id"):
                return str(event["session_id"])
        return None

    def observed_models(self, trace_path: Path) -> list[str]:
        # The result event's modelUsage is keyed by the models that were actually billed,
        # which includes the small model the harness uses for its own bookkeeping. Reporting
        # both is the honest answer to "which model answered".
        seen: set[str] = set()
        for event in jsonl_events(trace_path, limit=4000):
            if event.get("type") == "result":
                seen.update((event.get("modelUsage") or {}).keys())
        return sorted(seen)

    def classify(
        self, *, returncode: int, stdout_path: Path, stderr_path: Path, timed_out: bool
    ) -> StopVerdict:
        if timed_out:
            return self._timed_out_verdict()
        result = _last_result_event(stdout_path)
        # The limit rules only apply to a turn that actually failed. A clean turn's `result`
        # is the agent's own text, and an agent that happened to write about hitting a limit
        # would otherwise be recorded as having hit one.
        if result is not None and result.get("is_error"):
            limit = self._match_usage_limit(
                {
                    "result_json": json.dumps(result),
                    "result_text": str(result.get("result", "")),
                }
            )
            if limit is not None:
                return limit
        if result is None:
            return StopVerdict(
                StopKind.ERROR,
                "no result event in the stream trace",
                {"returncode": returncode, "stderr_tail": tail(stderr_path)[-2000:]},
            )
        evidence = {
            "returncode": returncode,
            "stderr_tail": tail(stderr_path)[-2000:],
            "subtype": result.get("subtype"),
            "is_error": result.get("is_error"),
            "stop_reason": result.get("stop_reason"),
            "terminal_reason": result.get("terminal_reason"),
            "api_error_status": result.get("api_error_status"),
            "num_turns": result.get("num_turns"),
        }
        reason = str(result.get("terminal_reason") or "")
        if returncode == 0 and not result.get("is_error"):
            return StopVerdict(StopKind.CHOSEN, "the session ended its turn cleanly", evidence)
        if reason in self.BOUNDED_STOPS:
            return StopVerdict(
                StopKind.CHOSEN, f"the turn ended at a bounded limit ({reason})", evidence
            )
        if reason in self.CONTEXT_ERRORS:
            return StopVerdict(
                StopKind.ERROR,
                f"the context limit ended the turn ({reason}), which resuming would not fix",
                evidence,
            )
        return StopVerdict(StopKind.ERROR, "the session ended with an error result", evidence)


class Codex(Harness):
    """OpenAI Codex CLI, non-interactive.

    Autonomy comes from the ``exec`` subcommand. Two of its defaults have to be overridden or
    the cell is not autonomous: ``codex exec`` sandboxes to read-only, so the agent cannot
    write without ``--dangerously-bypass-approvals-and-sandbox``, and the MCP server is
    optional by default, so a broker that fails to start leaves codex running toolless in
    silence. ``required = true`` turns that silent failure into an exit. Approvals need no
    flag: exec already defaults to never asking, and ``-a/--ask-for-approval`` is not an exec
    flag at all.

    Prior runs found codex unreliable over one long loop, which the runner already handles: a
    rollout is a sequence of bounded legs against one live stream, so codex is supervised
    episodically by the same mechanism every harness uses rather than by a special case.

    stdin is closed because codex reads it to end whenever it is not a TTY, prompt argument or
    not, so an inherited stdin hangs the process before the first turn.
    """

    name = "codex"
    # No base_env override: codex adds nothing to the base's clean NODE_OPTIONS.

    # codex exec ends a turn with a terminal `turn.completed` or `turn.failed` JSONL event. A
    # usage limit arrives as turn.failed carrying the usage_limit_reached error type, whose
    # message also names the reset time.
    # Read only from the TERMINAL turn.failed event. Mid-stream {"type":"error"} events are
    # retry chatter: codex retries transient failures internally and only a non-retryable one
    # is fatal, so a rule that matched any error event would false-positive constantly.
    usage_limit_rules = (
        UsageLimitRule(
            where="turn_failed",
            pattern=r"you'?ve hit your usage limit",
            citation="Display text of CodexErr::UsageLimitReached",
        ),
        UsageLimitRule(
            where="turn_failed",
            pattern=r"out of credits|spend cap|exceeded retry limit, last status: 429",
            citation="the other CodexErr::UsageLimitReached message arms",
        ),
        UsageLimitRule(
            where="turn_failed",
            pattern=r"^rate limit: ",
            citation="the rate-limit error prefix",
        ),
    )

    def session_id_from_trace(self, trace_path: Path) -> str | None:
        # codex announces its thread id as the first event and takes no id from the caller.
        started = _first_event_of_type(trace_path, ("thread.started",))
        return None if started is None else str(started.get("thread_id") or "") or None

    def version_probe(self) -> list[str]:
        return ["codex", "--version"]

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
    ) -> LaunchSpec:
        # codex exec has no separate system-prompt channel, so the standing instruction is
        # prepended to the turn. The bytes are the same as every other harness's system prompt
        # and the manifest records the digest, so the difference in placement stays visible.
        prompt = f"{system_prompt}\n\n{user_prompt}"
        # `codex exec resume <id>` takes the subcommand before the flags, and accepts neither
        # -s/--sandbox nor -C/--cd, which is why the sandbox is opened with the bypass flag
        # rather than with --sandbox: the bypass flag is accepted by both forms.
        argv = ["codex", "exec"]
        if resume and session_id:
            argv += ["resume", session_id]
        argv += [
            "--json",
            "-m",
            model,
            # Full autonomy: exec's default sandbox is read-only, which would leave the agent
            # unable to write anything durable about itself.
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "-c",
            f'mcp_servers.shogym.url="{mcp_url}"',
            "-c",
            'mcp_servers.shogym.default_tools_approval_mode="approve"',
            # Fail loudly rather than run toolless if the broker is not up.
            "-c",
            "mcp_servers.shogym.required=true",
            # The defaults are 10s startup and 60s per tool, and one get_task can pay for a
            # cold env and a first dataset load.
            "-c",
            "mcp_servers.shogym.startup_timeout_sec=60",
            "-c",
            "mcp_servers.shogym.tool_timeout_sec=900",
            # A container has no OS keyring, so credentials have to resolve from the file.
            "-c",
            'cli_auth_credentials_store="file"',
            prompt,
        ]
        return LaunchSpec(argv=argv, env=self.base_env())

    def classify(
        self, *, returncode: int, stdout_path: Path, stderr_path: Path, timed_out: bool
    ) -> StopVerdict:
        if timed_out:
            return self._timed_out_verdict()
        terminal = _last_event_of_type(stdout_path, ("turn.completed", "turn.failed"))
        failure = ""
        if terminal is not None and terminal.get("type") == "turn.failed":
            failure = str((terminal.get("error") or {}).get("message", ""))
        limit = self._match_usage_limit({"turn_failed": failure})
        if limit is not None:
            return limit
        evidence = {
            "returncode": returncode,
            "terminal_event": None if terminal is None else terminal.get("type"),
            "error_message": failure,
            "stderr_tail": tail(stderr_path)[-2000:],
        }
        if terminal is not None and terminal.get("type") == "turn.completed" and returncode == 0:
            return StopVerdict(StopKind.CHOSEN, "codex exec completed its turn", evidence)
        if terminal is None:
            # An interrupted turn emits neither terminal event and still exits 1, so a missing
            # terminal event is an interruption rather than a stop.
            return StopVerdict(
                StopKind.ERROR, "codex exec emitted no terminal turn event", evidence
            )
        return StopVerdict(StopKind.ERROR, "codex exec reported a failed turn", evidence)


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
    ) -> LaunchSpec:
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
        return LaunchSpec(
            argv=argv,
            env=self.base_env(),
            # Global settings live in the isolated HOME, so the endpoint is configured where
            # every session in this cell sees it and nowhere else. Only http servers are
            # honored; a stdio entry is dropped without an error.
            home_files={
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
            },
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


def _last_result_event(path: Path) -> dict | None:
    """The final ``type: result`` event of a stream-json trace."""
    return _last_event_of_type(path, ("result",))


def _first_event_of_type(path: Path, types: tuple[str, ...]) -> dict | None:
    """The first event of a JSONL trace whose ``type`` is one of ``types``."""
    for event in jsonl_events(path, limit=50):
        if event.get("type") in types:
            return event
    return None


def _last_event_of_type(path: Path, types: tuple[str, ...]) -> dict | None:
    """The last event of a JSONL trace whose ``type`` is one of ``types``."""
    for event in reversed(jsonl_events(path)):
        if event.get("type") in types:
            return event
    return None


_REGISTRY = {h.name: h for h in (ClaudeCode(), Codex(), PrimeAgent())}


def harness_for(name: str) -> Harness:
    if name not in _REGISTRY:
        raise ValueError(f"unknown harness {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


__all__ = ["BASE_ENV", "ClaudeCode", "Codex", "PrimeAgent", "harness_for"]
