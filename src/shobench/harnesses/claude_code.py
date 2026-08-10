"""Claude Code, the harness. The flags and stop signals are sourced in
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
    tail,
)
from shobench.harnesses._trace import _last_event_of_type


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


def _last_result_event(path: Path) -> dict | None:
    """The final ``type: result`` event of a stream-json trace."""
    return _last_event_of_type(path, ("result",))
