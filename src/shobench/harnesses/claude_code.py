"""Claude Code, the harness. The flags and stop signals are sourced in
``docs/harness-autonomy.md``, which cites where every one came from.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from shobench.harness import (
    Harness,
    LaunchSpec,
    StopKind,
    StopVerdict,
    UsageLimitRule,
    jsonl_events,
    stderr_evidence,
)
from shobench.harnesses._trace import _last_event_of_type, _strict_json_object


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
    # Where its recorded conversations live: --resume resolves the id against the transcripts
    # under projects/<cwd-slug>/<id>.jsonl in the HOME it runs with, fails loudly when none
    # matches, and appends the resumed turn to that same file. Every leg runs at cwd /work, so
    # a transcript recorded by the rollout is under the slug an eval fork resolves.
    session_state_dirs = (".claude/projects",)
    # The result event's modelUsage names every model that was billed.
    reports_observed_models = True
    effort_flag = "--effort"

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
        effort: str = "",
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
        # Pin reasoning effort only when the cell asks for it; otherwise the CLI default stands.
        if effort:
            argv += ["--effort", effort]
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

    def session_transcript(self, home: Path, session_id: str) -> Path | None:
        """The exactly-named project transcript, and it must hold a conversation the CLI can
        replay, not merely a line that names the session.

        The CLI resolves ``--resume <id>`` to ``projects/<slug>/<id>.jsonl`` and then demands
        an actual conversation record. The floor was established by minimizing a transcript
        the pinned CLI itself wrote and re-resuming after each cut (network off, zero
        tokens): a ``user`` line with a ``message`` carrying a role and non-empty content,
        a ``timestamp``, and the ``sessionId``. Every cut below that flips the CLI to a
        refusal: no ``message`` or no ``timestamp`` is "No conversation found with session
        ID" exactly as if the file were absent, and a ``message`` without content crashes
        the resume outright ("Failed to resume session"). An id in a line is not a
        conversation; requiring the whole floor also rejects a file recording some other
        session under this file name.
        """
        root = home / ".claude" / "projects"
        if not root.is_dir():
            return None
        for project in sorted(p for p in root.iterdir() if p.is_dir()):
            path = project / f"{session_id}.jsonl"
            if path.is_file() and _carries_resumable_conversation(path, session_id):
                return path
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
                {"returncode": returncode, "stderr": stderr_evidence(stderr_path)},
            )
        evidence = {
            "returncode": returncode,
            "stderr": stderr_evidence(stderr_path),
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


def _is_zoned_instant(value: object) -> bool:
    """Is this the complete, calendar-valid, timezone-carrying instant the CLI writes?

    A prefix pattern is not this check: it certified "2026-99-99T99:99:99Z", which the pinned
    CLI refuses as "No conversation found" (observed), exactly like an object, an epoch
    number, and a non-date string in the field. So the value is parsed as a datetime,
    end-anchored (a trailing remainder fails the parse), calendar-checked by the parse
    itself, and required to carry a timezone the way every CLI-written stamp does. The
    timezone requirement is a deliberate narrowing: a naive stamp was not probed, and no real
    transcript carries one, so refusing it costs nothing and claims nothing.
    """
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _carries_resumable_conversation(path: Path, session_id: str) -> bool:
    """Does any line of this transcript hold the conversation floor the pinned CLI requires?

    The qualifying line is the shape a real transcript's kickoff turn always has and the
    minimum the CLI was observed to accept: a ``user`` entry naming this ``sessionId``, with
    an ISO-instant ``timestamp`` (see :func:`_is_zoned_instant`) and a ``message`` carrying a
    role and non-empty content that is a string or a list. The value domains matter as much
    as the fields: a truthy timestamp of the wrong shape is refused as "No conversation
    found", and a truthy content of the wrong shape resolves and then crashes the resume
    ("e.map is not a function", whose own text shows the CLI's domain: a string is handled
    apart and anything else is mapped as an array). A list's elements are not constrained
    here because the CLI constrains none: a list of non-blocks was observed to resume, its
    entries contributing empty text. Lines are decoded in the shared strict dialect
    (``_strict_json_object``), because the CLI's own JSON.parse never yields a line carrying
    a NaN-family constant, so a qualifying line that needs one does not exist for it.

    Streamed and stopped at the first match, tolerant of a malformed line, because the file
    was written by another process and a crash can cut it anywhere; a transcript cut after
    its kickoff line is still one the CLI resumes.
    """
    try:
        with path.open(encoding="utf-8", errors="ignore") as lines:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                event = _strict_json_object(line)
                if event is None or event.get("sessionId") != session_id:
                    continue
                if event.get("type") != "user":
                    continue
                if not _is_zoned_instant(event.get("timestamp")):
                    continue
                message = event.get("message")
                if not isinstance(message, dict) or not message.get("role"):
                    continue
                content = message.get("content")
                if isinstance(content, str | list) and content:
                    return True
    except OSError:
        return False
    return False
