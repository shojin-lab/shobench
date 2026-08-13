"""OpenAI Codex CLI, the harness. The flags and stop signals are sourced in
``docs/harness-autonomy.md``, which cites where every one came from.
"""

from __future__ import annotations

from pathlib import Path

from shobench.harness import (
    Harness,
    LaunchSpec,
    StopKind,
    StopVerdict,
    UsageLimitRule,
    stderr_evidence,
)
from shobench.harnesses._trace import (
    _first_event_of_type,
    _first_parseable_event,
    _last_event_of_type,
)


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

    # Where its recorded threads live: `codex exec resume <thread-id>` resolves the id against
    # the rollout files under sessions/YYYY/MM/DD/ in the HOME it runs with (the thread id is
    # embedded in the filename), errors with "no rollout found for thread id" when none
    # matches, and appends the resumed turn to the file it found.
    session_state_dirs = (".codex/sessions",)

    # codex exec's JSONL does not name a model anywhere, and this is declared rather than left
    # to be inferred from an empty list, because the two read as opposite things. Checked
    # against the pinned CLI rather than assumed: `thread.started` carries a thread id alone,
    # the terminal `turn.completed` carries only a token-usage breakdown, and the item types the
    # stream emits (agent_message, reasoning, command_execution, file_change, mcp_tool_call,
    # web_search, todo_list) carry none either. There is also no `codex exec` flag or subcommand
    # that reports the resolved model, so for a codex cell the requested model is all there is
    # and the manifest says so instead of publishing an empty observed list.
    reports_observed_models = False
    effort_flag = "-c model_reasoning_effort"

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

    def session_transcript(self, home: Path, session_id: str) -> Path | None:
        """The rollout file whose name ends in this thread id, opening with a meta the CLI's
        own parser decodes.

        The floor was established against the pinned CLI (network off, zero tokens). The
        rollout must START with the session metadata: a parseable non-meta first line is
        refused as "does not start with session metadata" even when a valid meta follows it.
        And the meta must decode whole: its payload needs ``id``, ``timestamp``, ``cwd``,
        ``originator``, and ``cli_version``, because dropping any of them is refused as
        "failed to read session metadata" and an empty file as "rollout ... is empty". No
        items after the meta are required; a meta-only rollout resumed to the transport
        boundary. Requiring the payload id to equal the thread id also rejects a rollout
        recorded for some other thread, and the exact filename suffix keeps a longer id that
        merely contains this one from standing in.
        """
        root = home / ".codex" / "sessions"
        if not root.is_dir():
            return None
        suffix = f"-{session_id}.jsonl"
        for path in sorted(root.rglob("*.jsonl")):
            if not path.is_file() or not path.name.endswith(suffix):
                continue
            meta = _first_parseable_event(path)
            if meta is None or meta.get("type") != "session_meta":
                continue
            payload = meta.get("payload") or {}
            if payload.get("id") != session_id:
                continue
            if all(
                isinstance(payload.get(key), str) and payload.get(key)
                for key in ("timestamp", "cwd", "originator", "cli_version")
            ):
                return path
        return None

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
        effort: str = "",
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
        ]
        # Pin reasoning effort only when the cell asks for it; otherwise codex's config default
        # (or its built-in default) stands.
        if effort:
            argv += ["-c", f'model_reasoning_effort="{effort}"']
        # The prompt is exec's positional argument and has to come last.
        argv.append(prompt)
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
            "stderr": stderr_evidence(stderr_path),
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
