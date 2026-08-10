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
    Harness,
    LaunchSpec,
    StopKind,
    StopVerdict,
    UsageLimitRule,
    jsonl_events,
    tail,
)

# Every harness runs with a clean NODE_OPTIONS. An inherited one (a debugger port, a loader
# hook) reaches the harness's own Node runtime and has broken launches before.
BASE_ENV = {"NODE_OPTIONS": ""}


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

    # Established by running `claude -p --output-format json` (2.1.221): a clean finish is
    # is_error=false, subtype="success", terminal_reason="completed", api_error_status=null.
    usage_limit_rules = (
        UsageLimitRule(
            where="result_json",
            pattern=r'"api_error_status"\s*:\s*429',
            citation="result object field observed in `claude -p --output-format json` output",
        ),
        UsageLimitRule(
            where="result_json",
            pattern=r"rate[_ ]?limit",
            citation="`rate_limit` literal present in the 2.1.221 binary",
        ),
        UsageLimitRule(
            where="stderr",
            pattern=r"usage limit reached|limit will reset|rate.?limit|429",
            citation="stderr fallback for a limit reported before the result event",
        ),
    )

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
            env={
                **BASE_ENV,
                # The container runs as root; without this the CLI refuses bypassPermissions.
                "IS_SANDBOX": "1",
            },
            config_files={
                "claude.mcp.json": json.dumps(
                    {"mcpServers": {"shogym": {"type": "http", "url": mcp_url}}}, indent=2
                )
                + "\n"
            },
        )

    def classify(
        self, *, returncode: int, stdout_path: Path, stderr_path: Path, timed_out: bool
    ) -> StopVerdict:
        if timed_out:
            return StopVerdict(StopKind.LEG_TIMEOUT, "the runner ended the leg at its budget")
        result = _last_result_event(stdout_path)
        texts = {
            "result_json": json.dumps(result) if result else "",
            "stderr": tail(stderr_path),
        }
        limit = self._match_usage_limit(texts)
        if limit is not None:
            return limit
        if result is None:
            return StopVerdict(
                StopKind.ERROR,
                "no result event in the stream trace",
                {"returncode": returncode, "stderr_tail": texts["stderr"][-2000:]},
            )
        evidence = {
            "returncode": returncode,
            "subtype": result.get("subtype"),
            "is_error": result.get("is_error"),
            "stop_reason": result.get("stop_reason"),
            "terminal_reason": result.get("terminal_reason"),
            "api_error_status": result.get("api_error_status"),
            "num_turns": result.get("num_turns"),
        }
        if returncode == 0 and not result.get("is_error"):
            return StopVerdict(StopKind.CHOSEN, "the session ended its turn cleanly", evidence)
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

    # codex exec ends a turn with a terminal `turn.completed` or `turn.failed` JSONL event. A
    # usage limit arrives as turn.failed carrying the usage_limit_reached error type, whose
    # message also names the reset time.
    usage_limit_rules = (
        UsageLimitRule(
            where="stdout",
            pattern=r'"error_type"\s*:\s*"usage_limit_reached"',
            citation="CodexErr::UsageLimitReached, routed into turn.failed.error",
        ),
        UsageLimitRule(
            where="stdout",
            pattern=r"you'?ve hit your usage limit|usage limit reached",
            citation="Display text of CodexErr::UsageLimitReached",
        ),
        UsageLimitRule(
            where="stderr",
            pattern=r"you'?ve hit your usage limit|usage limit reached|429",
            citation="stderr fallback for a limit reported outside the event stream",
        ),
    )

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
        argv = ["codex", "exec", "--json", "-m", model]
        if resume and session_id:
            argv += ["resume", session_id]
        argv += [
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
        return LaunchSpec(argv=argv, env=dict(BASE_ENV))

    def classify(
        self, *, returncode: int, stdout_path: Path, stderr_path: Path, timed_out: bool
    ) -> StopVerdict:
        if timed_out:
            return StopVerdict(StopKind.LEG_TIMEOUT, "the runner ended the leg at its budget")
        texts = {"stdout": tail(stdout_path, lines=200), "stderr": tail(stderr_path)}
        limit = self._match_usage_limit(texts)
        if limit is not None:
            return limit
        terminal = _last_event_of_type(stdout_path, ("turn.completed", "turn.failed"))
        evidence = {
            "returncode": returncode,
            "terminal_event": None if terminal is None else terminal.get("type"),
            "stderr_tail": texts["stderr"][-2000:],
        }
        if terminal is not None and terminal.get("type") == "turn.completed" and returncode == 0:
            return StopVerdict(StopKind.CHOSEN, "codex exec completed its turn", evidence)
        if terminal is None:
            # An interrupted turn emits no terminal event and still exits 1, so a missing
            # terminal event is an interruption rather than a stop.
            return StopVerdict(
                StopKind.ERROR, "codex exec emitted no terminal turn event", evidence
            )
        evidence["error"] = terminal.get("error")
        return StopVerdict(StopKind.ERROR, "codex exec reported a failed turn", evidence)


class PrimeAgent(Harness):
    """Prime Intellect's prime-agent, non-interactive.

    prime-agent installs from its vendor script, never from npm: the ``prime-agent`` name does
    not exist on the npm registry, and the npm identity in its source tree installs Pi, a
    different agent. The image installs it from the vendor script accordingly.

    Autonomy needs more than a flag here. Autonomous mode starts disabled, and once enabled it
    carries five default budgets that are all far below an 8-hour rollout: 3 continuations, 12
    turns, 80,000 tokens, and 30 minutes. Leaving any of them at its default would end the
    rollout on an imposed cutoff and record it as the agent's own stop, which is exactly the
    confound the scope forbids. So the runner raises every budget past the cell's wall clock
    and treats a budget that was nonetheless reached as a cutoff, not a stop.

    Value-taking autonomous flags take a separate argument; ``--flag=value`` is rejected.

    stdin is closed, because print mode reads piped stdin and merges it into the prompt, so an
    inherited stdin hangs the process.
    """

    name = "prime_agent"

    usage_limit_rules = (
        UsageLimitRule(
            where="stdout",
            pattern=r'"kind"\s*:\s*"rate_limit"',
            citation="diagnostics[].details.kind on the last assistant message",
        ),
        UsageLimitRule(
            where="stdout",
            pattern=r"usage limit|rate.?limit|quota exceeded|429",
            citation="auto_retry_end.finalError text",
        ),
        UsageLimitRule(
            where="stderr",
            pattern=r"usage limit|rate.?limit|quota exceeded|429",
            citation="stderr fallback",
        ),
    )

    # The autonomous budgets, raised so far past any leg that reaching one is a real signal.
    # They are recorded in the manifest through the launch argv, so a reader can check that the
    # rollout was not silently truncated by a host policy.
    MAX_CONTINUATIONS = 100000
    MAX_TURNS = 100000
    MAX_TOKENS = 1000000000

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
        argv += [prompt]
        return LaunchSpec(
            argv=argv,
            env=dict(BASE_ENV),
            # Global settings live in the isolated HOME, so the MCP endpoint is configured
            # where every session in this cell sees it and nowhere else.
            home_files={
                ".prime/agent/settings.json": json.dumps(
                    {"mcpServers": {"shogym": {"type": "http", "url": mcp_url}}}, indent=2
                )
                + "\n"
            },
        )

    def classify(
        self, *, returncode: int, stdout_path: Path, stderr_path: Path, timed_out: bool
    ) -> StopVerdict:
        if timed_out:
            return StopVerdict(StopKind.LEG_TIMEOUT, "the runner ended the leg at its budget")
        texts = {"stdout": tail(stdout_path, lines=200), "stderr": tail(stderr_path)}
        limit = self._match_usage_limit(texts)
        if limit is not None:
            return limit
        ended = _last_event_of_type(stdout_path, ("agent_end",))
        evidence = {
            "returncode": returncode,
            "saw_agent_end": ended is not None,
            "stderr_tail": texts["stderr"][-2000:],
        }
        if returncode == 0 and ended is not None:
            return StopVerdict(StopKind.CHOSEN, "prime-agent ended its session", evidence)
        if returncode == 0:
            return StopVerdict(
                StopKind.UNKNOWN, "prime-agent exited 0 without an agent_end event", evidence
            )
        return StopVerdict(StopKind.ERROR, f"prime-agent exited {returncode}", evidence)


def _last_result_event(path: Path) -> dict | None:
    """The final ``type: result`` event of a stream-json trace."""
    return _last_event_of_type(path, ("result",))


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
