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

    Autonomy comes from the ``exec`` subcommand plus full-auto approval and sandbox settings.
    Prior runs found codex unreliable over one long loop, which the runner already handles: the
    rollout is a sequence of bounded legs against one live stream, so codex is supervised
    episodically without a special case.

    stdin is closed, because codex hangs on an open one.
    """

    name = "codex"

    usage_limit_rules = (
        UsageLimitRule(
            where="stdout",
            pattern=r"usage limit|rate limit|quota|429|resets? (at|in)",
            citation="see docs/harness-autonomy.md",
        ),
        UsageLimitRule(
            where="stderr",
            pattern=r"usage limit|rate limit|quota|429|resets? (at|in)",
            citation="see docs/harness-autonomy.md",
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
    ) -> LaunchSpec:
        # codex has no separate system-prompt channel in exec mode, so the standing instruction
        # is prepended to the turn. It is the same bytes as every other harness's system prompt;
        # the manifest records the digest so the difference in placement is visible.
        prompt = f"{system_prompt}\n\n{user_prompt}"
        argv = [
            "codex",
            "exec",
            "--json",
            "-m",
            model,
            "--dangerously-bypass-approvals-and-sandbox",
            "-c",
            f'mcp_servers.shogym.url="{mcp_url}"',
            "-c",
            'mcp_servers.shogym.default_tools_approval_mode="approve"',
            "-c",
            "mcp_servers.shogym.startup_timeout_sec=60",
            "-c",
            "mcp_servers.shogym.tool_timeout_sec=900",
        ]
        if resume and session_id:
            argv += ["resume", session_id]
        argv += [prompt]
        return LaunchSpec(argv=argv, env=dict(BASE_ENV))

    def classify(
        self, *, returncode: int, stdout_path: Path, stderr_path: Path, timed_out: bool
    ) -> StopVerdict:
        if timed_out:
            return StopVerdict(StopKind.LEG_TIMEOUT, "the runner ended the leg at its budget")
        texts = {"stdout": tail(stdout_path), "stderr": tail(stderr_path)}
        limit = self._match_usage_limit(texts)
        if limit is not None:
            return limit
        evidence = {"returncode": returncode, "stderr_tail": texts["stderr"][-2000:]}
        if returncode == 0:
            return StopVerdict(StopKind.CHOSEN, "codex exec exited cleanly", evidence)
        return StopVerdict(StopKind.ERROR, f"codex exec exited {returncode}", evidence)


class PrimeAgent(Harness):
    """Prime Intellect's prime-agent, non-interactive.

    prime-agent installs from the vendor script, never from npm: the npm name in its source
    tree installs Pi instead. It has no operational history in this program, so the manifest
    records ``prime-agent model list`` per cell and the cell does not start until the resolved
    model matches what the config asked for.

    stdin is closed, because prime-agent hangs on an open one.
    """

    name = "prime_agent"

    usage_limit_rules = (
        UsageLimitRule(
            where="stdout",
            pattern=r"usage limit|rate limit|quota|429|insufficient",
            citation="see docs/harness-autonomy.md",
        ),
        UsageLimitRule(
            where="stderr",
            pattern=r"usage limit|rate limit|quota|429|insufficient",
            citation="see docs/harness-autonomy.md",
        ),
    )

    def version_probe(self) -> list[str]:
        return ["prime-agent", "--version"]

    def model_probe(self) -> list[str] | None:
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
    ) -> LaunchSpec:
        prompt = f"{system_prompt}\n\n{user_prompt}"
        argv = [
            "prime-agent",
            "-p",
            "--mode",
            "json",
            prompt,
            "--model",
            model,
            # Autonomy: keep working past the first response rather than yielding to a human.
            "--autonomous",
            "--autonomous-max-continuations",
            "1000",
        ]
        return LaunchSpec(
            argv=argv,
            env=dict(BASE_ENV),
            config_files={
                "prime.settings.json": json.dumps(
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
        texts = {"stdout": tail(stdout_path), "stderr": tail(stderr_path)}
        limit = self._match_usage_limit(texts)
        if limit is not None:
            return limit
        evidence = {"returncode": returncode, "stderr_tail": texts["stderr"][-2000:]}
        if returncode == 0:
            return StopVerdict(StopKind.CHOSEN, "prime-agent exited cleanly", evidence)
        return StopVerdict(StopKind.ERROR, f"prime-agent exited {returncode}", evidence)


def _last_result_event(path: Path) -> dict | None:
    """The final ``type: result`` event of a stream-json trace."""
    for event in reversed(jsonl_events(path)):
        if event.get("type") == "result":
            return event
    return None


_REGISTRY = {h.name: h for h in (ClaudeCode(), Codex(), PrimeAgent())}


def harness_for(name: str) -> Harness:
    if name not in _REGISTRY:
        raise ValueError(f"unknown harness {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


__all__ = ["BASE_ENV", "ClaudeCode", "Codex", "PrimeAgent", "harness_for"]
