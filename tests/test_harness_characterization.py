"""A byte-level pin on what each harness asks the runner to run.

The runner drives every harness through the same handful of methods: it asks for a launch
spec (argv, env, files, stdin), for the version and model probes, and for a verdict on how a
leg ended. This file records the exact answers the three v0 harnesses give today, so a refactor
that reshapes where the code lives can prove it did not reshape what the code does. Every value
here was captured from the harnesses before they were split into a package; if one of these
assertions changes, the launch or the classification changed with it, and that is a behavior
change to be justified rather than absorbed.

The stop-classification rules have their own scenario coverage in ``test_runner_units.py``.
What this file adds there is the full verdict string, not only its kind, because the reason is
part of what the results JSON records.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shobench.harness import StopKind
from shobench.harnesses import (
    ClaudeCode,
    Codex,
    PrimeAgent,
    harness_for,
    shogym_stream_skill_files,
)

# One fixed set of launch inputs, reused for every harness so the specs line up column for
# column. The port in the url is arbitrary; it only has to be stable across the assertions.
_LAUNCH = dict(
    mcp_url="http://host.docker.internal:8973/mcp",
    system_prompt="SYS",
    user_prompt="USR",
    model="the-model",
    trace_path=Path("/trace/leg.stream.jsonl"),
    leg_timeout_s=3600,
)

_CLAUDE_MCP_CONFIG = (
    "{\n"
    '  "mcpServers": {\n'
    '    "shogym": {\n'
    '      "type": "http",\n'
    '      "url": "http://host.docker.internal:8973/mcp"\n'
    "    }\n"
    "  }\n"
    "}\n"
)

_PRIME_SETTINGS = (
    "{\n"
    '  "mcpServers": {\n'
    '    "shogym": {\n'
    '      "type": "http",\n'
    '      "url": "http://host.docker.internal:8973/mcp",\n'
    '      "bearerTokenEnvVar": "SHOBENCH_MCP_TOKEN"\n'
    "    }\n"
    "  }\n"
    "}\n"
)


# ----- the registry resolves the names the cells use -----------------------------------------


def test_the_registry_maps_each_name_to_its_class() -> None:
    assert type(harness_for("claude_code")) is ClaudeCode
    assert type(harness_for("codex")) is Codex
    assert type(harness_for("prime_agent")) is PrimeAgent


def test_an_unknown_harness_name_is_refused_with_the_known_set() -> None:
    with pytest.raises(ValueError, match="unknown harness 'nope'"):
        harness_for("nope")


# ----- claude_code ---------------------------------------------------------------------------


def test_claude_code_probes_and_env_are_pinned() -> None:
    h = harness_for("claude_code")
    assert h.name == "claude_code"
    assert h.pins_session_id is True
    assert h.base_env() == {"NODE_OPTIONS": "", "IS_SANDBOX": "1"}
    assert h.version_probe() == ["claude", "--version"]
    assert h.model_probe() is None


def test_claude_code_fresh_launch_is_pinned() -> None:
    spec = harness_for("claude_code").launch(**_LAUNCH)
    assert spec.argv == [
        "claude",
        "-p",
        "USR",
        "--model",
        "the-model",
        "--mcp-config",
        "/cfg/claude.mcp.json",
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--permission-mode",
        "bypassPermissions",
        "--append-system-prompt",
        "SYS",
        "--forward-subagent-text",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
    ]
    assert spec.env == {"NODE_OPTIONS": "", "IS_SANDBOX": "1"}
    assert spec.config_files == {"claude.mcp.json": _CLAUDE_MCP_CONFIG}
    assert spec.home_files == {}
    assert spec.stdin is None


def test_claude_code_resume_variants_pick_the_right_flag() -> None:
    h = harness_for("claude_code")
    resumed = h.launch(**_LAUNCH, session_id="SID-1", resume=True)
    assert resumed.argv[-2:] == ["--resume", "SID-1"]
    continued = h.launch(**_LAUNCH, resume=True)
    assert continued.argv[-1] == "--continue"
    pinned = h.launch(**_LAUNCH, session_id="SID-1")
    assert pinned.argv[-2:] == ["--session-id", "SID-1"]


# ----- codex ---------------------------------------------------------------------------------


def test_codex_probes_and_env_are_pinned() -> None:
    h = harness_for("codex")
    assert h.name == "codex"
    assert h.pins_session_id is False
    assert h.base_env() == {"NODE_OPTIONS": ""}
    assert h.version_probe() == ["codex", "--version"]
    assert h.model_probe() is None


def test_codex_fresh_launch_is_pinned() -> None:
    spec = harness_for("codex").launch(**_LAUNCH)
    assert spec.argv == [
        "codex",
        "exec",
        "--json",
        "-m",
        "the-model",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "-c",
        'mcp_servers.shogym.url="http://host.docker.internal:8973/mcp"',
        "-c",
        'mcp_servers.shogym.default_tools_approval_mode="approve"',
        "-c",
        "mcp_servers.shogym.required=true",
        "-c",
        "mcp_servers.shogym.startup_timeout_sec=60",
        "-c",
        "mcp_servers.shogym.tool_timeout_sec=900",
        "-c",
        'cli_auth_credentials_store="file"',
        "SYS\n\nUSR",
    ]
    assert spec.env == {"NODE_OPTIONS": ""}
    assert spec.config_files == {}
    assert spec.home_files == {}
    assert spec.stdin is None


def test_codex_resume_puts_the_subcommand_ahead_of_the_flags() -> None:
    spec = harness_for("codex").launch(**_LAUNCH, session_id="SID-1", resume=True)
    assert spec.argv[:4] == ["codex", "exec", "resume", "SID-1"]
    fresh = harness_for("codex").launch(**_LAUNCH)
    assert "resume" not in fresh.argv


# ----- prime_agent ---------------------------------------------------------------------------


def test_prime_agent_probes_and_env_are_pinned() -> None:
    h = harness_for("prime_agent")
    assert h.name == "prime_agent"
    assert h.pins_session_id is False
    assert h.base_env() == {"NODE_OPTIONS": "", "SHOBENCH_MCP_TOKEN": "local"}
    assert h.version_probe() == ["prime-agent", "--version"]
    assert h.model_probe() == ["prime-agent", "model", "list"]


def test_prime_agent_fresh_launch_is_pinned() -> None:
    spec = harness_for("prime_agent").launch(**_LAUNCH)
    assert spec.argv == [
        "prime-agent",
        "-p",
        "--mode",
        "json",
        "--model",
        "the-model",
        "--autonomous",
        "--autonomous-max-continuations",
        "100000",
        "--autonomous-max-turns",
        "100000",
        "--autonomous-max-tokens",
        "1000000000",
        "--autonomous-timeout-ms",
        "7200000",
        "--",
        "SYS\n\nUSR",
    ]
    assert spec.env == {"NODE_OPTIONS": "", "SHOBENCH_MCP_TOKEN": "local"}
    assert spec.config_files == {}
    # The HOME carries the settings entry and the vendored shogym-stream skill package beside
    # it; the skill is what actually reaches the server, since prime-agent's client is a
    # kernel-side import rather than a host-managed tool bridge.
    assert spec.home_files == {
        ".prime/agent/settings.json": _PRIME_SETTINGS,
        **shogym_stream_skill_files(),
    }
    assert spec.stdin is None
    # The settings entry is well-formed JSON naming the http server and the token variable.
    settings = json.loads(spec.home_files[".prime/agent/settings.json"])
    assert settings["mcpServers"]["shogym"]["type"] == "http"
    assert settings["mcpServers"]["shogym"]["bearerTokenEnvVar"] == "SHOBENCH_MCP_TOKEN"


def test_prime_agent_resume_variants_sit_before_the_prompt_separator() -> None:
    h = harness_for("prime_agent")
    resumed = h.launch(**_LAUNCH, session_id="SID-1", resume=True)
    sep = resumed.argv.index("--")
    assert resumed.argv[sep - 2 : sep] == ["--resume", "SID-1"]
    continued = h.launch(**_LAUNCH, resume=True)
    sep = continued.argv.index("--")
    assert continued.argv[sep - 1] == "--continue"
    pinned = h.launch(**_LAUNCH, session_id="SID-1")
    assert "--resume" not in pinned.argv
    assert "--continue" not in pinned.argv


# ----- the verdict text, not only its kind ---------------------------------------------------


def _write(path: Path, events: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


def test_a_leg_timeout_reads_the_same_for_every_harness(tmp_path: Path) -> None:
    for name in ("claude_code", "codex", "prime_agent"):
        verdict = harness_for(name).classify(
            returncode=-1,
            stdout_path=tmp_path / "missing.jsonl",
            stderr_path=tmp_path / "missing.txt",
            timed_out=True,
        )
        assert verdict.kind is StopKind.LEG_TIMEOUT
        assert verdict.reason == "the runner ended the leg at its budget"


def test_a_clean_finish_carries_each_harness_its_own_reason(tmp_path: Path) -> None:
    claude = harness_for("claude_code").classify(
        returncode=0,
        stdout_path=_write(
            tmp_path / "cc.jsonl",
            [
                {
                    "type": "result",
                    "is_error": False,
                    "subtype": "success",
                    "terminal_reason": "completed",
                    "api_error_status": None,
                    "result": "done",
                }
            ],
        ),
        stderr_path=tmp_path / "cc.err",
        timed_out=False,
    )
    assert claude.kind is StopKind.CHOSEN
    assert claude.reason == "the session ended its turn cleanly"

    codex = harness_for("codex").classify(
        returncode=0,
        stdout_path=_write(
            tmp_path / "cx.jsonl",
            [{"type": "thread.started", "thread_id": "t"}, {"type": "turn.completed"}],
        ),
        stderr_path=tmp_path / "cx.err",
        timed_out=False,
    )
    assert codex.kind is StopKind.CHOSEN
    assert codex.reason == "codex exec completed its turn"

    prime = harness_for("prime_agent").classify(
        returncode=0,
        stdout_path=_write(
            tmp_path / "pa.jsonl",
            [
                {"type": "session", "id": "s"},
                {
                    "type": "agent_end",
                    "messages": [{"role": "assistant", "stopReason": "stop"}],
                },
            ],
        ),
        stderr_path=tmp_path / "pa.err",
        timed_out=False,
    )
    assert prime.kind is StopKind.CHOSEN
    assert prime.reason == "the last message ended the turn"
