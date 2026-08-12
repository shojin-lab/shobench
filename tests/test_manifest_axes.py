"""The manifest says what was asked for and what actually happened, separately.

Three axes were recorded as though asking made them so. ``observed_models`` was implemented for
claude_code alone, so a codex or prime_agent cell published an empty list beside a comment
claiming it recorded what answered. Every cell recorded ``effort = "xhigh"``, including the
prime_agent cells whose harness accepts no effort at all and drops it. And ``credential_mode``
was copied out of the cell file without anyone looking at the credential that was seeded.

Where a harness really does report something, it is parsed from the shape that harness really
emits:

- claude_code's ``result`` event carries ``modelUsage`` keyed by the models that were billed;
- prime-agent's assistant messages carry ``provider``, ``model`` and an optional
  ``responseModel``, and arrive both as ``message_end`` and inside ``agent_end``'s message list.
  The shape is the ``AssistantMessage`` interface of the pi-ai package the pinned prime-agent
  bundles; there is no prime credential on this host to capture a live trace with, which is the
  same blocker the pending gate exists for;
- codex's ``exec --json`` names no model anywhere, which was checked against the pinned CLI
  rather than assumed, and is recorded as unobservable rather than as an empty observation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shobench.config import load_cell_by_name, load_instruction
from shobench.containers import CellSandbox
from shobench.credentials import effective_mode, spec_for
from shobench.harnesses import harness_for
from shobench.runner import RunContext, build_manifest
from shobench.runner import harness_for as _runner_harness_for
from shobench.splits import load_split_by_name

_LAUNCH = dict(
    mcp_url="http://host.docker.internal:8973/mcp",
    system_prompt="SYS",
    user_prompt="USR",
    model="the-model",
    trace_path=Path("/trace/leg.stream.jsonl"),
    leg_timeout_s=3600,
)


def _write(path: Path, events: list[dict]) -> Path:
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return path


# ----- which model answered ----------------------------------------------------------------


def test_prime_agent_reads_the_model_off_its_assistant_messages(tmp_path: Path) -> None:
    """A live-shaped stream: a session header, streamed messages, then the terminal agent_end."""
    trace = _write(
        tmp_path / "pa.jsonl",
        [
            {"type": "session", "id": "s-1", "timestamp": "2026-08-11T00:00:00Z", "cwd": "/work"},
            {"type": "message_start", "message": {"role": "user", "content": "go", "timestamp": 1}},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "on it"}],
                    "api": "anthropic-messages",
                    "provider": "anthropic",
                    "model": "claude-opus-5",
                    "usage": {"input": 10, "output": 3, "totalTokens": 13},
                    "stopReason": "toolUse",
                    "timestamp": 2,
                },
            },
            {
                "type": "agent_end",
                "messages": [
                    {"role": "user", "content": "go", "timestamp": 1},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "done"}],
                        "api": "anthropic-messages",
                        "provider": "anthropic",
                        "model": "claude-opus-5",
                        # What the provider says actually replied, which is the point of the axis.
                        "responseModel": "claude-opus-5-20260701",
                        "usage": {"input": 20, "output": 5, "totalTokens": 25},
                        "stopReason": "stop",
                        "timestamp": 3,
                    },
                ],
            },
        ],
    )

    observed = harness_for("prime_agent").observed_models(trace)

    assert observed == ["claude-opus-5", "claude-opus-5-20260701"]


def test_prime_agent_reports_nothing_from_a_leg_that_never_answered(tmp_path: Path) -> None:
    """A session that died before any assistant message has no model to report, and says so."""
    trace = _write(tmp_path / "pa.jsonl", [{"type": "session", "id": "s-1"}])

    assert harness_for("prime_agent").observed_models(trace) == []


def test_prime_agent_ignores_the_user_and_tool_messages(tmp_path: Path) -> None:
    """Only an assistant message says which model answered; the others carry no model at all."""
    trace = _write(
        tmp_path / "pa.jsonl",
        [
            {
                "type": "agent_end",
                "messages": [
                    {"role": "user", "content": "go", "timestamp": 1},
                    {
                        "role": "toolResult",
                        "toolCallId": "c1",
                        "toolName": "get_task",
                        "content": [{"type": "text", "text": "{}"}],
                        "isError": False,
                        "timestamp": 2,
                    },
                ],
            }
        ],
    )

    assert harness_for("prime_agent").observed_models(trace) == []


def test_codex_declares_that_its_trace_names_no_model(tmp_path: Path) -> None:
    """A real codex exec stream: a thread id, items, a terminal turn. No model anywhere.

    Recorded as unobservable, so an empty list in a codex manifest reads as "the harness does
    not report it" rather than as "no model answered".
    """
    codex = harness_for("codex")
    trace = _write(
        tmp_path / "cx.jsonl",
        [
            {"type": "thread.started", "thread_id": "019ff29a-f9f6-7b10-b861-9ebe26110a5e"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"id": "item_0", "type": "agent_message"}},
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 4},
            },
        ],
    )

    assert codex.reports_observed_models is False
    assert codex.observed_models(trace) == []


def test_claude_code_still_reads_the_models_it_was_billed_for(tmp_path: Path) -> None:
    """The one that already worked, kept in the same file so the three can be compared."""
    trace = _write(
        tmp_path / "cc.jsonl",
        [
            {
                "type": "result",
                "is_error": False,
                "modelUsage": {"claude-opus-5": {"inputTokens": 9}, "claude-haiku-4-5": {}},
            }
        ],
    )

    assert harness_for("claude_code").observed_models(trace) == [
        "claude-haiku-4-5",
        "claude-opus-5",
    ]


# ----- effort ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["claude_code", "codex", "prime_agent"])
def test_the_declared_effort_flag_matches_what_the_launch_actually_passes(name: str) -> None:
    """A harness that declares an effort flag must use it, and one that does not must not.

    The manifest reports ``applied`` from this declaration, so a drift between the declaration
    and the argv would put the wrong claim in the record for every cell of that harness.
    """
    harness = harness_for(name)
    argv = " ".join(harness.launch(**{**_LAUNCH, "model": "claude-opus-5"}, effort="xhigh").argv)

    if harness.effort_flag:
        assert harness.effort_flag in argv
        assert "xhigh" in argv
    else:
        assert "xhigh" not in argv


def test_prime_agent_records_the_effort_it_was_asked_for_and_that_it_was_not_applied(
    tmp_path: Path,
) -> None:
    """The v0 prime cells all ask for xhigh; prime-agent has no such control and drops it."""
    manifest = _manifest(tmp_path, "prime_agent", effort="xhigh")

    assert manifest["axes"]["effort"]["requested"] == "xhigh"
    assert manifest["axes"]["effort"]["applied"] is False
    assert "no reasoning-effort control" in manifest["axes"]["effort"]["how"]
    # The requested value stays in the cell record too, so the intent is not lost either.
    assert manifest["cell"]["effort"] == "xhigh"


def test_claude_code_records_the_same_effort_as_applied(tmp_path: Path) -> None:
    """The mutation check's other half: a harness with the control reports it as applied."""
    manifest = _manifest(tmp_path, "claude_code", effort="xhigh")

    assert manifest["axes"]["effort"] == {
        "requested": "xhigh",
        "applied": True,
        "how": "--effort",
    }


def test_a_cell_that_pins_no_effort_does_not_claim_to_have_applied_one(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, "claude_code", effort="")

    assert manifest["axes"]["effort"]["applied"] is False


# ----- the credential mode a cell really ran under ---------------------------------------------


def test_a_prime_oauth_seed_reads_as_the_subscription_it_claims(tmp_path: Path) -> None:
    spec = spec_for("prime_agent", "subscription")
    home = tmp_path / "home"
    (home / spec.seed_to).parent.mkdir(parents=True)
    (home / spec.seed_to).write_text(
        json.dumps({"anthropic": {"type": "oauth", "access": "a", "refresh": "r", "expires": 1}}),
        encoding="utf-8",
    )

    axis = effective_mode(spec, home)

    assert axis["effective"] == "subscription"
    assert axis["matches_requested"] is True
    assert axis["evidence"] == "providers=['anthropic']"


def test_a_prime_api_key_seed_is_not_published_as_a_subscription_run(tmp_path: Path) -> None:
    """Subscription billing is why the scope sets no token ceiling; api spend is another study."""
    spec = spec_for("prime_agent", "subscription")
    home = tmp_path / "home"
    (home / spec.seed_to).parent.mkdir(parents=True)
    (home / spec.seed_to).write_text(
        json.dumps({"anthropic": {"type": "api_key", "key": "sk-something"}}), encoding="utf-8"
    )

    axis = effective_mode(spec, home)

    assert axis["effective"] == "api_key"
    assert axis["matches_requested"] is False


def test_a_codex_api_key_login_is_not_published_as_a_subscription_run(tmp_path: Path) -> None:
    spec = spec_for("codex", "subscription")
    home = tmp_path / "home"
    (home / spec.seed_to).parent.mkdir(parents=True)
    (home / spec.seed_to).write_text(json.dumps({"auth_mode": "apikey"}), encoding="utf-8")

    axis = effective_mode(spec, home)

    assert axis["effective"] == "api_key"
    assert axis["matches_requested"] is False


def test_an_environment_credential_reports_the_name_it_arrived_by(tmp_path: Path) -> None:
    """claude_code seeds no file, so the variable that arrived is all the evidence there is."""
    spec = spec_for("claude_code", "subscription")

    present = effective_mode(spec, tmp_path, env_names=["CLAUDE_CODE_OAUTH_TOKEN"])
    absent = effective_mode(spec, tmp_path, env_names=[])

    assert present["effective"] == "subscription" and present["matches_requested"]
    assert absent["effective"] == "unknown" and not absent["matches_requested"]


# ----- helpers ---------------------------------------------------------------------------------


def _manifest(tmp_path: Path, harness: str, *, effort: str) -> dict:
    """A manifest built by the real builder, for a cell whose effort the test chooses."""
    from dataclasses import replace

    cell = replace(load_cell_by_name("smoke-automationbench-claude-code"), effort=effort)
    run_dir = tmp_path / "run"
    sandbox = CellSandbox(run_id="r", home=run_dir / "home", workdir=run_dir / "work")
    sandbox.home.mkdir(parents=True)
    sandbox.workdir.mkdir(parents=True)
    ctx = RunContext(
        cell=cell,
        split=load_split_by_name(cell.split),
        instruction=load_instruction(cell.instruction_arm),
        harness=_runner_harness_for(harness),
        run_id="r",
        run_dir=run_dir,
        sandbox=sandbox,
    )
    return build_manifest(ctx, probes={})
