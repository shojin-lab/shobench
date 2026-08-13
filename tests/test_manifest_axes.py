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


def _manifest(
    tmp_path: Path, harness: str, *, effort: str = "", eval_context: str = "resumed"
) -> dict:
    """A manifest built by the real builder, for a cell the test shapes."""
    from dataclasses import replace

    cell = replace(
        load_cell_by_name("smoke-automationbench-claude-code"),
        effort=effort,
        eval_context=eval_context,
    )
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


# ----- rollout feedback regime --------------------------------------------------------------


def test_rollout_feedback_defaults_to_immediate_and_reaches_the_manifest() -> None:
    """The premise is the default: a cell that says nothing gets feedback on done."""
    cell = load_cell_by_name("automationbench-claude_code-claude-opus-5")

    assert cell.rollout_feedback == "immediate"
    assert cell.to_manifest()["rollout_feedback"] == "immediate"


def test_a_never_cell_loads_and_an_unknown_regime_is_refused(tmp_path: Path) -> None:
    """The ablation arm is explicit, and a typo is a config error rather than a silent arm."""
    from shobench.config import load_cell

    base = "\n".join(
        [
            "[cell]",
            'name = "t"',
            'env = "automationbench"',
            'harness = "claude_code"',
            'model = "m"',
            'split = "automationbench"',
            "{regime}",
            "[budget]",
            "rollout_wall_clock_s = 60",
        ]
    )
    never = tmp_path / "never.toml"
    never.write_text(base.format(regime='rollout_feedback = "never"'), encoding="utf-8")
    assert load_cell(never).rollout_feedback == "never"

    typo = tmp_path / "typo.toml"
    typo.write_text(base.format(regime='rollout_feedback = "always"'), encoding="utf-8")
    with pytest.raises(ValueError, match="rollout_feedback"):
        load_cell(typo)


def test_eval_context_defaults_to_resumed_and_reaches_the_manifest() -> None:
    """The correction is the default: eval_after forks the rollout conversation unless a cell
    names the cold ablation, and the manifest records which of the two the run was."""
    cell = load_cell_by_name("automationbench-claude_code-claude-opus-5")

    assert cell.eval_context == "resumed"
    assert cell.to_manifest()["eval_context"] == "resumed"


def test_the_manifest_names_the_eval_prompt_the_context_selects(tmp_path: Path) -> None:
    """The artifact says which standing instruction its eval_after launched with, rather than
    leaving a reader to derive it from the axis: a resumed after carries the rollout
    instruction (its conversation already holds the objective, and swapping the instruction
    mid-conversation would measure an agent that never existed), a cold one stays blind."""
    resumed = _manifest(tmp_path / "resumed", "claude_code")
    assert resumed["cell"]["eval_context"] == "resumed"
    assert resumed["instruction"]["eval_prompt_used"] == "rollout_system"

    cold = _manifest(tmp_path / "cold", "claude_code", eval_context="cold")
    assert cold["cell"]["eval_context"] == "cold"
    assert cold["instruction"]["eval_prompt_used"] == "eval_system"


def test_a_cold_cell_loads_and_an_unknown_eval_context_is_refused(tmp_path: Path) -> None:
    """The ablation arm is explicit, and a typo is a config error rather than a silent arm."""
    from shobench.config import load_cell

    base = "\n".join(
        [
            "[cell]",
            'name = "t"',
            'env = "automationbench"',
            'harness = "claude_code"',
            'model = "m"',
            'split = "automationbench"',
            "{context}",
            "[budget]",
            "rollout_wall_clock_s = 60",
        ]
    )
    cold = tmp_path / "cold.toml"
    cold.write_text(base.format(context='eval_context = "cold"'), encoding="utf-8")
    assert load_cell(cold).eval_context == "cold"

    typo = tmp_path / "typo.toml"
    typo.write_text(base.format(context='eval_context = "warm"'), encoding="utf-8")
    with pytest.raises(ValueError, match="eval_context"):
        load_cell(typo)


def test_the_rollout_stream_regime_follows_the_cell_axis(monkeypatch, tmp_path: Path) -> None:
    """What the stream is actually constructed with, not what the config claims."""
    import dataclasses

    import shogym.serve as shogym_serve

    from shobench import serving

    captured: dict = {}

    class _FakeStream:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(shogym_serve, "TaskStream", _FakeStream)
    cell = load_cell_by_name("automationbench-claude_code-claude-opus-5")
    split = load_split_by_name(cell.split)

    serving.build_stream(cell, split, "rollout", tmp_path)
    assert type(captured["feedback"]).__name__ == "Immediate"

    serving.build_stream(
        dataclasses.replace(cell, rollout_feedback="never"), split, "rollout", tmp_path
    )
    assert type(captured["feedback"]).__name__ == "Never"


# ----- the image a run pinned itself to ---------------------------------------------------------
#
# A tag is a mutable name and a run is long, so one run has to resolve it once and then use the
# same bytes for every probe and every leg. The fresh path skipped that entirely: it probed and ran
# on the tag and recorded no content id, which left every future archive permanently unable to
# state the identity the pairing checks for.


def test_a_fresh_run_pins_its_image_and_records_both_names(tmp_path: Path, monkeypatch) -> None:
    """Driven through the real ``run_cell`` with no phases, because what is under test is what
    the call site resolves and hands on, not what a probe does with it."""
    import asyncio

    from shobench import runner

    probed: list[str] = []

    def fake_probe(argv, *, image, sandbox, env, redactor=None):
        probed.append(image)
        return "2.1.226 (Claude Code)"

    monkeypatch.setattr(runner, "_probe", fake_probe)
    monkeypatch.setattr(runner, "image_digest", lambda image: "sha256:" + "e" * 64)
    monkeypatch.setattr(CellSandbox, "up", lambda self, **kw: None)
    monkeypatch.setattr(CellSandbox, "down", lambda self: None)
    cell = load_cell_by_name("smoke-automationbench-claude-code")

    asyncio.run(
        runner.run_cell(
            cell,
            load_split_by_name(cell.split),
            runs_dir=tmp_path / "runs",
            results_dir=tmp_path / "results",
            agent_image="mutable-agent:latest",
            phases=(),
            capture_egress=False,
        )
    )

    # Every probe ran the resolved id, so a rebuild mid-run cannot put one image in the probe
    # and another in the legs.
    assert probed == ["sha256:" + "e" * 64]
    run_dir = next(p for p in (tmp_path / "runs").iterdir() if p.is_dir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    # The tag says what was asked for and the digest says what answered. An archive that states
    # only the tag can never prove its image to a later pairing.
    assert manifest["container"]["agent_image"] == "mutable-agent:latest"
    assert manifest["container"]["image_digest"] == "sha256:" + "e" * 64


def test_an_unresolvable_image_leaves_the_tag_in_place(tmp_path: Path, monkeypatch) -> None:
    """Docker not answering is an absence, not a failure: the run proceeds on the tag and the
    record says it could not name the bytes, which is what the unproven list is for."""
    import asyncio

    from shobench import runner

    probed: list[str] = []

    def fake_probe(argv, *, image, sandbox, env, redactor=None):
        probed.append(image)
        return "2.1.226 (Claude Code)"

    monkeypatch.setattr(runner, "_probe", fake_probe)
    monkeypatch.setattr(runner, "image_digest", lambda image: None)
    monkeypatch.setattr(CellSandbox, "up", lambda self, **kw: None)
    monkeypatch.setattr(CellSandbox, "down", lambda self: None)
    cell = load_cell_by_name("smoke-automationbench-claude-code")

    asyncio.run(
        runner.run_cell(
            cell,
            load_split_by_name(cell.split),
            runs_dir=tmp_path / "runs",
            results_dir=tmp_path / "results",
            agent_image="mutable-agent:latest",
            phases=(),
            capture_egress=False,
        )
    )

    assert probed == ["mutable-agent:latest"]
    run_dir = next(p for p in (tmp_path / "runs").iterdir() if p.is_dir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["container"]["agent_image"] == "mutable-agent:latest"
    assert manifest["container"]["image_digest"] is None


def test_resolving_an_image_asks_docker_every_time(monkeypatch) -> None:
    """The resolution is scoped to a run, not to the process. A cache that outlived one run made
    a second run in the same process pin the image the first one resolved, so a deliberate
    rebuild between two calls of the exported API published the old id for the new run's rows."""
    from shobench import containers

    answers = iter(["sha256:first", "sha256:second"])

    class _Result:
        returncode = 0
        stderr = ""

        def __init__(self) -> None:
            self.stdout = next(answers) + "\n"

    monkeypatch.setattr(containers, "docker", lambda *a, **kw: _Result())
    monkeypatch.setattr(containers.shutil, "which", lambda name: "/usr/bin/docker")

    assert containers.image_digest("mutable-agent:latest") == "sha256:first"
    assert containers.image_digest("mutable-agent:latest") == "sha256:second"
