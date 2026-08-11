"""Suspending a held-out eval a provider usage limit stopped, and continuing it later.

A subscription window can close in the middle of the held-out eval, not only the rollout. The
eval fan-out used to ignore each leg's verdict, so a usage limit followed the clean path: the
orderly stream close drained the in-flight task into a scored ``drained`` row, reward zero,
which reads as the model failing a held-out task when what failed was the window. In eval_after
that poisons the paired measurement after the eight-hour rollout is already paid for.

The fix mirrors the standalone held-out evaluator the runner descends from: completion is a
property of each task's own provenance, a finished id is never re-run, and a phase that cannot
account for every id is not published. The only thing added here is the trigger, a usage-limit
verdict that suspends the cell through the same hard-exit/resume plumbing the rollout uses.

None of this needs Docker. The eval streams are real (driven in-process, one per task, the way
the reviewer drained a real held-out stream), and the harness leg is stood in for by connecting
to the served stream over the loopback and pulling a task, which is all the leg's container does
over MCP anyway: a completed task terminates it, an interrupted one leaves it in flight so the
stream drains it exactly as a usage limit would.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from shobench import runner
from shobench.config import load_cell_by_name, load_instruction
from shobench.containers import CellSandbox
from shobench.harness import StopKind, StopVerdict
from shobench.results import read_phase
from shobench.runner import (
    ROLLOUT_STOPPING_FILE,
    SUSPENDED_EXIT_CODE,
    SUSPENSION_FILE,
    LegRecord,
    RunContext,
    _eval_pending_ids,
    _eval_task_valid_row,
    _run_phases,
    build_manifest,
)
from shobench.splits import Side, Split

_SMOKE_CELL = "smoke-automationbench-claude-code"


# ----- real single-task streams, no Docker ---------------------------------------------------


def _play_eval_task(prov_dir: Path, task_id: int, *, terminate: bool) -> None:
    """Drive one real single-task ``EvalStream`` to a real row, in process.

    ``terminate`` decides which. Terminating leaves an ``aborted`` scored row, a real outcome the
    agent reached. Not terminating leaves the task in flight, so closing the stream drains it into
    a ``drained`` row: exactly the state a usage limit mid-task leaves, and the row finding 2 is
    about.
    """
    import shogym
    from fastmcp import Client
    from shogym.serve import EvalStream, TaskRef, build_stream_server

    async def play() -> None:
        stream = EvalStream(shogym.make, [TaskRef("wordle_v1", task_id)], prov_dir=prov_dir)
        async with stream:
            server = build_stream_server(stream, name="shogym")
            async with Client(server) as client:
                await client.call_tool("get_task", {})
                if terminate:
                    await client.call_tool("terminate", {})

    asyncio.run(play())


def _play_rollout(prov_dir: Path, task_ids: tuple[str, ...]) -> None:
    """A real rollout record over the pool, played straight through, for the recorded-phase side."""
    import shogym
    from shogym.serve import Never, TaskRef, TaskStream

    async def play() -> None:
        stream = TaskStream(
            shogym.make,
            [TaskRef("wordle_v1", int(t)) for t in task_ids],
            prov_dir=prov_dir,
            feedback=Never(),
            max_in_flight=8,
        )
        async with stream:
            while (task := await stream.get_task()) is not None:
                await stream.dispatch("terminate", {}, lease=task.lease)

    asyncio.run(play())


def _http_leg(*, usage_limit_ids: tuple[int, ...] = (), dispatched: list[int]):
    """A stand-in for one eval leg: connect to the served stream over loopback and play the task.

    It records which id it was asked to run, so a test can prove a finished id is never
    re-dispensed. A ``usage_limit`` id pulls its task and leaves it in flight (the stream drains
    it) and returns a ``USAGE_LIMIT`` verdict; every other id terminates its task and returns a
    chosen stop. This is all the real container does over MCP, minus the container.
    """

    def leg(ctx: RunContext, *, phase: str, task_idx: int, session_id: str | None = None, **kw):
        from fastmcp import Client

        mcp_url = kw.get("mcp_url") or ctx.mcp_url
        port = int(mcp_url.split("://", 1)[1].split("/", 1)[0].rsplit(":", 1)[1])
        dispatched.append(task_idx)
        usage = task_idx in usage_limit_ids

        async def play() -> None:
            async with Client(f"http://127.0.0.1:{port}/mcp") as client:
                await client.call_tool("get_task", {})
                if not usage:
                    await client.call_tool("terminate", {})

        asyncio.run(play())
        verdict = (
            StopVerdict(StopKind.USAGE_LIMIT, "the window closed")
            if usage
            else StopVerdict(StopKind.CHOSEN, "it stopped on its own")
        )
        record = LegRecord(
            leg=task_idx if task_idx is not None else 0,
            phase=phase,
            task_idx=task_idx,
            started_at=0.0,
            ended_at=1.0,
            returncode=0,
            verdict=verdict,
            tasks_consumed_before=0,
            tasks_consumed_after=0,
            trace_path="t",
            run_dir=ctx.run_dir,
            session_id=session_id,
        )
        ctx.legs.append(record)
        return record

    return leg


def _wordle_ctx(
    tmp_path: Path, *, heldout: tuple[str, ...], pool: tuple[str, ...] = ("3", "4")
) -> RunContext:
    cell = load_cell_by_name(_SMOKE_CELL)
    cell = replace(
        cell,
        env="wordle_v1",
        budget=replace(cell.budget, eval_concurrency=1, eval_task_timeout_s=120),
    )
    run_dir = tmp_path / "run"
    sandbox = CellSandbox(run_id="test", home=run_dir / "home", workdir=run_dir / "work")
    sandbox.home.mkdir(parents=True)
    split = Split(
        env="wordle_v1",
        heldout=Side(task_ids=heldout),
        pool=Side(task_ids=pool),
        provenance={"kind": "adopted"},
        source=tmp_path / "split.json",
    )
    return RunContext(
        cell=cell,
        split=split,
        instruction=load_instruction(cell.instruction_arm),
        harness=runner.harness_for(cell.harness),
        run_id="test",
        run_dir=run_dir,
        sandbox=sandbox,
    )


# ----- completion is read off the per-task provenance ----------------------------------------


def test_completion_counts_a_scored_non_drained_row_and_nothing_else(tmp_path: Path) -> None:
    """The predicate a resume runs on: what "this held-out id is done" means, against real rows."""
    phase_dir = tmp_path / "eval_before"
    _play_eval_task(phase_dir / "task-00000", 0, terminate=True)  # aborted: a real outcome, done
    _play_eval_task(phase_dir / "task-00001", 1, terminate=False)  # drained: the poison, not done
    # task 2 never ran: its directory does not exist, which must read as not-done rather than raise.

    assert _eval_task_valid_row(phase_dir / "task-00000", 0) is not None
    assert _eval_task_valid_row(phase_dir / "task-00001", 1) is None
    assert _eval_task_valid_row(phase_dir / "task-00002", 2) is None

    # Only the finished id is skipped; the drained one and the unstarted one are both re-run. A
    # predicate that counted the drained row as scored would drop it from this list (the mutation).
    assert _eval_pending_ids(phase_dir, ("0", "1", "2")) == ["1", "2"]


# ----- a usage limit mid eval_before suspends, and the resume finishes only the incomplete -----


def test_usage_limit_mid_eval_before_suspends_then_resume_runs_only_the_incomplete(
    tmp_path: Path, monkeypatch
) -> None:
    ctx = _wordle_ctx(tmp_path, heldout=("0", "1", "2"))
    phase = "eval_before"
    monkeypatch.setattr(runner, "warm_env", lambda cell: None)

    # The suspension hard-exits the process; here it is captured so the test can see the phase and
    # the evidence the record would carry, and assert nothing published.
    captured: dict[str, object] = {}

    class _Suspended(Exception):
        pass

    def fake_suspend(ctx_arg, *, phase, verdict):
        captured["phase"] = phase
        captured["verdict"] = verdict
        raise _Suspended()

    monkeypatch.setattr(runner, "_suspend_eval_and_exit", fake_suspend)

    # Task 1 hits the usage limit. With one task at a time and the ids taken in order, task 0
    # finishes first, task 1 draws the limit, and task 2 must never be admitted after it.
    dispatched: list[int] = []
    monkeypatch.setattr(runner, "run_leg", _http_leg(usage_limit_ids=(1,), dispatched=dispatched))

    with pytest.raises(_Suspended):
        asyncio.run(runner.run_eval_phase(ctx, phase))

    assert captured["phase"] == "eval_before"
    assert captured["verdict"].kind is StopKind.USAGE_LIMIT  # type: ignore[union-attr]
    # Admission stopped at the limit: 0 and 1 ran, 2 was left for the resume (mutation: not
    # stopping admission would run task 2 as well and drain it too).
    assert dispatched == [0, 1]
    phase_dir = ctx.run_dir / phase
    assert _eval_task_valid_row(phase_dir / "task-00000", 0) is not None
    assert [r.closure for r in read_phase(phase_dir / "task-00001")] == ["drained"]
    assert _eval_pending_ids(phase_dir, ("0", "1", "2")) == ["1", "2"]

    # The window reopens: resume the same phase. Only the incomplete ids run, and the finished one
    # is not touched.
    dispatched.clear()
    monkeypatch.setattr(runner, "run_leg", _http_leg(dispatched=dispatched))
    rows = asyncio.run(runner.run_eval_phase(ctx, phase))

    # The completed id was not re-dispensed; only the interrupted and unstarted ids ran.
    assert sorted(dispatched) == [1, 2]
    assert 0 not in dispatched
    # Exactly one valid, non-drained, scored row per held-out id, in task-id order, no drained
    # leftover. Not clearing the interrupted id would leave its drained row beside the replay, so
    # the count would be four and a drained closure would survive (the quarantine mutation).
    assert len(rows) == 3
    assert [r.task_idx for r in rows] == [0, 1, 2]
    assert all(r.scored and r.closure != "drained" for r in rows)


# ----- an eval_after usage limit must not lose the rollout ------------------------------------


def test_eval_after_resume_republishes_the_rollout_and_eval_before_intact(
    tmp_path: Path, monkeypatch
) -> None:
    ctx = _wordle_ctx(tmp_path, heldout=("0", "1"), pool=("3", "4"))
    run_dir = ctx.run_dir
    monkeypatch.setattr(runner, "warm_env", lambda cell: None)

    # The state an eval_after usage limit leaves: eval_before measured, the rollout run and its
    # stop classification persisted, and eval_after half done with one task drained.
    _play_eval_task(run_dir / "eval_before" / "task-00000", 0, terminate=True)
    _play_eval_task(run_dir / "eval_before" / "task-00001", 1, terminate=True)
    _play_rollout(run_dir / "rollout", ("3", "4"))
    runner.write_json(
        run_dir / ROLLOUT_STOPPING_FILE,
        {
            "stop_reason": "pool_exhausted",
            "tasks_dispensed": 2,
            "stopped_with_tasks_available": False,
        },
    )
    _play_eval_task(run_dir / "eval_after" / "task-00000", 0, terminate=True)  # done
    _play_eval_task(run_dir / "eval_after" / "task-00001", 1, terminate=False)  # drained, pending

    manifest = build_manifest(ctx, probes={"version": "test"})
    # This process is the resume that owns the eval_after continuation.
    manifest["resumptions"] = [{"phase": "eval_after", "resumed_at": 2.0}]

    dispatched: list[int] = []
    monkeypatch.setattr(runner, "run_leg", _http_leg(dispatched=dispatched))
    observer = runner._Egress(None, run_dir)

    results_path = asyncio.run(
        _run_phases(
            ctx,
            manifest=manifest,
            phases=("eval_after",),
            results_dir=tmp_path / "results",
            observer=observer,
            suspended=None,
            recorded_phases=("eval_before", "rollout"),
        )
    )
    published = json.loads(results_path.read_text())

    # Only the pending eval_after id re-ran; the completed one was not re-dispensed.
    assert dispatched == [1]

    # eval_after: one valid, non-drained row per held-out id, no drained leftover.
    after = published["eval_after"]["tasks"]
    assert [r["task_idx"] for r in after] == [0, 1]
    assert all(r["closure"] != "drained" for r in after)
    assert published["eval_after"]["summary"]["n_scored"] == 2

    # The eight-hour rollout survived the suspension: its stop reason and its rows republish
    # intact rather than blank (the point of persisting the stop classification and reading the
    # recorded phases back).
    assert published["rollout"]["stopping"]["stop_reason"] == "pool_exhausted"
    assert published["rollout"]["stopping"]["usage_limit_resumes"] == 1
    assert len(published["rollout"]["tasks"]) >= 1

    # And so did eval_before: both halves of the paired measurement are present and pair.
    assert [r["task_idx"] for r in published["eval_before"]["tasks"]] == [0, 1]
    assert published["eval_before"]["summary"]["n_scored"] == 2
    assert len(published["paired"]) == 2
    assert not published["unpaired"]


# ----- a held-out id that produced no row at all ----------------------------------------------
#
# The failure this section exists for is not a task that scored badly. It is a task that left the
# record holding nothing: the runner caught its exception and wrote a breadcrumb, or the harness
# exited before ever calling get_task and nothing raised anywhere. Counted by rows, that id is not
# a failure but an absence, and an absence used to leave the published file entirely, taking the
# denominator with it: a paired mean and a bootstrap over a silently selected subset, under a
# manifest still saying 120 or 40 tasks were requested.


def _publish(ctx: RunContext, tmp_path: Path) -> Path:
    """Publish this run through the real tail: eval_after runs, the recorded phases carry over."""
    return asyncio.run(
        _run_phases(
            ctx,
            manifest=build_manifest(ctx, probes={"version": "test"}),
            phases=("eval_after",),
            results_dir=tmp_path / "results",
            observer=runner._Egress(None, ctx.run_dir),
            recorded_phases=("eval_before", "rollout"),
        )
    )


def test_a_held_out_id_with_no_row_is_published_and_loses_the_finished_name(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The whole of the fix, through the real publish path.

    eval_before recorded one of its two committed ids and nothing at all for the other, which is
    what a task that never ran leaves behind. The published file has to say so: the lost id
    present and unscored, the requested count still two, the pair it cannot form in the unpaired
    list, and the file itself under a name no reader mistakes for a finished measurement.
    """
    ctx = _wordle_ctx(tmp_path, heldout=("0", "1"), pool=("3", "4"))
    monkeypatch.setattr(runner, "warm_env", lambda cell: None)

    _play_eval_task(ctx.run_dir / "eval_before" / "task-00000", 0, terminate=True)
    # Held-out id 1 has no directory and no row: nothing recorded an outcome for it.
    _play_rollout(ctx.run_dir / "rollout", ("3", "4"))
    runner.write_json(ctx.run_dir / ROLLOUT_STOPPING_FILE, {"stop_reason": "pool_exhausted"})
    monkeypatch.setattr(runner, "run_leg", _http_leg(dispatched=[]))

    results_path = _publish(ctx, tmp_path)
    published = json.loads(results_path.read_text())

    # The requested count is the committed set, and the id that recorded nothing is in the file
    # rather than absent from it.
    assert published["eval_before"]["summary"]["n_requested"] == 2
    assert [r["task_idx"] for r in published["eval_before"]["tasks"]] == [0, 1]
    lost = published["eval_before"]["tasks"][1]
    assert lost["closure"] == "missing"
    assert lost["reward"] is None and lost["success"] is None
    # It cannot pair, and the pair it cannot form is reported rather than dropped.
    assert [p["task_idx"] for p in published["paired"]] == [0]
    assert [u["task_idx"] for u in published["unpaired"]] == [1]
    # And the cell is not a finished measurement, by field and by name.
    assert published["heldout"]["complete"] is False
    assert published["heldout"]["eval_before"]["missing_task_ids"] == [1]
    assert results_path.name.endswith(".incomplete.json")
    assert not (tmp_path / "results" / f"{ctx.cell.name}.json").exists()
    assert "INCOMPLETE" in capsys.readouterr().err


def test_a_cell_that_accounts_for_every_id_publishes_under_its_own_name(
    tmp_path: Path, monkeypatch
) -> None:
    """The sibling assertion: the same path with nothing lost keeps the finished name.

    Without it, a rule that called every cell incomplete would look exactly like one that calls
    the right ones incomplete.
    """
    ctx = _wordle_ctx(tmp_path, heldout=("0", "1"), pool=("3", "4"))
    monkeypatch.setattr(runner, "warm_env", lambda cell: None)

    _play_eval_task(ctx.run_dir / "eval_before" / "task-00000", 0, terminate=True)
    _play_eval_task(ctx.run_dir / "eval_before" / "task-00001", 1, terminate=True)
    _play_rollout(ctx.run_dir / "rollout", ("3", "4"))
    runner.write_json(ctx.run_dir / ROLLOUT_STOPPING_FILE, {"stop_reason": "pool_exhausted"})
    monkeypatch.setattr(runner, "run_leg", _http_leg(dispatched=[]))

    results_path = _publish(ctx, tmp_path)
    published = json.loads(results_path.read_text())

    assert results_path.name == f"{ctx.cell.name}.json"
    assert published["heldout"]["complete"] is True
    assert published["eval_before"]["summary"]["n_missing"] == 0
    assert len(published["paired"]) == 2
    assert not published["unpaired"]


# ----- resume_cell routes each interrupted phase to the phases that remain --------------------


def _write_eval_suspension(run_dir: Path, phase: str) -> None:
    """A run directory suspended in an eval phase: the manifest that names the cell and a
    suspension record naming the interrupted phase. Enough for ``resume_cell`` to route it."""
    from shobench.splits import load_split_by_name

    cell = load_cell_by_name(_SMOKE_CELL)
    run_dir.mkdir(parents=True)
    (run_dir / "home").mkdir()
    # A manifest built from the checkout's own cell, split, and instruction, so the resume's drift
    # check matches rather than refusing the record it is meant to route.
    ctx = RunContext(
        cell=cell,
        split=load_split_by_name(cell.split),
        instruction=load_instruction(cell.instruction_arm),
        harness=runner.harness_for(cell.harness),
        run_id="run-1",
        run_dir=run_dir,
        sandbox=CellSandbox(run_id="run-1", home=run_dir / "home", workdir=run_dir / "w"),
    )
    runner.write_json(run_dir / "manifest.json", build_manifest(ctx, probes={"version": "test"}))
    runner.write_json(
        run_dir / SUSPENSION_FILE,
        {
            "schema": "shobench.suspension/1",
            "run_id": "run-1",
            "cell": cell.name,
            "harness": cell.harness,
            "phase": phase,
            "legs_before": 1,
            "completed_task_ids": [0],
            "pending_task_ids": [1, 2],
            "stop_evidence": StopVerdict(StopKind.USAGE_LIMIT, "the window closed").to_json(),
            "suspended_at": 1.0,
            "resume_with": f"uv run shobench resume --run {run_dir} --go",
        },
    )


@pytest.mark.parametrize(
    ("phase", "want_phases", "want_recorded"),
    [
        ("eval_before", ("eval_before", "rollout", "eval_after"), ()),
        ("eval_after", ("eval_after",), ("eval_before", "rollout")),
    ],
)
def test_resume_cell_routes_an_eval_suspension_to_the_phases_that_remain(
    tmp_path: Path, monkeypatch, phase, want_phases, want_recorded
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_phases(
        ctx, *, manifest, phases, results_dir, observer, suspended=None, recorded_phases=()
    ):
        captured["phases"] = phases
        captured["recorded_phases"] = recorded_phases
        captured["suspended"] = suspended
        captured["resumptions"] = list(manifest.get("resumptions", []))
        return results_dir / "x.json"

    monkeypatch.setattr(runner, "_run_phases", fake_run_phases)
    monkeypatch.setattr(
        CellSandbox, "up", lambda self, **kw: self.home.mkdir(parents=True, exist_ok=True)
    )
    monkeypatch.setattr(CellSandbox, "down", lambda self: None)
    monkeypatch.setattr(runner, "seed_home", lambda spec, home: {})
    monkeypatch.setattr(runner, "_start_egress", lambda sandbox, run_dir: None)

    run_dir = tmp_path / phase
    _write_eval_suspension(run_dir, phase)
    asyncio.run(runner.resume_cell(run_dir, results_dir=tmp_path / "res", capture_egress=False))

    # An eval suspension carries no rollout clock and no session to reattach to, so it runs the
    # phases that remain with no suspension object, and re-runs only the ids provenance shows
    # pending. The recorded phases it carries forward differ by where the limit fell.
    assert captured["phases"] == want_phases
    assert captured["recorded_phases"] == want_recorded
    assert captured["suspended"] is None
    assert captured["resumptions"][-1]["phase"] == phase  # type: ignore[index]


# ----- the guaranteed hard exit, and publishing nothing --------------------------------------


def test_an_eval_suspension_records_the_phase_hard_exits_and_publishes_nothing(
    tmp_path: Path,
) -> None:
    """The eval counterpart of the rollout suspension: a record naming the phase, a hard exit
    through the shared tail, and nothing published or removed. Run as a child because the whole
    point is a process that leaves without unwinding, which cannot be done in-process."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "home").mkdir()
    (run_dir / "eval_before").mkdir()
    program = f"""
    import sys
    from pathlib import Path
    from shobench.config import load_cell_by_name
    from shobench.containers import CellSandbox
    from shobench.harness import StopKind, StopVerdict
    from shobench.runner import RunContext, _suspend_eval_and_exit
    from shobench.splits import Side, Split

    run_dir = Path({str(run_dir)!r})
    cell = load_cell_by_name({_SMOKE_CELL!r})
    split = Split(
        env=cell.env,
        heldout=Side(task_ids=("0", "1", "2")),
        pool=Side(task_ids=("3",)),
        provenance={{"kind": "adopted"}},
        source=run_dir / "s.json",
    )
    ctx = RunContext(
        cell=cell,
        split=split,
        instruction=None,
        harness=None,
        run_id="run-under-test",
        run_dir=run_dir,
        sandbox=CellSandbox(run_id="r", home=run_dir / "home", workdir=run_dir / "w"),
    )
    ctx.teardown = lambda: Path(run_dir / "torn-down").write_text("yes")
    _suspend_eval_and_exit(
        ctx, phase="eval_before", verdict=StopVerdict(StopKind.USAGE_LIMIT, "the window closed")
    )
    print("returned", file=sys.stderr)
    """
    ended = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)], capture_output=True, text=True
    )
    assert ended.returncode == SUSPENDED_EXIT_CODE, ended.stderr[-2000:]
    assert "returned" not in ended.stderr, "the suspension must not return into the phase"

    record = json.loads((run_dir / SUSPENSION_FILE).read_text())
    assert record["phase"] == "eval_before"
    # No task ran, so every requested id is still pending and none is recorded done.
    assert record["pending_task_ids"] == [0, 1, 2]
    assert record["completed_task_ids"] == []
    assert record["stop_evidence"]["kind"] == StopKind.USAGE_LIMIT.value

    assert (run_dir / "torn-down").is_file()
    assert (run_dir / "home").is_dir() and (run_dir / "eval_before").is_dir()
    assert not list(tmp_path.glob("**/results/*.json"))


# ----- the CLI plan for an eval suspension ----------------------------------------------------


def test_resume_plan_for_an_eval_suspension_skips_the_rollout_clock_guard(
    tmp_path: Path, capsys
) -> None:
    """An eval suspension has no rollout wall clock, so the spent-clock refusal must not apply to
    it, and its plan says which held-out ids remain rather than a dispense count."""
    from shobench.cli import main as cli_main

    cell = load_cell_by_name(_SMOKE_CELL)
    (tmp_path / SUSPENSION_FILE).write_text(
        json.dumps(
            {
                "schema": "shobench.suspension/1",
                "run_id": "run-1",
                "cell": cell.name,
                "harness": cell.harness,
                "phase": "eval_after",
                "legs_before": 3,
                "completed_task_ids": [0],
                "pending_task_ids": [1, 2],
                "stop_evidence": StopVerdict(StopKind.USAGE_LIMIT, "the window closed").to_json(),
                "suspended_at": 1.0,
                "resume_with": f"uv run shobench resume --run {tmp_path} --go",
            }
        )
    )

    assert cli_main(["resume", "--run", str(tmp_path)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["interrupted_phase"] == "eval_after"
    assert plan["pending_task_ids"] == [1, 2]
    assert plan["held_out_tasks_left"] == 2
    assert plan["phases_left"] == ["eval_after"]
    # The rollout-only fields are absent: an eval suspension is not bounded by that clock.
    assert "remaining_rollout_s" not in plan
