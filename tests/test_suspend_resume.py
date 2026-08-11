"""Suspending a rollout a provider stopped, and continuing it later.

A subscription window can close in the middle of an eight-hour rollout. The run is not over
when that happens, so the cell suspends instead of finishing: no eval_after, no results file,
nothing removed, and a record on disk saying what a continuation needs. What makes that
correct is a property of shogym rather than of this runner, and the first test here is the one
that pins it, because everything else assumes it.

None of this needs Docker. What needs Docker is the harness leg and the containers around it,
so those are stood in for and everything else is the real code: the suspension record, the
continuation's arithmetic, the provenance it reads back, and the results file it publishes.
That last part matters most. A continuation that runs and publishes a result missing half the
measurement is worse than one that refuses to start, and only a test that follows a resume all
the way to ``write_results`` can see the difference.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from shobench import runner
from shobench.cli import main as cli_main
from shobench.config import Budget, load_cell_by_name, load_instruction
from shobench.containers import CellSandbox
from shobench.harness import StopKind, StopVerdict
from shobench.report import report_cell
from shobench.results import TaskResult
from shobench.runner import (
    SUSPENDED_EXIT_CODE,
    SUSPENSION_FILE,
    LegRecord,
    RunContext,
    Suspension,
    build_manifest,
    resume_cell,
)
from shobench.splits import load_split_by_name

# Any real cell will do for the fields these read; the smoke cell is the cheapest to load.
_SMOKE_CELL = "smoke-automationbench-claude-code"

# The child that leaves a stream the way a suspension leaves it: a task in flight and no
# unwinding. Written as a program rather than a fixture because the whole point is a process
# that ends without running its cleanup, which cannot be done inside the test process.
_HOLD_A_TASK_THEN = """
import asyncio, json, os, sys
from pathlib import Path
import shogym
from shogym.serve import Immediate, TaskRef, TaskStream

async def main():
    prov, ending = Path(sys.argv[1]), sys.argv[2]
    stream = TaskStream(
        shogym.make,
        [TaskRef("wordle_v1", 0), TaskRef("wordle_v1", 1)],
        prov_dir=prov,
        feedback=Immediate(),
    )
    async with stream:
        await stream.get_task()
        if ending == "suspended":
            # What _suspend_and_exit does: leave without unwinding the stream.
            sys.stdout.flush()
            os._exit({code})

asyncio.run(main())
"""


def _rows(prov: Path) -> list[dict]:
    path = prov / "results.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _hold_a_task_then(prov: Path, ending: str) -> subprocess.CompletedProcess[str]:
    program = _HOLD_A_TASK_THEN.format(code=SUSPENDED_EXIT_CODE)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program), str(prov), ending],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_only_an_unwound_stream_forfeits_the_task_it_was_holding(tmp_path: Path) -> None:
    """Why a suspension exits hard, stated as the difference it makes to the record.

    shogym's orderly close drains whatever is in flight into a scored row, and its resume
    replays queue positions that have no row, so a tidy exit spends the task the agent was
    working on and a resumed stream never offers it again. Ending without unwinding leaves the
    claim on disk and the position row-less, which is the state ``resume=True`` exists for.
    This is the premise the suspension design rests on, so it is checked against the real
    shogym rather than assumed from its documentation.
    """
    import asyncio

    import shogym
    from shogym.serve import Immediate, TaskRef, TaskStream

    async def reopen(prov: Path) -> int:
        stream = TaskStream(
            shogym.make,
            [TaskRef("wordle_v1", 0), TaskRef("wordle_v1", 1)],
            prov_dir=prov,
            feedback=Immediate(),
            resume=True,
        )
        async with stream:
            return stream.queue_info().remaining

    suspended_dir = tmp_path / "suspended"
    suspended_dir.mkdir()
    ended = _hold_a_task_then(suspended_dir, "suspended")
    assert ended.returncode == SUSPENDED_EXIT_CODE, ended.stderr[-2000:]
    assert (suspended_dir / "claim.json").is_file(), "the claim a resume reclaims"
    assert _rows(suspended_dir) == [], "the task in flight must reach no row at all"
    assert asyncio.run(reopen(suspended_dir)) == 2, "both positions are still to serve"

    orderly_dir = tmp_path / "orderly"
    orderly_dir.mkdir()
    assert _hold_a_task_then(orderly_dir, "orderly").returncode == 0
    assert [row["closure"] for row in _rows(orderly_dir)] == ["drained"]
    assert asyncio.run(reopen(orderly_dir)) == 1, "the drained position is gone for good"


def test_a_suspension_records_the_run_and_publishes_nothing(tmp_path: Path) -> None:
    """A suspended cell leaves a plan, not an ending.

    The record has to carry what the process that wrote it knew and no longer will: which
    session to reattach to, how much of the wall clock is spent, and how far the pool got. What
    it must not leave is a results file, because a rollout stopped by a provider has not
    reached the terminus that eval_after is supposed to follow.
    """
    run_dir = tmp_path / "run"
    program = f"""
    import json, sys
    from pathlib import Path
    from shobench.config import load_cell_by_name
    from shobench.containers import CellSandbox
    from shobench.harness import StopKind, StopVerdict
    from shobench.runner import LegRecord, RunContext, _suspend_and_exit

    run_dir = Path({str(run_dir)!r})
    cell = load_cell_by_name({_SMOKE_CELL!r})
    ctx = RunContext(
        cell=cell,
        split=None,
        instruction=None,
        harness=None,
        run_id="run-under-test",
        run_dir=run_dir,
        sandbox=CellSandbox(run_id="run-under-test", home=run_dir / "home", workdir=run_dir / "w"),
    )
    record = LegRecord(
        leg=0,
        phase="rollout",
        task_idx=None,
        started_at=1000.0,
        ended_at=2800.0,
        returncode=0,
        verdict=StopVerdict(StopKind.USAGE_LIMIT, "the window closed", {{"where": "stdout"}}),
        tasks_consumed_before=0,
        tasks_consumed_after=4,
        trace_path=str(run_dir / "rollout" / "traces" / "leg-0000.stream.jsonl"),
        run_dir=run_dir,
        session_id="sess-1",
    )
    ctx.legs.append(record)
    ctx.teardown = lambda: Path(run_dir / "torn-down").write_text("yes")
    _suspend_and_exit(
        ctx,
        record=record,
        session_id="sess-1",
        elapsed_rollout_s=600.0,
        tasks_dispensed=4,
        pool_queued=40,
        rollout_wall_clock_s=cell.budget.rollout_wall_clock_s,
    )
    print("returned", file=sys.stderr)
    """
    run_dir.mkdir()
    # What a real run would have on disk by now: the agent's home and the provenance it
    # has been writing. A suspension must leave both where they are.
    (run_dir / "home").mkdir()
    (run_dir / "rollout").mkdir()
    ended = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)], capture_output=True, text=True
    )
    assert ended.returncode == SUSPENDED_EXIT_CODE, ended.stderr[-2000:]
    assert "returned" not in ended.stderr, "the suspension must not return into the phase"

    record = json.loads((run_dir / SUSPENSION_FILE).read_text())
    assert record["session_id"] == "sess-1"
    assert record["tasks_dispensed"] == 4
    assert record["elapsed_rollout_s"] == 600.0
    assert record["remaining_rollout_s"] == record["rollout_wall_clock_s"] - 600.0
    assert record["stop_evidence"]["kind"] == StopKind.USAGE_LIMIT.value
    assert record["stop_evidence"]["resumable"] is True
    assert "shobench resume --run" in record["resume_with"]

    # The containers stop, the record is complete, and nothing is published or removed.
    assert (run_dir / "torn-down").is_file()
    assert json.loads((run_dir / "legs.json").read_text())[0]["session_id"] == "sess-1"
    assert (run_dir / "home").is_dir() and (run_dir / "rollout").is_dir()
    assert not list(tmp_path.glob("**/results/*.json"))


def test_the_exit_happens_even_when_everything_after_the_record_fails(tmp_path: Path) -> None:
    """Once the record is on disk, nothing may keep the process from leaving.

    Teardown and the console line are courtesies; the exit is the correctness property. If a
    docker call hangs up or the output pipe is closed, an exception here would unwind back
    through the stream, shogym would drain the task in flight into a scored row, and the
    position the suspension exists to preserve would be spent. So this child breaks both: its
    teardown raises and its stderr is closed before the suspension runs.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    program = f"""
    import os, sys
    from pathlib import Path
    from shobench.config import load_cell_by_name
    from shobench.containers import CellSandbox
    from shobench.harness import StopKind, StopVerdict
    from shobench.runner import LegRecord, RunContext, _suspend_and_exit

    run_dir = Path({str(run_dir)!r})
    cell = load_cell_by_name({_SMOKE_CELL!r})
    ctx = RunContext(
        cell=cell,
        split=None,
        instruction=None,
        harness=None,
        run_id="run-under-test",
        run_dir=run_dir,
        sandbox=CellSandbox(run_id="r", home=run_dir / "home", workdir=run_dir / "w"),
    )

    def teardown():
        raise RuntimeError("the docker daemon is not answering")

    ctx.teardown = teardown
    record = LegRecord(
        leg=0,
        phase="rollout",
        task_idx=None,
        started_at=0.0,
        ended_at=1.0,
        returncode=0,
        verdict=StopVerdict(StopKind.USAGE_LIMIT, "the window closed"),
        tasks_consumed_before=0,
        tasks_consumed_after=1,
        trace_path=str(run_dir / "t"),
        run_dir=run_dir,
        session_id="sess-1",
    )
    ctx.legs.append(record)
    os.close(2)  # a closed pipe: the console line cannot be written either
    _suspend_and_exit(
        ctx,
        record=record,
        session_id="sess-1",
        elapsed_rollout_s=60.0,
        tasks_dispensed=1,
        pool_queued=10,
        rollout_wall_clock_s=cell.budget.rollout_wall_clock_s,
    )
    os._exit(0)  # reached only if the suspension returned, which it must not
    """
    ended = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)], capture_output=True, text=True
    )
    assert ended.returncode == SUSPENDED_EXIT_CODE, ended.stdout[-2000:]
    assert json.loads((run_dir / SUSPENSION_FILE).read_text())["session_id"] == "sess-1"


def test_a_continuation_inherits_what_is_left_of_the_one_wall_clock() -> None:
    """The rollout budget is the run's, not the invocation's.

    Each suspension folds the time already spent into the record, so the clock a continuation
    is given shrinks with every interruption instead of starting over. A cell interrupted three
    times is the same length of experiment as one that ran straight through.
    """
    budget = Budget(rollout_wall_clock_s=8 * 3600).rollout_wall_clock_s
    first = Suspension(
        run_id="r",
        session_id="s",
        elapsed_rollout_s=3600.0,
        tasks_dispensed=3,
        pool_queued=40,
        legs_before=1,
        suspended_at=1.0,
        rollout_wall_clock_s=budget,
    )
    assert first.remaining_rollout_s == budget - 3600

    # The second suspension carries the first one's hour plus its own.
    second = Suspension(**{**first.__dict__, "elapsed_rollout_s": 3600.0 + 5400.0})
    assert second.remaining_rollout_s == budget - 9000

    spent = Suspension(**{**first.__dict__, "elapsed_rollout_s": float(budget) + 60.0})
    assert spent.remaining_rollout_s == 0, "an overspent clock is zero, never negative"

    # The clock is the one the interrupted run was given, not the one the cell file says today.
    edited = Suspension(**{**first.__dict__, "rollout_wall_clock_s": 16 * 3600})
    assert first.remaining_rollout_s == budget - 3600
    assert edited.remaining_rollout_s == 16 * 3600 - 3600


def test_a_continuation_appends_to_the_record_rather_than_replacing_it(tmp_path: Path) -> None:
    """A resumed cell is a second process writing one run's record.

    The legs the suspended run wrote are read back and carried forward, so a finished cell shows
    its whole rollout. Writing only this process's legs would erase the stretch that was
    interrupted, which is the part a reader most wants when asking why a run took two goes.
    """
    from shobench.containers import CellSandbox
    from shobench.runner import LegRecord, RunContext

    ctx = RunContext(
        cell=load_cell_by_name(_SMOKE_CELL),
        split=None,
        instruction=None,
        harness=None,
        run_id="run-1",
        run_dir=tmp_path,
        sandbox=CellSandbox(run_id="run-1", home=tmp_path / "home", workdir=tmp_path / "w"),
        prior_legs=[{"leg": 0, "phase": "rollout", "session_id": "sess-1"}],
    )
    ctx.legs.append(
        LegRecord(
            leg=1,
            phase="rollout",
            task_idx=None,
            started_at=0.0,
            ended_at=1.0,
            returncode=0,
            verdict=StopVerdict(StopKind.CHOSEN, "it finished"),
            tasks_consumed_before=4,
            tasks_consumed_after=9,
            trace_path=str(tmp_path / "t"),
            run_dir=tmp_path,
            session_id="sess-1",
        )
    )
    records = ctx.leg_records()
    assert [leg["leg"] for leg in records] == [0, 1]
    assert {leg["session_id"] for leg in records} == {"sess-1"}


def test_resume_needs_a_suspension_and_spends_nothing_without_one(tmp_path: Path, capsys) -> None:
    """A run that was not suspended has nothing to continue, and says so before spending."""
    assert cli_main(["resume", "--run", str(tmp_path), "--go"]) == 1
    assert "no suspension record" in capsys.readouterr().err


def test_resume_prints_the_plan_and_refuses_an_exhausted_clock(tmp_path: Path, capsys) -> None:
    """The same ``--go`` contract as a fresh run, plus the one refusal a continuation needs.

    Without ``--go`` the suspension record is the plan and nothing runs. With it, a rollout
    whose wall clock is already gone is refused rather than handed a second budget, since that
    would make the continuation a longer experiment than the one that was interrupted.
    """
    cell = load_cell_by_name(_SMOKE_CELL)
    record = {
        "schema": "shobench.suspension/1",
        "run_id": "run-1",
        "cell": cell.name,
        "harness": cell.harness,
        "phase": "rollout",
        "session_id": "sess-1",
        "legs_before": 1,
        "tasks_dispensed": 4,
        "pool_queued": 40,
        "elapsed_rollout_s": float(cell.budget.rollout_wall_clock_s),
        "rollout_wall_clock_s": cell.budget.rollout_wall_clock_s,
        "remaining_rollout_s": 0.0,
        "stop_evidence": StopVerdict(StopKind.USAGE_LIMIT, "the window closed").to_json(),
        "suspended_at": 1.0,
        "resume_with": f"shobench resume --run {tmp_path} --go",
    }
    (tmp_path / SUSPENSION_FILE).write_text(json.dumps(record))

    assert cli_main(["resume", "--run", str(tmp_path)]) == 0
    printed = capsys.readouterr()
    assert json.loads(printed.out)["tasks_dispensed_so_far"] == 4
    assert "plan only" in printed.err

    assert cli_main(["resume", "--run", str(tmp_path), "--go"]) == 1
    assert "wall clock is already spent" in capsys.readouterr().err
    assert (tmp_path / SUSPENSION_FILE).is_file(), "a refusal leaves the run continuable"


# ----- a suspension followed by a continuation, end to end ------------------------------------
#
# The seams that need Docker are stood in for and nothing else is: the sandbox, the stream and
# its server, and the harness leg. What runs for real is the part that was wrong, which is
# everything the continuation does with the record it inherits.


def _record_eval_phase(phase_dir: Path, task_ids: tuple[int, ...]) -> list[TaskResult]:
    """Record a finished eval phase in the layout ``run_eval_phase`` actually writes.

    Each task gets its own stream under ``<phase>/task-<id>/``, because one fresh session per
    task is enforced by serving one task per stream, so a phase's rows are spread across those
    per-task directories and never sit in one ``results.jsonl`` at the phase root. An earlier
    fixture wrote a single flat directory the runner never produces, and a hydration bug that
    read only the phase root passed against it: a real run returned zero before rows. This
    writes the real layout, driven by actual streams, so the continuation is tested against the
    provenance it will meet.
    """
    import shogym
    from fastmcp import Client
    from shogym.serve import EvalStream, TaskRef, build_stream_server

    async def play(prov_dir: Path, task_id: int) -> None:
        stream = EvalStream(shogym.make, [TaskRef("wordle_v1", task_id)], prov_dir=prov_dir)
        async with stream:
            server = build_stream_server(stream, name="shogym")
            async with Client(server) as client:
                await client.call_tool("get_task", {})
                await client.call_tool("terminate", {})

    for task_id in task_ids:
        asyncio.run(play(phase_dir / f"task-{task_id:05d}", task_id))
    return runner.read_eval_phase(phase_dir, task_ids)


class _FakeStream:
    """A queue that answers what the rollout asks it, without an env or a server."""

    def __init__(self, remaining: int) -> None:
        from shogym.serve import QueueInfo

        self._info = QueueInfo(remaining=remaining, consumed=0, in_flight=0)

    def queue_info(self):
        return self._info

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _stand_in_for_docker(monkeypatch, *, leg: object) -> dict[str, object]:
    """Replace the seams a leg needs a daemon for, and report what the leg was asked to do."""
    import contextlib

    calls: dict[str, object] = {}
    monkeypatch.setattr(CellSandbox, "up", lambda self, **kw: None)
    monkeypatch.setattr(CellSandbox, "down", lambda self: calls.__setitem__("sandbox_down", True))
    monkeypatch.setattr(runner, "build_stream", lambda *a, **kw: _FakeStream(remaining=7))
    monkeypatch.setattr(runner, "free_port", lambda: 0)

    @contextlib.asynccontextmanager
    async def served(stream, port, host="0.0.0.0"):  # noqa: S104 - mirrors the real signature
        yield

    monkeypatch.setattr(runner, "_served", served)
    monkeypatch.setattr(runner, "run_leg", leg)
    return calls


def _suspended_run(
    tmp_path: Path, *, elapsed: float = 300.0
) -> tuple[Path, dict, list[TaskResult]]:
    """A run directory in the state a usage limit leaves: measured, half-run, and waiting."""
    cell = load_cell_by_name(_SMOKE_CELL)
    split = load_split_by_name(cell.split)
    run_dir = tmp_path / "runs" / "run-1"
    (run_dir / "home").mkdir(parents=True)
    (run_dir / "work").mkdir(parents=True)
    ctx = RunContext(
        cell=cell,
        split=split,
        instruction=load_instruction(cell.instruction_arm),
        harness=runner.harness_for(cell.harness),
        run_id="run-1",
        run_dir=run_dir,
        sandbox=CellSandbox(run_id="run-1", home=run_dir / "home", workdir=run_dir / "work"),
    )
    manifest = build_manifest(ctx, probes={"version": "test"})
    runner.write_json(run_dir / "manifest.json", manifest)

    # The four dispenses the first process durably wrote before the window closed, one per queue
    # position it reached. The published dispense count is read back from these records rather
    # than summed off a counter (see `dispensed_positions`), so the fake continuation needs a real
    # dispenses file to count even though its stand-in stream hands out nothing more. Distinct
    # positions, because the count is of positions and a resumed run's whole hazard is a position
    # that appears twice.
    rollout_dir = run_dir / "rollout"
    rollout_dir.mkdir(parents=True)
    dispenses = [
        {
            "lease": f"lease-{position}",
            "seq": position + 1,
            "position": position,
            "env": cell.env,
            "task_idx": position,
            "dispensed_at": 1.0,
            "feedback_regime": "never",
            "extensions": {},
        }
        for position in range(4)
    ]
    (rollout_dir / "dispenses.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in dispenses), encoding="utf-8"
    )

    # The cell's own held-out ids, in the per-task layout run_eval_phase writes: the continuation
    # has to gather rows across the task-<id>/ subdirectories, not from one flat directory the
    # runner never produces. More than one so the gathering is exercised rather than an accident
    # of a single directory that happened to be the phase root. They are the committed ids rather
    # than a convenient 0 and 1 because the published result is counted against that set: rows for
    # ids nobody requested are as wrong as ids with no rows.
    heldout = tuple(int(task_id) for task_id in split.heldout.task_ids)
    before = _record_eval_phase(run_dir / "eval_before", task_ids=heldout)
    runner.write_json(
        run_dir / "legs.json",
        [
            {
                "leg": 0,
                "phase": "eval_before",
                "session_id": "sess-eval",
                "task_idx": 0,
                "observed_models": ["model-from-the-first-process"],
            },
            {
                "leg": 0,
                "phase": "rollout",
                "session_id": "sess-1",
                "task_idx": None,
                "observed_models": ["model-from-the-first-process"],
            },
        ],
    )
    runner.write_json(
        run_dir / SUSPENSION_FILE,
        {
            "schema": "shobench.suspension/1",
            "run_id": "run-1",
            "cell": cell.name,
            "harness": cell.harness,
            "phase": "rollout",
            "session_id": "sess-1",
            "legs_before": 2,
            "tasks_dispensed": 4,
            "pool_queued": 11,
            "elapsed_rollout_s": elapsed,
            "rollout_wall_clock_s": cell.budget.rollout_wall_clock_s,
            "remaining_rollout_s": cell.budget.rollout_wall_clock_s - elapsed,
            "stop_evidence": StopVerdict(StopKind.USAGE_LIMIT, "the window closed").to_json(),
            "suspended_at": 1.0,
            "resume_with": f"uv run shobench resume --run {run_dir} --go",
        },
    )
    return run_dir, manifest, before


def test_a_continuation_publishes_the_measurement_the_first_process_took(
    tmp_path: Path, monkeypatch
) -> None:
    """The result a resumed cell publishes has to be the result an uninterrupted cell publishes.

    eval_before ran before the window closed, and its rows are on disk under the run directory.
    A continuation that starts its phase table from nothing publishes a file whose before side
    is empty, whose paired deltas do not exist, and whose after rows are all unpaired: the one
    question the benchmark asks, silently unanswered, in a file that otherwise looks complete.
    So the continuation reads that phase back, pairs it, and carries the models and legs of the
    stretch it did not run.
    """
    run_dir, _, before = _suspended_run(tmp_path)
    cell = load_cell_by_name(_SMOKE_CELL)
    split = load_split_by_name(cell.split)
    seen: dict[str, object] = {}

    def leg(ctx, *, phase, leg, session_id, resume, timeout_s, user_prompt, **kw) -> LegRecord:
        seen[phase] = {
            "session_id": session_id,
            "resume": resume,
            "timeout_s": timeout_s,
            "user_prompt": user_prompt,
        }
        record = LegRecord(
            leg=leg,
            phase=phase,
            task_idx=kw.get("task_idx"),
            started_at=10.0,
            ended_at=20.0,
            returncode=0,
            verdict=StopVerdict(StopKind.CHOSEN, "it stopped on its own"),
            tasks_consumed_before=0,
            tasks_consumed_after=0,
            trace_path=str(ctx.run_dir / "t.jsonl"),
            run_dir=ctx.run_dir,
            observed_models=["model-from-the-continuation"],
            session_id=session_id,
        )
        # As the real one does: a leg is on the run's record whatever else happens to it.
        ctx.legs.append(record)
        return record

    _stand_in_for_docker(monkeypatch, leg=leg)
    # The eval that follows the rollout is a phase of its own with its own coverage; here it
    # only has to produce rows the published file can pair against.
    after = [TaskResult(**{**row.__dict__, "reward": 1.0, "success": True}) for row in before]

    async def eval_phase(ctx, phase):
        return after

    monkeypatch.setattr(runner, "run_eval_phase", eval_phase)

    results_path = asyncio.run(
        resume_cell(run_dir, results_dir=tmp_path / "results", capture_egress=False)
    )
    published = json.loads(results_path.read_text())

    # Both halves of the measurement, and the pairing between them. eval_before was recorded
    # across two per-task directories, so a published before side with fewer than both rows
    # means the continuation read the wrong directory level and the paired result is lost.
    assert [row["task_idx"] for row in published["eval_before"]["tasks"]] == sorted(
        int(task_id) for task_id in split.heldout.task_ids
    )
    assert published["eval_before"]["summary"]["n_requested"] == 2
    assert len(published["paired"]) == 2, "both recorded tasks must pair against the after side"
    assert not published["unpaired"]

    # The whole run's models and legs, not just this process's share.
    assert "model-from-the-first-process" in published["manifest"]["observed_models"]
    assert "model-from-the-continuation" in published["manifest"]["observed_models"]
    legs = json.loads((run_dir / "legs.json").read_text())
    assert [leg["phase"] for leg in legs] == ["eval_before", "rollout", "rollout"]

    # The rollout continued the recorded session on what was left of the recorded clock, and
    # sent the continuation cue rather than the opener: a resumed run is continued, not begun.
    instruction = load_instruction(cell.instruction_arm)
    assert seen["rollout"] == {
        "session_id": "sess-1",
        "resume": True,
        "timeout_s": cell.budget.rollout_wall_clock_s - 300,
        "user_prompt": instruction.continuation,
    }
    assert instruction.continuation != instruction.kickoff, "the two cues must be distinct"
    # The published count is the distinct positions in the record's real dispenses file, which is
    # the four the first process wrote. That the overcount a resume introduces is collapsed away
    # is the real-stream test's job below; this one only pins that the count is read from the
    # record rather than from a counter the stand-in stream cannot move.
    assert published["rollout"]["stopping"]["tasks_dispensed"] == 4
    assert published["manifest"]["resumptions"][0]["session_id"] == "sess-1"
    # The record was this run's retry handle, and the continuation owns the ending now.
    assert not (run_dir / SUSPENSION_FILE).exists()

    # The published number an operator reads: a resumed cell says it resumed once, not never.
    assert published["rollout"]["stopping"]["usage_limit_resumes"] == 1
    report = report_cell(published)
    assert report.resumes == 1, "report_cell must see the resume the run actually took"


def test_a_continuation_that_fails_can_still_be_tried_again(tmp_path: Path, monkeypatch) -> None:
    """The suspension record is the only handle a waiting cell has, so it outlives a bad attempt.

    A continuation is normally started hours later in a new shell, which is where a serving-side
    variable goes missing and a stream refuses to build. Consuming the record before the
    continuation had produced anything turned every such failure into a cell that could neither
    finish nor be tried again.
    """
    run_dir, _, _ = _suspended_run(tmp_path)

    def leg(*a, **kw):
        raise RuntimeError("the stream never came up")

    calls = _stand_in_for_docker(monkeypatch, leg=leg)
    with pytest.raises(RuntimeError, match="never came up"):
        asyncio.run(resume_cell(run_dir, results_dir=tmp_path / "results", capture_egress=False))

    assert (run_dir / SUSPENSION_FILE).is_file(), "the run must still be resumable"
    assert not (tmp_path / "results").exists(), "a failed attempt publishes nothing"
    assert calls.get("sandbox_down"), "the containers come down even when the attempt fails"


def test_a_continuation_refuses_an_experiment_that_changed_under_it(
    tmp_path: Path, monkeypatch
) -> None:
    """A cell edited while the run waited is a different experiment wearing its run id.

    The manifest recorded what this run is, digest by digest, before anything spent. If the
    checkout no longer matches, continuing would publish one run id describing two experiments:
    half a rollout under one budget or one pool, half under another. The refusal names what
    moved and leaves the run resumable.
    """
    run_dir, manifest, _ = _suspended_run(tmp_path)
    manifest["cell"]["config_sha256"] = "0" * 64
    runner.write_json(run_dir / "manifest.json", manifest)
    _stand_in_for_docker(monkeypatch, leg=lambda *a, **kw: None)

    with pytest.raises(RuntimeError, match="no longer matches"):
        asyncio.run(resume_cell(run_dir, results_dir=tmp_path / "results", capture_egress=False))
    assert (run_dir / SUSPENSION_FILE).is_file()


# ----- resume against a REAL TaskStream, the boundary the stand-in cannot reach ---------------
#
# The stand-in stream above has no live registry and no leases, so it cannot show either way a
# resumed rollout differs from an uninterrupted one: the interrupted task's lease going dead when
# the stream reopens, and the replayed position landing twice in the record. Both are properties
# of pinned shogym, so these drive the real thing. Every v0 cell serves max_in_flight 8, so the
# stream is lease-wrapped and this is the path they take, not a hypothetical one. No Docker: the
# stream and its record are the real code, and the harness leg is stood in for by pulling and
# terminating tasks directly, which is all the leg's process does over MCP anyway.

# A child that leaves a real max_in_flight-8 rollout stream the way a suspension leaves it: one
# task pulled, its lease written out, and a hard exit with no unwinding so the position stays
# row-less. Never() because the rollout serves under that regime, and a resume refuses a
# directory recorded under any other.
_ABANDON_A_TASK = """
import asyncio, os, sys
from pathlib import Path
import shogym
from shogym.serve import Never, TaskRef, TaskStream

async def main():
    prov = Path(sys.argv[1])
    stream = TaskStream(
        shogym.make,
        [TaskRef("wordle_v1", 0), TaskRef("wordle_v1", 1)],
        prov_dir=prov,
        feedback=Never(),
        max_in_flight=8,
    )
    async with stream:
        task = await stream.get_task()
        (prov / "OLD_LEASE").write_text(task.lease)
        sys.stdout.flush()
        os._exit({code})

asyncio.run(main())
"""


def _abandon_a_task(prov: Path) -> str:
    """Run the child and return the lease it was holding when it was cut off."""
    program = _ABANDON_A_TASK.format(code=SUSPENDED_EXIT_CODE)
    ended = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program), str(prov)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert ended.returncode == SUSPENDED_EXIT_CODE, ended.stderr[-2000:]
    assert _rows(prov) == [], "the interrupted position must reach no result row"
    return (prov / "OLD_LEASE").read_text().strip()


def _answer(result: object) -> str:
    """The text an env/stream tool result carries, however the transport wrapped it."""
    parts = getattr(result, "content", None) or []
    return "".join(getattr(part, "text", "") for part in parts) or str(result)


async def _resume_and_drain(prov: Path, old_lease: str) -> dict:
    """Reopen the abandoned record with resume=True and drive it to the end by re-pulling.

    This is what a resumed harness session has to do once its old lease is refused: pull the next
    task rather than retry the one it was holding. It records the facts each finding asserts on.
    """
    import shogym
    from shogym.serve import Never, TaskRef, TaskStream

    stream = TaskStream(
        shogym.make,
        [TaskRef("wordle_v1", 0), TaskRef("wordle_v1", 1)],
        prov_dir=prov,
        feedback=Never(),
        max_in_flight=8,
        resume=True,
    )
    facts: dict = {}
    async with stream:
        facts["remaining_on_open"] = stream.queue_info().remaining
        # The stale lease from before the suspension: the reopened stream has an empty live
        # registry, so it denotes no episode and cannot continue the interrupted task.
        facts["stale_lease_answer"] = _answer(
            await stream.dispatch("terminate", {}, lease=old_lease)
        )
        pulled: list[str] = []
        while (task := await stream.get_task()) is not None:
            pulled.append(task.lease)
            await stream.dispatch("terminate", {}, lease=task.lease)
        facts["pulled_leases"] = pulled
        facts["remaining_after"] = stream.queue_info().remaining
    return facts


async def _play_two_tasks(prov: Path) -> None:
    """An uninterrupted two-position rollout, played straight through, for the baseline."""
    import shogym
    from shogym.serve import Never, TaskRef, TaskStream

    stream = TaskStream(
        shogym.make,
        [TaskRef("wordle_v1", 0), TaskRef("wordle_v1", 1)],
        prov_dir=prov,
        feedback=Never(),
        max_in_flight=8,
    )
    async with stream:
        while (task := await stream.get_task()) is not None:
            await stream.dispatch("terminate", {}, lease=task.lease)


def test_a_resumed_stream_recovers_the_interrupted_position_by_re_pulling(tmp_path: Path) -> None:
    """The interrupted task is abandoned and replayed, and re-pulling is what recovers it.

    Pinned shogym reopens with an empty live registry, so the lease the suspended session was
    holding denotes nothing: at max_in_flight above 1 (every v0 cell) its next call under that
    lease is refused as ``unknown_lease``, and the runner cannot make a tool call to re-attach it.
    What recovers the position is the resumed session pulling again: shogym re-offers the row-less
    position on the next ``get_task`` under a fresh lease, and once both positions play the queue
    is exhausted, so nothing is lost or left stalled. The continuation cue is the runner's only
    lever on a real agent, so it must tell it to do exactly this rather than retry a dead handle.
    """
    prov = tmp_path / "rollout"
    prov.mkdir()
    old_lease = _abandon_a_task(prov)

    facts = asyncio.run(_resume_and_drain(prov, old_lease))

    # The interrupted position is still to serve when the stream reopens, and so is the whole
    # two-position pool: the suspension left it row-less on purpose.
    assert facts["remaining_on_open"] == 2
    # The old lease is dead here, so continuing the in-flight task through it is impossible.
    assert "unknown_lease" in facts["stale_lease_answer"]
    # Re-pulling recovers the position under a new lease, never the old one, and both positions
    # play out to an exhausted queue: the resume mechanism loses no task and stalls on none.
    assert old_lease not in facts["pulled_leases"]
    assert len(facts["pulled_leases"]) == 2
    assert facts["remaining_after"] == 0

    # The cue must carry the instruction that makes a real agent re-pull rather than retry its
    # dead lease; a bare "Continue." would leave it acting on the interrupted task.
    instruction = load_instruction("get-better")
    assert "get_task" in instruction.continuation
    assert instruction.continuation != instruction.kickoff, "the two cues must be distinct"


def test_a_resumed_cell_publishes_the_uninterrupted_counts(tmp_path: Path) -> None:
    """A resumed two-position pool publishes the counts an uninterrupted one does.

    The replay leaves the record holding both the abandoned dispense (kept as a ``broker_abort``
    for audit) and the replay's real closure for one position, so the raw rows outnumber the
    positions. The published counts collapse that chain by position, and the dispensed total is
    the distinct positions rather than each process's counter summed, so the resumed run reports
    the two an uninterrupted run reports instead of three.
    """
    from shobench.results import (
        collapse_replays,
        dispensed_positions,
        read_phase,
        rollout_summary,
    )

    baseline = tmp_path / "uninterrupted"
    baseline.mkdir()
    asyncio.run(_play_two_tasks(baseline))
    base_summary = rollout_summary(read_phase(baseline))
    assert base_summary["tasks_attempted"] == 2
    assert dispensed_positions(baseline) == 2

    resumed = tmp_path / "resumed"
    resumed.mkdir()
    old_lease = _abandon_a_task(resumed)
    asyncio.run(_resume_and_drain(resumed, old_lease))
    rows = read_phase(resumed)

    # The abandoned dispense survives in the raw record for audit: three rows for two positions,
    # one position carrying both its abandonment and the replay that superseded it.
    assert [r.closure for r in rows] == ["broker_abort", "aborted", "aborted"]
    assert sorted(r.position for r in rows) == [0, 0, 1]
    assert len(collapse_replays(rows)) == 2

    # Collapsed, the resumed run matches the uninterrupted one exactly: one attempt per position,
    # both scored, no broker_abort in the tally, and the same distinct-position dispense total.
    summary = rollout_summary(rows)
    assert summary["tasks_attempted"] == base_summary["tasks_attempted"] == 2
    assert summary["tasks_scored"] == base_summary["tasks_scored"] == 2
    assert "broker_abort" not in summary["closures"]
    assert summary["closures"] == base_summary["closures"]
    assert dispensed_positions(resumed) == 2

    # Mutation guard: the pre-fix shaping counted rows and summed counters, both of which are
    # three for this record. That the raw rows outnumber the collapsed count is the overcount.
    assert len(rows) == 3
    assert len(rows) != summary["tasks_attempted"]
