"""Suspending a rollout a provider stopped, and continuing it later.

A subscription window can close in the middle of an eight-hour rollout. The run is not over
when that happens, so the cell suspends instead of finishing: no eval_after, no results file,
nothing removed, and a record on disk saying what a continuation needs. What makes that
correct is a property of shogym rather than of this runner, and the first test here is the one
that pins it, because everything else assumes it.

None of this needs Docker. What needs Docker is the leg itself, so these drive the pieces
either side of it: the suspension record, the arithmetic a continuation runs on, and the
command an operator types.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from shobench.cli import main as cli_main
from shobench.config import Budget, load_cell_by_name
from shobench.harness import StopKind, StopVerdict
from shobench.runner import SUSPENDED_EXIT_CODE, SUSPENSION_FILE, Suspension

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
    )
    assert first.remaining_rollout_s(budget) == budget - 3600

    # The second suspension carries the first one's hour plus its own.
    second = Suspension(**{**first.__dict__, "elapsed_rollout_s": 3600.0 + 5400.0})
    assert second.remaining_rollout_s(budget) == budget - 9000

    spent = Suspension(**{**first.__dict__, "elapsed_rollout_s": float(budget) + 60.0})
    assert spent.remaining_rollout_s(budget) == 0, "an overspent clock is zero, never negative"


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
