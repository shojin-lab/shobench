"""Ending a run on purpose, and ending one that has stopped getting anywhere.

Two endings, one seam. Before either existed the only way an operator could stop a run was
``pkill`` plus ``docker rm -f``, which ends the runner before it can write ``legs.json`` and
``rollout_stopping.json``. A run stopped that way has no terminus, so ``rebookend`` refuses it
forever and the cell can never produce an eval_after: the cheap way to stop a wedged run destroys
the measurement while leaving it to burn its whole clock preserves it. Both tests below are
therefore about the RECORDS an ending leaves, not only about the ending.

The stall half is the one that has to be careful about what it does NOT do. Trace silence is not
evidence of a stall: a task can legitimately take an hour inside one tool call, and an agent that
delegates goes quiet in its own trace by design while its children work. So the reading is over
four sources and any one of them resets the clock, and three of the tests here exist to hold the
detector to that: a leg whose only activity is child artifacts, a leg whose only activity is
``/work``, and a leg that seals a row late are each doing exactly what the benchmark wants.

Nothing here needs Docker, a provider, or a bound port. The container is stood in for at the one
seam the runner owns, which is the leg supervisor; the stream is a fake that never drains, so the
drain watchdog stays out of tests that are not about it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

# Imported here rather than inside the fake queue below, so the first import of shogym happens at
# collection like every other import in the suite.
from shogym.serve import QueueInfo

from shobench import runner
from shobench.cli import main
from shobench.config import Budget, load_all_cells, load_cell_by_name, load_instruction
from shobench.containers import CellSandbox
from shobench.harness import StopKind
from shobench.harnesses import harness_for
from shobench.results import TaskResult
from shobench.runner import (
    STOP_REQUEST_FILE,
    EarlyEnding,
    LegRecord,
    RunContext,
    build_manifest,
)
from shobench.splits import Side, Split

_SMOKE_CELL = "smoke-automationbench-claude-code"
_PRIME_CELL = "hle-prime_agent-claude-opus-5"


def _ctx(
    tmp_path: Path, *, cell_name: str = _SMOKE_CELL, heldout: tuple[str, ...] = ("1",)
) -> RunContext:
    cell = load_cell_by_name(cell_name)
    run_dir = tmp_path / "run"
    sandbox = CellSandbox(run_id="test", home=run_dir / "home", workdir=run_dir / "work")
    sandbox.home.mkdir(parents=True)
    sandbox.workdir.mkdir(parents=True)
    split = Split(
        env=cell.env,
        heldout=Side(task_ids=heldout),
        pool=Side(task_ids=("900", "901")),
        provenance={"kind": "adopted"},
        source=tmp_path / "split.json",
    )
    return RunContext(
        cell=cell,
        split=split,
        instruction=load_instruction(cell.instruction_arm),
        harness=harness_for(cell.harness),
        run_id="test",
        run_dir=run_dir,
        sandbox=sandbox,
    )


def _with_bound(ctx: RunContext, bound_s: int) -> RunContext:
    """The same cell with a different no-progress bound, which is a per-cell budget field."""
    budget = replace(ctx.cell.budget, rollout_no_progress_s=bound_s)
    return replace(ctx, cell=replace(ctx.cell, budget=budget))


class _FakeStream:
    """A stream that never drains, so no ending but the one under test can reach a leg."""

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def queue_info(self) -> QueueInfo:
        return QueueInfo(remaining=1, consumed=0, in_flight=0)


@contextlib.asynccontextmanager
async def _fake_served(stream: object, port: int):
    yield


def _offline_rollout(monkeypatch) -> None:
    """Everything a rollout phase reaches outside this process, stood in for."""
    monkeypatch.setattr(runner, "build_stream", lambda *a, **k: _FakeStream())
    monkeypatch.setattr(runner, "_served", _fake_served)
    monkeypatch.setattr(runner, "read_phase", lambda prov_dir: [])
    monkeypatch.setattr(runner, "dispensed_positions", lambda prov_dir: 0)


class _AsksToStopMidLeg:
    """A container that runs until an operator asks it to end, and asks on its own first wait.

    Writing the request from inside the leg's own supervision is what makes "mid-leg" the fact
    under test rather than a race: the leg is provably launched and being waited on before the ask
    exists at all, so nothing here can pass by stopping a run that had not started.
    """

    stdin = None

    def __init__(self, run_dir: Path, reason: str) -> None:
        self.run_dir = run_dir
        self.reason = reason
        self.asked = False
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        if self.killed:
            return -9
        if not self.asked:
            self.asked = True
            runner.write_stop_request(self.run_dir, reason=self.reason)
        raise subprocess.TimeoutExpired("docker", timeout or 0)

    def kill(self) -> None:
        self.killed = True


def _supervising(monkeypatch, proc: object, removed: list[str]) -> None:
    monkeypatch.setattr(runner, "LEG_POLL_S", 0.01)
    monkeypatch.setattr(runner, "STOP_POLL_S", 0.01)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda argv, **kw: proc)
    monkeypatch.setattr(runner, "docker", lambda *args, **kw: removed.append(args[-1]))


def _stopped_run(tmp_path: Path, monkeypatch, *, reason: str = "wedged on a negamax cell") -> Path:
    """One rollout an operator's ask ended, through the real phase tail. Returns the run dir."""
    ctx = _ctx(tmp_path)
    _offline_rollout(monkeypatch)
    removed: list[str] = []
    # Built before the supervisor's fakes go in: standing in for Popen stands in for the one
    # `subprocess.run` uses too, and the manifest reads this checkout's revision through it.
    manifest = build_manifest(ctx, probes={"version": "test"})
    _supervising(monkeypatch, _AsksToStopMidLeg(ctx.run_dir, reason), removed)
    asyncio.run(
        runner._run_phases(
            ctx,
            manifest=manifest,
            # Both phases, so the record also says what the stop kept from starting.
            phases=("rollout", "eval_after"),
            results_dir=tmp_path / "results",
            observer=runner._Egress(None, ctx.run_dir),
        )
    )
    # The container was removed by name, not merely orphaned: one that outlived its leg would keep
    # spending and keep holding whatever it was handed.
    assert removed
    return ctx.run_dir


# ----- an operator's stop, and the records it leaves ------------------------------------------


def test_a_stop_mid_leg_writes_both_records_and_its_own_verdict_kind(
    tmp_path: Path, monkeypatch
) -> None:
    """The whole point of the entry: the ending happens through the normal path, so the two files
    a killed run never gets to write are on disk, and the leg says who ended it."""
    run_dir = _stopped_run(tmp_path, monkeypatch)

    legs = json.loads((run_dir / "legs.json").read_text())
    stopping = json.loads((run_dir / runner.ROLLOUT_STOPPING_FILE).read_text())

    verdict = legs[0]["verdict"]
    assert verdict["kind"] == "operator_stop"
    # Its own kind, and specifically none of the four it sits between: each of those means
    # something different to whoever reads the run.
    assert verdict["kind"] not in {"chosen_stop", "leg_timeout", "usage_limit", "stream_drained"}
    # Not resumable: nothing is waiting for a window to reopen, the run is over by a decision.
    assert verdict["resumable"] is False
    # The operator's own words about why reach the record, which is the only place they land.
    assert verdict["evidence"]["reason"] == "wedged on a negamax cell"
    assert stopping["stop_reason"] == "operator_stopped"
    assert stopping["stop_evidence"]["kind"] == "operator_stop"
    # Not counted as the agent's own stop, which is the metric the charter asks about.
    assert stopping["stopped_with_tasks_available"] is False


def test_a_stop_between_phases_keeps_the_next_phase_from_starting(
    tmp_path: Path, monkeypatch
) -> None:
    """A stop is an ask about the RUN. The rollout it interrupted still publishes, and eval_after,
    which had not begun, does not begin."""
    run_dir = _stopped_run(tmp_path, monkeypatch)
    manifest = json.loads((run_dir / "manifest.json").read_text())

    assert manifest["operator_stop"]["phases_not_run"] == ["eval_after"]
    assert manifest["operator_stop"]["reason"] == "wedged on a negamax cell"
    assert not (run_dir / "eval_after").exists()


def test_the_stopped_run_carries_the_terminus_a_rebookend_requires(
    tmp_path: Path, monkeypatch
) -> None:
    """The refusal this entry exists to remove, checked against its own two predicates.

    ``rebookend`` blocks on exactly these: a rollout terminus on disk, and a terminal session in
    it to fork. An operator-ended rollout has both, because the agent's state at the moment it was
    stopped is a real ending; a killed run has neither, because nothing got to write them.
    """
    run_dir = _stopped_run(tmp_path, monkeypatch)

    assert (run_dir / runner.ROLLOUT_STOPPING_FILE).is_file()
    assert runner.terminal_session_in(run_dir) is not None


def test_a_run_with_no_terminus_is_still_refused(tmp_path: Path) -> None:
    """The sibling assertion: the guard above is real, and a run stopped the old way still fails
    it. A test that only showed the stopped run passing would pass with the guard removed."""
    bare = tmp_path / "killed-run"
    bare.mkdir()

    with pytest.raises(RuntimeError, match="no rollout terminus"):
        asyncio.run(runner.rerun_eval(bare, results_dir=tmp_path / "results"))
    assert runner.terminal_session_in(bare) is None


def test_the_ask_is_consumed_so_a_later_reopening_is_not_stopped_by_it(
    tmp_path: Path, monkeypatch
) -> None:
    """The request is a one-shot signal and the leg verdict is where it becomes durable. Left on
    disk it would end the next process to open the directory, so an operator repairing an eval
    hole in a run they had stopped would watch the repair stop itself."""
    run_dir = _stopped_run(tmp_path, monkeypatch)

    assert not (run_dir / STOP_REQUEST_FILE).exists()
    assert json.loads((run_dir / "legs.json").read_text())[0]["verdict"]["evidence"]["reason"]


def test_a_stop_during_an_eval_phase_admits_no_further_tasks(tmp_path: Path, monkeypatch) -> None:
    """A stop is an ask about the run, and the eval fan-out is where ignoring it costs most.

    Every leg the supervisor kills a second after launching still paid for a home copy and a
    container start, once per remaining held-out id, so a phase that kept admitting would spend
    the whole fan-out on nothing. Admission closes on the ask exactly as it does on a usage limit.
    """
    ctx = _ctx(tmp_path, heldout=tuple(str(i) for i in range(1, 13)))
    launched: list[int] = []

    def fake_run_leg(ctx_arg: RunContext, **kw: object) -> LegRecord:
        idx = int(kw["task_idx"])  # type: ignore[arg-type]
        launched.append(idx)
        # The first leg to run is the moment the operator's ask lands.
        ctx_arg.operator_stop.fire(ctx_arg.harness.operator_verdict(request={"reason": "enough"}))
        return LegRecord(
            leg=idx,
            phase=str(kw["phase"]),
            task_idx=idx,
            started_at=0.0,
            ended_at=1.0,
            returncode=-1,
            verdict=ctx_arg.operator_stop.verdict,  # type: ignore[arg-type]
            tasks_consumed_before=0,
            tasks_consumed_after=0,
            trace_path="t",
            run_dir=ctx_arg.run_dir,
        )

    monkeypatch.setattr(runner, "warm_env", lambda cell: None)
    monkeypatch.setattr(runner, "build_stream", lambda *a, **k: _FakeStream())
    monkeypatch.setattr(runner, "_served", _fake_served)
    monkeypatch.setattr(runner, "EVAL_LAUNCH_STAGGER_S", 0.0)
    monkeypatch.setattr(runner, "read_phase", lambda prov_dir: [])
    monkeypatch.setattr(runner, "run_leg", fake_run_leg)
    asyncio.run(runner.run_eval_phase(ctx, "eval_before"))

    # The first wave was already admitted when the ask landed; nothing behind it was.
    assert 0 < len(launched) <= ctx.cell.budget.eval_concurrency < 12


def test_the_watcher_fires_on_an_ask_and_leaves_a_quiet_run_alone(tmp_path: Path) -> None:
    """The two halves of the poller, without a phase around them."""
    ctx = _ctx(tmp_path)
    ctx.run_dir.mkdir(parents=True, exist_ok=True)

    with runner._watching_for_stop(ctx):
        assert not ctx.operator_stop.fired.wait(timeout=0.2)

    ctx = _ctx(tmp_path / "second")
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    runner.write_stop_request(ctx.run_dir, reason="by hand")
    with runner._watching_for_stop(ctx):
        assert ctx.operator_stop.fired.wait(timeout=5.0)

    assert ctx.operator_stop.verdict is not None
    assert ctx.operator_stop.verdict.kind is StopKind.OPERATOR
    assert not (ctx.run_dir / STOP_REQUEST_FILE).exists()


def test_every_harness_reports_an_operator_stop_the_same_way() -> None:
    """The decision was neither the agent's nor the harness's, so the verdict is identical across
    harnesses and is never any of the kinds it sits between."""
    for name in ("claude_code", "codex", "prime_agent"):
        verdict = harness_for(name).operator_verdict(request={"reason": "r", "requested_at": 1.0})
        assert verdict.kind is StopKind.OPERATOR
        assert verdict.kind.value == "operator_stop"
        assert verdict.kind not in {
            StopKind.CHOSEN,
            StopKind.LEG_TIMEOUT,
            StopKind.USAGE_LIMIT,
            StopKind.DRAINED,
        }
        assert verdict.resumable is False
        assert verdict.evidence["reason"] == "r"


# ----- the CLI, which must not write an ask nobody will read ----------------------------------


def test_stopping_a_finished_run_is_a_clean_no_op(tmp_path: Path, capsys) -> None:
    """A run nobody owns will never read a request, and a file left behind would end the next
    process to reopen the directory. So the no-op writes nothing at all."""
    run_dir = tmp_path / "finished"
    run_dir.mkdir()
    (run_dir / runner.RUN_LOCK_FILE).write_text("{}", encoding="utf-8")

    assert main(["stop", "--run", str(run_dir)]) == 0
    assert not (run_dir / STOP_REQUEST_FILE).exists()
    assert "not owned by a live process" in capsys.readouterr().err


def test_stopping_a_live_run_writes_the_ask(tmp_path: Path) -> None:
    """Liveness is proven by the same shared flock a rebookend proves it with, never by a pid: the
    lock file is never unlinked, so a finished run names a pid the system may have reissued."""
    run_dir = tmp_path / "live"
    lock_fd = runner._acquire_run_lock(run_dir)
    try:
        assert main(["stop", "--run", str(run_dir), "--reason", "needed the machine"]) == 0
    finally:
        runner._release_run_lock(lock_fd)

    request = json.loads((run_dir / STOP_REQUEST_FILE).read_text())
    assert request["reason"] == "needed the machine"
    assert request["requested_at"] > 0


def test_stopping_a_directory_no_run_ever_owned_is_refused(tmp_path: Path, capsys) -> None:
    """No lock file means no shobench process was ever here, which is a mistyped path far more
    often than it is a run, so it refuses rather than quietly writing a file nothing reads."""
    empty = tmp_path / "not-a-run"
    empty.mkdir()

    assert main(["stop", "--run", str(empty)]) == 1
    assert not (empty / STOP_REQUEST_FILE).exists()
    assert "not a live run directory" in capsys.readouterr().err


# ----- the stall detector: what it fires on ----------------------------------------------------


def _watching(ctx: RunContext, *, bound_s: float, poll_s: float, ending: EarlyEnding):
    trace = runner.leg_trace_path(ctx.run_dir, "rollout", 0)
    trace.parent.mkdir(parents=True, exist_ok=True)
    prov = ctx.run_dir / "rollout"
    prov.mkdir(parents=True, exist_ok=True)
    return runner._watch_for_no_progress(
        ctx, ending, trace_path=trace, prov_dir=prov, bound_s=bound_s, poll_s=poll_s
    )


def test_a_leg_with_no_progress_anywhere_is_stalled(tmp_path: Path) -> None:
    """The observed wedge: a container alive and holding a core, writing nothing anywhere."""
    ctx = _ctx(tmp_path, cell_name=_PRIME_CELL)
    ending = EarlyEnding()

    assert asyncio.run(_watching(ctx, bound_s=0.2, poll_s=0.01, ending=ending)) is True

    assert ending.verdict is not None
    assert ending.verdict.kind is StopKind.STALLED
    assert ending.verdict.kind.value == "no_progress"
    # Distinguishable from an operator stop, which is the other ending that reaches this seam.
    assert ending.verdict.kind is not StopKind.OPERATOR
    assert ending.verdict.resumable is False
    # The rule and what actually happened, both: a leg ended a moment past its bound and one
    # silent for twice it are different observations about the same rule.
    assert ending.verdict.evidence["bound_s"] == 0.2
    assert ending.verdict.evidence["silent_s"] >= 0.2


def _never_stalls(ctx: RunContext, write, *, bound_s: float, poll_s: float) -> None:
    """Run the watcher against a leg whose ONLY activity is ``write``, and require it never fires.

    Four times the bound, so a detector that ignored the source would have fired three times over
    by the time this gives up waiting for it.
    """
    ending = EarlyEnding()

    async def drive() -> None:
        async def writer() -> None:
            while True:
                await asyncio.sleep(poll_s)
                write()

        writing = asyncio.create_task(writer())
        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(
                    _watching(ctx, bound_s=bound_s, poll_s=poll_s, ending=ending),
                    timeout=bound_s * 4,
                )
        finally:
            writing.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await writing

    asyncio.run(drive())
    assert not ending.fired.is_set()
    assert ending.verdict is None


def _appender(path: Path):
    """A write that moves both the size and the mtime of a tree, so no test rests on clock
    granularity: an appended line changes the byte count whatever the filesystem stores."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def write() -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write("a line\n")

    return write


def test_a_leg_whose_only_activity_is_child_artifacts_is_not_stalled(tmp_path: Path) -> None:
    """A delegating agent goes quiet in its own trace by design while its children work. prime
    keeps its RLM children under ``session-artifacts/<id>/sub-*``, and a detector that read the
    parent trace alone would end exactly the legs doing the most substantive work."""
    ctx = _ctx(tmp_path, cell_name=_PRIME_CELL)
    child = ctx.sandbox.home / ".prime/agent/session-artifacts/sess-1/sub-0/messages.jsonl"

    _never_stalls(ctx, _appender(child), bound_s=0.2, poll_s=0.02)


def test_a_leg_whose_only_activity_is_work_files_is_not_stalled(tmp_path: Path) -> None:
    """``/work`` is the writable cwd every harness runs in, so an agent building something has to
    touch it even while its trace says nothing."""
    ctx = _ctx(tmp_path, cell_name=_PRIME_CELL)

    _never_stalls(ctx, _appender(ctx.sandbox.workdir / "solver.py"), bound_s=0.2, poll_s=0.02)


def test_a_leg_that_seals_a_row_late_is_not_stalled(tmp_path: Path) -> None:
    """A task can legitimately take an hour inside one tool call. The clock is over the condition
    rather than over the start of the leg, so a row that lands late resets it rather than arriving
    after a leg the detector already ended."""
    ctx = _ctx(tmp_path, cell_name=_PRIME_CELL)
    sealed = ctx.run_dir / "rollout" / "results.jsonl"

    # Half the bound between seals: every row is late by the standard of the poll, and none of
    # them is late by the standard of the bound.
    _never_stalls(ctx, _appender(sealed), bound_s=0.2, poll_s=0.1)


def test_a_leg_whose_only_activity_is_its_own_trace_is_not_stalled(tmp_path: Path) -> None:
    """The obvious source is still a source: the detector adds three, it does not replace one."""
    ctx = _ctx(tmp_path, cell_name=_PRIME_CELL)
    trace = runner.leg_trace_path(ctx.run_dir, "rollout", 0)

    _never_stalls(ctx, _appender(trace), bound_s=0.2, poll_s=0.02)


def test_an_unreadably_large_work_tree_makes_the_check_inert_rather_than_firing(
    tmp_path: Path, monkeypatch
) -> None:
    """The fail-safe direction, stated as a test. An agent that npm-installs into /work makes the
    walk arbitrarily large; the worst that may produce is a rollout with no stall detection, which
    is what every rollout had before this existed."""
    ctx = _ctx(tmp_path, cell_name=_PRIME_CELL)
    monkeypatch.setattr(runner, "PROGRESS_WALK_LIMIT", 1)
    for name in ("a", "b", "c"):
        (ctx.sandbox.workdir / name).write_text("x", encoding="utf-8")

    ending = EarlyEnding()

    async def drive() -> None:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                _watching(ctx, bound_s=0.1, poll_s=0.01, ending=ending), timeout=0.5
            )

    asyncio.run(drive())
    assert not ending.fired.is_set()


# ----- the stall detector: where it applies, and what says how long -----------------------------


def _watched_bounds(monkeypatch) -> list[float]:
    """Record the bound every progress watcher a phase builds is given."""
    seen: list[float] = []
    real = runner._watch_for_no_progress

    async def watch(*args: object, **kw: object) -> bool:
        seen.append(float(kw["bound_s"]))  # type: ignore[arg-type]
        return await real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(runner, "_watch_for_no_progress", watch)
    return seen


def _one_rollout(ctx: RunContext, monkeypatch) -> None:
    _offline_rollout(monkeypatch)
    monkeypatch.setattr(runner, "_supervise", lambda *a, **kw: (0, False, False))
    asyncio.run(runner.run_rollout_phase(ctx))


def test_the_bound_the_rollout_watches_against_comes_from_the_cell(
    tmp_path: Path, monkeypatch
) -> None:
    """A per-cell budget field, not a constant in the runner: 'no progress for fifteen minutes'
    means one thing for tasks measured in minutes and another for tasks measured in hours."""
    seen = _watched_bounds(monkeypatch)

    _one_rollout(_with_bound(_ctx(tmp_path), 1234), monkeypatch)

    assert seen == [1234.0]


def test_a_cell_that_asks_for_no_bound_gets_no_watcher(tmp_path: Path, monkeypatch) -> None:
    """Zero is not a zero-length bound, it is the absence of one, which is also what every run
    recorded before the field existed ran under."""
    seen = _watched_bounds(monkeypatch)

    _one_rollout(_with_bound(_ctx(tmp_path), 0), monkeypatch)

    assert seen == []


def test_the_eval_phase_is_untouched(tmp_path: Path, monkeypatch) -> None:
    """Scoped to the rollout, which is the only leg with no other bound. Every eval leg is already
    bounded per task by ``eval_task_timeout_s``, and where that bound is legitimately an hour a
    second guard adds nothing but a way to end a task that is still working."""
    seen = _watched_bounds(monkeypatch)
    ctx = _ctx(tmp_path)

    monkeypatch.setattr(runner, "warm_env", lambda cell: None)
    monkeypatch.setattr(runner, "build_stream", lambda *a, **k: _FakeStream())
    monkeypatch.setattr(runner, "_served", _fake_served)
    monkeypatch.setattr(runner, "EVAL_LAUNCH_STAGGER_S", 0.0)
    monkeypatch.setattr(
        runner,
        "read_phase",
        lambda prov_dir: [
            TaskResult(seq=1, position=0, task_idx=1, closure="sealed", reward=1.0, success=True)
        ],
    )
    monkeypatch.setattr(runner, "_supervise", lambda *a, **kw: (0, False, False))
    asyncio.run(runner.run_eval_phase(ctx, "eval_before"))

    assert seen == []


def test_the_shipped_cells_bound_their_rollouts_generously(tmp_path: Path) -> None:
    """The default has to sit far above anything observed to be legitimate work, or the guard
    becomes the thing that ends the agents doing the most computation. The longest single tool
    call in the archives is a yc_bench hour."""
    assert Budget(rollout_wall_clock_s=1).rollout_no_progress_s == 7200
    for cell in load_all_cells():
        bound = cell.budget.rollout_no_progress_s
        assert bound == 0 or bound >= 3600, cell.name
        if cell.budget.rollout_wall_clock_s > 3600:
            # On a cell whose rollout is longer than the bound, the bound has to be reachable, or
            # the wall clock is the only ending there ever was. A smoke cell is the other case:
            # its whole rollout is shorter than any legitimate silence, so nothing is lost.
            assert bound < cell.budget.rollout_wall_clock_s, cell.name


def test_the_bound_is_read_off_the_cell_file(tmp_path: Path) -> None:
    """It is a budget key like the others, so a cell whose tasks run longer than the default can
    say so in the one place every other bound is written down."""
    path = tmp_path / "cell.toml"
    path.write_text(
        "\n".join(
            [
                "[cell]",
                'name = "t"',
                'env = "wordle_v1"',
                'harness = "claude_code"',
                'model = "claude-opus-5"',
                'split = "s"',
                "[budget]",
                "rollout_wall_clock_s = 28800",
                "rollout_no_progress_s = 14400",
            ]
        ),
        encoding="utf-8",
    )
    from shobench.config import load_cell

    cell = load_cell(path)
    assert cell.budget.rollout_no_progress_s == 14400
    assert cell.to_manifest()["budget"]["rollout_no_progress_s"] == 14400


# ----- the new field must not refuse the runs that predate it -----------------------------------


def test_a_record_that_predates_the_bound_still_bookends(tmp_path: Path) -> None:
    """A field added to the cell exists on one side of a bookend's comparison only, and the
    comparison fails closed on exactly that. Every archived run predates this one, so absence has
    to read as the value it truly means: a rollout that ran under no such bound.
    """
    cell = load_cell_by_name(_PRIME_CELL)
    recorded = cell.to_manifest()
    del recorded["budget"]["rollout_no_progress_s"]
    manifest = {"cell": recorded}

    # Recovered as unbounded, which is what the recorded rollout actually ran under.
    assert runner.recorded_no_progress_bound(manifest) == 0
    assert runner.bookend_cell(cell, manifest).budget.rollout_no_progress_s == 0
    # And so the comparison the rebookend refuses on says nothing about this field.
    inherited = runner.bookend_cell(cell, manifest).to_manifest()
    assert "budget.rollout_no_progress_s" not in runner.cell_field_drift(recorded, inherited)


def test_a_recorded_bound_is_inherited_rather_than_reread_from_the_checkout() -> None:
    """The sibling: a bookend runs no rollout, so publishing today's number would label the
    source's finished rollout with a bound nobody applied to it."""
    cell = load_cell_by_name(_PRIME_CELL)
    recorded = cell.to_manifest()
    recorded["budget"]["rollout_no_progress_s"] = 999
    manifest = {"cell": recorded}

    assert runner.bookend_cell(cell, manifest).budget.rollout_no_progress_s == 999
    assert runner.reopened_cell(cell, manifest).budget.rollout_no_progress_s == 999
