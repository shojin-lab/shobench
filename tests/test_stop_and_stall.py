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
import os
import subprocess
import time
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


def _gone_within(path: Path, seconds: float) -> bool:
    """Did this file disappear inside the window, polled the way the CLI polls for it?"""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not path.exists():
            return True
        time.sleep(0.02)
    return not path.exists()


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


def _stopped_run(
    tmp_path: Path,
    monkeypatch,
    *,
    reason: str = "wedged on a negamax cell",
    transcript: bool = True,
) -> Path:
    """One rollout an operator's ask ended, through the real ownership and the real phase tail.

    Ownership is taken for real rather than faked, because the watcher that consumes the ask
    belongs to it: a run driven through ``_run_phases`` alone has no watcher at all, which is the
    state every already-running process is in and exactly what the CLI must refuse.

    ``transcript`` decides whether the leg got far enough to leave a resumable conversation
    behind, which is what separates a stop that can be bookended from one that cannot.
    """
    ctx = _ctx(tmp_path)
    # A short clock, so a regression that stops consuming the ask fails in seconds rather than
    # hanging the suite for the cell's real wall clock.
    budget = replace(ctx.cell.budget, rollout_wall_clock_s=30)
    ctx = replace(ctx, cell=replace(ctx.cell, budget=budget))
    _offline_rollout(monkeypatch)
    removed: list[str] = []
    # Built before the supervisor's fakes go in: standing in for Popen stands in for the one
    # `subprocess.run` uses too, and the manifest reads this checkout's revision through it.
    manifest = build_manifest(ctx, probes={"version": "test"})
    with runner.owning_run(ctx.run_dir) as stop:
        ctx = replace(ctx, operator_stop=stop)
        if transcript:
            _seed_claude_transcript(ctx, monkeypatch)
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


_PINNED_SESSION = "11111111-2222-3333-4444-555555555555"


def _seed_claude_transcript(ctx: RunContext, monkeypatch) -> None:
    """Put a resumable transcript where a leg that got that far would have left one.

    The leg is stood in for at the supervisor, so nothing writes the conversation a real one
    would. What the harness requires to reopen a session is the harness's own rule, and the file
    written here is one that rule accepts rather than an empty one wearing the right name: that is
    the whole difference between a terminus a bookend can fork and an id in a record.

    The session id is pinned so the seeded file and the leg agree on it; claude lets the runner
    choose one, which is exactly why an id alone proves nothing about a transcript existing.
    """
    monkeypatch.setattr(runner, "_fresh_session_id", lambda ctx_arg: _PINNED_SESSION)
    root = ctx.sandbox.home / ".claude" / "projects" / "-work"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{_PINNED_SESSION}.jsonl").write_text(
        json.dumps(
            {
                "type": "user",
                "sessionId": _PINNED_SESSION,
                "timestamp": "2026-08-14T00:00:00.000Z",
                "message": {"role": "user", "content": "Get Better"},
            }
        )
        + "\n",
        encoding="utf-8",
    )


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
    """The refusal this entry exists to remove, checked against the predicates rebookend uses.

    All three of them, not the easy two. A record naming a session is not a terminus a bookend can
    fork: the plan also resolves the transcript through the harness's own validation, and a
    stopped leg is exactly where an id can exist with no conversation behind it.
    """
    run_dir = _stopped_run(tmp_path, monkeypatch)
    session = runner.terminal_session_in(run_dir)

    assert (run_dir / runner.ROLLOUT_STOPPING_FILE).is_file()
    assert session is not None
    # The predicate the rebookend plan actually blocks on.
    assert harness_for("claude_code").session_transcript(run_dir / "home", session) is not None
    stopping = json.loads((run_dir / runner.ROLLOUT_STOPPING_FILE).read_text())
    assert stopping["terminus_rebookendable"] is True
    assert stopping["terminus_not_rebookendable_because"] == ""


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
    """The two halves of the poller that ownership runs, without a phase around them."""
    quiet = tmp_path / "quiet"
    with runner.owning_run(quiet) as stop:
        assert not stop.fired.wait(timeout=0.2)

    asked = tmp_path / "asked"
    asked.mkdir(parents=True)
    (asked / runner.RUN_LOCK_FILE).write_text("{}", encoding="utf-8")
    runner.write_stop_request(asked, reason="by hand")
    with runner.owning_run(asked) as stop:
        assert stop.fired.wait(timeout=5.0)
        assert stop.verdict is not None
        assert stop.verdict.kind is StopKind.OPERATOR
        # The unlink is the acknowledgment, and the CLI waits on exactly this. It follows the
        # fire rather than preceding it, so this waits for it the way the CLI does.
        assert _gone_within(asked / STOP_REQUEST_FILE, 5.0)


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


def test_stopping_a_watching_owner_writes_the_ask_and_waits_to_be_acknowledged(
    tmp_path: Path,
) -> None:
    """The whole protocol on the happy path: the owner advertises, the ask is written, the owner
    consumes it, and only then does the command report success.

    Liveness is proven by the same shared flock a rebookend proves it with, never by a pid: the
    lock file is never unlinked, so a finished run names a pid the system may have reissued.
    """
    run_dir = tmp_path / "live"
    seen: list[dict] = []
    with runner.owning_run(run_dir) as stop:
        assert main(["stop", "--run", str(run_dir), "--reason", "needed the machine"]) == 0
        seen.append(dict(stop.verdict.evidence)) if stop.verdict else None

    # Reported success only because the owner consumed it, which is also what leaves nothing
    # behind for a later reopening to latch.
    assert not (run_dir / STOP_REQUEST_FILE).exists()
    assert seen and seen[0]["reason"] == "needed the machine"
    assert seen[0]["requested_at"] > 0


def test_a_stop_against_an_owner_that_cannot_consume_it_is_refused(
    tmp_path: Path, capsys
) -> None:
    """The case that matters most in practice, and the one a busy lock alone gets wrong.

    A process started before the stop path existed holds its directory exactly as a current one
    does, and reads nothing. Writing an ask there does not stop it, reports success, and leaves a
    file for the next resume or rerun-eval to act on. So support is advertised in the lock and an
    owner that does not advertise is refused with nothing written.
    """
    run_dir = tmp_path / "older-build"
    # Ownership taken the way every entry took it before the watcher existed.
    lock_fd = runner._acquire_run_lock(run_dir)
    try:
        assert main(["stop", "--run", str(run_dir), "--reason", "wedged"]) == 1
    finally:
        runner._release_run_lock(lock_fd)

    assert not (run_dir / STOP_REQUEST_FILE).exists()
    err = capsys.readouterr().err
    assert "advertises no stop protocol" in err
    assert "would never be read" in err


def test_only_an_owner_that_watches_advertises_that_it_does(tmp_path: Path) -> None:
    """The advertisement is written by the one entry that starts the watcher, so it cannot claim
    support a directory does not have."""
    watching = tmp_path / "watching"
    with runner.owning_run(watching):
        assert runner.read_lock_holder(watching)["stop_protocol"] == runner.STOP_PROTOCOL

    bare = tmp_path / "bare"
    lock_fd = runner._acquire_run_lock(bare)
    try:
        assert "stop_protocol" not in runner.read_lock_holder(bare)
    finally:
        runner._release_run_lock(lock_fd)


def test_an_owner_that_leaves_without_reading_the_ask_has_it_withdrawn(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The window between the liveness probe and the write, where the owner can finish in between.

    An owner that goes away with the request still on disk never saw it, so the command takes the
    ask back under the now-free shared lock and says nothing was stopped. Leaving it would be the
    same landmine the advertisement check exists to prevent, reached by a race instead of by an
    old build.
    """
    run_dir = tmp_path / "finishes-mid-write"
    run_dir.mkdir()
    holder = {"pid": 1, "at": 1.0, "stop_protocol": runner.STOP_PROTOCOL}
    (run_dir / runner.RUN_LOCK_FILE).write_text(json.dumps(holder), encoding="utf-8")

    from shobench import cli

    answers = iter([True])  # live for the probe, gone by the time the ask is written

    monkeypatch.setattr(cli, "_owner_is_live", lambda d: next(answers, False))

    assert main(["stop", "--run", str(run_dir), "--reason", "too late"]) == 1

    assert not (run_dir / STOP_REQUEST_FILE).exists()
    err = capsys.readouterr().err
    assert "released it without reading the stop" in err
    assert "was withdrawn" in err


def test_an_ask_outside_the_phases_is_still_consumed(tmp_path: Path) -> None:
    """The watcher's lifetime is the LOCK's lifetime, not the phase loop's.

    An owner holds its directory through setup, between phases, and through publication and
    teardown. A watcher scoped to the phases left every ask outside that window accepted and
    unread, which is the state this asserts against: no phase runs here at all.
    """
    run_dir = tmp_path / "between-phases"
    with runner.owning_run(run_dir) as stop:
        runner.write_stop_request(run_dir, reason="while nothing is running")
        assert stop.fired.wait(timeout=5.0)
        assert _gone_within(run_dir / STOP_REQUEST_FILE, 5.0)
    assert stop.verdict is not None
    assert stop.verdict.kind is StopKind.OPERATOR


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


# ----- what a reading of a tree the host cannot see must never say -----------------------------


def test_an_unreadable_subtree_reads_as_progress_rather_than_as_silence(tmp_path: Path) -> None:
    """``os.walk`` swallows a directory whose listing fails and carries on as though it were not
    there, so a mode-000 subtree used to produce a perfectly stable ``(0, 0, 0)`` on every reading
    while the container went on writing inside it.

    The container runs as root and the watcher does not, so this is the ordinary case and not an
    exotic one: an agent that chmods a working directory would have had its rollout ended for
    silence while it was busy. A partial walk must never be reported as a complete one.
    """
    root = tmp_path / "work"
    locked = root / "locked"
    locked.mkdir(parents=True)
    (locked / "growing.txt").write_text("one\n", encoding="utf-8")
    os.chmod(locked, 0o000)
    try:
        readings = [runner._tree_pulse(root) for _ in range(3)]
    finally:
        os.chmod(locked, 0o700)

    assert len(set(readings)) == len(readings), readings
    assert all(reading[:2] == (-1, -1) for reading in readings)


def test_an_unreadable_root_reads_as_progress_too(tmp_path: Path) -> None:
    """The sibling: the tree the walk is pointed AT can be the unreadable one, and a check that
    only handled unreadable children would call that one empty."""
    root = tmp_path / "sealed"
    root.mkdir()
    (root / "inside.txt").write_text("x", encoding="utf-8")
    os.chmod(root, 0o000)
    try:
        assert runner._tree_pulse(root) != runner._tree_pulse(root)
    finally:
        os.chmod(root, 0o700)


def test_a_tree_that_does_not_exist_yet_reads_as_stable_emptiness(tmp_path: Path) -> None:
    """The regression the obvious fix introduces, held down.

    Making every unreadable tree reset the clock is only correct if ABSENCE is not unreadable.
    prime's session-artifact tree appears the first time a child session runs, so a reading that
    called absence unknowable would disable the detector for every prime cell until then, which is
    precisely the cell the detector was written for.
    """
    absent = tmp_path / "home" / ".prime" / "agent" / "session-artifacts"

    assert runner._tree_pulse(absent) == runner._tree_pulse(absent) == (0, 0, 0)


def test_a_leg_writing_under_an_unreadable_subtree_is_not_stalled(tmp_path: Path) -> None:
    """The whole point of the reading, end to end: work the host cannot see is still work."""
    ctx = _ctx(tmp_path, cell_name=_PRIME_CELL)
    locked = ctx.sandbox.workdir / "locked"
    locked.mkdir(parents=True)
    write = _appender(locked / "solver.py")
    write()
    os.chmod(locked, 0o000)
    try:
        ending = EarlyEnding()

        async def drive() -> None:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(
                    _watching(ctx, bound_s=0.2, poll_s=0.02, ending=ending), timeout=0.8
                )

        asyncio.run(drive())
    finally:
        os.chmod(locked, 0o700)

    assert not ending.fired.is_set()


# ----- which of two endings a leg reports ------------------------------------------------------


def test_the_ending_reported_is_the_one_that_fired_first(tmp_path: Path) -> None:
    """Not the one the caller listed first. ``run_leg`` appends the run's operator handle after the
    leg's own, so tuple order always preferred the stall or the drain; an operator who got there
    first was then recorded as a stall in the leg while the manifest recorded the ask, which is one
    run described two ways.
    """
    harness = harness_for("prime_agent")

    operator, stalled = EarlyEnding(), EarlyEnding()
    operator.fire(harness.operator_verdict(request={"reason": "needed the machine"}))
    stalled.fire(harness.no_progress_verdict(bound_s=7200, silent_s=7201.0))
    # The order run_leg builds: the leg's own handle first, the run's operator handle last.
    assert runner._fired_verdict((stalled, operator)).kind is StopKind.OPERATOR

    operator, stalled = EarlyEnding(), EarlyEnding()
    stalled.fire(harness.no_progress_verdict(bound_s=7200, silent_s=7201.0))
    operator.fire(harness.operator_verdict(request={"reason": "too late"}))
    assert runner._fired_verdict((stalled, operator)).kind is StopKind.STALLED


def test_a_handle_that_already_fired_keeps_its_first_verdict(tmp_path: Path) -> None:
    """First writer wins within one handle as well as between them."""
    harness = harness_for("prime_agent")
    ending = EarlyEnding()

    assert ending.fire(harness.operator_verdict(request={"reason": "first"})) is True
    assert ending.fire(harness.no_progress_verdict(bound_s=1, silent_s=2)) is False
    assert ending.verdict is not None
    assert ending.verdict.kind is StopKind.OPERATOR


# ----- whether the ending it reached can be bookended -------------------------------------------


def test_a_stop_before_a_transcript_lands_says_it_cannot_be_bookended(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The promise the command used to make unconditionally, and could not keep.

    A stop can end a leg before a resumable conversation exists. claude runs under an id the
    RUNNER pinned, so the record names a session whatever the leg wrote, and an operator told the
    terminus was bookendable would find out otherwise only when `rebookend` refused it. So the run
    checks the predicate `rebookend` checks, and says which of the two endings it got.
    """
    run_dir = _stopped_run(tmp_path, monkeypatch, transcript=False)
    stopping = json.loads((run_dir / runner.ROLLOUT_STOPPING_FILE).read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())

    # The record names a session, and that is exactly what is not sufficient.
    assert runner.terminal_session_in(run_dir) is not None
    assert stopping["terminus_rebookendable"] is False
    assert "no resumable transcript" in stopping["terminus_not_rebookendable_because"]
    assert manifest["operator_stop"]["terminus_rebookendable"] is False
    assert "NOT resumable" in capsys.readouterr().err


def test_a_stop_after_a_transcript_lands_says_it_can_be(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The sibling assertion, so the check above is not simply always false."""
    _stopped_run(tmp_path, monkeypatch, transcript=True)

    err = capsys.readouterr().err
    assert "terminus is resumable" in err
    assert "rebookend --run" in err
