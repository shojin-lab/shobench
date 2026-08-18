"""What the concurrency reader may claim from a record that holds no sealing time.

The floor is the whole design, so these tests are mostly about where it stops. A stream whose
tasks sealed in the order they were pulled reads as unknown, never as sequential, because that
record is equally consistent with an agent holding eight tasks and finishing them in turn. A
seal out of pull order is the case the record does settle, and a lease with no result row at all
is the case it settles only as far as a leg record or a replay of its queue position reaches.

The floor has to hold in both directions, so the cases that must NOT read as overlap are here
beside the ones that must: a lease its replay proves was over, a one-slot ceiling, and the
trailing drained rows a saturated ending leaves, which are bracketed rather than assigned.

The fixtures are the shapes shogym writes: a dispense record carrying the lease, its queue
position and the moment it was handed out, and a result row per seal, appended in seal order.
One case is taken from a real ``TaskStream`` instead, because the trailing-row shape is the one
worth checking against the writer rather than against a belief about it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shobench import cli
from shobench.concurrency import find_runs, render_table, run_concurrency, stream_concurrency

CEILING = 8


def _dispense(seq: int, at: float, position: int | None = None) -> dict:
    return {
        "lease": f"lease-{seq:04d}",
        "seq": seq,
        "position": seq - 1 if position is None else position,
        "env": "toy",
        "task_idx": seq,
        "dispensed_at": at,
        "feedback_regime": "never",
        "extensions": {},
    }


def _result(seq: int, closure: str = "sealed", position: int | None = None) -> dict:
    return {
        "seq": seq,
        "lease": f"lease-{seq:04d}",
        "position": seq - 1 if position is None else position,
        "env": "toy",
        "task_idx": seq,
        "closure": closure,
        "score": {"reward": 1.0, "success": True, "feedback": []},
        "observed": [],
        "diagnostic": None,
        "extensions": {},
        "feedback_regime": "never",
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _run(
    tmp_path: Path,
    *,
    dispenses: list[dict],
    results: list[dict],
    phase: str = "rollout",
    legs: list[dict] | None = None,
    ceiling: int = CEILING,
) -> Path:
    run_dir = tmp_path / "run"
    _write_jsonl(run_dir / phase / "dispenses.jsonl", dispenses)
    _write_jsonl(run_dir / phase / "results.jsonl", results)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "toy-run", "cell": {"max_in_flight": ceiling}}), encoding="utf-8"
    )
    if legs is not None:
        (run_dir / "legs.json").write_text(json.dumps(legs), encoding="utf-8")
    return run_dir


def _rollout(run_dir: Path):
    (row,) = [row for row in run_concurrency(run_dir) if row.phase == "rollout"]
    return row


def test_one_task_at_a_time_is_reported_as_unknown_not_sequential(tmp_path):
    """Sealing in pull order is what a sequential run looks like, and not only that run."""
    run_dir = _run(
        tmp_path,
        dispenses=[_dispense(1, 100.0), _dispense(2, 200.0), _dispense(3, 300.0)],
        results=[_result(1), _result(2), _result(3)],
    )
    row = _rollout(run_dir)
    assert row.dispensed == 3
    assert row.max_in_flight == CEILING
    assert row.max_open == 1
    assert row.mean_open_at_pull == 0.0
    assert row.strictly_sequential is None
    assert row.to_json()["seal_times"] == "unrecorded"


def test_a_stream_that_dispensed_one_task_is_provably_sequential(tmp_path):
    row = _rollout(_run(tmp_path, dispenses=[_dispense(1, 100.0)], results=[_result(1)]))
    assert row.strictly_sequential is True
    assert row.max_open == 1


def test_a_one_slot_ceiling_is_provably_sequential(tmp_path):
    """One slot serves one lease, whatever the seal order leaves open to interpretation."""
    run_dir = _run(
        tmp_path,
        dispenses=[_dispense(1, 100.0), _dispense(2, 200.0)],
        results=[_result(1), _result(2)],
        ceiling=1,
    )
    row = _rollout(run_dir)
    assert row.max_in_flight == 1
    assert row.strictly_sequential is True


def test_a_seal_out_of_pull_order_proves_two_leases_were_open(tmp_path):
    """Task 2 sealed after task 3, so it was open across task 3's whole life."""
    run_dir = _run(
        tmp_path,
        dispenses=[_dispense(1, 100.0), _dispense(2, 200.0), _dispense(3, 300.0)],
        results=[_result(1), _result(3), _result(2)],
    )
    row = _rollout(run_dir)
    assert row.max_open == 2
    assert row.strictly_sequential is False
    assert row.mean_open_at_pull == pytest.approx(1 / 3)
    assert row.never_sealed == 0


def test_a_forced_row_above_an_earned_one_is_certainly_a_displacement(tmp_path):
    """The close claims every live task in one pass, so nothing is earned after it begins."""
    run_dir = _run(
        tmp_path,
        dispenses=[_dispense(seq, 100.0 * seq) for seq in range(1, 5)],
        results=[
            _result(1),
            _result(2, closure="drained"),
            _result(4),
            _result(3, closure="drained"),
        ],
    )
    row = _rollout(run_dir)
    assert row.drained == 2
    assert (row.displaced_at_least, row.displaced_at_most) == (1, 2)
    assert row.strictly_sequential is False


def test_a_saturated_ending_reports_displacement_as_a_range(tmp_path):
    """A stream that ends holding a full registry mixes both causes in its trailing rows, and
    only the ceiling bounds how many of them the close can account for."""
    run_dir = _run(
        tmp_path,
        dispenses=[_dispense(seq, 100.0 * seq) for seq in range(1, 4)],
        results=[_result(seq, closure="drained") for seq in (1, 2, 3)],
        ceiling=2,
    )
    row = _rollout(run_dir)
    assert row.drained == 3
    assert (row.displaced_at_least, row.displaced_at_most) == (1, 3)


def test_without_a_ceiling_the_trailing_rows_bound_nothing(tmp_path):
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "rollout" / "dispenses.jsonl", [_dispense(seq, 100.0 * seq) for seq in (1, 2)]
    )
    _write_jsonl(
        run_dir / "rollout" / "results.jsonl",
        [_result(seq, closure="drained") for seq in (1, 2)],
    )
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "toy-run"}), encoding="utf-8")
    (row,) = run_concurrency(run_dir)
    assert row.max_in_flight is None
    assert (row.displaced_at_least, row.displaced_at_most) == (0, 2)


def test_a_lease_with_no_result_row_stays_open_when_nothing_says_when_it_ended(tmp_path):
    """No result row and no leg record is the case the reader may not close on its own."""
    run_dir = _run(
        tmp_path,
        dispenses=[_dispense(1, 100.0), _dispense(2, 200.0), _dispense(3, 300.0)],
        results=[_result(1), _result(3)],
    )
    row = _rollout(run_dir)
    assert row.never_sealed == 1
    assert row.max_open == 2
    assert row.strictly_sequential is False


def test_a_replayed_position_bounds_the_lease_it_abandoned(tmp_path):
    """Only a reopened stream replays a position, and it reopens with an empty registry, so the
    abandoned lease was over before the replay went out. No leg file is needed to say so."""
    run_dir = _run(
        tmp_path,
        dispenses=[_dispense(1, 100.0, position=0), _dispense(2, 200.0, position=0)],
        results=[_result(2, position=0)],
    )
    row = _rollout(run_dir)
    assert row.never_sealed == 1
    assert row.max_open == 1
    assert row.mean_open_at_pull == 0.0
    assert row.strictly_sequential is True


def test_a_replay_bounds_an_abandoned_lease_without_ending_the_ones_beside_it(tmp_path):
    """The replay closes the lease it replaced and nothing else, so a genuine overlap somewhere
    else in the same record still reads as one."""
    run_dir = _run(
        tmp_path,
        dispenses=[
            _dispense(1, 100.0, position=0),
            _dispense(2, 200.0, position=1),
            _dispense(3, 300.0, position=0),
            _dispense(4, 400.0, position=2),
        ],
        results=[_result(2, position=1), _result(4, position=2), _result(3, position=0)],
    )
    row = _rollout(run_dir)
    assert row.never_sealed == 1
    assert row.max_open == 2
    assert row.strictly_sequential is False


def test_a_lease_the_harness_died_holding_ends_with_its_leg(tmp_path):
    """A resumed stream starts with an empty registry, so an abandoned lease is not open past
    the session that held it, and crediting it to the agent as concurrency would be wrong."""
    run_dir = _run(
        tmp_path,
        dispenses=[_dispense(1, 100.0), _dispense(2, 200.0), _dispense(3, 400.0)],
        results=[_result(1), _result(3)],
        legs=[
            {"phase": "rollout", "started_at": 50.0, "ended_at": 250.0},
            {"phase": "rollout", "started_at": 350.0, "ended_at": 500.0},
        ],
    )
    row = _rollout(run_dir)
    assert row.never_sealed == 1
    assert row.max_open == 1
    assert row.strictly_sequential is None


def test_an_eval_phase_reads_its_per_task_streams(tmp_path):
    """Eval runs one session per task and gives each its own directory, so no lease there can
    overlap another whatever the ceiling says."""
    run_dir = tmp_path / "run"
    (run_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "toy-run", "cell": {"max_in_flight": CEILING}}), encoding="utf-8"
    )
    for seq in (1, 2, 3):
        task = run_dir / "eval_after" / f"task-{seq:05d}"
        _write_jsonl(task / "dispenses.jsonl", [_dispense(seq, 100.0 * seq)])
        _write_jsonl(task / "results.jsonl", [_result(seq)])
    (row,) = run_concurrency(run_dir)
    assert (row.phase, row.streams, row.dispensed) == ("eval_after", 3, 3)
    assert row.max_open == 1
    assert row.strictly_sequential is True


def test_the_reader_writes_nothing_into_the_run_directory(tmp_path):
    run_dir = _run(
        tmp_path,
        dispenses=[_dispense(1, 100.0), _dispense(2, 200.0)],
        results=[_result(1), _result(2)],
    )
    before = {p: p.stat().st_mtime_ns for p in sorted(run_dir.rglob("*"))}
    run_concurrency(run_dir)
    assert {p: p.stat().st_mtime_ns for p in sorted(run_dir.rglob("*"))} == before


def test_find_runs_takes_a_run_or_a_directory_of_them(tmp_path):
    run_dir = _run(tmp_path, dispenses=[_dispense(1, 100.0)], results=[_result(1)])
    assert find_runs(run_dir) == [run_dir]
    assert find_runs(run_dir.parent) == [run_dir]


def test_the_table_says_the_numbers_are_floors(tmp_path):
    run_dir = _run(
        tmp_path,
        dispenses=[_dispense(1, 100.0), _dispense(2, 200.0)],
        results=[_result(1), _result(2)],
    )
    rendered = render_table(run_concurrency(run_dir))
    assert ">=1" in rendered
    assert "floors" in rendered
    assert "unknown" in rendered


def test_the_table_renders_an_unsettled_displacement_count_as_a_range(tmp_path):
    run_dir = _run(
        tmp_path,
        dispenses=[_dispense(seq, 100.0 * seq) for seq in range(1, 4)],
        results=[_result(seq, closure="drained") for seq in (1, 2, 3)],
        ceiling=2,
    )
    rendered = render_table(run_concurrency(run_dir))
    assert "1..3" in rendered
    assert "a range wherever the record cannot separate them" in rendered


def test_the_cli_serves_it_read_only(tmp_path, capsys):
    run_dir = _run(
        tmp_path,
        dispenses=[_dispense(1, 100.0), _dispense(2, 200.0), _dispense(3, 300.0)],
        results=[_result(1), _result(3), _result(2)],
    )
    assert cli.main(["concurrency", str(run_dir), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    (phase,) = payload["phases"]
    assert phase["strictly_sequential"] is False
    assert phase["max_open"] == 2
    assert phase["max_open_is_floor"] is True
    assert phase["seal_times"] == "unrecorded"
    assert phase["displaced_at_least"] == 0
    assert phase["displaced_at_most"] == 0


def test_an_empty_target_is_reported_rather_than_rendered(tmp_path, capsys):
    assert cli.main(["concurrency", str(tmp_path)]) == 1
    assert "no run directory" in capsys.readouterr().out


def test_a_stream_alone_can_be_read(tmp_path):
    """The per-stream reader is the unit the phase roll-up is built from."""
    prov = tmp_path / "rollout"
    _write_jsonl(prov / "dispenses.jsonl", [_dispense(1, 10.0), _dispense(2, 20.0)])
    _write_jsonl(prov / "results.jsonl", [_result(2), _result(1)])
    stream = stream_concurrency(prov, label="rollout")
    assert stream.max_open == 2
    assert stream.sealed == 2
    assert stream.strictly_sequential is False


def test_a_real_saturated_stream_is_read_the_way_shogym_wrote_it(tmp_path):
    """The shape the range exists for, taken from a real stream rather than a hand-built file.

    Three pulls into two slots is one displacement at the third pull and two tasks the close
    drains, and shogym records all three under the one closure. The reader may not report the
    split as though the file held it, and its lower bound has to find the displacement.
    """
    import asyncio

    import shogym
    from shogym.serve import Immediate, TaskRef, TaskStream

    prov = tmp_path / "run" / "rollout"
    prov.mkdir(parents=True)

    async def saturate() -> None:
        stream = TaskStream(
            shogym.make,
            [TaskRef("wordle_v1", index) for index in range(3)],
            prov_dir=prov,
            feedback=Immediate(),
            max_in_flight=2,
        )
        async with stream:
            for _ in range(3):
                await stream.get_task()

    asyncio.run(saturate())
    (tmp_path / "run" / "manifest.json").write_text(
        json.dumps({"run_id": "toy-run", "cell": {"max_in_flight": 2}}), encoding="utf-8"
    )
    row = _rollout(tmp_path / "run")
    assert row.drained == 3
    assert (row.displaced_at_least, row.displaced_at_most) == (1, 3)
