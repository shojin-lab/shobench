"""A damaged exam is repaired by naming the ids to measure again.

``rerun-eval --redo-task`` is the operator's scalpel. A hole in a phase is already pending and
needs no naming, while a settled row is complete by every test the runner has and would never
run again on its own, so an id whose measurement nobody should quote has to be named, and
nothing detects which ids deserve it. The rows and the trace that id holds are moved rather
than deleted: a redo says the leg measured the wrong thing, not that it never happened.

No Docker. The repair's phase runner is stood in for, because what it does with a pending id is
already covered and what is under test here is which ids become pending.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from shobench import runner
from shobench.cli import main as cli_main
from shobench.config import load_cell_by_name, load_instruction
from shobench.results import INCOMPLETE_SUFFIX, MISSING_CLOSURE, write_results
from shobench.runner import REDONE_DIR, _eval_pending_ids, read_eval_phase
from shobench.splits import load_split_by_name

_SMOKE_CELL = "smoke-automationbench-claude-code"


def _settled_row_wire(task_idx: int, *, lease: str) -> str:
    """A recorded outcome for one held-out id, in shogym's own wire shape."""
    return json.dumps(
        {
            "seq": 1,
            "lease": lease,
            "position": 0,
            "env": "automationbench",
            "task_idx": task_idx,
            "closure": "sealed",
            "score": {"reward": 0.0, "success": False, "feedback": []},
            "observed": [],
            "diagnostic": None,
            "extensions": {},
            "feedback_regime": "never",
        }
    )


def _run_with_a_measured_eval_after(tmp_path: Path) -> Path:
    """A finished run holding one settled row per committed held-out id, in both eval phases.

    Both phases, because an artifact is complete only when each of them accounts for every id,
    and what a redo does to a COMPLETE artifact is the thing under test below.

    The run's ending is published too, into ``tmp_path / "results"``, the directory the repairs
    below hand in: a run that ended has an artifact somewhere, and a redo runs only where that
    artifact is.
    """
    run_dir = tmp_path / "run"
    (run_dir / "home").mkdir(parents=True)
    (run_dir / "work").mkdir()
    cell = load_cell_by_name(_SMOKE_CELL)
    split = load_split_by_name(cell.split)
    instruction = load_instruction(cell.instruction_arm)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "r-1",
                "cell": cell.to_manifest(),
                "split": split.to_manifest(),
                "instruction": {
                    "rollout_system_sha256": instruction.rollout_system_sha256,
                    "eval_system_sha256": instruction.eval_system_sha256,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "rollout_stopping.json").write_text("{}", encoding="utf-8")
    for phase in ("eval_before", "eval_after"):
        for position, task_id in enumerate(split.heldout.task_ids):
            task_dir = run_dir / phase / f"task-{int(task_id):05d}"
            task_dir.mkdir(parents=True)
            (task_dir / "results.jsonl").write_text(
                _settled_row_wire(int(task_id), lease=f"{phase}-lease-{position}") + "\n",
                encoding="utf-8",
            )
    _publish_the_finished_run(run_dir, tmp_path / "results")
    return run_dir


def _publish_the_finished_run(run_dir: Path, results_dir: Path) -> Path:
    """The artifact that run's own ending published: complete, quoting every held-out row."""
    ids = [int(task_id) for task_id in _heldout_ids(run_dir)]
    return write_results(
        results_dir / f"{_SMOKE_CELL}.json",
        manifest=json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")),
        phases={
            phase: read_eval_phase(run_dir / phase, ids)
            for phase in ("eval_before", "eval_after")
        },
        stopping={},
        heldout_ids=ids,
    )


def _heldout_ids(run_dir: Path) -> list[str]:
    cell = load_cell_by_name(_SMOKE_CELL)
    return [str(task_id) for task_id in load_split_by_name(cell.split).heldout.task_ids]


def test_the_plan_lists_exactly_what_a_go_would_redo_and_spends_nothing(
    tmp_path: Path, capsys
) -> None:
    """The dry plan is the whole of what an operator has to check before spending."""
    run_dir = _run_with_a_measured_eval_after(tmp_path)
    redo, kept = _heldout_ids(run_dir)

    assert cli_main(["rerun-eval", "--run", str(run_dir), "--redo-task", redo]) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan["redo_tasks"] == [redo]
    assert plan["redo_tasks_with_rows_to_set_aside"] == [redo]
    assert plan["unknown_redo_tasks"] == []
    # Nothing was pending, so the redo is the whole of what a --go would pay for.
    assert plan["pending"] == 0
    assert plan["tasks_a_go_would_run"] == 1
    # And the plan really is dry: both ids still hold the rows they held.
    assert (run_dir / "eval_after" / f"task-{int(redo):05d}" / "results.jsonl").is_file()
    assert (run_dir / "eval_after" / f"task-{int(kept):05d}" / "results.jsonl").is_file()
    assert not (run_dir / "eval_after" / REDONE_DIR).exists()


def test_an_id_the_run_never_committed_to_blocks_the_go(tmp_path: Path) -> None:
    """A typo in the one command that moves a measurement out of the artifact stops it."""
    run_dir = _run_with_a_measured_eval_after(tmp_path)

    assert cli_main(["rerun-eval", "--run", str(run_dir), "--redo-task", "999999", "--go"]) == 1
    assert not (run_dir / "eval_after" / REDONE_DIR).exists()

    # And the API refuses it too, for a caller that never went through the CLI.
    with pytest.raises(RuntimeError, match="never committed to measuring"):
        asyncio.run(
            runner.rerun_eval(
                run_dir, results_dir=tmp_path / "results", redo_tasks=["999999"],
                capture_egress=False,
            )
        )


def test_a_redo_sets_the_old_rows_aside_and_makes_the_id_pending_again(
    tmp_path: Path, monkeypatch
) -> None:
    """The scalpel: one named id is measured again, its old rows are kept, the rest is untouched."""
    run_dir = _run_with_a_measured_eval_after(tmp_path)
    redo, kept = _heldout_ids(run_dir)
    before = (run_dir / "eval_after" / f"task-{int(redo):05d}" / "results.jsonl").read_bytes()

    captured: dict = {}

    async def fake_run_phases(ctx, *, manifest, phases, results_dir, observer, **kwargs):
        captured.update(phases=phases, pending=_eval_pending_ids(ctx.run_dir / "eval_after",
                                                                _heldout_ids(ctx.run_dir)))
        return results_dir / "cell.json"

    monkeypatch.setattr(runner, "_run_phases", fake_run_phases)
    monkeypatch.setattr(runner, "CellSandbox", _FakeSandbox)
    monkeypatch.setattr(runner, "_watch_cell_credential", lambda ctx, spec: None)

    asyncio.run(
        runner.rerun_eval(
            run_dir,
            results_dir=tmp_path / "results",
            redo_tasks=[redo],
            capture_egress=False,
        )
    )

    # The redone id is pending when the phase runs, and it is the only one: a repair measures
    # what it was asked to and never the ids beside it.
    assert captured["phases"] == ("eval_after",)
    assert captured["pending"] == [redo]

    # Set aside, not deleted: the rows are somewhere a reader can find them, byte for byte.
    archives = sorted((run_dir / "eval_after" / REDONE_DIR).iterdir())
    assert [path.name.split(".")[0] for path in archives] == [f"task-{int(redo):05d}"]
    assert (archives[0] / "results.jsonl").read_bytes() == before
    assert not (run_dir / "eval_after" / f"task-{int(redo):05d}").exists()
    assert (run_dir / "eval_after" / f"task-{int(kept):05d}" / "results.jsonl").is_file()

    # And the manifest says a redo happened, beside the repair it was part of.
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["eval_reruns"][-1]
    assert entry["phase"] == "eval_after"
    assert entry["redone"] == [
        {
            "task_id": int(redo),
            "rows_moved_to": runner.run_relative(archives[0], run_dir),
        }
    ]


def test_a_redo_that_fails_before_the_replacement_leaves_an_honest_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    """The move is the commitment, so the publication cannot wait for an ending that may not come.

    From the moment the rows are set aside the run directory says that measurement was rejected,
    and a results directory still holding it as a complete row of the cell contradicts it. What
    can end a repair between the move and its own publication is ordinary: a sandbox that will
    not come up, a server that fails, a provider limit that hard-exits the process.
    """
    run_dir = _run_with_a_measured_eval_after(tmp_path)
    redo, kept = _heldout_ids(run_dir)
    results_dir = tmp_path / "results"
    finished = results_dir / f"{_SMOKE_CELL}.json"
    assert json.loads(finished.read_text(encoding="utf-8"))["heldout"]["complete"]

    class _SandboxThatWillNotStart(_FakeSandbox):
        def up(self) -> None:
            raise RuntimeError("the sandbox could not be brought up")

    monkeypatch.setattr(runner, "CellSandbox", _SandboxThatWillNotStart)
    monkeypatch.setattr(runner, "_watch_cell_credential", lambda ctx, spec: None)

    with pytest.raises(RuntimeError, match="could not be brought up"):
        asyncio.run(
            runner.rerun_eval(
                run_dir, results_dir=results_dir, redo_tasks=[redo], capture_egress=False
            )
        )

    # Gone by name and by body: a cell publishes one artifact, and this one can no longer
    # account for the id it just rejected.
    assert not finished.exists()
    published = json.loads(
        (results_dir / f"{_SMOKE_CELL}{INCOMPLETE_SUFFIX}").read_text(encoding="utf-8")
    )
    assert published["heldout"]["complete"] is False
    assert published["heldout"]["eval_after"]["missing_task_ids"] == [int(redo)]
    rows = {task["task_idx"]: task for task in published["eval_after"]["tasks"]}
    assert rows[int(redo)]["closure"] == MISSING_CLOSURE
    assert rows[int(redo)]["reward"] is None
    # And the scalpel is still a scalpel: the id beside it is published as measured.
    assert rows[int(kept)]["closure"] == "sealed"


def test_a_redo_is_refused_where_this_run_never_published(tmp_path: Path, capsys) -> None:
    """A redo supersedes an artifact by republishing over it, so it runs only where that one is.

    The results directory is a free parameter: ``--results`` takes any path and its default is
    relative to the current working directory, while a run records nowhere the artifact it
    published. Pointed anywhere else, the early republication lands BESIDE that artifact and
    leaves it complete, still quoting the row the operator just rejected, which is the whole of
    what publishing early was for. So the requirement is checked before a row moves, and an
    operator who cannot meet it pays nothing to find out.
    """
    run_dir = _run_with_a_measured_eval_after(tmp_path)
    redo, _kept = _heldout_ids(run_dir)
    published = tmp_path / "results" / f"{_SMOKE_CELL}.json"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    where_it_published = ["--results", str(tmp_path / "results")]
    args = ["rerun-eval", "--run", str(run_dir), "--redo-task", redo]

    # The redo is refused nothing where the artifact it would supersede actually is.
    assert cli_main([*args, *where_it_published]) == 0
    assert json.loads(capsys.readouterr().out)["redo_artifact_refusal"] is None

    # Anywhere else, the plan carries the refusal, naming both spellings of the stem this run
    # publishes under and the directory that was searched for them.
    assert cli_main([*args, "--results", str(elsewhere)]) == 0
    refusal = json.loads(capsys.readouterr().out)["redo_artifact_refusal"]
    assert f"{_SMOKE_CELL}.json" in refusal
    assert f"{_SMOKE_CELL}{INCOMPLETE_SUFFIX}" in refusal
    assert str(elsewhere) in refusal

    # And a --go stops on it, before the move it would otherwise have made.
    assert cli_main([*args, "--results", str(elsewhere), "--go"]) == 1
    assert "holds no artifact of run" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="holds no artifact of run"):
        asyncio.run(
            runner.rerun_eval(
                run_dir, results_dir=elsewhere, redo_tasks=[redo], capture_egress=False
            )
        )
    assert not (run_dir / "eval_after" / REDONE_DIR).exists()
    assert (run_dir / "eval_after" / f"task-{int(redo):05d}" / "results.jsonl").is_file()
    assert list(elsewhere.iterdir()) == []
    # The artifact the run really has says what it always said, rather than being contradicted
    # by a run directory that had moved on without it.
    assert json.loads(published.read_text(encoding="utf-8"))["heldout"]["complete"]

    # An artifact of ANOTHER run of this cell wears exactly this name, and the name is not what
    # is being asked for.
    another = json.loads(published.read_text(encoding="utf-8"))
    another["manifest"]["run_id"] = "r-2"
    (elsewhere / f"{_SMOKE_CELL}.json").write_text(json.dumps(another), encoding="utf-8")

    assert cli_main([*args, "--results", str(elsewhere), "--go"]) == 1
    assert "holds no artifact of run" in capsys.readouterr().err
    assert not (run_dir / "eval_after" / REDONE_DIR).exists()


def _record_a_leg(run_dir: Path, phase: str, task_id: str, mark: str) -> dict:
    """One eval leg's trace, stderr and record, written where ``run_leg`` writes them."""
    idx = int(task_id)
    stem = runner.leg_stem(idx, idx)
    traces = run_dir / phase / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    (traces / f"{stem}.stream.jsonl").write_text(f'{{"attempt": "{mark}"}}\n', encoding="utf-8")
    (traces / f"{stem}.err.txt").write_text(f"{mark}\n", encoding="utf-8")
    return {
        "leg": idx,
        "phase": phase,
        "task_idx": idx,
        "trace_path": f"{phase}/traces/{stem}.stream.jsonl",
    }


def test_a_redo_archives_the_rejected_attempts_trace_and_leg_record_with_its_rows(
    tmp_path: Path, monkeypatch
) -> None:
    """The archive is the whole attempt: rows, trace, stderr, and the record that names them.

    The replacement runs under the same phase, task and leg number as the attempt it replaces,
    and ``run_leg`` opens the trace and the stderr to APPEND. A trace left in the phase would
    hold both attempts in one file, with two leg records pointing at it and nothing saying which
    bytes belonged to which measurement.
    """
    run_dir = _run_with_a_measured_eval_after(tmp_path)
    redo, kept = _heldout_ids(run_dir)
    rejected_leg = _record_a_leg(run_dir, "eval_after", redo, "rejected")
    kept_leg = _record_a_leg(run_dir, "eval_after", kept, "kept")
    (run_dir / "legs.json").write_text(json.dumps([rejected_leg, kept_leg]), encoding="utf-8")
    stem = runner.leg_stem(int(redo), int(redo))
    replacement = {**rejected_leg, "session_id": "the-replacement"}

    async def fake_run_phases(ctx, *, manifest, phases, results_dir, observer, **kwargs):
        # What the replacement leg does: the same computed paths, opened the same way.
        traces = ctx.run_dir / "eval_after" / "traces"
        traces.mkdir(parents=True, exist_ok=True)
        with (traces / f"{stem}.stream.jsonl").open("a", encoding="utf-8") as out:
            out.write('{"attempt": "replacement"}\n')
        with (traces / f"{stem}.err.txt").open("a", encoding="utf-8") as err:
            err.write("replacement\n")
        ctx.publish_json(ctx.run_dir / "legs.json", [*ctx.leg_records(), replacement])
        return results_dir / "cell.json"

    monkeypatch.setattr(runner, "_run_phases", fake_run_phases)
    monkeypatch.setattr(runner, "CellSandbox", _FakeSandbox)
    monkeypatch.setattr(runner, "_watch_cell_credential", lambda ctx, spec: None)

    asyncio.run(
        runner.rerun_eval(
            run_dir,
            results_dir=tmp_path / "results",
            redo_tasks=[redo],
            capture_egress=False,
        )
    )

    archive = next(iter((run_dir / "eval_after" / REDONE_DIR).iterdir()))
    archived, live = archive / "traces", run_dir / "eval_after" / "traces"
    assert (archived / f"{stem}.stream.jsonl").read_bytes() == b'{"attempt": "rejected"}\n'
    assert (archived / f"{stem}.err.txt").read_bytes() == b"rejected\n"
    # The replacement's trace is its own, though it was opened exactly as an appending leg
    # opens it: the file it appended to was no longer there.
    assert (live / f"{stem}.stream.jsonl").read_bytes() == b'{"attempt": "replacement"}\n'
    assert (live / f"{stem}.err.txt").read_bytes() == b"replacement\n"
    # The record went with the evidence it names, re-pointed at the archived copy.
    moved_to = f"{runner.run_relative(archive, run_dir)}/traces/{stem}.stream.jsonl"
    assert json.loads((archive / "legs.json").read_text(encoding="utf-8")) == [
        {**rejected_leg, "trace_path": moved_to}
    ]
    # And the run's own record holds the replacement and the untouched id, with no second
    # entry for the redone one.
    assert json.loads((run_dir / "legs.json").read_text(encoding="utf-8")) == [
        kept_leg,
        replacement,
    ]
    # The id beside it kept its trace where it was.
    kept_stem = runner.leg_stem(int(kept), int(kept))
    assert (live / f"{kept_stem}.stream.jsonl").read_bytes() == b'{"attempt": "kept"}\n'


class _FakeSandbox:
    """The container work a repair does not need in order to decide which ids to run."""

    def __init__(self, run_id: str, home, workdir):
        self.run_id, self.home, self.workdir = run_id, home, workdir

    def up(self) -> None:
        pass

    def down(self) -> None:
        pass


def test_a_padded_or_repeated_spelling_names_the_task_once(tmp_path: Path, capsys) -> None:
    """An id copied off a task directory (zero padded) is the task it names, and naming a task
    twice redoes it once, so the plan stays an exact description of a --go."""
    run_dir = _run_with_a_measured_eval_after(tmp_path)
    redo, _kept = _heldout_ids(run_dir)
    padded = f"{int(redo):05d}"

    assert (
        cli_main(
            ["rerun-eval", "--run", str(run_dir), "--redo-task", padded, "--redo-task", redo]
        )
        == 0
    )

    plan = json.loads(capsys.readouterr().out)
    assert plan["redo_tasks"] == [redo]
    assert plan["redo_tasks_with_rows_to_set_aside"] == [redo]
    assert plan["unknown_redo_tasks"] == []
    assert plan["tasks_a_go_would_run"] == 1
