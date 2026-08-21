"""A damaged exam is repaired by naming the ids to measure again.

``rerun-eval --redo-task`` is the operator's scalpel. A hole in a phase is already pending and
needs no naming, while a settled row is complete by every test the runner has and would never
run again on its own, so an id whose measurement nobody should quote has to be named, and
nothing detects which ids deserve it. The rows that id holds are moved rather than deleted: a
redo says the leg measured the wrong thing, not that it never happened.

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
from shobench.runner import REDONE_DIR, _eval_pending_ids
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
    """A finished run whose eval_after holds one settled row per committed held-out id."""
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
    for position, task_id in enumerate(split.heldout.task_ids):
        task_dir = run_dir / "eval_after" / f"task-{int(task_id):05d}"
        task_dir.mkdir(parents=True)
        (task_dir / "results.jsonl").write_text(
            _settled_row_wire(int(task_id), lease=f"lease-{position}") + "\n", encoding="utf-8"
        )
    return run_dir


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


class _FakeSandbox:
    """The container work a repair does not need in order to decide which ids to run."""

    def __init__(self, run_id: str, home, workdir):
        self.run_id, self.home, self.workdir = run_id, home, workdir

    def up(self) -> None:
        pass

    def down(self) -> None:
        pass
