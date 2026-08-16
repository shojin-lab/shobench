"""The guard that decides whether a replication arm may be written at all.

An arm republishes another manifest's membership rather than drawing one, so the check inside
``build_order2`` is the only thing between it and a parent that moved underneath it. These tests
put a doctored rebuild to that check: they are what says it compares tasks rather than integers.

They are offline and keyless. The builder is driven with manifests written here and a rebuild
that returns whatever the case under test needs, so no env, dataset or upstream is involved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from shobench.config import repo_root
from shobench.splits import load_split, write_split

sys.path.insert(0, str(repo_root() / "tools"))

import build_splits  # noqa: E402

# The builder looks the parent's pool_ceiling up in the cells that read it, so the fixture takes
# the name of a split some cell actually reads. Nothing else about that split is used here.
PARENT = "hle"

IDS = [str(i) for i in range(12)]
LABELS = [f"question-{i:02d}" for i in range(12)]
ENV_KWARGS = {"task_split": "train"}


def _write(
    path: Path,
    *,
    ids: list[str] | None = None,
    labels: list[str] | None = None,
    env_kwargs: dict[str, str] | None = None,
) -> Path:
    ids = IDS if ids is None else ids
    labels = LABELS if labels is None else labels
    env_kwargs = ENV_KWARGS if env_kwargs is None else env_kwargs
    return write_split(
        path,
        env="hle",
        total_tasks=len(ids),
        heldout=ids[:4],
        pool=ids[4:],
        heldout_env_kwargs=env_kwargs,
        pool_env_kwargs=env_kwargs,
        heldout_labels=labels[:4],
        pool_labels=labels[4:],
        provenance={"kind": "seeded", "seed": 1},
    )


def test_an_unmoved_parent_is_replicated_and_reruns_byte_identically(tmp_path: Path) -> None:
    parent = _write(tmp_path / f"{PARENT}.json")
    out = tmp_path / f"{PARENT}_order2.json"

    build_splits.build_order2(out, parent=parent, rebuild=_write)
    first = out.read_bytes()
    build_splits.build_order2(out, parent=parent, rebuild=_write)
    assert out.read_bytes() == first

    arm = load_split(out)
    committed = load_split(parent)
    assert list(arm.heldout.task_ids) == list(committed.heldout.task_ids)
    assert set(arm.pool.task_ids) == set(committed.pool.task_ids)
    assert list(arm.pool.task_ids) != list(committed.pool.task_ids)
    # The label a position resolves to travels with it, which is the same property the drift
    # check reads: an arm that reordered ids alone would name the wrong task at every position.
    assert dict(zip(arm.pool.task_ids, arm.pool.labels, strict=True)) == dict(
        zip(committed.pool.task_ids, committed.pool.labels, strict=True)
    )


@pytest.mark.parametrize(
    ("moved", "rebuilt"),
    [
        ("a pool position resolving to another task", {"labels": [*LABELS[:4], "x", *LABELS[5:]]}),
        ("a held-out position resolving to another task", {"labels": ["x", *LABELS[1:]]}),
        ("the env the positions index into", {"env_kwargs": {"task_split": "test"}}),
        ("the drawn ids themselves", {"ids": [*IDS[:11], "99"]}),
    ],
)
def test_a_parent_whose_tasks_moved_is_refused(
    tmp_path: Path, moved: str, rebuilt: dict[str, object]
) -> None:
    """Every one of these leaves the id list a fresh build produces intact or nearly so, which is
    what makes them worth testing: an upstream reorder that keeps its row count hands the builder
    the same integers pointing at different questions, and the arm would publish the parent's
    stale labels over them."""
    parent = _write(tmp_path / f"{PARENT}.json")
    out = tmp_path / f"{PARENT}_order2.json"

    with pytest.raises(SystemExit, match="no longer holds the tasks"):
        build_splits.build_order2(
            out, parent=parent, rebuild=lambda path: _write(path, **rebuilt)
        )
    assert not out.exists(), moved
