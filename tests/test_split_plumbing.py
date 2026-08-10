"""Does the committed manifest address the tasks it says it does?

A split manifest is only worth anything if the ids in it resolve, inside the real env, to the
tasks the manifest names. These tests construct the env the way the runner does, through the
same ``env_factory`` closure, and check the manifest against it.

They need the env's data, so each one skips itself when that data is absent rather than
failing. Run them before a cell runs; a manifest that disagrees with its env is the kind of
error that produces numbers rather than an exception.
"""

from __future__ import annotations

import os

import pytest

from shobench.serving import env_factory
from shobench.splits import load_split_by_name


def _env(name: str, kwargs: dict):
    try:
        return env_factory(name, kwargs)(name)
    except Exception as exc:  # noqa: BLE001 - the message is the skip reason
        pytest.skip(f"{name} data not provisioned here: {type(exc).__name__}: {exc}")


@pytest.mark.skipif(
    not os.environ.get("TAU2_DATA_DIR"), reason="needs TAU2_DATA_DIR at the pinned sha"
)
def test_tau2_manifest_ids_resolve_to_the_labels_it_records() -> None:
    split = load_split_by_name("tau2_telecom")
    for side in (split.heldout, split.pool):
        env = _env("tau2_telecom", side.env_kwargs)
        assert env.num_tasks == len(side)
        resolved = [env._resolve_task_id(t) for t in side.task_ids]
        assert resolved == list(side.labels)


def test_automationbench_manifest_covers_the_env_exactly() -> None:
    split = load_split_by_name("automationbench")
    env = _env("automationbench", split.heldout.env_kwargs)
    assert env.num_tasks == split.total_tasks
    ids = {int(i) for i in split.heldout.task_ids} | {int(i) for i in split.pool.task_ids}
    assert ids == set(range(env.num_tasks))


def test_hle_manifest_ids_resolve_to_the_question_ids_it_records() -> None:
    split = load_split_by_name("hle")
    env = _env("hle", split.heldout.env_kwargs)
    assert env.num_tasks == split.total_tasks
    for side in (split.heldout, split.pool):
        for task_id, label in list(zip(side.task_ids, side.labels, strict=True))[:20]:
            assert str(env._tasks[int(task_id)]["id"]) == label
