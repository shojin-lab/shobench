"""Does the committed manifest address the tasks it says it does?

A split manifest is only worth anything if the ids in it resolve, inside the real env, to the
tasks the manifest names. These tests construct the env the way the runner does, through the
same ``env_factory`` closure, and check the manifest against it.

They need the env's data, so each one skips itself when that data is absent rather than
failing. Run them before a cell runs; a manifest that disagrees with its env is the kind of
error that produces numbers rather than an exception.

Absent has to mean absent, though, and not "absent until something fetches it". Constructing
one of these envs on a cold cache is a download: automationbench and tau2 pull a sha-pinned
tarball from GitHub, and hle pulls a gated dataset from the Hugging Face Hub. A test that
quietly paid for one would tie this file's verdict to a third-party host being reachable and
would make the same test pass on a machine with a network and skip on one without, which is a
skip reason nobody can read. So the fetch is refused here and the absence is what gets
reported. The tests themselves are unchanged: where the data is provisioned, they run in full.
"""

from __future__ import annotations

import os

import pytest

from shobench import tau2_data
from shobench.serving import env_factory
from shobench.splits import load_split_by_name


@pytest.fixture(autouse=True)
def _no_upstream_fetch(monkeypatch):
    """Refuse the pinned-tarball download an env construction would otherwise pay for.

    shogym provisions automationbench's and tau2's upstream *source* into ``~/.cache/shogym`` on
    first construction, so the seam is its downloader and not anything shobench owns. Patching a
    private name is the deliberate choice here: it is the single place that source reaches the
    network, and the day shogym renames it this fixture fails at setup rather than letting the
    fetch back in unnoticed.
    """
    from shogym.envs import _upstream

    def refuse(package: str, *_rest: object) -> None:
        raise RuntimeError(
            f"the pinned {package} source is not cached here, and a test will not download it"
        )

    monkeypatch.setattr(_upstream, "_download_package", refuse)


def _env(name: str, kwargs: dict):
    try:
        return env_factory(name, kwargs)(name)
    except Exception as exc:  # noqa: BLE001 - the message is the skip reason
        pytest.skip(f"{name} data not provisioned here: {type(exc).__name__}: {exc}")


def test_tau2_manifest_ids_resolve_to_the_labels_it_records() -> None:
    # The runner points TAU2_DATA_DIR at the provisioned cache; do the same here so this runs
    # whenever the data is present, not only when an operator exported the variable. Skips
    # cleanly on a host that has not provisioned it.
    if not tau2_data.is_present():
        pytest.skip(f"tau2 data not provisioned; run {tau2_data.PROVISION_COMMAND}")
    os.environ["TAU2_DATA_DIR"] = str(tau2_data.resolve_data_dir())
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
    # hle's tasks come off the Hub through ``datasets``, which the fixture above cannot cover:
    # huggingface_hub reads HF_HUB_OFFLINE once, at import, so setting it from a test is read
    # too late to stop anything. The cache the loader is pointed at is checked instead, before
    # the loader is called at all. A dir that is present but stale still reaches the Hub, which
    # is the developer machine this runs on and not a runner.
    from shogym.envs.hle import data as hle_data

    cache = hle_data.cache_dir()
    if not (cache.is_dir() and any(cache.iterdir())):
        pytest.skip(f"the gated hle dataset is not cached at {cache}, and a test will not fetch it")
    split = load_split_by_name("hle")
    env = _env("hle", split.heldout.env_kwargs)
    assert env.num_tasks == split.total_tasks
    for side in (split.heldout, split.pool):
        for task_id, label in list(zip(side.task_ids, side.labels, strict=True))[:20]:
            assert str(env._tasks[int(task_id)]["id"]) == label
