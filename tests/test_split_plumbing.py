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

Two doors, so two mechanisms: the tarball fetch is shogym's own function, refused by the fixture
below, and the Hub fetch belongs to ``datasets``, stopped process-wide by ``conftest.py``.

What is left here is naming the absence, and naming it only when it is real. These tests skip on
a refused fetch and on a cache the loader would not reach for, and on nothing else. Data that is
present and will not load fails instead, because a green run reporting absent data the machine is
holding is worse than a red one.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from shogym.envs.hle import data as hle_data

from shobench import tau2_data
from shobench.serving import env_factory
from shobench.splits import load_split_by_name


class _RefusedFetch(RuntimeError):
    """Raised where a download would have been. It is the one absence this file can skip on."""


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
        raise _RefusedFetch(
            f"the pinned {package} source is not cached here, and a test will not download it"
        )

    monkeypatch.setattr(_upstream, "_download_package", refuse)


def _hle_is_cached(cache: Path) -> bool:
    """Whether ``datasets`` considers the hle dataset cached under ``cache``.

    Put to the factory the offline load path falls back to rather than to a copy of its glob. Its
    answer is coarser than "loadable" on purpose: a directory where builds live is enough, even a
    half-written one. The factory reaching for a directory is what makes absence untrue, and
    everything past this line is then free to fail rather than obliged to skip.
    """
    from datasets.load import CachedDatasetModuleFactory

    try:
        CachedDatasetModuleFactory(hle_data.HF_DATASET, cache_dir=str(cache)).get_module()
    except FileNotFoundError:
        return False
    return True


def _hle_builder(cache: Path):
    """The builder object the loader will read through, built the way the loader builds it.

    Predicting where the loader ends up is the mistake this exists to stop making: selection is
    only its first move, and the builder can then replace the selected directory with a
    legacy-layout one beside it and read the files under a different name. ``load_dataset`` builds
    this object and reads through it, so this builds the same object from the same arguments and
    asks it what it resolved to. Constructing a builder settles paths and metadata and opens
    nothing; the read comes later, through the env the runner would use.

    An error here fails rather than skips, absence having been settled before it was called.
    """
    from datasets import load_dataset_builder

    try:
        return load_dataset_builder(hle_data.HF_DATASET, cache_dir=str(cache))
    except Exception as exc:
        raise AssertionError(
            f"datasets cannot open a builder over {cache}, and the hle data is cached there: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _hle_files_datasets_will_read(builder) -> list[Path]:
    """The exact files the reader is instructed to open for the requested split.

    The reader is handed a file list rather than globbing the directory, so that list is what has
    to be there, and the arrow files that can sit beside it (another split's shards, residue from
    an older prepare) are not this test's business. The list comes from the reader's own
    instruction builder, off the resolved directory and split metadata, under the name the read
    path would use, which on the legacy branch is the builder's rather than the dataset's.
    """
    from datasets.arrow_reader import make_file_instructions

    name = builder.name if builder._check_legacy_cache() else builder.dataset_name
    instructions = make_file_instructions(
        name=name,
        split_infos=list(builder.info.splits.values()),
        instruction=hle_data.HF_SPLIT,
        filetype_suffix="arrow",
        prefix_path=str(builder.cache_dir),
    )
    return [Path(instruction["filename"]) for instruction in instructions.file_instructions]


def _env(name: str, kwargs: dict):
    """Construct the env the way the runner does.

    One failure skips here, the refused fetch, which says the source is not on this machine. Every
    other failure fails: a test reaches this line having established that its data is present, so
    "not provisioned" over a machine holding the data would be a false reason for a green run.
    """
    try:
        return env_factory(name, kwargs)(name)
    except _RefusedFetch as exc:
        pytest.skip(f"{name} data not provisioned here: {exc}")


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


def test_tau2_banking_manifest_ids_resolve_to_the_labels_it_records() -> None:
    # banking declares no train/test split, so both sides index the one task list and carry the
    # same env kwargs. One env answers for both, and the manifest has to cover that list exactly.
    if not tau2_data.is_present():
        pytest.skip(f"tau2 data not provisioned; run {tau2_data.PROVISION_COMMAND}")
    os.environ["TAU2_DATA_DIR"] = str(tau2_data.resolve_data_dir())
    split = load_split_by_name("tau2_banking_knowledge")
    assert split.heldout.env_kwargs == split.pool.env_kwargs
    env = _env("tau2_banking_knowledge", split.heldout.env_kwargs)
    assert env.num_tasks == split.total_tasks
    for side in (split.heldout, split.pool):
        resolved = [env._resolve_task_id(t) for t in side.task_ids]
        assert resolved == list(side.labels)


def test_tau2_banking_split_serves_only_tasks_the_env_evaluator_scores() -> None:
    """Every served banking task declares a basis tau2's ``env`` evaluator actually scores.

    The cells run ``evaluation_type = "env"``, and that evaluator starts a reward at 1.0 and
    multiplies it only for DB and ENV_ASSERTION. A task whose basis holds anything else is not
    scored down, it is not scored at all: an ACTION-only task returns 1.0 for any run that
    terminates normally. Such a task on either side would publish a free success as a
    measurement, so the split excludes it, and this holds that exclusion to the data rather than
    to a count. A domain bump that adds tasks, or a pin that moves, fails here instead of
    quietly reintroducing one.
    """
    if not tau2_data.is_present():
        pytest.skip(f"tau2 data not provisioned; run {tau2_data.PROVISION_COMMAND}")
    os.environ["TAU2_DATA_DIR"] = str(tau2_data.resolve_data_dir())
    from shogym.envs.tau2 import mcp_server

    honored = {"DB", "ENV_ASSERTION"}
    tasks = mcp_server.load_tasks("banking_knowledge")
    basis_by_id = {
        str(task.id): {
            getattr(member, "value", member)
            for member in (getattr(task.evaluation_criteria, "reward_basis", None) or ())
        }
        for task in tasks
    }
    split = load_split_by_name("tau2_banking_knowledge")
    served = list(split.heldout.labels) + list(split.pool.labels)
    unscored = {
        label: sorted(basis_by_id[label])
        for label in served
        if not basis_by_id[label] or not basis_by_id[label] <= honored
    }
    assert not unscored, f"served tasks the env evaluator does not score in full: {unscored}"

    # The other half of the claim: everything excluded was excluded for that reason alone, so
    # the draw is over the whole eligible population rather than an arbitrary subset of it.
    eligible = {
        label for label, basis in basis_by_id.items() if basis and basis <= honored
    }
    assert set(served) == eligible
    assert split.provenance["excluded_tasks"].keys() == basis_by_id.keys() - eligible


def test_automationbench_manifest_covers_the_env_exactly() -> None:
    split = load_split_by_name("automationbench")
    env = _env("automationbench", split.heldout.env_kwargs)
    assert env.num_tasks == split.total_tasks
    ids = {int(i) for i in split.heldout.task_ids} | {int(i) for i in split.pool.task_ids}
    assert ids == set(range(env.num_tasks))


def test_hle_manifest_ids_resolve_to_the_question_ids_it_records() -> None:
    # hle's tasks come off the Hub through ``datasets``, which the fixture above cannot cover: the
    # download is not shogym's function. conftest.py pins that client offline for the whole
    # process, so the loader below reads this cache or raises, and every check between here and it
    # is about that cache. Whether anything is cached at all decides the skip; past that, every
    # disagreement with what the loader resolved to is a failure.
    cache = hle_data.cache_dir()
    if not _hle_is_cached(cache):
        pytest.skip(f"the gated hle dataset is not cached at {cache}, and a test will not fetch it")
    builder = _hle_builder(cache)
    build = Path(builder.cache_dir)
    assert hle_data.HF_SPLIT in (builder.info.splits or {}), (
        f"datasets resolves to {build}, which records no {hle_data.HF_SPLIT!r} split"
    )
    files = _hle_files_datasets_will_read(builder)
    unusable = [f.name for f in files if not (f.is_file() and f.stat().st_size)]
    assert not unusable, (
        f"datasets reads {hle_data.HF_SPLIT!r} out of {build}, and {unusable} is missing or empty"
    )
    split = load_split_by_name("hle")
    env = _env("hle", split.heldout.env_kwargs)
    assert env.num_tasks == split.total_tasks
    for side in (split.heldout, split.pool):
        for task_id, label in list(zip(side.task_ids, side.labels, strict=True))[:20]:
            assert str(env._tasks[int(task_id)]["id"]) == label
