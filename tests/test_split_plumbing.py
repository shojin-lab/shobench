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

Two mechanisms hold that line, because the two downloads leave through different doors. The
tarball fetch is shogym's own function, refused by the fixture below. The Hub fetch belongs to
``datasets``, and it is stopped at the process level by ``conftest.py``, which pins the Hub
client offline before anything imports it.

What is left here is naming the absence, and naming it only when it is real. A skip is a claim
about the machine, so these tests skip on the two states that make the claim true, a refused
fetch and a cache with nothing prepared in it, and on nothing else. Data that is present and
then will not load is a failure. It reads as the harsher choice and is the honest one: a green
run reporting absent data the machine is holding is worse than a red one.
"""

from __future__ import annotations

import json
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


def _hle_cache_root(cache: Path) -> Path:
    """Where ``datasets`` keeps its builds of the hle dataset under ``cache``, named its way.

    The dataset's own id is the input, so a rename upstream moves this with it. Builds land in
    ``<root>/<config>/<version>/<hash>/``.
    """
    from datasets.naming import camelcase_to_snakecase

    parts = hle_data.HF_DATASET.split("/")
    parts[-1] = camelcase_to_snakecase(parts[-1])
    return cache / "___".join(parts)


def _hle_build_datasets_will_load(cache: Path) -> Path:
    """The one build ``datasets`` will read out of ``cache``, chosen the way it chooses.

    Which builds under the root look prepared is the wrong question, because the loader does not
    ask it. It globs the candidates, keeps the ones whose recorded config matches the directory
    they sit in, and takes whichever was modified last. A finished build beside a newer broken one
    therefore loses to the broken one, and a check that reads any prepared build it can find has
    checked something the loader will not open.

    So the choice is made by the loader's own function rather than by a copy of it here. It is
    private, and the day it moves this raises at import rather than quietly going back to
    inspecting one build while the loader reads another. It raises on a cache it cannot choose
    from, which is a failure and not a skip: by the time this is called, something under the root
    has been prepared, so absence is not what is wrong.
    """
    from datasets.packaged_modules.cache.cache import _find_hash_in_cache

    try:
        config_name, version, build_hash = _find_hash_in_cache(
            dataset_name=hle_data.HF_DATASET,
            config_name=None,
            cache_dir=str(cache),
            config_kwargs={},
            custom_features=None,
        )
    except Exception as exc:
        raise AssertionError(
            f"datasets cannot choose which build to load out of {cache}, and the hle data is "
            f"cached there: {type(exc).__name__}: {exc}"
        ) from exc
    return _hle_cache_root(cache) / config_name / version / build_hash


def _hle_shards_datasets_will_read(build: Path, info: dict) -> list[Path]:
    """The arrow files the requested split will be read from, named the way the reader names them.

    A build directory can hold arrow files this load never opens: another split's shards, residue
    from an older prepare that the recorded split no longer refers to. The reader does not glob the
    directory. It takes the split's recorded ``shard_lengths`` and builds the filenames from the
    dataset name and the split name, so those files are the whole of what has to be there, and
    what else sits beside them is not this test's business. Rejecting a build over a file the
    loader will not touch would fail a cache that works, which is the same class of untrue report
    as skipping over a cache that is present.
    """
    from datasets.naming import filenames_for_dataset_split

    split_info = info["splits"][hle_data.HF_SPLIT]
    return [
        Path(name)
        for name in filenames_for_dataset_split(
            build,
            dataset_name=hle_data.HF_DATASET.split("/")[-1],
            split=hle_data.HF_SPLIT,
            filetype_suffix="arrow",
            shard_lengths=split_info.get("shard_lengths"),
        )
    ]


def _env(name: str, kwargs: dict):
    """Construct the env the way the runner does.

    One failure is a skip here, and it is the one this file arranges: the refused fetch, which
    says the source is not on this machine. Every other failure fails. A test only reaches this
    line once it has established that the data it needs is present, and printing "not provisioned"
    over a machine holding the data would be a false reason for a green run, which is the exact
    report these tests exist to make impossible.
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


def test_automationbench_manifest_covers_the_env_exactly() -> None:
    split = load_split_by_name("automationbench")
    env = _env("automationbench", split.heldout.env_kwargs)
    assert env.num_tasks == split.total_tasks
    ids = {int(i) for i in split.heldout.task_ids} | {int(i) for i in split.pool.task_ids}
    assert ids == set(range(env.num_tasks))


def test_hle_manifest_ids_resolve_to_the_question_ids_it_records() -> None:
    # hle's tasks come off the Hub through ``datasets``, which the fixture above cannot cover:
    # the download is not shogym's function. conftest.py pins the Hub client offline for the whole
    # process, so the loader below reads this cache or raises, and reading this cache is what the
    # pin also makes checkable: offline is the branch that selects a build from disk, and it is
    # the branch every run takes.
    #
    # Nothing prepared under the root means nothing was ever cached, which is the only state that
    # can honestly skip. Past that line the data is here, so the build the loader will select has
    # to hold up, and if it does not, that is a failure with a true reason rather than a green run
    # claiming data this machine has.
    cache = hle_data.cache_dir()
    root = _hle_cache_root(cache)
    if not any((c / "dataset_info.json").is_file() for c in root.glob("*/*/*")):
        pytest.skip(f"the gated hle dataset is not cached at {cache}, and a test will not fetch it")
    build = _hle_build_datasets_will_load(cache)
    info = json.loads((build / "dataset_info.json").read_text())
    assert hle_data.HF_SPLIT in (info.get("splits") or {}), (
        f"datasets will load {build}, which records no {hle_data.HF_SPLIT!r} split"
    )
    shards = _hle_shards_datasets_will_read(build, info)
    unusable = [s.name for s in shards if not (s.is_file() and s.stat().st_size)]
    assert not unusable, (
        f"datasets reads {hle_data.HF_SPLIT!r} out of {build}, and {unusable} is missing or empty"
    )
    split = load_split_by_name("hle")
    env = _env("hle", split.heldout.env_kwargs)
    assert env.num_tasks == split.total_tasks
    for side in (split.heldout, split.pool):
        for task_id, label in list(zip(side.task_ids, side.labels, strict=True))[:20]:
            assert str(env._tasks[int(task_id)]["id"]) == label
