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


def _hle_builder(cache: Path):
    """The builder object the loader will read through, built the way the loader builds it.

    Predicting where the loader ends up is the mistake this function exists to stop making. The
    selection is only the first move: after it picks a directory, the builder may replace that
    directory with a legacy-layout one it finds beside it, and on that branch it reads the files
    under a different name as well. Any copy of that reasoning kept here is a divergence waiting
    to be found, so nothing is copied. ``load_dataset`` builds this object and then reads through
    it, so this builds the same object from the same arguments and asks it what it resolved to.

    Cheap, and it opens nothing: constructing a builder settles paths and metadata. The read comes
    later, in the test, through the env the runner would use.

    An error here is a failure rather than a skip. Nothing is cached is checked before this is
    called, so a builder that cannot be constructed over a cache holding a prepared build is a
    broken cache, and the exception says so.
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

    A build directory can hold arrow files this read never opens: another split's shards, residue
    from an older prepare the recorded split no longer refers to. The reader does not glob the
    directory, it is handed a file list, so that list is what has to be there and what else sits
    beside it is not this test's business. Failing over a file the loader will not touch is the
    same untrue report as skipping over a cache the machine is holding.

    The list is built by the reader's own instruction builder, off the resolved directory, the
    resolved split metadata, and the name the read path would use, which is the builder's name
    rather than the dataset's on the legacy branch.
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
    # can honestly skip. Past that line the data is here, so what the loader resolves to has to
    # hold up, and if it does not, that is a failure with a true reason rather than a green run
    # claiming data this machine has.
    cache = hle_data.cache_dir()
    root = _hle_cache_root(cache)
    if not any((c / "dataset_info.json").is_file() for c in root.glob("*/*/*")):
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
