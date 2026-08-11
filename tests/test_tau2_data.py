"""The tau2 data provisioner: what it fetches, what it refuses, and where it lands.

Two layers, like shogym's own upstream-provisioning tests. The mechanics run offline against a
``file://`` tarball built in a temp dir, so they need no network and no 730 MB. The real-data
layer runs only when the pinned tree is provisioned here: it skips cleanly otherwise, exactly as
shogym's runtime-provisioned env tests do, and ties the fetched bytes to the committed split when
it is there.

A temp dir cannot hold the pinned commit's 14 MB, so the offline layer stands a manifest over the
same paths, carrying the stand-ins' own digests, in for the committed one. The mechanism is what
is under test either way: a tree is accepted only when every file matches the manifest it is
judged against, byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

import pytest

from shobench import cli, tau2_data
from shobench.pins import TAU2_UPSTREAM_SHA
from shobench.splits import load_split_by_name

# Stand-ins at the real relative paths, one per manifest entry, so the fixtures exercise the same
# file set the runtime reads (including the two files a construction reads that are not task
# files: the tech support manual and the user simulator's guidelines).
_TELECOM = "tau2/domains/telecom"
_SPLIT_BODY = json.dumps({"train": ["t0", "t1"], "test": ["t2"], "base": ["t0", "t1", "t2"]})
_FILES = {
    f"{_TELECOM}/tasks.json": "[]\n",
    f"{_TELECOM}/split_tasks.json": _SPLIT_BODY,
    f"{_TELECOM}/db.toml": "x = 1\n",
    f"{_TELECOM}/user_db.toml": "y = 2\n",
    f"{_TELECOM}/main_policy.md": "# policy\n",
    f"{_TELECOM}/tech_support_manual.md": "# manual\n",
    "tau2/user_simulator/simulation_guidelines_tools.md": "# guidelines\n",
}


def _manifest_for(files: dict[str, str]) -> dict[str, dict[str, object]]:
    """The manifest those bodies are the pinned bytes of."""
    return {
        rel: {"size": len(body.encode()), "sha256": hashlib.sha256(body.encode()).hexdigest()}
        for rel, body in files.items()
    }


def _make_tarball(tmp_path: Path, *, files: dict[str, str] | None = None) -> str:
    """Build a GitHub-shaped ``tau2-bench-<sha>/`` tarball; return its ``file://`` URL."""
    root = tmp_path / f"build-{len(list(tmp_path.glob('build-*')))}" / "tau2-bench-somesha"
    # Siblings that must never be kept: the source and the tests. Another domain's file rides
    # along under data/, standing in for the bulk the manifest does not gate.
    members = {
        "src/tau2/__init__.py": "VERSION = 'pinned'\n",
        "tests/test_x.py": "assert True\n",
        "data/tau2/domains/airline/tasks.json": "[]\n",
    }
    members.update(
        {f"data/{rel}": body for rel, body in (files if files is not None else _FILES).items()}
    )
    for rel, body in members.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    archive = root.parent / "archive.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(root, arcname=root.name)
    return archive.as_uri()


@pytest.fixture
def _stand_in_manifest(monkeypatch):
    """Judge the offline fixtures against the stand-ins' digests, not the pinned commit's."""
    monkeypatch.setattr(tau2_data, "_MANIFEST", _manifest_for(_FILES))


@pytest.fixture
def _cache(tmp_path, monkeypatch, _stand_in_manifest):
    """Point the provisioner at a temp cache and clear any TAU2_DATA_DIR override."""
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("TAU2_DATA_DIR", raising=False)
    return tmp_path


# ----- the sha pin: the data is the same commit as the source -----


def test_the_data_sha_is_the_source_pin() -> None:
    """The data must come from the same commit shogym's tau2 *source* is pinned to, or a run
    mixes two upstreams. Importing the adapter is side-effect-free (it fetches nothing). The
    committed digest manifest is held to the same sha, so moving the pin without re-recording
    the digests is a failing test rather than a gate that quietly checks the wrong commit."""
    from shogym.envs.tau2.adapter import UPSTREAM_SHA as source_sha

    assert tau2_data.UPSTREAM_SHA == source_sha
    assert tau2_data.UPSTREAM_SHA.startswith(TAU2_UPSTREAM_SHA)
    manifest = json.loads(tau2_data._MANIFEST_PATH.read_text())
    assert manifest["upstream_sha"] == tau2_data.UPSTREAM_SHA


# ----- resolve, needs -----


def test_resolve_prefers_an_explicit_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TAU2_DATA_DIR", str(tmp_path / "mine" / "data"))
    assert tau2_data.resolve_data_dir() == tmp_path / "mine" / "data"


def test_resolve_falls_back_to_the_pinned_cache_path(_cache) -> None:
    resolved = tau2_data.resolve_data_dir()
    assert resolved == _cache / "cache" / "tau2-data" / tau2_data.UPSTREAM_SHA / "data"


def test_needs_tau2_data_is_by_prefix() -> None:
    assert tau2_data.needs_tau2_data("tau2_telecom")
    assert tau2_data.needs_tau2_data("tau2")
    assert not tau2_data.needs_tau2_data("automationbench")
    assert not tau2_data.needs_tau2_data("hle")


# ----- verify: neither a partial fetch nor a tree that is not the pinned bytes -----


def _tree(tmp_path: Path, *, drop: str | None = None) -> Path:
    """A data tree on disk (no tarball) holding every manifest file, for verify() cases."""
    data = tmp_path / "data"
    for rel, body in _FILES.items():
        if rel == drop:
            continue
        path = data / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return data


def test_verify_accepts_the_manifest_tree(tmp_path, _stand_in_manifest) -> None:
    tau2_data.verify(_tree(tmp_path))  # does not raise
    assert tau2_data.is_present(_tree(tmp_path))


@pytest.mark.parametrize(
    "rel",
    [
        # The task files, and the two the gate has to reach past task loading to cover: the
        # manual the env reads at construction and the guidelines the user simulator reads.
        "tau2/domains/telecom/split_tasks.json",
        "tau2/domains/telecom/tech_support_manual.md",
        "tau2/user_simulator/simulation_guidelines_tools.md",
    ],
)
def test_verify_rejects_a_tree_missing_a_file_the_runtime_reads(
    tmp_path, _stand_in_manifest, rel
) -> None:
    """Absent here means absent at env setup or, worse, partway into a spending run."""
    with pytest.raises(tau2_data.Tau2DataError, match="missing"):
        tau2_data.verify(_tree(tmp_path, drop=rel))


def test_verify_rejects_a_truncated_file(tmp_path, _stand_in_manifest) -> None:
    data = _tree(tmp_path)
    (data / "tau2/domains/telecom/split_tasks.json").write_text(_SPLIT_BODY[:20])
    with pytest.raises(tau2_data.Tau2DataError, match="bytes, the pinned commit"):
        tau2_data.verify(data)


def test_verify_rejects_edited_bytes_at_the_same_size(tmp_path, _stand_in_manifest) -> None:
    """The drift that structure cannot catch: a policy the right length and the wrong content
    still moves the benchmark's numbers, so identity is what is checked, not shape."""
    data = _tree(tmp_path)
    policy = data / "tau2/domains/telecom/main_policy.md"
    policy.write_text("# POLICY\n")
    with pytest.raises(tau2_data.Tau2DataError, match="not the bytes of upstream"):
        tau2_data.verify(data)


def test_verify_rejects_an_absent_dir(tmp_path, _stand_in_manifest) -> None:
    with pytest.raises(tau2_data.Tau2DataError, match="no tau2 data"):
        tau2_data.verify(tmp_path / "nope")


def test_require_names_the_provisioning_command(_cache) -> None:
    with pytest.raises(tau2_data.Tau2DataError, match=tau2_data.PROVISION_COMMAND):
        tau2_data.require()


# ----- provision: only data/, atomically, idempotently -----


def test_provision_keeps_only_the_data_subtree(_cache, monkeypatch) -> None:
    monkeypatch.setattr(tau2_data, "_TARBALL_URL", _make_tarball(_cache))
    data_dir = tau2_data.provision(log=lambda *a: None)

    assert data_dir == tau2_data.resolve_data_dir()
    tau2_data.verify(data_dir)
    # The source and tests siblings were filtered out; only data/ was kept.
    assert not (data_dir.parent / "src").exists()
    assert not (data_dir.parent / "tests").exists()
    # And the provenance marker records the pin.
    marker = json.loads((data_dir.parent / tau2_data._MARKER).read_text())
    assert marker["upstream_sha"] == tau2_data.UPSTREAM_SHA


def test_provision_is_idempotent_and_does_not_refetch(_cache, monkeypatch) -> None:
    monkeypatch.setattr(tau2_data, "_TARBALL_URL", _make_tarball(_cache))
    first = tau2_data.provision(log=lambda *a: None)
    # A present tree must skip the fetch: point the url somewhere unreachable and prove it is
    # never opened.
    monkeypatch.setattr(tau2_data, "_TARBALL_URL", "http://unreachable.invalid/x.tar.gz")
    assert tau2_data.provision(log=lambda *a: None) == first


def test_provision_refuses_an_archive_missing_the_split_file(_cache, monkeypatch) -> None:
    partial = {k: v for k, v in _FILES.items() if not k.endswith("split_tasks.json")}
    monkeypatch.setattr(tau2_data, "_TARBALL_URL", _make_tarball(_cache, files=partial))
    with pytest.raises(tau2_data.Tau2DataError, match="missing"):
        tau2_data.provision(log=lambda *a: None)
    # Nothing incomplete was published.
    assert not tau2_data.is_present(tau2_data.resolve_data_dir())


def test_provision_reclaims_staging_an_interrupted_run_left(_cache, monkeypatch) -> None:
    """A killed fetch leaves most of a 730 MB tree in the cache, because TemporaryDirectory
    unwinds and a signal does not, and every retry stacks another. The next provision reclaims
    them. The stale dir is pre-created rather than produced by a signal: what is under test is
    the reclaiming, and a suite that fired signals to get there would test the harness."""
    monkeypatch.setattr(tau2_data, "_TARBALL_URL", _make_tarball(_cache))
    sha_dir = tau2_data.resolve_data_dir().parent
    stale = sha_dir / f"{tau2_data._STAGING_PREFIX}abandoned"
    (stale / "data" / _TELECOM).mkdir(parents=True)
    (stale / "data" / _TELECOM / "tasks.json").write_text("half a download\n")

    data_dir = tau2_data.provision(log=lambda *a: None)

    assert not stale.exists()
    tau2_data.verify(data_dir)


def test_force_replaces_the_tree_it_paid_to_download(_cache, monkeypatch) -> None:
    """Forcing is the repair path, so the fetched tree must land. What it repairs is the part of
    the 730 MB the manifest does not gate: the gated files pass verify by definition, and a run
    that is otherwise unhappy with its tree has only this flag. Keeping the old tree would make
    the flag cost a full download and change nothing, and report success either way.
    """
    canary = "tau2/domains/airline/tasks.json"

    def tarball(body: str) -> str:
        return _make_tarball(_cache, files=dict(_FILES, **{canary: body}))

    monkeypatch.setattr(tau2_data, "_TARBALL_URL", tarball('["old"]\n'))
    data_dir = tau2_data.provision(log=lambda *a: None)
    assert (data_dir / canary).read_text() == '["old"]\n'

    monkeypatch.setattr(tau2_data, "_TARBALL_URL", tarball('["new"]\n'))
    assert tau2_data.provision(force=True, log=lambda *a: None) == data_dir
    assert (data_dir / canary).read_text() == '["new"]\n'


def _override_checkout(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """An operator's own tree at TAU2_DATA_DIR: not the pinned data, and lived in."""
    data = _tree(tmp_path / "operator")
    (data / "tau2/domains/telecom/main_policy.md").write_text("# POLICY\n")
    work = data / "my-uncommitted-work.txt"
    work.write_text("hours of it\n")
    monkeypatch.setenv("TAU2_DATA_DIR", str(data))
    return data, work


def test_an_invalid_override_is_refused_rather_than_replaced(_cache, monkeypatch) -> None:
    """A tree named by TAU2_DATA_DIR belongs to the operator, so failing verify is a reason to
    stop, not a licence to delete it and everything living beside it. The refusal also comes
    before the download: the unreachable url proves nothing was fetched to justify it."""
    data, work = _override_checkout(_cache, monkeypatch)
    monkeypatch.setattr(tau2_data, "_TARBALL_URL", "http://unreachable.invalid/x.tar.gz")

    with pytest.raises(tau2_data.Tau2DataError, match="--force"):
        tau2_data.provision(log=lambda *a: None)
    assert work.read_text() == "hours of it\n"
    assert (data / "tau2/domains/telecom/main_policy.md").read_text() == "# POLICY\n"


def test_force_replaces_an_override_the_operator_asked_to_replace(_cache, monkeypatch) -> None:
    """Forcing is the operator saying so, so the same tree is replaced without argument."""
    data, work = _override_checkout(_cache, monkeypatch)
    monkeypatch.setattr(tau2_data, "_TARBALL_URL", _make_tarball(_cache))

    assert tau2_data.provision(force=True, log=lambda *a: None) == data
    tau2_data.verify(data)
    assert not work.exists()


def test_a_valid_override_is_verified_and_not_written_to(_cache, monkeypatch) -> None:
    """The operator owns a tree they named down to its provenance, so verify-and-skip records
    nothing beside it. Otherwise the marker lands in their checkout root, outside the dir they
    pointed at, and a checkout mounted read-only fails the one command advertised as safe. The
    read-only parent is the assertion: nothing is written, so nothing can fail to be."""
    checkout = _cache / "operator"
    data = _tree(checkout)
    (checkout / "README.md").write_text("the operator's own checkout\n")
    monkeypatch.setenv("TAU2_DATA_DIR", str(data))
    monkeypatch.setattr(tau2_data, "_TARBALL_URL", "http://unreachable.invalid/x.tar.gz")

    checkout.chmod(0o555)
    try:
        assert tau2_data.provision(log=lambda *a: None) == data
    finally:
        checkout.chmod(0o755)
    assert sorted(p.name for p in checkout.iterdir()) == ["README.md", "data"]


def test_a_lost_publish_race_adopts_the_tree_that_won(_cache, monkeypatch) -> None:
    """Two provisioners of one pin want the same bytes, so losing the publish is not a failure to
    report to the caller. The tree that landed is adopted, and adopted on the verify below the
    publish block rather than on trust. Simulated without threads by publishing the winner while
    this run is still unpacking, which is the window the race opens."""
    monkeypatch.setattr(tau2_data, "_TARBALL_URL", _make_tarball(_cache))
    data_dir = tau2_data.resolve_data_dir()
    unpack = tau2_data._extract_data_subtree

    def unpack_then_lose_the_race(archive: Path, staged_data: Path) -> None:
        unpack(archive, staged_data)
        shutil.copytree(_tree(_cache / "winner"), data_dir)
        (data_dir / "published-by-the-winner").write_text("yes\n")

    monkeypatch.setattr(tau2_data, "_extract_data_subtree", unpack_then_lose_the_race)
    assert tau2_data.provision(log=lambda *a: None) == data_dir
    # Adopted, not overwritten: the winner's tree is the one still in place.
    assert (data_dir / "published-by-the-winner").exists()


# ----- the gate every command that starts a tau2 cell has to pass -----
#
# The data dir is a process-local assignment, and a continuation is a new process started in a
# new shell hours after the one it continues. So the gate that resolves the cache belongs on the
# resume path exactly as much as on the fresh one, and these drive the real command rather than
# the helper under it.

_TAU2_CELL = "tau2_telecom-claude_code-claude-opus-5"


@pytest.fixture
def _unset_data_dir(monkeypatch):
    """Run with TAU2_DATA_DIR unset, and put the host's state back however the command leaves it.

    The command under test is the one that assigns this variable, and ``monkeypatch`` restores
    only the names it was told about, so clearing a variable that was already absent would record
    nothing and let the assignment leak into every test that runs after. Setting it first is what
    registers the name.
    """
    monkeypatch.setenv("TAU2_DATA_DIR", "")
    monkeypatch.delenv("TAU2_DATA_DIR")


def _suspended_tau2_run(run_dir: Path) -> Path:
    """A run directory a usage limit suspended during a tau2 cell's held-out eval."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "suspended.json").write_text(
        json.dumps(
            {
                "schema": "shobench.suspension/1",
                "run_id": "run-1",
                "cell": _TAU2_CELL,
                "harness": "claude_code",
                "phase": "eval_after",
                "completed_task_ids": [0],
                "pending_task_ids": [1],
                "stop_evidence": {"kind": "usage_limit"},
                "suspended_at": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _provision_offline(cache: Path, monkeypatch) -> Path:
    """Put the stand-in data where the managed cache would be, without a network."""
    monkeypatch.setattr(tau2_data, "_TARBALL_URL", _make_tarball(cache))
    return tau2_data.provision(log=lambda *a: None)


def _resumed_with(monkeypatch) -> dict[str, str]:
    """Stand in for the continuation itself and report the environment it was handed."""
    seen: dict[str, str] = {}

    async def fake_resume_cell(run_dir, **kwargs):
        seen["TAU2_DATA_DIR"] = os.environ.get("TAU2_DATA_DIR", "")
        return Path(run_dir) / "results.json"

    monkeypatch.setattr(cli, "resume_cell", fake_resume_cell)
    # A tau2 cell's user simulator needs a key of its own, and the resume refuses without it
    # before it ever reaches the data gate. It authenticates nothing here: the continuation is
    # stood in for.
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-key")
    return seen


def test_a_suspended_tau2_cell_resumes_against_the_provisioned_cache(
    _cache, _unset_data_dir, monkeypatch
) -> None:
    """The bug this exists for: a continuation is a new process, so it inherits nothing from the
    process that suspended, and a tau2 cell resumed with TAU2_DATA_DIR unset used to reach env
    construction with no data dir at all and fail on a path nobody configured."""
    _provision_offline(_cache, monkeypatch)
    run_dir = _suspended_tau2_run(_cache / "run")
    seen = _resumed_with(monkeypatch)

    assert cli.main(["resume", "--run", str(run_dir), "--go"]) == 0
    assert seen["TAU2_DATA_DIR"] == str(tau2_data.resolve_data_dir())


def test_the_resume_plan_names_the_data_the_continuation_will_serve(
    _cache, _unset_data_dir, monkeypatch, capsys
) -> None:
    """Without ``--go`` the plan is the whole output, so the tree a continuation would point the
    stream at is in it, exactly as it is for a fresh run."""
    _provision_offline(_cache, monkeypatch)
    run_dir = _suspended_tau2_run(_cache / "run")

    assert cli.main(["resume", "--run", str(run_dir)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["tau2_data"]["data_dir"] == str(tau2_data.resolve_data_dir())
    assert plan["tau2_data"]["present"] is True


def test_a_resume_with_no_provisioned_data_refuses_and_stays_resumable(
    _cache, _unset_data_dir, monkeypatch, capsys
) -> None:
    """Unprovisioned data stops the continuation before the sandbox comes up, names the command
    that fixes it, and leaves the record that makes another attempt possible."""
    run_dir = _suspended_tau2_run(_cache / "run")
    seen = _resumed_with(monkeypatch)

    assert cli.main(["resume", "--run", str(run_dir), "--go"]) == 1
    err = capsys.readouterr().err
    assert tau2_data.PROVISION_COMMAND in err
    assert seen == {}, "nothing was spent"
    assert (run_dir / "suspended.json").is_file(), "the run is still resumable"


def test_a_resumed_tau2_cell_can_construct_its_env_from_what_the_resume_resolved(
    tmp_path, _unset_data_dir, monkeypatch
) -> None:
    """The real path, end to end, when the pinned data is provisioned here: what the resume
    command puts in the environment is what the env reads, so constructing the cell's env after it
    succeeds. Without the gate this raises FileNotFoundError on the source cache's missing
    ``telecom/tasks.json``, which is the failure a suspended tau2 cell used to meet."""
    if not tau2_data.is_present(
        tau2_data._cache_root() / "tau2-data" / tau2_data.UPSTREAM_SHA / "data"
    ):
        pytest.skip(f"tau2 data not provisioned; run {tau2_data.PROVISION_COMMAND}")
    import shogym

    from shobench.config import load_cell_by_name

    run_dir = _suspended_tau2_run(tmp_path / "run")
    _resumed_with(monkeypatch)

    assert cli.main(["resume", "--run", str(run_dir), "--go"]) == 0
    cell = load_cell_by_name(_TAU2_CELL)
    shogym.make(cell.env, config=(dict(cell.env_kwargs) or None))


# ----- the real data, when it is provisioned here -----


def test_the_provisioned_data_matches_the_committed_split() -> None:
    """Skips cleanly when the data is not provisioned here; asserts identity when it is: the
    telecom split file's train/test sizes are exactly the committed split manifest's pool and
    held-out sizes, tying the fetched bytes to the pinned split."""
    if not tau2_data.is_present():
        pytest.skip(f"tau2 data not provisioned; run {tau2_data.PROVISION_COMMAND}")
    data_dir = tau2_data.resolve_data_dir()
    tau2_data.verify(data_dir)
    splits = json.loads((data_dir / "tau2/domains/telecom/split_tasks.json").read_text())
    manifest = load_split_by_name("tau2_telecom")
    assert len(splits["train"]) == len(manifest.pool)
    assert len(splits["test"]) == len(manifest.heldout)
