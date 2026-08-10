"""The tau2 data provisioner: what it fetches, what it refuses, and where it lands.

Two layers, like shogym's own upstream-provisioning tests. The provisioning mechanics run
offline against a ``file://`` tarball built in a temp dir, so they need no network and no 730 MB.
The identity layer runs only when the real data is present: it skips cleanly otherwise, exactly
as shogym's runtime-provisioned env tests do, and asserts the pinned data when it is there.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from shobench import tau2_data
from shobench.pins import TAU2_UPSTREAM_SHA
from shobench.splits import load_split_by_name

# The telecom files a real construction reads, plus the two archive siblings the fetch must not
# keep. Small stand-ins; only their presence and shape matter to the provisioner.
_TELECOM = "data/tau2/domains/telecom"
_SPLIT_BODY = json.dumps({"train": ["t0", "t1"], "test": ["t2"], "base": ["t0", "t1", "t2"]})


def _make_tarball(tmp_path: Path, *, complete: bool = True) -> str:
    """Build a GitHub-shaped ``tau2-bench-<sha>/`` tarball; return its ``file://`` URL."""
    root = tmp_path / "build" / "tau2-bench-somesha"
    files = {
        f"{_TELECOM}/tasks.json": "[]\n",
        f"{_TELECOM}/split_tasks.json": _SPLIT_BODY,
        f"{_TELECOM}/db.toml": "x = 1\n",
        f"{_TELECOM}/user_db.toml": "y = 2\n",
        f"{_TELECOM}/main_policy.md": "# policy\n",
        # Siblings that must never be kept: the source, tests, and other domains' bulk.
        "src/tau2/__init__.py": "VERSION = 'pinned'\n",
        "tests/test_x.py": "assert True\n",
        "data/tau2/domains/airline/tasks.json": "[]\n",
    }
    if not complete:
        del files[f"{_TELECOM}/split_tasks.json"]
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    archive = tmp_path / "archive.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(root, arcname=root.name)
    return archive.as_uri()


@pytest.fixture
def _cache(tmp_path, monkeypatch):
    """Point the provisioner at a temp cache and clear any TAU2_DATA_DIR override."""
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("TAU2_DATA_DIR", raising=False)
    return tmp_path


# ----- the sha pin: the data is the same commit as the source -----


def test_the_data_sha_is_the_source_pin() -> None:
    """The data must come from the same commit shogym's tau2 *source* is pinned to, or a run
    mixes two upstreams. Importing the adapter is side-effect-free (it fetches nothing)."""
    from shogym.envs.tau2.adapter import UPSTREAM_SHA as source_sha

    assert tau2_data.UPSTREAM_SHA == source_sha
    assert tau2_data.UPSTREAM_SHA.startswith(TAU2_UPSTREAM_SHA)


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


# ----- verify: a partial fetch is not trusted -----


def _tree(tmp_path: Path) -> Path:
    """A minimal complete data tree on disk (no tarball), for verify() cases."""
    data = tmp_path / "data"
    for rel, body in {
        "tau2/domains/telecom/tasks.json": "[]\n",
        "tau2/domains/telecom/split_tasks.json": _SPLIT_BODY,
        "tau2/domains/telecom/db.toml": "x = 1\n",
        "tau2/domains/telecom/user_db.toml": "y = 2\n",
        "tau2/domains/telecom/main_policy.md": "# policy\n",
    }.items():
        path = data / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return data


def test_verify_accepts_a_complete_tree(tmp_path) -> None:
    tau2_data.verify(_tree(tmp_path))  # does not raise
    assert tau2_data.is_present(_tree(tmp_path))


def test_verify_rejects_a_missing_file(tmp_path) -> None:
    data = _tree(tmp_path)
    (data / "tau2/domains/telecom/tasks.json").unlink()
    with pytest.raises(tau2_data.Tau2DataError, match="missing"):
        tau2_data.verify(data)


def test_verify_rejects_an_empty_file(tmp_path) -> None:
    data = _tree(tmp_path)
    (data / "tau2/domains/telecom/db.toml").write_text("")
    with pytest.raises(tau2_data.Tau2DataError, match="empty"):
        tau2_data.verify(data)


def test_verify_rejects_an_unparseable_split_file(tmp_path) -> None:
    data = _tree(tmp_path)
    (data / "tau2/domains/telecom/split_tasks.json").write_text("{not json")
    with pytest.raises(tau2_data.Tau2DataError, match="did not parse"):
        tau2_data.verify(data)


def test_verify_rejects_an_absent_dir(tmp_path) -> None:
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
    # And the completeness marker records the pin.
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
    monkeypatch.setattr(tau2_data, "_TARBALL_URL", _make_tarball(_cache, complete=False))
    with pytest.raises(tau2_data.Tau2DataError, match="missing"):
        tau2_data.provision(log=lambda *a: None)
    # Nothing incomplete was published.
    assert not tau2_data.is_present(tau2_data.resolve_data_dir())


# ----- identity, only when the real data is present -----


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
