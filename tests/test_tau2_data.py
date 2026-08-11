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
import tarfile
from pathlib import Path

import pytest

from shobench import tau2_data
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
