"""Provisioning tau2-bench's ``data/`` at the pinned upstream sha.

shogym's tau2 port provisions the upstream *source* (``src/tau2``) at runtime and deliberately
filters out the archive's ~700 MB of benchmark ``data/`` (see ``shogym.envs._upstream``). The
data is a separate concern: upstream resolves it from ``TAU2_DATA_DIR``, and the three
tau2_telecom cells need it. This module fetches exactly that subtree, once, into a cache the
runner points ``TAU2_DATA_DIR`` at.

The data is ~730 MB, so it is provisioned rather than committed. It is fetched from the same
SHA-pinned GitHub archive the source comes from, so the data's identity is the same pin as the
source (a test asserts the two shas agree). The provisioning is idempotent: a complete tree is
recognized and skipped; a partial one is not trusted, because completeness is decided by the
files a tau2_telecom construction actually reads, checked present and non-empty, and the
published tree is renamed into place atomically after that check passes.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# The pinned upstream commit whose data these cells use. This is the full sha; a test asserts it
# equals shogym's own tau2 source pin (``shogym.envs.tau2.adapter.UPSTREAM_SHA``) and begins with
# the short sha recorded in ``pins.TAU2_UPSTREAM_SHA`` and in the split manifest, so the data and
# the source can never silently come from different commits.
UPSTREAM_SHA = "1d244f5dca42944b67a379b44bfeb9f5748f189d"
_TARBALL_URL = f"https://github.com/sierra-research/tau2-bench/archive/{UPSTREAM_SHA}.tar.gz"

# Envs whose construction reads tau2 data. v0 ships only tau2_telecom, but every tau2 domain
# resolves its tasks from this tree, so the gate is by prefix rather than by an explicit list.
_TAU2_PREFIX = "tau2"

# The files a tau2_telecom env reads at construction time (task ids come out of these). If any is
# absent or empty the tree is a partial fetch, not a usable one. Relative to the ``data/`` dir
# ``TAU2_DATA_DIR`` names; upstream reads ``DATA_DIR / "tau2" / "domains" / <domain>`` from there.
_REQUIRED_FILES = (
    "tau2/domains/telecom/tasks.json",
    "tau2/domains/telecom/split_tasks.json",
    "tau2/domains/telecom/db.toml",
    "tau2/domains/telecom/user_db.toml",
    "tau2/domains/telecom/main_policy.md",
)

# How a partial-fetch's split file is caught: it must parse and be non-empty, not merely exist.
_PARSE_CHECK = "tau2/domains/telecom/split_tasks.json"

# The record written next to a completed tree. Its presence is not what completeness is judged on
# (an externally provisioned tree without it is still recognized), but it carries provenance.
_MARKER = ".shobench-tau2-data.json"

# What the runner and the docs tell an operator to run when the data is missing.
PROVISION_COMMAND = "uv run python tools/provision_tau2_data.py"

_DOWNLOAD_TIMEOUT_S = 300.0


class Tau2DataError(RuntimeError):
    """The tau2 data is missing or incomplete. The message names the fix."""


def _cache_root() -> Path:
    base = os.environ.get("SHOGYM_CACHE")
    return Path(base) if base else Path.home() / ".cache" / "shogym"


def resolve_data_dir() -> Path:
    """Where ``TAU2_DATA_DIR`` should point: an operator override, else the pinned cache path.

    An explicit ``TAU2_DATA_DIR`` in the environment wins, exactly as ``TAU2_SRC`` overrides the
    source cache, so a checkout provisioned elsewhere is honored without a re-fetch. Otherwise the
    path is ``<SHOGYM_CACHE or ~/.cache/shogym>/tau2-data/<sha>/data`` -- a sibling of shogym's
    own ``tau2/<sha>`` source cache, never the same directory.
    """
    override = os.environ.get("TAU2_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return _cache_root() / "tau2-data" / UPSTREAM_SHA / "data"


def verify(data_dir: Path) -> None:
    """Raise :class:`Tau2DataError` unless ``data_dir`` is a complete tau2 data tree.

    Completeness is the files a tau2_telecom construction reads, present and non-empty, with the
    split file additionally required to parse -- so a download truncated mid-file is caught rather
    than trusted. This is the identity check too: the required files are the pinned commit's
    telecom data, and the cache path carries the sha.
    """
    if not data_dir.is_dir():
        raise Tau2DataError(f"no tau2 data at {data_dir}")
    for rel in _REQUIRED_FILES:
        path = data_dir / rel
        if not path.is_file():
            raise Tau2DataError(f"tau2 data at {data_dir} is incomplete: missing {rel}")
        if path.stat().st_size == 0:
            raise Tau2DataError(f"tau2 data at {data_dir} is incomplete: {rel} is empty")
    parse_path = data_dir / _PARSE_CHECK
    try:
        parsed = json.loads(parse_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Tau2DataError(
            f"tau2 data at {data_dir}: {_PARSE_CHECK} did not parse ({exc})"
        ) from exc
    if not parsed:
        raise Tau2DataError(f"tau2 data at {data_dir}: {_PARSE_CHECK} is empty")


def is_present(data_dir: Path | None = None) -> bool:
    """Whether a complete tau2 data tree is where the runner would look for it."""
    try:
        verify(data_dir if data_dir is not None else resolve_data_dir())
    except Tau2DataError:
        return False
    return True


def require() -> Path:
    """Return the data dir if complete, else raise with the provisioning command.

    This is the runner's loud failure: a tau2 cell that reached serving with no data would build,
    spend, and fail partway with an upstream message that names none of this. So the check is made
    before anything spends, and the message says exactly what to run.
    """
    data_dir = resolve_data_dir()
    try:
        verify(data_dir)
    except Tau2DataError as exc:
        raise Tau2DataError(
            f"{exc}. Provision it first (about 730 MB, one time): {PROVISION_COMMAND}"
        ) from exc
    return data_dir


def needs_tau2_data(env: str) -> bool:
    """Whether an env's construction reads tau2 data."""
    return env == _TAU2_PREFIX or env.startswith(_TAU2_PREFIX + "_")


def _write_marker(sha_dir: Path, data_dir: Path) -> None:
    (sha_dir / _MARKER).write_text(
        json.dumps(
            {
                "upstream": "sierra-research/tau2-bench",
                "upstream_sha": UPSTREAM_SHA,
                "tarball": _TARBALL_URL,
                "data_dir": str(data_dir),
                "required_files": list(_REQUIRED_FILES),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _extract_data_subtree(archive: Path, staged_data: Path) -> None:
    """Extract only the archive's top-level ``data/`` into ``staged_data``.

    A GitHub archive extracts under a single ``<repo>-<sha>/`` root; the data sits at
    ``<root>/data``. Members are filtered by that prefix and streamed, so the ~5 MB of source and
    the rest of the archive are never written. ``filter="data"`` rejects path traversal.
    """
    wanted: str | None = None
    with tarfile.open(archive, mode="r:gz") as tf:
        for member in tf:
            if wanted is None:
                root = member.name.split("/", 1)[0]
                wanted = f"{root}/data"
            if member.name == wanted or member.name.startswith(wanted + "/"):
                tf.extract(member, staged_data.parent, filter="data")
    produced = staged_data.parent / (wanted or "")
    if wanted is None or not produced.is_dir():
        raise Tau2DataError(
            f"unexpected tau2-bench archive layout: no '{wanted}' in {_TARBALL_URL}"
        )
    produced.replace(staged_data)


def provision(*, force: bool = False, log=print) -> Path:
    """Fetch tau2's ``data/`` at the pinned sha into the cache; return the data dir.

    Idempotent: an already-complete tree is verified and its marker refreshed, and no network is
    touched, unless ``force``. Otherwise the tarball is downloaded, the ``data/`` subtree is
    extracted to a staging dir, verified, and only then renamed into place, so an interrupted run
    leaves no tree that :func:`verify` would accept.
    """
    data_dir = resolve_data_dir()
    if not force and is_present(data_dir):
        sha_dir = data_dir.parent
        if sha_dir.is_dir() and not (sha_dir / _MARKER).is_file():
            _write_marker(sha_dir, data_dir)
        log(f"[tau2-data] already provisioned at {data_dir}")
        return data_dir

    sha_dir = data_dir.parent
    sha_dir.mkdir(parents=True, exist_ok=True)
    log(f"[tau2-data] fetching {_TARBALL_URL}")
    with tempfile.TemporaryDirectory(dir=str(sha_dir), prefix=".dl-") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "archive.tar.gz"
        with urllib.request.urlopen(_TARBALL_URL, timeout=_DOWNLOAD_TIMEOUT_S) as resp:
            with archive.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
        log(f"[tau2-data] extracting the data/ subtree ({archive.stat().st_size >> 20} MB archive)")
        staged_data = tmp_path / "data"
        _extract_data_subtree(archive, staged_data)
        archive.unlink()
        verify(staged_data)
        # Publish atomically. If a concurrent provisioner already published a complete tree, adopt
        # it rather than fail on the non-empty rename.
        if data_dir.exists():
            if is_present(data_dir):
                log(f"[tau2-data] another tree is already in place at {data_dir}; keeping it")
            else:
                shutil.rmtree(data_dir)
                staged_data.replace(data_dir)
        else:
            staged_data.replace(data_dir)
    verify(data_dir)
    _write_marker(sha_dir, data_dir)
    log(f"[tau2-data] provisioned {data_dir}")
    return data_dir


__all__ = [
    "PROVISION_COMMAND",
    "UPSTREAM_SHA",
    "Tau2DataError",
    "is_present",
    "needs_tau2_data",
    "provision",
    "require",
    "resolve_data_dir",
    "verify",
]
