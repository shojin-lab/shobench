"""Provisioning tau2-bench's ``data/`` at the pinned upstream sha.

shogym's tau2 port provisions the upstream *source* (``src/tau2``) at runtime and deliberately
filters out the archive's ~700 MB of benchmark ``data/`` (see ``shogym.envs._upstream``). The
data is a separate concern: upstream resolves it from ``TAU2_DATA_DIR``, and the three
tau2_telecom cells need it. This module fetches exactly that subtree, once, into a cache the
runner points ``TAU2_DATA_DIR`` at.

The data is ~730 MB, so it is provisioned rather than committed. It is fetched from the same
SHA-pinned GitHub archive the source comes from, so the data's identity is the same pin as the
source (a test asserts the two shas agree). The provisioning is idempotent: a tree that already
is the pinned data is recognized and skipped, and a freshly fetched one is renamed into place
only after it validates, so an interrupted fetch leaves nothing that would be accepted.

What "is the pinned data" means is a byte question, not a structural one, so a committed manifest
of sizes and sha256s answers it. Existence and non-emptiness would accept an older checkout handed
in through ``TAU2_DATA_DIR``, or a tree whose policy or DB was edited, and either changes what the
benchmark measures while every gate still reports green. Digests of the pinned commit's own bytes
cannot be talked into that: a tree either is that commit's data or it is refused by name.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

# The pinned upstream commit whose data these cells use. This is the full sha; a test asserts it
# equals shogym's own tau2 source pin (``shogym.envs.tau2.adapter.UPSTREAM_SHA``) and begins with
# the short sha recorded in ``pins.TAU2_UPSTREAM_SHA`` and in the split manifest, so the data and
# the source can never silently come from different commits.
UPSTREAM_SHA = "1d244f5dca42944b67a379b44bfeb9f5748f189d"
_TARBALL_URL = f"https://github.com/sierra-research/tau2-bench/archive/{UPSTREAM_SHA}.tar.gz"

# Envs whose construction reads tau2 data. v0 ships only tau2_telecom, but every tau2 domain
# resolves its tasks from this tree, so the gate is by prefix rather than by an explicit list.
_TAU2_PREFIX = "tau2"

# The manifest: every file the pinned tau2_telecom runtime reads, with the size and sha256 of the
# pinned commit's bytes. It is committed data and it is both halves of the gate at once. The file
# list says what has to exist, and it was traced rather than guessed (every ``open`` under
# ``TAU2_DATA_DIR`` recorded across env construction, session start, and a user turn), which is
# why it carries the tech support manual the env reads unconditionally and the guidelines the user
# simulator reads when it builds its system prompt, and carries no voice or other-domain bulk. The
# digests say what those files have to *be*. Paths are relative to the ``data/`` dir
# ``TAU2_DATA_DIR`` names; upstream reads ``DATA_DIR / "tau2" / ...`` from there.
#
# The digests come from the pinned commit itself, each file fetched from upstream at
# ``UPSTREAM_SHA`` and hashed, so they are independent of whatever a local cache happens to hold;
# moving the pin means re-recording them, and a test holds the manifest to the sha so a bumped pin
# cannot keep checking the old commit's bytes. What the manifest does not do is digest all
# ~730 MB: that would gate files no cell touches at a cost on every check, so ``--force`` is what
# replaces a tree whose ungated bulk is suspect.
_MANIFEST_PATH = Path(__file__).with_name("tau2_data_manifest.json")
_MANIFEST: dict[str, dict[str, Any]] = json.loads(
    _MANIFEST_PATH.read_text(encoding="utf-8")
)["files"]

# The record written beside the managed cache, and only there: a tree the operator named through
# TAU2_DATA_DIR owns its own provenance, and a marker beside it would be this module writing into
# their checkout. Its presence is not what any tree is judged on (an externally provisioned tree
# without it is still recognized), which is what makes leaving it out free.
_MARKER = ".shobench-tau2-data.json"

# What a fetch in flight is named while it stages, and therefore what an interrupted one leaves
# behind. Dotted so it is out of the way, prefixed so an abandoned one is recognizable as ours.
_STAGING_PREFIX = ".dl-"

# What the runner and the docs tell an operator to run when the data is missing.
PROVISION_COMMAND = "uv run python tools/provision_tau2_data.py"

_DOWNLOAD_TIMEOUT_S = 300.0


class Tau2DataError(RuntimeError):
    """The tau2 data is missing, incomplete, or not the pinned bytes. The message names the fix."""


def _cache_root() -> Path:
    base = os.environ.get("SHOGYM_CACHE")
    return Path(base) if base else Path.home() / ".cache" / "shogym"


def _override_data_dir() -> Path | None:
    """The operator's explicit ``TAU2_DATA_DIR``, if any: a tree this module may read, not own.

    Whether the path was named by an operator or derived by this module is what decides who may
    delete it, so the two cases are distinguishable rather than collapsed into one resolved path.
    """
    override = os.environ.get("TAU2_DATA_DIR")
    return Path(override).expanduser() if override else None


def resolve_data_dir() -> Path:
    """Where ``TAU2_DATA_DIR`` should point: an operator override, else the pinned cache path.

    An explicit ``TAU2_DATA_DIR`` in the environment wins, exactly as ``TAU2_SRC`` overrides the
    source cache, so a checkout provisioned elsewhere is honored without a re-fetch. Otherwise the
    path is ``<SHOGYM_CACHE or ~/.cache/shogym>/tau2-data/<sha>/data`` -- a sibling of shogym's
    own ``tau2/<sha>`` source cache, never the same directory.
    """
    override = _override_data_dir()
    if override is not None:
        return override
    return _cache_root() / "tau2-data" / UPSTREAM_SHA / "data"


def verify(data_dir: Path) -> None:
    """Raise :class:`Tau2DataError` unless ``data_dir`` is the pinned commit's tau2 data.

    Every manifest file must be present and byte-identical to the pinned commit. That is one
    check doing two jobs. Completeness: a tree missing a file the runtime reads is refused here
    rather than at env setup, or later still, after a spending run has begun. Identity: a tree of
    the right shape but the wrong bytes (a stale checkout pointed at by ``TAU2_DATA_DIR``, an
    edited policy or DB, a fetch truncated mid-file) is data drift that would move the benchmark's
    numbers silently, so it is refused by name too. The size is compared before the digest only
    because it makes the common failure, a partial fetch, say so.
    """
    if not data_dir.is_dir():
        raise Tau2DataError(f"no tau2 data at {data_dir}")
    for rel, pinned in _MANIFEST.items():
        path = data_dir / rel
        if not path.is_file():
            raise Tau2DataError(f"tau2 data at {data_dir} is incomplete: missing {rel}")
        try:
            body = path.read_bytes()
        except OSError as exc:
            raise Tau2DataError(f"tau2 data at {data_dir}: {rel} is unreadable ({exc})") from exc
        if len(body) != pinned["size"]:
            raise Tau2DataError(
                f"tau2 data at {data_dir}: {rel} is {len(body)} bytes, the pinned commit's is "
                f"{pinned['size']} (a partial fetch, or not the pinned data)"
            )
        if hashlib.sha256(body).hexdigest() != pinned["sha256"]:
            raise Tau2DataError(
                f"tau2 data at {data_dir}: {rel} is not the bytes of upstream {UPSTREAM_SHA} "
                "(right size, different content), so this tree is not the pinned data"
            )


def is_present(data_dir: Path | None = None) -> bool:
    """Whether the pinned tau2 data is where the runner would look for it."""
    try:
        verify(data_dir if data_dir is not None else resolve_data_dir())
    except Tau2DataError:
        return False
    return True


def require() -> Path:
    """Return the data dir if it is the pinned data, else raise with the provisioning command.

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
                "verified_files": sorted(_MANIFEST),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _sweep_abandoned_staging(sha_dir: Path, log) -> None:
    """Reclaim staging dirs a killed provisioner left in the managed cache. Best effort.

    ``TemporaryDirectory`` cleans up when Python unwinds, and a signal does not unwind. A run
    killed mid-fetch therefore leaves its archive and its partly staged tree, most of ~730 MB
    once extraction is under way, sitting in the cache, and every retry stacks another one.
    Nothing else ever reclaims them, so the next provision does it before staging its own.

    Deliberately without a lock. A sweep can only harm a second live provisioner, and this
    module already assumes there is none: a single operator provisions once per host, the same
    assumption behind publishing without one. Serializing per sha is the upgrade point if that
    stops holding. Failures are swallowed rather than raised, since reclaiming space is a
    courtesy on the way to the real work and must not be what fails the command.
    """
    for stale in sorted(sha_dir.glob(_STAGING_PREFIX + "*")):
        if stale.is_dir():
            log(f"[tau2-data] reclaiming {stale.name}, left by an interrupted fetch")
            shutil.rmtree(stale, ignore_errors=True)


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

    Idempotent: a tree that already is the pinned data is verified and touches no network, unless
    ``force``. Otherwise the tarball is downloaded, the ``data/`` subtree is extracted to a
    staging dir, verified, and only then renamed into place, so an interrupted run leaves no tree
    that :func:`verify` would accept.

    ``force`` is the repair path, so it replaces the managed cache. Paying for a download and then
    keeping the tree the operator asked to be rid of would make the flag do nothing at full price.
    It is also the only thing that replaces a tree under an explicit ``TAU2_DATA_DIR``: that one
    belongs to the operator, so an unforced run refuses it rather than overwrites it. Nothing at
    all is written beside such a tree, not even the provenance marker, so pointing this command at
    a read-only checkout verifies it and stops there.
    """
    data_dir = resolve_data_dir()
    override = _override_data_dir()
    if not force:
        try:
            verify(data_dir)
        except Tau2DataError as exc:
            # An explicit TAU2_DATA_DIR names a tree the operator owns, and the only way to be
            # here is that the tree is not the pinned data. Replacing it would take everything
            # else living in it too: a checkout's other domains, notes, unfinished work. So it is
            # refused, and only --force accepts that cost on the operator's say-so. Refusing here
            # rather than after publishing costs nothing, because nothing has been fetched yet.
            # The managed cache gets no such protection and needs none: this module created it,
            # and replacing our own cache destroys no one's work.
            if override is not None and data_dir.exists():
                raise Tau2DataError(
                    f"{exc}. TAU2_DATA_DIR names a tree this command did not create, so it is "
                    f"left alone: fix it, unset TAU2_DATA_DIR to provision the managed cache "
                    f"instead, or re-run with --force to replace {data_dir}"
                ) from exc
        else:
            # Verified, and for an override that is the whole of it: reading a tree the operator
            # named must not write to it, and the marker would land in their checkout root rather
            # than in the data dir they pointed at, which a read-only checkout would refuse
            # outright. So the advertised verify-and-skip does exactly that, and nothing else.
            if override is None:
                sha_dir = data_dir.parent
                if sha_dir.is_dir() and not (sha_dir / _MARKER).is_file():
                    _write_marker(sha_dir, data_dir)
            log(f"[tau2-data] already provisioned at {data_dir}")
            return data_dir

    sha_dir = data_dir.parent
    sha_dir.mkdir(parents=True, exist_ok=True)
    # Only the managed cache is swept, by the ownership rule that governs everything else here:
    # a staging dir abandoned beside a tree the operator named is still in their directory, so it
    # stays for them to remove rather than being deleted by a pattern match on their disk.
    if override is None:
        _sweep_abandoned_staging(sha_dir, log)
    log(f"[tau2-data] fetching {_TARBALL_URL}")
    with tempfile.TemporaryDirectory(dir=str(sha_dir), prefix=_STAGING_PREFIX) as tmp:
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
        # Publish by rename, with two questions settled first. Is what is there already the
        # pinned data: then adopt it, which is how a provisioner that published while this one
        # downloaded is accepted, since two provisioners of one pin want the same bytes. If not,
        # may this run delete it: what may be deleted is what forcing authorized, plus the managed
        # cache, which this module owns and reached here only by being absent or invalid. A tree
        # under an unforced override never is, having been refused above if it existed and still
        # being the operator's if it appeared since. What is deleted moves aside into the staging
        # dir first (same filesystem, so it is a rename) and leaves with it, so the moment nothing
        # is published is a rename rather than a recursive delete.
        if data_dir.exists() and not force and is_present(data_dir):
            log(f"[tau2-data] another tree is already in place at {data_dir}; keeping it")
        else:
            if data_dir.exists() and (force or override is None):
                data_dir.replace(tmp_path / "superseded")
            try:
                staged_data.replace(data_dir)
            except OSError:
                # The same race one step later: a directory arrived between that check and this
                # rename. Adopt it for the same reason, on the verify below this block rather than
                # on trust, and without retrying, because a second failure is a real problem and
                # not a race. If the destination is not a directory the rename failed for its own
                # reasons, and hiding those behind an adoption would be worse than raising them.
                if not data_dir.is_dir():
                    raise
                log(f"[tau2-data] another provisioner published {data_dir} first; adopting it")
    verify(data_dir)
    # Same rule after a fetch, forced or not: provenance is recorded for the cache this module
    # owns, and a tree the operator named keeps its own.
    if override is None:
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
