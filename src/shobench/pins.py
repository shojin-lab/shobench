"""The substrate pins every cell manifest records.

A cell's numbers are only interpretable against the code that produced them, so the runner
writes these into the manifest before a cell starts and refuses to start when the installed
substrate disagrees with the pin (``shobench doctor``). Changing a pin is a reviewable commit
here, never a silent environment difference between two cells.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

# shogym, the environment substrate. Pinned by commit because PyPI's shogym 0.0.1 predates the
# serve layer this runner uses. `[tool.uv.sources]` in pyproject.toml installs this exact rev.
SHOGYM_REV = "9eb9edb88087af9a08520482a2d1de5831870944"
SHOGYM_REPO = "https://github.com/shojin-lab/shogym.git"

# Upstream data pins the envs themselves carry, recorded here so a manifest reader does not have
# to open shogym to learn which upstream produced the split.
TAU2_UPSTREAM_SHA = "1d244f5d"


@lru_cache(maxsize=1)
def shobench_revision() -> tuple[str | None, bool]:
    """The runner's own commit, and whether the tree it ran from carried uncommitted changes.

    shogym serves and scores a task; this package decides how the task is launched and
    supervised, which is the other half of what produced a row: the deadline handed to the
    stream, the launch stagger, the drain watchdog, the row reconciliation, the flags the
    harness adapter builds. Two archives from different runner code can otherwise agree on
    every recorded field, so from here on every run records the revision it ran as.

    The dirty flag is not decoration. A modified tree's commit does not identify the code that
    ran, so a comparison has to treat it as unproven rather than as a match, and recording it is
    what lets the comparison know. A checkout git cannot answer for (an installed wheel, a
    tarball, no git on the host) records nothing rather than guessing, and nothing reads as the
    honest absence it is.

    Cached because a process writes several manifests and the answer cannot change under it.
    """
    root = Path(__file__).resolve().parents[2]
    try:
        rev = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if rev.returncode != 0:
            return None, False
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return rev.stdout.strip() or None, bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, False


__all__ = ["SHOGYM_REPO", "SHOGYM_REV", "TAU2_UPSTREAM_SHA", "shobench_revision"]
