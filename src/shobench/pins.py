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

    A wrong answer is worse than none, so two ways to give one are closed. Every git command has
    to succeed, since a failed ``status`` beside a successful ``rev-parse`` would attest "this
    commit, clean" about bytes nobody looked at. And the repository has to be THIS package's own:
    git searches upward, so a wheel installed under someone else's checkout finds their
    repository, and the toplevel counts only when it tracks the file being imported.

    Cached for the process, unlike the agent image: the imported code cannot change under a
    running process, so the answer at import time is the answer for every run in it.
    """
    module = Path(__file__).resolve()
    top = _git(module.parent, "rev-parse", "--show-toplevel")
    if top is None:
        return None, False
    root = Path(top)
    if (root / "src" / "shobench" / "pins.py").resolve() != module:
        # An installed build, or a checkout inside another repository. There is no revision that
        # identifies these bytes, and the static package version identifies
        # nothing either, so the honest answer is that nothing is known.
        return None, False
    rev = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    if rev is None or status is None:
        return None, False
    return rev or None, bool(status)


def _git(cwd: Path, *args: str) -> str | None:
    """One git command's output, or nothing at all when it did not succeed."""
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


__all__ = ["SHOGYM_REPO", "SHOGYM_REV", "TAU2_UPSTREAM_SHA", "shobench_revision"]
