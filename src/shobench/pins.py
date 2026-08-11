"""The substrate pins every cell manifest records.

A cell's numbers are only interpretable against the code that produced them, so the runner
writes these into the manifest before a cell starts and refuses to start when the installed
substrate disagrees with the pin (``shobench doctor``). Changing a pin is a reviewable commit
here, never a silent environment difference between two cells.
"""

from __future__ import annotations

# shogym, the environment substrate. Pinned by commit because PyPI's shogym 0.0.1 predates the
# serve layer this runner uses. `[tool.uv.sources]` in pyproject.toml installs this exact rev.
SHOGYM_REV = "9eb9edb88087af9a08520482a2d1de5831870944"
SHOGYM_REPO = "https://github.com/shojin-lab/shogym.git"

# Upstream data pins the envs themselves carry, recorded here so a manifest reader does not have
# to open shogym to learn which upstream produced the split.
TAU2_UPSTREAM_SHA = "1d244f5d"

__all__ = ["SHOGYM_REPO", "SHOGYM_REV", "TAU2_UPSTREAM_SHA"]
