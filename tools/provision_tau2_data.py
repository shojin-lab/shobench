"""Provision tau2-bench's ``data/`` at the pinned upstream sha into shogym's cache.

shogym provisions the tau2 *source* at runtime but deliberately does not carry the ~730 MB of
benchmark ``data/``. The three tau2_telecom cells need it, so this fetches exactly that subtree,
once, into ``<SHOGYM_CACHE or ~/.cache/shogym>/tau2-data/<sha>/data`` -- the path the runner
points ``TAU2_DATA_DIR`` at. It is idempotent: a tree that already is the pinned data is verified
and skipped.

    uv run python tools/provision_tau2_data.py            # fetch if missing, else verify
    uv run python tools/provision_tau2_data.py --force    # re-fetch, replacing what is there
    uv run python tools/provision_tau2_data.py --check     # verify only, never fetch

Verification is against a committed manifest of the pinned commit's sizes and sha256s, so it
answers "is this that commit's data", not merely "are some files here". A tree that fails it is
named file by file, which is what makes ``--force`` the repair: it replaces the tree it fetched
over.

An explicit ``TAU2_DATA_DIR`` is honored, and a tree there is the operator's rather than this
command's. So a failing one is reported and left alone, with everything else living in it, and
only ``--force`` replaces it. The managed cache carries no such claim: an invalid one is replaced
without asking, because this command is the only thing that put it there.

The runner names this command when a tau2 cell is asked to run without the data present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shobench import tau2_data  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-fetch and replace the tree that is there"
    )
    parser.add_argument(
        "--check", action="store_true", help="verify the data is the pinned tree; never fetch"
    )
    args = parser.parse_args(argv)

    if args.check:
        data_dir = tau2_data.resolve_data_dir()
        try:
            tau2_data.verify(data_dir)
        except tau2_data.Tau2DataError as exc:
            print(f"UNUSABLE: {exc}", file=sys.stderr)
            print(
                f"provision it with: {tau2_data.PROVISION_COMMAND}"
                " (add --force to replace a tree that is already there)",
                file=sys.stderr,
            )
            return 1
        print(f"verified against upstream {tau2_data.UPSTREAM_SHA}: {data_dir}")
        return 0

    data_dir = tau2_data.provision(force=args.force)
    print(f"\nTAU2_DATA_DIR -> {data_dir}")
    print("The runner sets TAU2_DATA_DIR to this path itself; no export is needed for a cell.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
