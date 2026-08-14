"""Derive the committed split manifests under ``splits/``.

Run this to regenerate a manifest, never to serve one: the runner reads the committed JSON and
nothing else. Each builder records in ``provenance`` what it read and how, so a reviewer can
rerun it and get byte-identical output.

    uv run python tools/build_splits.py automationbench --shorep ../shorep
    uv run python tools/build_splits.py tau2_telecom
    uv run python tools/build_splits.py hle
    uv run python tools/build_splits.py tau2_banking_knowledge

The builders differ because the splits have different authority. automationbench adopts a
published split, tau2_telecom honors upstream's declared one, and hle and
tau2_banking_knowledge have none to honor so this repo draws and publishes its own.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shobench.pins import SHOGYM_REV, TAU2_UPSTREAM_SHA  # noqa: E402
from shobench.splits import splits_dir, write_split  # noqa: E402

# One seed for every derivation in this repo, so "which seed" is never a per-file question.
SEED = 20260807

# ----- automationbench -------------------------------------------------------------------

# The exploratory probes in the published package are not measured arms and do not carry the
# full held-out set, so they are excluded from the agreement check.
SHOREP_NON_ARMS = {"ctx-25"}
AUTOMATIONBENCH_TOTAL = 600


def build_automationbench(shorep: Path, out: Path) -> Path:
    """Adopt the conversation-not-memories held-out 120 verbatim.

    Every measured arm in the published package scored the same 120 task indices, so the split
    is recoverable by intersecting them, and the agreement is itself the check: if the arms
    disagreed, the adoption claim would be false and this raises instead of picking one.
    """
    heldout_dir = shorep / "studies" / "conversation-not-memories" / "data" / "heldout"
    if not heldout_dir.is_dir():
        raise SystemExit(f"no shorep held-out data at {heldout_dir}")

    per_arm: dict[str, set[int]] = {}
    for path in sorted(heldout_dir.glob("*.jsonl")):
        if path.stem in SHOREP_NON_ARMS:
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        per_arm[path.stem] = {int(r["task_idx"]) for r in rows}
    if not per_arm:
        raise SystemExit(f"no arms found under {heldout_dir}")

    id_sets = list(per_arm.values())
    if any(s != id_sets[0] for s in id_sets):
        disagreeing = [name for name, s in per_arm.items() if s != id_sets[0]]
        raise SystemExit(f"arms disagree on the held-out set: {disagreeing}")
    heldout = sorted(id_sets[0])
    if len(heldout) != 120:
        raise SystemExit(f"expected 120 held-out ids, found {len(heldout)}")

    # The pool is everything else. Its ORDER is seeded because the rollout serves it as a
    # stream and the order is part of the initial conditions; its MEMBERSHIP is not a draw.
    remaining = [i for i in range(AUTOMATIONBENCH_TOTAL) if i not in set(heldout)]
    if len(remaining) != 480:
        raise SystemExit(f"expected 480 pool ids, found {len(remaining)}")
    random.Random(SEED).shuffle(remaining)

    return write_split(
        out,
        env="automationbench",
        total_tasks=AUTOMATIONBENCH_TOTAL,
        heldout=[str(i) for i in heldout],
        pool=[str(i) for i in remaining],
        provenance={
            "kind": "adopted",
            "seed": SEED,
            "source": "shorep studies/conversation-not-memories/data/heldout",
            "arms_agreeing": sorted(per_arm),
            "excluded_from_agreement": sorted(SHOREP_NON_ARMS),
            "procedure": (
                "Held-out ids are the task_idx values every measured arm of the published "
                "study scored, adopted verbatim and served in ascending order. The pool is the "
                "remaining 480 of 600, shuffled once with the recorded seed to fix the "
                "rollout's serving order."
            ),
            "id_meaning": (
                "A shogym automationbench task_id is the stringified index into the env's 600 "
                "tasks, which is the same integer the study recorded as task_idx."
            ),
            "shogym_rev": SHOGYM_REV,
        },
    )


# ----- tau2_telecom ----------------------------------------------------------------------

TAU2_RAW = (
    "https://raw.githubusercontent.com/sierra-research/tau2-bench/"
    f"{TAU2_UPSTREAM_SHA}/data/tau2/domains/telecom"
)


def _fetch(url: str, cache: Path) -> Any:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (pinned https)
            cache.write_bytes(response.read())
    return json.loads(cache.read_text(encoding="utf-8"))


def build_tau2_telecom(out: Path, cache_dir: Path) -> Path:
    """Honor tau2's declared telecom split at the pinned upstream sha.

    shogym's port loads the split through upstream's own ``get_tasks(task_split_name=...)``,
    and its serve-layer task id is the position inside that filtered list. So the manifest
    records positions plus upstream's ids as labels, and the sides differ in ``task_split``.
    """
    tasks_doc = _fetch(f"{TAU2_RAW}/tasks.json", cache_dir / "telecom_tasks.json")
    splits_doc = _fetch(f"{TAU2_RAW}/split_tasks.json", cache_dir / "telecom_split_tasks.json")
    tasks = tasks_doc["tasks"] if isinstance(tasks_doc, dict) else tasks_doc

    def ordered(split_name: str) -> list[str]:
        # Upstream filters the file-order task list by split membership, so the position of a
        # task inside a split is its rank in file order among that split's members.
        members = set(splits_doc[split_name])
        return [str(t["id"]) for t in tasks if str(t["id"]) in members]

    train_ids = ordered("train")
    test_ids = ordered("test")
    if (len(train_ids), len(test_ids)) != (74, 40):
        raise SystemExit(
            f"upstream split changed: train={len(train_ids)} test={len(test_ids)}, expected 74/40"
        )

    return write_split(
        out,
        env="tau2_telecom",
        total_tasks=len(splits_doc.get("base", [])) or None,
        heldout=[str(i) for i in range(len(test_ids))],
        pool=[str(i) for i in range(len(train_ids))],
        heldout_env_kwargs={"task_split": "test"},
        pool_env_kwargs={"task_split": "train"},
        heldout_labels=test_ids,
        pool_labels=train_ids,
        provenance={
            "kind": "upstream",
            "upstream": "sierra-research/tau2-bench",
            "upstream_sha": TAU2_UPSTREAM_SHA,
            "source": "data/tau2/domains/telecom/split_tasks.json",
            "declared_split_sizes": {k: len(v) for k, v in splits_doc.items()},
            "disjoint_by": (
                "upstream's train and test lists are disjoint sets of tau2 task ids, and the "
                "two sides construct the env with different task_split values, so the "
                "positional ids never address the same task"
            ),
            "procedure": (
                "Held-out is upstream's declared test split (40) and the pool is its train "
                "split (74), each served in upstream's file order. Ids are positions inside "
                "the split, which is how shogym's port addresses a tau2 task; labels carry "
                "upstream's own task ids."
            ),
            "id_meaning": (
                "A shogym tau2 task_id is the stringified index into the env's task list for "
                "the constructed task_split, so an id is only meaningful with its env_kwargs."
            ),
            "shogym_rev": SHOGYM_REV,
        },
    )


# ----- hle -------------------------------------------------------------------------------

HLE_HELDOUT_N = 120
HLE_POOL_N = 300


def build_hle(out: Path) -> Path:
    """Draw and publish a split, because hle is an eval set with no canonical one.

    shogym slices the text-only questions 80/20 into ``train`` and ``test``; the 1726-task
    ``train`` slice is the population v0 draws from, which leaves shogym's own ``test`` slice
    untouched and out of this benchmark entirely.
    """
    from shogym.envs.hle.data import HF_DATASET, load_hle_tasks, split_tasks

    population = split_tasks(load_hle_tasks(text_only=True), "train")
    n = len(population)
    if n != 1726:
        raise SystemExit(
            f"hle train slice is {n} tasks and the manifest expects 1726: the dataset drifted"
        )

    order = list(range(n))
    random.Random(SEED).shuffle(order)
    heldout_idx = sorted(order[:HLE_HELDOUT_N])
    pool_idx = sorted(order[HLE_HELDOUT_N : HLE_HELDOUT_N + HLE_POOL_N])

    return write_split(
        out,
        env="hle",
        total_tasks=n,
        heldout=[str(i) for i in heldout_idx],
        pool=[str(i) for i in pool_idx],
        heldout_env_kwargs={"task_split": "train"},
        pool_env_kwargs={"task_split": "train"},
        heldout_labels=[str(population[i]["id"]) for i in heldout_idx],
        pool_labels=[str(population[i]["id"]) for i in pool_idx],
        provenance={
            "kind": "seeded",
            "seed": SEED,
            "dataset": HF_DATASET,
            "population": (
                "the text-only questions shogym assigns to its train slice "
                "(80 percent of 2158, so 1726 tasks)"
            ),
            "procedure": (
                "Shuffle the population's positions once with the recorded seed, take the "
                "first 120 as held-out and the next 300 as the improvement pool, then sort "
                "each side ascending so serving order is stable and reviewable. The pool is a "
                "ceiling on what the rollout may serve, not a quota."
            ),
            "id_meaning": (
                "A shogym hle task_id is the stringified index into the constructed "
                "task_split's task list; labels carry the dataset's own question ids."
            ),
            "leakage": (
                "hle answers are public. v0 observes leakage rather than gating it, so this "
                "split is published in full and the runner records network egress instead."
            ),
            "shogym_rev": SHOGYM_REV,
        },
    )


# ----- tau2_banking_knowledge ------------------------------------------------------------

BANKING_HELDOUT_N = 40
BANKING_POOL_N = 47
BANKING_TOTAL = 97
BANKING_ELIGIBLE_N = 87

# The reward bases tau2's ``env`` evaluator actually scores. It starts a task's reward at 1.0 and
# multiplies it only for these two, so a task whose basis holds anything else is scored on less
# than it declares: an ACTION-only task returns 1.0 for any run that terminates normally, and a
# DB + NL_ASSERTION task keeps only its DB half. The offline cells serve this evaluator, so the
# population is the tasks it scores in full and the rest are excluded rather than reported.
BANKING_HONORED_BASES = frozenset({"DB", "ENV_ASSERTION"})


def _banking_basis(task: Any) -> frozenset[str]:
    """One task's reward basis as plain strings, empty when it declares none."""
    criteria = getattr(task, "evaluation_criteria", None)
    basis = getattr(criteria, "reward_basis", None) or ()
    return frozenset(getattr(member, "value", member) for member in basis)


def build_tau2_banking_knowledge(out: Path) -> Path:
    """Draw and publish a split, because banking ships no split file to honor.

    telecom's sizes are upstream's; these are chosen. Held-out is 40, the size upstream declares
    for telecom, so the two tau2 envs are read against the same held-out N, and the pool is the
    87-task eligible population's remaining 47.
    """
    import os

    from shobench import tau2_data

    # Read the ids through the loader the env reads them through. banking's tasks come from a
    # directory of per-task files in sorted order rather than from tasks.json, and a manifest id
    # is a position in the list that loader returns.
    os.environ["TAU2_DATA_DIR"] = str(tau2_data.require())
    from shogym.envs.tau2 import mcp_server

    tasks = mcp_server.load_tasks("banking_knowledge")
    task_ids = [str(task.id) for task in tasks]
    n = len(task_ids)
    if n != BANKING_TOTAL:
        raise SystemExit(
            f"banking_knowledge has {n} tasks and the manifest expects {BANKING_TOTAL}: "
            "the domain drifted"
        )

    # Eligible positions, in the env's own task order. A task with no declared basis is excluded
    # too: the evaluator hands those a free 1.0 by the same arithmetic.
    bases = [_banking_basis(task) for task in tasks]
    eligible = [i for i, basis in enumerate(bases) if basis and basis <= BANKING_HONORED_BASES]
    excluded = {
        task_ids[i]: sorted(basis)
        for i, basis in enumerate(bases)
        if i not in set(eligible)
    }
    if len(eligible) != BANKING_ELIGIBLE_N:
        raise SystemExit(
            f"banking_knowledge has {len(eligible)} tasks the env evaluator scores in full and "
            f"the manifest expects {BANKING_ELIGIBLE_N}: the domain drifted"
        )

    order = list(eligible)
    random.Random(SEED).shuffle(order)
    heldout_idx = sorted(order[:BANKING_HELDOUT_N])
    pool_idx = sorted(order[BANKING_HELDOUT_N : BANKING_HELDOUT_N + BANKING_POOL_N])

    return write_split(
        out,
        env="tau2_banking_knowledge",
        total_tasks=n,
        heldout=[str(i) for i in heldout_idx],
        pool=[str(i) for i in pool_idx],
        heldout_labels=[task_ids[i] for i in heldout_idx],
        pool_labels=[task_ids[i] for i in pool_idx],
        provenance={
            "kind": "seeded",
            "seed": SEED,
            "upstream": "sierra-research/tau2-bench",
            "upstream_sha": TAU2_UPSTREAM_SHA,
            "source": "data/tau2/domains/banking_knowledge/tasks",
            "population": (
                "the 87 of 97 banking_knowledge tasks whose reward_basis tau2's env evaluator "
                "scores in full, which is a basis drawn from DB and ENV_ASSERTION alone"
            ),
            "authority": (
                "banking_knowledge ships no split_tasks.json, so there is no declared split to "
                "honor the way telecom's is honored, and no prior published run over these "
                "tasks to adopt. This repo draws one and publishes it."
            ),
            "excluded": (
                "The cells serve tau2's env evaluator, which starts a reward at 1.0 and "
                "multiplies it only for the DB and ENV_ASSERTION bases. Nine ACTION-only tasks "
                "are therefore not scored at all under it: every normally terminated run returns "
                "1.0 whatever the agent did. One DB + NL_ASSERTION task would be scored on its "
                "DB half alone. Both would be published as measurements of something they do "
                "not measure, so they are out of the population rather than in it with a "
                "caveat. Serving them honestly needs the keyed evaluator, which is a different "
                "cell."
            ),
            "excluded_tasks": excluded,
            "procedure": (
                "Shuffle the eligible positions once with the recorded seed, take the first 40 "
                "as held-out and the remaining 47 as the improvement pool, then sort each side "
                "ascending so serving order is stable and reviewable. Held-out is 40 because "
                "upstream declares 40 for telecom, so the two tau2 envs are read against the "
                "same held-out N. The pool is a ceiling on what the rollout may serve, not a "
                "quota."
            ),
            "id_meaning": (
                "A shogym tau2 task_id is the stringified index into the env's task list. "
                "banking_knowledge declares no train/test split, so both sides index the same "
                "97-task list, positions the split does not draw are served by neither side, "
                "and disjointness is checked on the ids themselves; labels carry upstream's own "
                "task ids."
            ),
            "shogym_rev": SHOGYM_REV,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "env",
        choices=["automationbench", "tau2_telecom", "hle", "tau2_banking_knowledge", "all"],
    )
    parser.add_argument("--shorep", type=Path, default=Path("../shorep"))
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "shobench")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    out_dir = args.out_dir or splits_dir()
    all_envs = ["automationbench", "tau2_telecom", "hle", "tau2_banking_knowledge"]
    targets = all_envs if args.env == "all" else [args.env]
    for env in targets:
        out = out_dir / f"{env}.json"
        if env == "automationbench":
            build_automationbench(args.shorep.resolve(), out)
        elif env == "tau2_telecom":
            build_tau2_telecom(out, args.cache)
        elif env == "tau2_banking_knowledge":
            build_tau2_banking_knowledge(out)
        else:
            build_hle(out)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
