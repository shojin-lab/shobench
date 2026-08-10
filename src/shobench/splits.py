"""Split manifests: the exact task ids on each side of a cell's split.

A manifest is committed data, not a computation. The runner reads ids from it and serves those
ids in that order, so a rerun serves the identical split and a reviewer can see what was held
out without running anything. Manifests that were derived rather than adopted also carry the
seed and the procedure that produced them, so the derivation is reproducible.

Three provenance kinds appear in v0:

``adopted``
    The ids come from a prior published study and are reused verbatim, so numbers stay
    comparable. automationbench does this with the conversation-not-memories held-out 120.
``upstream``
    The benchmark declares its own split and we honor it. tau2_telecom does this.
``seeded``
    No canonical split exists, so this repo draws one, records the seed and the algorithm, and
    publishes it. hle does this.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "shobench.split/1"
PROVENANCE_KINDS = ("adopted", "upstream", "seeded")


def splits_dir(root: Path | None = None) -> Path:
    from shobench.config import repo_root

    return (root or repo_root()) / "splits"


@dataclass(frozen=True)
class Side:
    """One side of a split: the ids, in served order, plus how to construct the env for them.

    ``env_kwargs`` exists because two of the three v0 envs address a task by its position
    inside an env-level split rather than by a global id. tau2_telecom serves its held-out set
    as ``task_split="test"`` and its pool as ``task_split="train"``; hle does the same over its
    own 80/20 slice. Carrying the kwargs on the side, not on the cell, is what keeps ids
    unambiguous: an id means nothing without the env it indexes into.
    """

    task_ids: tuple[str, ...]
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    # The env's own name for each id, positionally aligned with ``task_ids``. Serving never
    # uses it; it exists so a reviewer can read the manifest and see which upstream tasks were
    # held out, rather than a list of integers.
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.labels and len(self.labels) != len(self.task_ids):
            raise ValueError(
                f"labels ({len(self.labels)}) and task_ids ({len(self.task_ids)}) disagree"
            )

    def __len__(self) -> int:
        return len(self.task_ids)

    def key(self) -> tuple[tuple[str, Any], ...]:
        """The env-identity part of this side, for disjointness reasoning."""
        return tuple(sorted(self.env_kwargs.items()))


@dataclass(frozen=True)
class Split:
    """One env's improvement pool and held-out set, as ids the runner serves."""

    env: str
    heldout: Side
    pool: Side
    provenance: dict[str, Any]
    source: Path
    total_tasks: int | None = None

    def __post_init__(self) -> None:
        # Ids collide only when both sides index into the same env construction. When the sides
        # carry different env_kwargs they address disjoint task lists by construction, and the
        # manifest records that reasoning in provenance.disjoint_by.
        if self.heldout.key() == self.pool.key():
            overlap = set(self.heldout.task_ids) & set(self.pool.task_ids)
            if overlap:
                raise ValueError(
                    f"{self.source}: split is not disjoint; {len(overlap)} shared ids "
                    f"(first few: {sorted(overlap)[:5]})"
                )
        for label, side in (("held-out", self.heldout), ("pool", self.pool)):
            if len(set(side.task_ids)) != len(side.task_ids):
                raise ValueError(f"{self.source}: {label} ids contain duplicates")

    @property
    def id_digest(self) -> str:
        """A digest over the served ids in served order.

        Two runs of a cell agree on their split exactly when this matches, which is cheaper to
        compare in a results table than two id lists.
        """
        payload = json.dumps(
            {
                "env": self.env,
                "heldout": {
                    "env_kwargs": self.heldout.env_kwargs,
                    "task_ids": list(self.heldout.task_ids),
                },
                "pool": {
                    "env_kwargs": self.pool.env_kwargs,
                    "task_ids": list(self.pool.task_ids),
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_manifest(self) -> dict[str, Any]:
        return {
            "env": self.env,
            "path": str(self.source),
            "n_heldout": len(self.heldout),
            "n_pool": len(self.pool),
            "heldout_env_kwargs": dict(self.heldout.env_kwargs),
            "pool_env_kwargs": dict(self.pool.env_kwargs),
            "total_tasks": self.total_tasks,
            "id_digest": self.id_digest,
            "provenance": self.provenance,
        }


def _side(raw: dict[str, Any]) -> Side:
    return Side(
        task_ids=tuple(str(x) for x in raw["task_ids"]),
        env_kwargs=dict(raw.get("env_kwargs", {})),
        labels=tuple(str(x) for x in raw.get("labels", ())),
    )


def load_split(path: Path) -> Split:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    schema = raw.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"{path}: schema {schema!r} is not {SCHEMA!r}")
    provenance = raw.get("provenance", {})
    kind = provenance.get("kind")
    if kind not in PROVENANCE_KINDS:
        raise ValueError(f"{path}: provenance.kind {kind!r} is not one of {PROVENANCE_KINDS}")
    if kind == "seeded" and "seed" not in provenance:
        raise ValueError(f"{path}: a seeded split must record provenance.seed")
    return Split(
        env=raw["env"],
        heldout=_side(raw["heldout"]),
        pool=_side(raw["pool"]),
        provenance=provenance,
        source=path,
        total_tasks=raw.get("total_tasks"),
    )


def load_split_by_name(name: str, *, root: Path | None = None) -> Split:
    path = splits_dir(root) / f"{name}.json"
    if not path.is_file():
        available = sorted(p.stem for p in splits_dir(root).glob("*.json"))
        raise FileNotFoundError(f"unknown split {name!r}; available: {', '.join(available)}")
    return load_split(path)


def write_split(
    path: Path,
    *,
    env: str,
    heldout: list[str],
    pool: list[str],
    provenance: dict[str, Any],
    total_tasks: int | None = None,
    heldout_env_kwargs: dict[str, Any] | None = None,
    pool_env_kwargs: dict[str, Any] | None = None,
    heldout_labels: list[str] | None = None,
    pool_labels: list[str] | None = None,
) -> Path:
    """Write a manifest. Round-tripped through :func:`load_split` so a bad one never lands."""
    body = {
        "schema": SCHEMA,
        "env": env,
        "total_tasks": total_tasks,
        "provenance": provenance,
        "heldout": {
            "n": len(heldout),
            "env_kwargs": heldout_env_kwargs or {},
            "task_ids": heldout,
            "labels": heldout_labels or [],
        },
        "pool": {
            "n": len(pool),
            "env_kwargs": pool_env_kwargs or {},
            "task_ids": pool,
            "labels": pool_labels or [],
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    load_split(path)
    return path


__all__ = [
    "PROVENANCE_KINDS",
    "SCHEMA",
    "Side",
    "Split",
    "load_split",
    "load_split_by_name",
    "splits_dir",
    "write_split",
]
