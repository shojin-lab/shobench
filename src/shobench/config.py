"""Cell configuration: what a cell is, and how one is read off disk.

A cell is the unit of the benchmark: one environment, one harness, one model, one split
manifest, one instruction arm, one budget, one credential mode. Everything that distinguishes
two cells lives in a TOML file under ``cells/``, so the matrix is reviewable as data and a new
experiment arm is a new file rather than a new code path. The generation-effect study rides
this runner by varying ``[instruction]`` alone.

Nothing here talks to Docker, shogym, or a harness. It reads, validates, and hashes.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The three phases a cell runs, in order. `runner` executes exactly these names.
PHASES = ("eval_before", "rollout", "eval_after")

# Credential modes the runner knows how to provision. Subscription is the preferred mode for
# every v0 cell; api_key exists because two harnesses can only be smoke-tested that way today.
CREDENTIAL_MODES = ("subscription", "api_key")

HARNESSES = ("claude_code", "codex", "prime_agent")


def repo_root() -> Path:
    """The checkout root, located from this file rather than from the working directory.

    ``src/shobench/config.py`` -> ``src/shobench`` -> ``src`` -> root.
    """
    return Path(__file__).resolve().parents[2]


def repo_relative(path: Path) -> str:
    """A repo path recorded portably: relative to the checkout root, in POSIX form.

    A durable record must not carry the operator's absolute path, which leaks a username and a
    machine layout and is wrong on any other checkout anyway. The manifest lives beside the
    files it names, so the checkout-relative form is what a reader on another machine resolves.
    A path that somehow lies outside the checkout falls back to its basename rather than leaking
    the absolute path it came from.
    """
    try:
        return path.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return path.name


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Instruction:
    """One instruction arm, resolved to its bytes and its digest.

    The digest is what makes "byte-identical in every cell" checkable after the fact: it goes
    into the manifest, so two cells claiming the same arm can be proven to have carried the
    same prompt.
    """

    arm: str
    rollout_system_path: Path
    rollout_system: str
    eval_system_path: Path
    eval_system: str
    kickoff: str
    continuation: str

    @property
    def rollout_system_sha256(self) -> str:
        return _sha256_text(self.rollout_system)

    @property
    def eval_system_sha256(self) -> str:
        return _sha256_text(self.eval_system)

    @property
    def continuation_sha256(self) -> str:
        return _sha256_text(self.continuation)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "rollout_system": self.rollout_system,
            "rollout_system_sha256": self.rollout_system_sha256,
            "eval_system": self.eval_system,
            "eval_system_sha256": self.eval_system_sha256,
            "kickoff": self.kickoff,
            "continuation": self.continuation,
            "continuation_sha256": self.continuation_sha256,
        }


def load_instruction(arm: str, *, root: Path | None = None) -> Instruction:
    """Read an instruction arm from ``instructions/<arm>/``.

    Every arm supplies four files. ``rollout.system.txt`` is the standing instruction for the
    improvement rollout, ``eval.system.txt`` the one for both eval phases (never carrying the
    improvement objective, because an eval is a measurement), ``kickoff.txt`` the minimal user
    turn, and ``continue.txt`` the cue the runner sends to resume a rollout.
    """
    base = (root or repo_root()) / "instructions" / arm
    if not base.is_dir():
        raise FileNotFoundError(f"unknown instruction arm {arm!r}: no directory at {base}")
    parts: dict[str, tuple[Path, str]] = {}
    for name in ("rollout.system.txt", "eval.system.txt", "kickoff.txt", "continue.txt"):
        path = base / name
        if not path.is_file():
            raise FileNotFoundError(f"instruction arm {arm!r} is missing {name} ({path})")
        parts[name] = (path, path.read_text(encoding="utf-8"))
    return Instruction(
        arm=arm,
        rollout_system_path=parts["rollout.system.txt"][0],
        rollout_system=parts["rollout.system.txt"][1],
        eval_system_path=parts["eval.system.txt"][0],
        eval_system=parts["eval.system.txt"][1],
        kickoff=parts["kickoff.txt"][1],
        continuation=parts["continue.txt"][1],
    )


@dataclass(frozen=True)
class Budget:
    """The rollout budget, and the eval-phase guard rails.

    ``rollout_wall_clock_s`` is the parity axis the scope settled on: identical wall clock
    across harnesses within an env, no token ceiling. ``pool_ceiling`` is the maximum number of
    improvement tasks the runner will serve; the agent stopping short of it is an outcome, not
    a failure, so nothing re-serves to reach it.
    """

    rollout_wall_clock_s: int
    pool_ceiling: int | None = None
    eval_task_timeout_s: int = 3600
    # A rollout leg is one harness invocation inside the wall clock. Bounding it keeps a
    # harness that wedges from consuming the whole budget in one unrecoverable process, and it
    # is what makes codex's episodic supervision the same mechanism as everyone else's.
    rollout_leg_timeout_s: int = 3600
    # How many consecutive legs may end without the stream advancing before the runner calls
    # the rollout over. This is what separates "chose to stop" from "wedged".
    max_stalled_legs: int = 3

    def to_manifest(self) -> dict[str, Any]:
        return {
            "rollout_wall_clock_s": self.rollout_wall_clock_s,
            "pool_ceiling": self.pool_ceiling,
            "eval_task_timeout_s": self.eval_task_timeout_s,
            "rollout_leg_timeout_s": self.rollout_leg_timeout_s,
            "max_stalled_legs": self.max_stalled_legs,
        }


@dataclass(frozen=True)
class Cell:
    """One (env, harness, model) cell of the benchmark matrix."""

    name: str
    env: str
    harness: str
    model: str
    split: str
    instruction_arm: str
    budget: Budget
    credential_mode: str
    source: Path
    # Extra keyword arguments handed to shogym env construction, e.g. tau2's task_split_name.
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    # Environment variables the cell's containers need beyond credentials, e.g. HF_TOKEN's
    # presence requirement. Values are never stored here, only names.
    required_env: tuple[str, ...] = ()
    # Free-text note carried into the manifest, for anything a reader of the results needs.
    note: str = ""

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "env": self.env,
            "harness": self.harness,
            "model": self.model,
            "split": self.split,
            "instruction_arm": self.instruction_arm,
            "credential_mode": self.credential_mode,
            "env_kwargs": dict(self.env_kwargs),
            "required_env": list(self.required_env),
            "budget": self.budget.to_manifest(),
            "config_path": repo_relative(self.source),
            "config_sha256": _sha256_file(self.source),
            "note": self.note,
        }


def _require(table: dict[str, Any], key: str, path: Path) -> Any:
    if key not in table:
        raise ValueError(f"{path}: missing required key {key!r}")
    return table[key]


def load_cell(path: Path) -> Cell:
    """Read one cell config. Every validation failure names the file and the key."""
    path = Path(path).resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    cell = _require(raw, "cell", path)
    budget_table = _require(raw, "budget", path)

    harness = _require(cell, "harness", path)
    if harness not in HARNESSES:
        raise ValueError(f"{path}: harness {harness!r} is not one of {HARNESSES}")
    mode = cell.get("credential_mode", "subscription")
    if mode not in CREDENTIAL_MODES:
        raise ValueError(f"{path}: credential_mode {mode!r} is not one of {CREDENTIAL_MODES}")

    budget = Budget(
        rollout_wall_clock_s=int(_require(budget_table, "rollout_wall_clock_s", path)),
        pool_ceiling=budget_table.get("pool_ceiling"),
        eval_task_timeout_s=int(budget_table.get("eval_task_timeout_s", 3600)),
        rollout_leg_timeout_s=int(budget_table.get("rollout_leg_timeout_s", 3600)),
        max_stalled_legs=int(budget_table.get("max_stalled_legs", 3)),
    )
    return Cell(
        name=_require(cell, "name", path),
        env=_require(cell, "env", path),
        harness=harness,
        model=_require(cell, "model", path),
        split=_require(cell, "split", path),
        instruction_arm=cell.get("instruction_arm", "get-better"),
        budget=budget,
        credential_mode=mode,
        source=path,
        env_kwargs=dict(raw.get("env_kwargs", {})),
        required_env=tuple(cell.get("required_env", ())),
        note=cell.get("note", ""),
    )


def cells_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "cells"


def load_cell_by_name(name: str, *, root: Path | None = None) -> Cell:
    """Resolve a cell by file stem under ``cells/``."""
    path = cells_dir(root) / f"{name}.toml"
    if not path.is_file():
        available = sorted(p.stem for p in cells_dir(root).glob("*.toml"))
        raise FileNotFoundError(f"unknown cell {name!r}; available: {', '.join(available)}")
    return load_cell(path)


def load_all_cells(*, root: Path | None = None) -> list[Cell]:
    return [load_cell(p) for p in sorted(cells_dir(root).glob("*.toml"))]


__all__ = [
    "CREDENTIAL_MODES",
    "HARNESSES",
    "PHASES",
    "Budget",
    "Cell",
    "Instruction",
    "cells_dir",
    "load_all_cells",
    "load_cell",
    "load_cell_by_name",
    "load_instruction",
    "repo_relative",
    "repo_root",
]
