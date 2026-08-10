"""The reporting script: paired bootstrap over the held-out deltas.

The interval is paired because both eval phases score the same held-out tasks, so resampling
task-level pairs cancels per-task difficulty out of the interval instead of leaving it in as
between-task variance. The procedure is the one the scope settled: resample the held-out tasks
with replacement, recompute the mean before-to-after delta on each resample, and report the
2.5th and 97.5th percentiles.

    uv run python -m shobench.report results/ --format table
    uv run python -m shobench.report results/ --format json > forest.json

The JSON output is the forest plot's input: one record per cell with the point estimate and
the interval, which is everything a plot needs and nothing it does not.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_RESAMPLES = 10000
DEFAULT_SEED = 20260807


@dataclass(frozen=True)
class CellReport:
    """One cell's four numbers and an interval, plus what the rollout did to earn them."""

    cell: str
    env: str
    harness: str
    model: str
    n_paired: int
    n_unpaired: int
    mean_before: float | None
    mean_after: float | None
    mean_delta: float | None
    ci_low: float | None
    ci_high: float | None
    solve_before: float | None
    solve_after: float | None
    rollout_attempted: int
    rollout_scored: int
    stop_reason: str
    resumes: int

    def to_json(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "env": self.env,
            "harness": self.harness,
            "model": self.model,
            "n_paired": self.n_paired,
            "n_unpaired": self.n_unpaired,
            "mean_before": self.mean_before,
            "mean_after": self.mean_after,
            "mean_delta": self.mean_delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "full_solve_before": self.solve_before,
            "full_solve_after": self.solve_after,
            "rollout_tasks_attempted": self.rollout_attempted,
            "rollout_tasks_scored": self.rollout_scored,
            "stop_reason": self.stop_reason,
            "usage_limit_resumes": self.resumes,
        }


def paired_bootstrap(
    deltas: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` for the mean of the paired deltas.

    Each resample draws ``len(deltas)`` pairs with replacement and takes their mean, which is
    the paired part: a pair moves as a unit, so a task that is hard in both phases contributes
    its delta and not its difficulty.
    """
    if not deltas:
        raise ValueError("no paired deltas to bootstrap")
    values = list(deltas)
    n = len(values)
    mean = sum(values) / n
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    low = means[int((alpha / 2) * resamples)]
    high = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return mean, low, high


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return (sum(present) / len(present)) if present else None


def report_cell(
    doc: dict[str, Any], *, resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED
) -> CellReport:
    manifest = doc.get("manifest", {})
    cell = manifest.get("cell", {})
    paired = doc.get("paired", [])
    deltas = [p["reward_delta"] for p in paired if p.get("reward_delta") is not None]

    mean_delta = ci_low = ci_high = None
    if deltas:
        mean_delta, ci_low, ci_high = paired_bootstrap(deltas, resamples=resamples, seed=seed)

    rollout = doc.get("rollout", {})
    stopping = rollout.get("stopping", {})
    return CellReport(
        cell=cell.get("name", "?"),
        env=cell.get("env", "?"),
        harness=cell.get("harness", "?"),
        model=cell.get("model", "?"),
        n_paired=len(paired),
        n_unpaired=len(doc.get("unpaired", [])),
        mean_before=_mean([p.get("reward_before") for p in paired]),
        mean_after=_mean([p.get("reward_after") for p in paired]),
        mean_delta=mean_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        solve_before=doc.get("eval_before", {}).get("summary", {}).get("full_solve_rate"),
        solve_after=doc.get("eval_after", {}).get("summary", {}).get("full_solve_rate"),
        rollout_attempted=rollout.get("summary", {}).get("tasks_attempted", 0),
        rollout_scored=rollout.get("summary", {}).get("tasks_scored", 0),
        stop_reason=stopping.get("stop_reason", "unrecorded"),
        resumes=stopping.get("usage_limit_resumes", 0),
    )


def _fmt(value: float | None, places: int = 3) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def render_table(reports: Sequence[CellReport]) -> str:
    header = (
        "cell",
        "env",
        "harness",
        "model",
        "N",
        "before",
        "after",
        "delta",
        "95% CI",
        "solve b/a",
        "rollout",
        "stop",
    )
    rows = [
        (
            r.cell,
            r.env,
            r.harness,
            r.model,
            str(r.n_paired),
            _fmt(r.mean_before),
            _fmt(r.mean_after),
            _fmt(r.mean_delta),
            f"[{_fmt(r.ci_low)}, {_fmt(r.ci_high)}]",
            f"{_fmt(r.solve_before, 2)}/{_fmt(r.solve_after, 2)}",
            f"{r.rollout_scored}/{r.rollout_attempted}",
            r.stop_reason,
        )
        for r in reports
    ]
    widths = [max(len(str(cell)) for cell in column) for column in zip(header, *rows, strict=True)]
    lines = [
        "  ".join(str(cell).ljust(width) for cell, width in zip(header, widths, strict=True)),
        "  ".join("-" * width for width in widths),
    ]
    lines += [
        "  ".join(str(cell).ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in rows
    ]
    return "\n".join(lines)


def load_results(target: Path) -> list[dict[str, Any]]:
    paths = sorted(target.glob("*.json")) if target.is_dir() else [target]
    docs = []
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("schema", "").startswith("shobench.results/"):
            docs.append(doc)
    return docs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=Path("results"))
    parser.add_argument("--format", choices=["table", "json"], default="table")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    docs = load_results(args.results)
    if not docs:
        print(f"no shobench results JSON found under {args.results}")
        return 1
    reports = [report_cell(d, resamples=args.resamples, seed=args.seed) for d in docs]
    reports.sort(key=lambda r: (r.env, r.harness, r.model))
    if args.format == "json":
        print(
            json.dumps(
                {
                    "bootstrap": {"resamples": args.resamples, "seed": args.seed, "alpha": 0.05},
                    "cells": [r.to_json() for r in reports],
                },
                indent=2,
            )
        )
    else:
        print(render_table(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CellReport", "paired_bootstrap", "render_table", "report_cell"]
