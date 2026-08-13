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

from shobench.runner import recorded_eval_context, recorded_rollout_feedback

DEFAULT_RESAMPLES = 10000
DEFAULT_SEED = 20260807


@dataclass(frozen=True)
class CellReport:
    """One cell's four numbers and an interval, plus what the rollout did to earn them.

    Identity is more than the cell name: rebookends put several runs of one cell beside each
    other on purpose, so every row carries its ``run_id``, its arm axes, and its pairing. A
    row whose ``pairing`` is "self" measured its own before and after; "assembled" is a
    bookend whose after was paired with its SOURCE's before by the assembler; and
    "source_missing" is a bookend whose source artifact was not among the loaded files, kept
    visible rather than silently reported as an unpaired zero.
    """

    cell: str
    env: str
    harness: str
    model: str
    run_id: str
    rollout_feedback: str
    eval_context: str
    rebookend_of: str | None
    baseline_run_id: str | None
    pairing: str
    n_paired: int
    n_unpaired: int
    # What the cell asked for, and whether the file can account for it. A paired mean over 118
    # of 120 held-out tasks is a different number from a paired mean over 120, and a table that
    # shows only the first reads as the second.
    n_requested: int
    n_missing: int
    complete: bool
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
            "run_id": self.run_id,
            "rollout_feedback": self.rollout_feedback,
            "eval_context": self.eval_context,
            "rebookend_of": self.rebookend_of,
            "baseline_run_id": self.baseline_run_id,
            "pairing": self.pairing,
            "n_paired": self.n_paired,
            "n_unpaired": self.n_unpaired,
            "n_requested": self.n_requested,
            "n_missing": self.n_missing,
            "complete": self.complete,
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


def assemble(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join every bookend artifact to its source before any reporting math runs.

    A bookend artifact is honest but incomplete by construction: its eval_before is all
    missing rows and its own pairing is empty, because the before side belongs to the SOURCE
    run it names in ``manifest.rebookend.rebookend_of``. Fed to the reporter raw, it showed
    ``n_paired 0`` beside a duplicate cell row, which is not the measurement the rebookend
    exists to create. So the assembler runs first: each bookend is matched to the loaded
    artifact whose ``manifest.run_id`` its marker's BASELINE identity names (falling back to
    the lineage id for markers written before the field existed, whose source was their
    baseline), the baseline's eval_before block replaces the bookend's empty one, and the
    pairing is recomputed through the same ``pair_evals`` every publisher uses, over the
    bookend's committed held-out ids.

    A bookend whose baseline is not among the loaded files passes through UNASSEMBLED, with
    an annotation the reporter surfaces as ``pairing: baseline_missing``: an explicit row,
    never a silent unpaired zero, because a measurement that cannot find its other half is a
    fact the reader needs. Non-bookend documents pass through untouched. Assembly is
    in-memory only; no artifact is rewritten.
    """
    from shobench.results import TaskResult, heldout_accounting, pair_evals

    by_run_id = {
        doc.get("manifest", {}).get("run_id"): doc
        for doc in docs
        if doc.get("manifest", {}).get("run_id")
    }
    assembled: list[dict[str, Any]] = []
    for doc in docs:
        marker = doc.get("manifest", {}).get("rebookend")
        if not marker:
            assembled.append(doc)
            continue
        # The before-side comes from the BASELINE identity, which is the pairing partner the
        # marker records beside the lineage: the rollout source (rebookend_of) may be an
        # after-only or rollout-only run whose eval_before never existed, measured instead by
        # a separate deferred-baseline run. A marker from before the field existed named only
        # the source, and for those the source IS the baseline, so the fallback keeps them
        # assembling with their old meaning.
        baseline_id = marker.get("baseline_run_id") or marker.get("rebookend_of")
        baseline = by_run_id.get(baseline_id)
        if baseline is not None and (
            baseline is doc or "rebookend" in baseline.get("manifest", {})
        ):
            # The named baseline is itself a bookend (a chain, a cycle, or a self-loop). Its
            # eval_before is all missing by construction, so pairing against it would report
            # an assembled measurement whose before-side never existed. The runner refuses to
            # create such an artifact at the acceptance boundary; one that exists anyway (an
            # older branch, a hand-made file) is labeled for what it is, never "assembled".
            invalid = dict(doc)
            invalid["assembly"] = {"invalid_provenance": str(baseline_id or "")}
            assembled.append(invalid)
            continue
        if baseline is None:
            assembled.append(doc)
            continue
        task_ids = [int(t) for t in doc.get("heldout", {}).get("task_ids", [])]
        before = [TaskResult(**row) for row in baseline.get("eval_before", {}).get("tasks", [])]
        after = [TaskResult(**row) for row in doc.get("eval_after", {}).get("tasks", [])]
        paired, unpaired = pair_evals(before, after, task_ids=task_ids)
        accounting = {
            "eval_before": heldout_accounting(before, task_ids=task_ids),
            "eval_after": heldout_accounting(after, task_ids=task_ids),
        }
        joined = dict(doc)
        joined["eval_before"] = baseline.get("eval_before", {})
        joined["paired"] = paired
        joined["unpaired"] = unpaired
        joined["heldout"] = {
            **doc.get("heldout", {}),
            **accounting,
            "complete": all(entry["complete"] for entry in accounting.values()),
        }
        joined["assembly"] = {"paired_with": baseline.get("manifest", {}).get("run_id")}
        assembled.append(joined)
    return assembled


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
    # Read from the published accounting rather than recomputed from the rows: the ids a cell
    # never measured are exactly what its own rows cannot show. A file carrying no accounting at
    # all is from some other version of this writer, so it is marked rather than trusted, and its
    # ratio falls back to the ids it does know about instead of reading as a denominator of zero.
    heldout = doc.get("heldout", {})
    unpaired = doc.get("unpaired", [])
    missing = sorted(
        {
            idx
            for phase in ("eval_before", "eval_after")
            for idx in heldout.get(phase, {}).get("missing_task_ids", [])
        }
    )
    marker = manifest.get("rebookend")
    assembly = doc.get("assembly", {})
    if not marker:
        pairing = "self"
    elif assembly.get("paired_with"):
        pairing = "assembled"
    elif "invalid_provenance" in assembly:
        pairing = "invalid_provenance"
    else:
        pairing = "baseline_missing"
    return CellReport(
        cell=cell.get("name", "?"),
        env=cell.get("env", "?"),
        harness=cell.get("harness", "?"),
        model=cell.get("model", "?"),
        run_id=str(manifest.get("run_id", "?")),
        # The recorded-axis semantics, not a guess: a manifest written before an axis existed
        # could only have run that axis's one pre-axis posture, which is exactly what
        # ``recorded_rollout_feedback`` and ``recorded_eval_context`` define (never, cold),
        # so a legacy artifact renders its real arm rather than a question mark.
        rollout_feedback=recorded_rollout_feedback(manifest),
        eval_context=recorded_eval_context(manifest),
        rebookend_of=(str(marker.get("rebookend_of")) if marker else None),
        baseline_run_id=(
            str(marker.get("baseline_run_id") or marker.get("rebookend_of"))
            if marker
            else None
        ),
        pairing=pairing,
        n_paired=len(paired),
        n_unpaired=len(unpaired),
        n_requested=heldout.get("n_requested", len(paired) + len(unpaired)),
        n_missing=len(missing),
        complete=bool(heldout.get("complete", False)),
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


def _run_suffix(report: CellReport, run_id: str | None = None) -> str:
    """The run id with its cell-name prefix stripped, which is what distinguishes duplicate
    cell rows without tripling the table's width; the JSON output carries the full id."""
    value = report.run_id if run_id is None else run_id
    prefix = f"{report.cell}-"
    return value[len(prefix):] if value.startswith(prefix) else value


def render_table(reports: Sequence[CellReport]) -> str:
    header = (
        "cell",
        "run",
        "arm",
        "pairing",
        "env",
        "harness",
        "model",
        "N",
        "missing",
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
            # A cell that cannot account for every held-out task is marked where its name is,
            # because every other number on its line is a mean over a subset of what it asked for.
            r.cell if r.complete else f"{r.cell} *",
            _run_suffix(r),
            f"{r.rollout_feedback}+{r.eval_context}",
            (
                "self"
                if r.pairing == "self"
                else f"of {_run_suffix(r, r.baseline_run_id)}"
                if r.pairing == "assembled"
                else "INVALID PROVENANCE"
                if r.pairing == "invalid_provenance"
                else "BASELINE MISSING"
            ),
            r.env,
            r.harness,
            r.model,
            # Paired over requested, so the denominator the number is a mean over is in the
            # table rather than in the file the table came from.
            f"{r.n_paired}/{r.n_requested}",
            str(r.n_missing),
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
    if any(r.pairing == "invalid_provenance" for r in reports):
        lines += [
            "",
            "INVALID PROVENANCE: this row is a rebookend whose named source is itself a "
            "rebookend (a chain or a cycle). A bookend's before-side never exists, so no "
            "assembled measurement can be built from it; rebookend the original run instead.",
        ]
    if any(r.pairing == "baseline_missing" for r in reports):
        lines += [
            "",
            "BASELINE MISSING: this row is a rebookend whose baseline artifact (named in its "
            "manifest.rebookend.baseline_run_id) was not among the loaded results, so its "
            "after side could not be paired with the baseline's before side. Load both files "
            "together to assemble the measurement.",
        ]
    if any(not r.complete for r in reports):
        lines += [
            "",
            "* INCOMPLETE: this cell could not account for every held-out task it requested (or "
            "its results file carries no such accounting at all), so its numbers may be over a "
            "subset. The ids it lost are named in its results file, under "
            "heldout.missing_task_ids, and that file is a .incomplete.json.",
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
    docs = assemble(docs)
    reports = [report_cell(d, resamples=args.resamples, seed=args.seed) for d in docs]
    reports.sort(key=lambda r: (r.env, r.harness, r.model, r.run_id))
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


__all__ = ["CellReport", "assemble", "paired_bootstrap", "render_table", "report_cell"]
