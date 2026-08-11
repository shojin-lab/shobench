"""Reading a phase's durable record back, and writing the cell's results JSON.

The server keeps the score, so the record is what shogym wrote to the phase's provenance
directory and nothing the agent said about itself. This module reads those rows through
shogym's own readers, pairs the two eval phases by task index, and writes one JSON per cell in
the shape the reporting script and the results page consume.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "shobench.results/1"

# Closures shogym records for a task that reached a verdict. Anything else carries no score and
# is reported as unscored rather than averaged as a zero.
SCORED_CLOSURES = frozenset({"sealed", "aborted", "drained"})


@dataclass
class TaskResult:
    """One task's outcome in one phase, flattened out of a shogym ``ResultRow``."""

    seq: int
    position: int
    task_idx: int
    closure: str
    reward: float | None
    success: bool | None
    diagnostic: str | None = None
    observed: list[dict[str, Any]] = field(default_factory=list)

    @property
    def scored(self) -> bool:
        return self.closure in SCORED_CLOSURES and (
            self.reward is not None or self.success is not None
        )


def read_phase(prov_dir: Path) -> list[TaskResult]:
    """Every task the phase recorded, in seal order, including the ones that never sealed.

    ``reconcile`` is what turns a dispense with no result into a visible ``broker_abort`` row
    rather than a silent absence, which matters because a cell that lost tasks to a crash must
    not read as a cell that served fewer.
    """
    from shogym.serve import read_results, reconcile

    rows = sorted([*read_results(prov_dir), *reconcile(prov_dir)], key=lambda row: row.seq)
    out: list[TaskResult] = []
    for row in rows:
        score = row.score
        out.append(
            TaskResult(
                seq=row.seq,
                position=row.position,
                task_idx=row.task_idx,
                closure=str(row.closure),
                reward=None if score is None else score.reward,
                success=None if score is None else score.success,
                diagnostic=row.diagnostic,
                observed=[dict(item) for item in (row.observed or ())],
            )
        )
    return out


def collapse_replays(rows: list[TaskResult]) -> list[TaskResult]:
    """One row per queue position: the settled outcome when there is one, else the abandonment.

    A rollout resumed after a usage limit redispenses the position its suspension left in flight,
    so shogym's record holds two rows for that one position: the reconciled ``broker_abort`` for
    the abandoned dispense, which it keeps as provenance, and the replay's real closure. Counting
    both would turn one queue position into two headline tasks and inflate every closure tally, so
    a resumed cell would publish a different measurement than the uninterrupted run it is meant to
    match. Collapsing by position is what makes the two agree: the settled closure supersedes the
    abandonment it replaced, and a position that only ever aborted keeps its ``broker_abort`` as
    its single row. The raw rows, abandonment included, stay in the published ``tasks`` list for
    audit; only the counts derived here are per position.

    ``rows`` arrive in seq order (see :func:`read_phase`), so a later real closure is seen after
    the earlier ``broker_abort`` it supersedes and replaces it, and a second abandonment of a
    still-unfinished position does not displace the first.
    """
    chosen: dict[int, TaskResult] = {}
    for row in rows:
        current = chosen.get(row.position)
        if current is None or (
            current.closure == "broker_abort" and row.closure != "broker_abort"
        ):
            chosen[row.position] = row
    return [chosen[position] for position in sorted(chosen)]


def rollout_summary(rows: list[TaskResult]) -> dict[str, Any]:
    """The stopping metrics, which are the charter's own question.

    ``tasks_attempted`` counts queue positions the stream dispensed and sealed, so a rollout that
    stopped early reads as a smaller number rather than as a truncated one. The count is per
    position rather than per row (see :func:`collapse_replays`), so a position a resume replayed
    is one attempt here even though the record keeps both its abandonment and its replay. The
    runner supplies the stop classification separately, because only it saw how each harness leg
    ended.
    """
    collapsed = collapse_replays(rows)
    scored = [r for r in collapsed if r.scored]
    rewards = [r.reward for r in scored if r.reward is not None]
    successes = [r.success for r in scored if r.success is not None]
    return {
        "tasks_attempted": len(collapsed),
        "tasks_scored": len(scored),
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
        "full_solve_rate": (sum(successes) / len(successes)) if successes else None,
        "closures": _closure_counts(collapsed),
    }


def dispensed_positions(prov_dir: Path) -> int:
    """How many distinct queue positions this rollout ever dispensed, across every process.

    A rollout resumed after a usage limit redispenses the position its suspension abandoned, so
    that position is written to ``dispenses.jsonl`` twice: once for the abandoned attempt, once
    for the replay. Summing each process's own dispense counter double-counts it and reports a
    two-position pool as three dispensed. Counting distinct positions instead makes a resumed run
    report the total an uninterrupted one does, one dispense per position it reached.
    """
    from shogym.serve import read_dispenses

    return len({int(record["position"]) for record in read_dispenses(prov_dir)})


def _closure_counts(rows: list[TaskResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.closure] = counts.get(row.closure, 0) + 1
    return dict(sorted(counts.items()))


def eval_summary(rows: list[TaskResult]) -> dict[str, Any]:
    scored = [r for r in rows if r.scored]
    rewards = [r.reward for r in scored if r.reward is not None]
    successes = [r.success for r in scored if r.success is not None]
    return {
        "n_requested": len(rows),
        "n_scored": len(scored),
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
        "full_solve_rate": (sum(successes) / len(successes)) if successes else None,
        "closures": _closure_counts(rows),
    }


def pair_evals(
    before: list[TaskResult], after: list[TaskResult]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair the two eval phases by task index.

    Both phases serve the same held-out ids in the same order exactly once, so the index is a
    key. A task scored in only one phase cannot contribute a paired delta, so it is returned in
    the unpaired list instead of being dropped, and the report says how many there were.
    """
    by_idx_before = {r.task_idx: r for r in before}
    by_idx_after = {r.task_idx: r for r in after}
    paired: list[dict[str, Any]] = []
    unpaired: list[dict[str, Any]] = []
    for idx in sorted(set(by_idx_before) | set(by_idx_after)):
        lhs = by_idx_before.get(idx)
        rhs = by_idx_after.get(idx)
        if lhs is None or rhs is None or not lhs.scored or not rhs.scored:
            unpaired.append(
                {
                    "task_idx": idx,
                    "before": None if lhs is None else asdict(lhs),
                    "after": None if rhs is None else asdict(rhs),
                }
            )
            continue
        delta = None
        if lhs.reward is not None and rhs.reward is not None:
            delta = rhs.reward - lhs.reward
        paired.append(
            {
                "task_idx": idx,
                "reward_before": lhs.reward,
                "reward_after": rhs.reward,
                "reward_delta": delta,
                "success_before": lhs.success,
                "success_after": rhs.success,
            }
        )
    return paired, unpaired


def write_results(
    path: Path,
    *,
    manifest: dict[str, Any],
    phases: dict[str, list[TaskResult]],
    stopping: dict[str, Any],
    egress: dict[str, Any] | None = None,
) -> Path:
    """Write the cell's results JSON: the manifest, the per-task scores, and the summaries."""
    before = phases.get("eval_before", [])
    after = phases.get("eval_after", [])
    paired, unpaired = pair_evals(before, after)
    body = {
        "schema": SCHEMA,
        "manifest": manifest,
        "eval_before": {
            "summary": eval_summary(before),
            "tasks": [asdict(r) for r in before],
        },
        "eval_after": {
            "summary": eval_summary(after),
            "tasks": [asdict(r) for r in after],
        },
        "rollout": {
            "summary": rollout_summary(phases.get("rollout", [])),
            "stopping": stopping,
            "tasks": [asdict(r) for r in phases.get("rollout", [])],
        },
        "paired": paired,
        "unpaired": unpaired,
        "egress": egress or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "SCHEMA",
    "SCORED_CLOSURES",
    "TaskResult",
    "collapse_replays",
    "dispensed_positions",
    "eval_summary",
    "pair_evals",
    "read_phase",
    "rollout_summary",
    "write_results",
]
