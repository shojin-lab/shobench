"""Reading a phase's durable record back, and writing the cell's results JSON.

The server keeps the score, so the record is what shogym wrote to the phase's provenance
directory and nothing the agent said about itself. This module reads those rows through
shogym's own readers, pairs the two eval phases by task index, and writes one JSON per cell in
the shape the reporting script and the results page consume.

**Every published number is counted against the committed held-out set, never against the rows
that happened to arrive.** A held-out task can produce no row at all: the runner catches the
exception and writes a breadcrumb, or a harness exits before it ever calls ``get_task`` and
nothing raises anywhere. Counted by rows, that id is not a failure but an absence, and an
absence changes what the denominator means: an id missing from both phases used to disappear
from the measurement entirely, leaving a paired mean and a bootstrap over a silently selected
subset while the manifest went on saying 120 or 40 tasks were requested. So the committed ids
are an input here, an id that produced nothing is carried as an explicit unscored row, and a
result that cannot account for all of them is published under a name no reader can mistake for
a finished measurement.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "shobench.results/1"

# Closures shogym records for a task that reached a verdict. Anything else carries no score and
# is reported as unscored rather than averaged as a zero.
SCORED_CLOSURES = frozenset({"sealed", "aborted", "drained"})

# The closure this runner gives a requested held-out id that produced no row of its own. It is
# not one of shogym's, because shogym never saw the task: a harness that died before calling
# ``get_task`` leaves the broker nothing to record. Naming the absence is the whole point, since
# the alternative is an id that is simply not in the published file.
MISSING_CLOSURE = "missing"

# What a results file is called when it cannot account for every committed held-out id. A
# measurement with a hole in it is not the cell's result, and a reader reaching for
# ``<cell>.json`` must not find one silently standing in for it.
INCOMPLETE_SUFFIX = ".incomplete.json"


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
    # The regime the stream served this row under, verbatim from shogym's record: row-level
    # evidence of which feedback arm produced the number, independent of the manifest's claim,
    # so a manifest/row disagreement stays auditable. None only for filled rows no stream served.
    feedback_regime: str | None = None

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
                feedback_regime=(
                    None if row.feedback_regime is None else str(row.feedback_regime)
                ),
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


def missing_row(task_idx: int, *, diagnostic: str) -> TaskResult:
    """The row a requested held-out id gets when it produced none of its own.

    Unscored by construction, so it can never be averaged as a zero, and carrying the reason in
    the diagnostic so a reader of the published file sees what happened to that id rather than
    inferring something from a gap in the task list.
    """
    return TaskResult(
        # Negative, because ``seq`` and ``position`` are the broker's own numbering and this row
        # never reached the broker. A reader can tell a synthesized row from a recorded one
        # without knowing the closure name.
        seq=-1,
        position=-1,
        task_idx=task_idx,
        closure=MISSING_CLOSURE,
        reward=None,
        success=None,
        diagnostic=diagnostic,
    )


def heldout_accounting(
    rows: list[TaskResult], *, task_ids: Sequence[int]
) -> dict[str, Any]:
    """Whether one eval phase holds exactly one outcome for every committed held-out id.

    Three ways it does not, and each is a different lie if it goes unsaid. An id with no row is
    a task that vanished from the measurement. An id with several is an outcome nobody can read,
    since which of them the pairing picks up is an implementation detail. An id nobody requested
    is a row from some other split, which means this file is not the experiment its manifest
    describes.

    A synthesized ``missing`` row does not account for anything: it is this runner saying an id
    produced nothing, not a record of the id being measured. Ignoring it here is what makes the
    verdict the same whether it is asked before or after :func:`fill_missing` has run, which
    matters because both the reader and the writer fill.
    """
    counts = Counter(row.task_idx for row in rows if row.closure != MISSING_CLOSURE)
    missing = [idx for idx in task_ids if counts[idx] == 0]
    ambiguous = [idx for idx in task_ids if counts[idx] > 1]
    unrequested = sorted(set(counts) - set(task_ids))
    return {
        "complete": not (missing or ambiguous or unrequested),
        "missing_task_ids": missing,
        "ambiguous_task_ids": ambiguous,
        "unrequested_task_ids": unrequested,
    }


def fill_missing(
    rows: list[TaskResult],
    *,
    task_ids: Sequence[int],
    diagnostic: Callable[[int], str] | None = None,
) -> list[TaskResult]:
    """Every committed held-out id, in id order, with a row for the ones that produced none.

    This is the whole of the fix for a task that vanished: an id the record has nothing for is
    still an id the cell requested, so it is carried as an unscored ``missing`` row rather than
    left out. Nothing that did arrive is touched, including a row for an id nobody requested,
    which stays visible instead of being quietly dropped by the same pass.
    """
    present = {row.task_idx for row in rows}
    absent = [idx for idx in task_ids if idx not in present]
    if not absent:
        return list(rows)
    say = diagnostic or (lambda idx: "this held-out id produced no row")
    return sorted(
        [*rows, *(missing_row(idx, diagnostic=say(idx)) for idx in absent)],
        key=lambda row: row.task_idx,
    )


def eval_summary(rows: list[TaskResult], *, task_ids: Sequence[int]) -> dict[str, Any]:
    """One eval phase's numbers, counted against the ids the split committed to.

    ``n_requested`` is the size of that committed set and never the number of rows that happened
    to arrive. The two used to be the same expression, which is exactly how a phase that lost a
    task published as a smaller phase that lost nothing.
    """
    scored = [r for r in rows if r.scored]
    rewards = [r.reward for r in scored if r.reward is not None]
    successes = [r.success for r in scored if r.success is not None]
    accounting = heldout_accounting(rows, task_ids=task_ids)
    return {
        "n_requested": len(task_ids),
        "n_scored": len(scored),
        "n_missing": len(accounting["missing_task_ids"]),
        "complete": accounting["complete"],
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
        "full_solve_rate": (sum(successes) / len(successes)) if successes else None,
        "closures": _closure_counts(rows),
    }


def pair_evals(
    before: list[TaskResult], after: list[TaskResult], *, task_ids: Sequence[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair the two eval phases by task index, over the committed held-out set.

    Both phases serve the same held-out ids in the same order exactly once, so the index is a
    key. A task scored in only one phase cannot contribute a paired delta, so it is returned in
    the unpaired list instead of being dropped, and the report says how many there were.

    The ids are walked from the committed set rather than from the rows, so an id absent from
    both phases lands in the unpaired list as the empty pair it is. Walking the union of what
    arrived is what made that id vanish instead: it was in neither dict, so it was in neither
    output, and ``n_unpaired`` stayed at zero while the pairing quietly shrank.
    """
    by_idx_before = {r.task_idx: r for r in before}
    by_idx_after = {r.task_idx: r for r in after}
    paired: list[dict[str, Any]] = []
    unpaired: list[dict[str, Any]] = []
    for idx in sorted(set(task_ids) | set(by_idx_before) | set(by_idx_after)):
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


def _creation_mode(directory: Path) -> int:
    """The mode an ordinarily created file receives in this directory: 0666 under the umask.

    Read by creating one rather than by flipping ``os.umask``, because the flip is a
    process-global write and two concurrent publishers interleaving it could corrupt the
    umask for everything else in the process; publication is exactly the boundary that runs
    concurrently. The probe is exclusive-created, read, and removed, all inside the leaf's
    own directory.
    """
    probe = directory / f".mode-probe.{uuid.uuid4().hex}.tmp"
    fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    try:
        return stat.S_IMODE(os.fstat(fd).st_mode)
    finally:
        os.close(fd)
        probe.unlink(missing_ok=True)


def write_results(
    path: Path,
    *,
    manifest: dict[str, Any],
    phases: dict[str, list[TaskResult]],
    stopping: dict[str, Any],
    heldout_ids: Sequence[int],
    egress: dict[str, Any] | None = None,
    redact: Callable[[Any], Any] | None = None,
    before_source_run_id: str | None = None,
) -> Path:
    """Write the cell's results JSON: the manifest, the per-task scores, and the summaries.

    ``heldout_ids`` is the committed held-out set, and it is required rather than derived because
    it is the denominator of everything published here. Both eval phases are filled out against
    it, so an id that produced no row carries an explicit unscored one, the pairing walks it
    rather than the rows, and the summaries count requested tasks rather than arrivals.

    A result that still cannot account for every id is written under
    ``<cell>.incomplete.json``: the numbers stay readable and the missing ids are named in
    ``heldout``, but nothing reaching for the cell's result finds a partial measurement standing
    in for it. A cell publishes one artifact, so whichever name this run did not take is removed
    rather than left behind for a reader to find and believe.

    ``redact`` is applied to the whole assembled body immediately before it is serialized. It
    takes the body rather than each part because this file is the one artifact assembled from
    every other: stop evidence carrying a stderr tail, per-task diagnostics carrying whatever a
    harness said, and the manifest's probe output. Redacting once at the end covers all of them
    and cannot be forgotten for a field added later.
    """
    cell_manifest = manifest.get("cell")
    if isinstance(cell_manifest, dict):
        # A manifest written before an axis existed could only have run that axis's one
        # pre-axis posture, so absence is backfilled explicitly rather than left for every
        # reader to infer: never for the feedback arm, cold for the eval context.
        backfill = {}
        if "rollout_feedback" not in cell_manifest:
            backfill["rollout_feedback"] = "never"
        if "eval_context" not in cell_manifest:
            backfill["eval_context"] = "cold"
        if backfill:
            manifest = {**manifest, "cell": {**cell_manifest, **backfill}}
    ids = [int(task_id) for task_id in heldout_ids]
    # Filled here as well as by the reader that produced the rows, so a caller that assembles a
    # phase some other way still cannot publish a hole. Filling twice is free: the second pass
    # finds nothing absent.
    before = fill_missing(phases.get("eval_before", []), task_ids=ids)
    after = fill_missing(phases.get("eval_after", []), task_ids=ids)
    accounting = {
        phase: heldout_accounting(rows, task_ids=ids)
        for phase, rows in (("eval_before", before), ("eval_after", after))
    }
    complete = all(entry["complete"] for entry in accounting.values())
    paired, unpaired = pair_evals(before, after, task_ids=ids)
    body = {
        "schema": SCHEMA,
        "manifest": manifest,
        # What was asked for, and whether the file below can account for it. First of the
        # measurement fields because it is the one that says whether to trust the rest.
        "heldout": {
            "n_requested": len(ids),
            "task_ids": ids,
            "complete": complete,
            **accounting,
        },
        "eval_before": {
            "summary": eval_summary(before, task_ids=ids),
            "tasks": [asdict(r) for r in before],
            # Present only when the rows were measured by ANOTHER run and carried in (a
            # rebookend publishing its baseline's before block): the label is what keeps a
            # reader from taking them for rows this run measured.
            **({"source_run_id": before_source_run_id} if before_source_run_id else {}),
        },
        "eval_after": {
            "summary": eval_summary(after, task_ids=ids),
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
    if redact is not None:
        body = redact(body)
    path = Path(path)
    finished, partial = path, path.with_name(path.stem + INCOMPLETE_SUFFIX)
    path = finished if complete else partial
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the leaf and swapped in atomically, so publication REPLACES whatever
    # holds the name: a stale file, a hard link, or a symlink someone left at the
    # deterministic leaf. A plain write follows an existing symlink, which turned a link
    # planted at the leaf name into a write through the results directory into wherever it
    # pointed (an archived source run, in review); ``os.replace`` swaps the directory entry
    # itself and follows nothing. Owned here rather than by each caller, so every publisher
    # (a fresh cell, a resume, a rerun, a rebookend) inherits the same guarantee.
    #
    # The scratch entry is per CALL, minted exclusively by mkstemp, in the same directory so
    # the swap can never cross devices. A per-process name was not enough: two publishers in
    # one process shared it, and the pre-unlink one needed to reuse the name let publisher B
    # unlink A's scratch mid-write, so A swapped B's half-written inode into the leaf and
    # reported success (reproduced). Exclusive creation of a fresh name has no such window,
    # and the ``finally`` keeps a failed publication from leaving the scratch behind.
    scratch_fd, scratch_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    scratch = Path(scratch_name)
    try:
        with os.fdopen(scratch_fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(body, indent=2) + "\n")
        # The swap carries the scratch's mode onto the leaf, and mkstemp minted it 0600: left
        # alone, a fresh artifact landed owner-only and a republish DOWNGRADED an existing
        # 0644 or 0664 result to 0600. So the scratch takes the destination's intended mode
        # first: a regular file already at the leaf keeps its mode (an operator who opened a
        # result up, or locked one down, keeps that choice across republication), and
        # anything else gets what an ordinary creation would have gotten here, 0666 under the
        # process umask. lstat, because the entry being replaced can be a planted symlink and
        # its TARGET's mode means nothing for the regular file about to take the name.
        try:
            existing = os.lstat(path)
            mode = stat.S_IMODE(existing.st_mode) if stat.S_ISREG(existing.st_mode) else None
        except FileNotFoundError:
            mode = None
        os.chmod(scratch, _creation_mode(path.parent) if mode is None else mode)
        os.replace(scratch, path)
    finally:
        scratch.unlink(missing_ok=True)
    # A results directory holds one artifact per cell, and a rerun already replaces what the
    # last run wrote. Leaving the other name in place would leave two files describing one cell
    # from two runs, which is how a reader ends up reporting the one that reads better.
    (partial if complete else finished).unlink(missing_ok=True)
    return path


__all__ = [
    "INCOMPLETE_SUFFIX",
    "MISSING_CLOSURE",
    "SCHEMA",
    "SCORED_CLOSURES",
    "TaskResult",
    "collapse_replays",
    "dispensed_positions",
    "eval_summary",
    "fill_missing",
    "heldout_accounting",
    "missing_row",
    "pair_evals",
    "read_phase",
    "rollout_summary",
    "write_results",
]
