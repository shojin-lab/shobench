"""How much of the stream's concurrency an agent actually used.

The task stream leases up to ``max_in_flight`` tasks at once, so an agent may pull the next task
while an earlier one is still open, and the stream force-ends the oldest lease when a pull
arrives at capacity. Whether an agent takes that choice is recorded as a number nowhere, so it
is reconstructed here from the files the stream already writes.

    shobench concurrency runs/<run-dir> [...] --format table

**A result row carries no sealing time.** ``dispenses.jsonl`` stamps ``dispensed_at`` on every
task it hands out, but ``results.jsonl`` records only that a task sealed, and shobench registers
no provenance extension that would stamp the moment. The harness traces do not supply it
either: of the three, claude_code's stream-json is the only one that timestamps its events at
all, and a seal happens inside a tool call rather than as one. So the exact open interval of a
lease is not in the record, and nothing here invents one.

Two things the record does settle exactly: when each lease opened, and the order the leases
closed in, because ``results.jsonl`` is appended one row per seal in seal order. Every history
the record admits therefore closes each lease after its own dispense and in the file's order,
and placing every close as early as those two facts allow is the history with the least
overlap. That floor is what this module reports. ``max_open`` and ``mean_open_at_pull`` are
lower bounds, and ``strictly_sequential`` is ``False`` only where the floor rules sequential
out, ``True`` only where a stream could not have held two leases at all, and ``None``, rendered
as "unknown", everywhere else. An agent that pulled a second task and then sealed the two in
the order it pulled them reads as unknown, never as sequential.

The time-weighted mean the same question would want is not among the numbers, for the same
reason: weighting by time needs the interval a lease was open, and the close half of that
interval is what the record does not hold. ``mean_open_at_pull`` stands in for it, over the
instants the record does stamp: how many tasks the agent already had open each time it pulled.

The tasks a pull force-ended are bracketed rather than counted, because that ending and the one
the stream's close performs are written under a single closure name. A run that finishes holding
a full registry mixes the two in its trailing rows, so ``displaced`` is reported as the range
the record and the ceiling force and never as an exact number the file cannot support.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DISPENSES_FILE = "dispenses.jsonl"
MANIFEST_FILE = "manifest.json"
LEGS_FILE = "legs.json"

# The closure a task ends in when the stream ended it at a pull, and also the one it ends in
# when the stream closed. shogym writes one name for both (see TaskStream.get_task and
# TaskStream.aclose), so the two are told apart by where the row sits, not by what it says.
FORCED_CLOSURE = "drained"

# The closures only the agent's own call produces. `aclose` claims every unsettled task in the
# critical section that closes the stream, so no task can reach one of these after the closing
# drain has begun: a row carrying one proves every forced row above it was ended at a pull.
EARNED_CLOSURES = frozenset({"sealed", "aborted"})


@dataclass(frozen=True)
class StreamConcurrency:
    """One provenance directory's concurrency, floored.

    ``dispensed`` counts leases rather than queue positions, which is what a lease count has to
    be: a resumed run redispenses the position its suspension abandoned, and those are two
    leases that were open at different times.

    ``never_sealed`` counts leases the record holds a dispense for and no result row: the stream
    died holding them. Their close is bounded by whichever of two records reaches them, the leg
    that dispensed them ending or their queue position being replayed, and where neither does
    they stay open for the rest of the timeline.

    ``displaced_at_least`` and ``displaced_at_most`` bracket the tasks a pull force-ended,
    because displacement and the closing drain are one closure name (see :func:`_drain_split`).
    """

    prov_dir: str
    dispensed: int
    sealed: int
    max_open: int
    mean_open_at_pull: float
    drained: int
    displaced_at_least: int
    displaced_at_most: int
    never_sealed: int
    strictly_sequential: bool | None


@dataclass(frozen=True)
class PhaseConcurrency:
    """One phase of one run, rolled up over the streams it ran.

    Leases only compete inside a stream, because ``max_in_flight`` bounds one stream's registry,
    so ``max_open`` is the largest any single stream reached rather than a sum. An eval phase
    runs one session per task and gives each its own directory, so its streams hold one lease
    each whatever the ceiling says.
    """

    run_id: str
    phase: str
    streams: int
    dispensed: int
    max_in_flight: int | None
    max_open: int
    mean_open_at_pull: float
    drained: int
    displaced_at_least: int
    displaced_at_most: int
    never_sealed: int
    strictly_sequential: bool | None

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "streams": self.streams,
            "dispensed": self.dispensed,
            "max_in_flight": self.max_in_flight,
            "seal_times": "unrecorded",
            "max_open": self.max_open,
            "max_open_is_floor": True,
            "mean_open_at_pull": self.mean_open_at_pull,
            "drained": self.drained,
            "displaced_at_least": self.displaced_at_least,
            "displaced_at_most": self.displaced_at_most,
            "never_sealed": self.never_sealed,
            "strictly_sequential": self.strictly_sequential,
        }


def _drain_split(closures: Sequence[str], ceiling: int | None) -> tuple[int, int, int]:
    """Bracket the forced closures a pull caused, as ``(total, at_least, at_most)``.

    Displacement and the closing drain are one closure name, and the record does not separate
    them. What it does settle is a boundary and a ceiling. A row carrying a closure the agent
    earned proves the stream was still serving when it landed, so every forced row above the
    last of those was ended by a pull. Below it the two causes are mixed, and only the registry
    bounds them: the live set never exceeds ``max_in_flight``, so at most that many of the
    trailing forced rows can be the closing drain and the rest were pulls.

    A run at capacity that ends holding a full registry is exactly where the trailing block
    stops being readable as the close alone, and it is the common shape rather than a rare one.
    """
    boundary = 0
    for index in range(len(closures) - 1, -1, -1):
        if closures[index] in EARNED_CLOSURES:
            boundary = index + 1
            break
    above = sum(1 for closure in closures[:boundary] if closure == FORCED_CLOSURE)
    trailing = sum(1 for closure in closures[boundary:] if closure == FORCED_CLOSURE)
    at_least = above + (max(0, trailing - ceiling) if ceiling is not None else 0)
    # A pull and the close are the only two producers of this closure, so every row carrying it
    # is a candidate and the whole count is the upper bound.
    return above + trailing, at_least, above + trailing


def _floor_timeline(
    opens: Sequence[tuple[str, float]],
    seal_order: Sequence[str],
    unsealed_close: dict[str, float],
) -> list[tuple[float, int]]:
    """The history with the least overlap the record admits, as ``(time, delta)`` events.

    A close may be no earlier than its own open and no earlier than the close before it in
    ``results.jsonl``, and taking that lower bound for every close is what makes the counts read
    off this timeline floors rather than estimates. A close is emitted as soon as both bounds
    allow and never past the next dispense, so a lease that could have closed before the next
    pull does.

    A lease with no result row closes at the bound ``unsealed_close`` carries for it, and it is
    merged in AHEAD of everything sharing its instant rather than behind. The bound is the moment
    something else proves the lease was already over, and a replay of its queue position is
    exactly such a moment stamped at the instant of another dispense, so trailing that dispense
    would invent the one overlap the bound rules out. Its own open is the one event it may never
    precede.
    """
    opened = dict(opens)
    order = [lease for lease in seal_order if lease in opened]
    events: list[tuple[float, int]] = []
    opened_at_index: dict[str, int] = {}
    live: set[str] = set()
    i = j = 0
    previous = float("-inf")
    while True:
        closing = order[j] if j < len(order) else None
        if closing is not None and closing in live:
            at = max(opened[closing], previous)
            if i >= len(opens) or at <= opens[i][1]:
                events.append((at, -1))
                live.discard(closing)
                previous = at
                j += 1
                continue
        if i < len(opens):
            lease, at = opens[i]
            opened_at_index[lease] = len(events)
            events.append((at, 1))
            live.add(lease)
            i += 1
            continue
        if closing is not None:
            # Every dispense has been read and this lease is not open, so no history places this
            # close anywhere: it is a second row for a lease already sealed.
            j += 1
            continue
        break

    bounded = sorted(
        (max(unsealed_close[lease], opened[lease]), opened_at_index[lease])
        for lease in live
        if lease in unsealed_close
    )
    merged: list[tuple[float, int]] = []
    k = 0
    for index, event in enumerate(events):
        while k < len(bounded) and bounded[k][0] <= event[0] and bounded[k][1] < index:
            merged.append((bounded[k][0], -1))
            k += 1
        merged.append(event)
    merged += [(at, -1) for at, _ in bounded[k:]]
    return merged


def _leg_windows(run_dir: Path, phase: str) -> list[tuple[float, float]]:
    """When each of this phase's harness sessions ran, from the run's own leg record."""
    path = run_dir / LEGS_FILE
    if not path.exists():
        return []
    try:
        legs = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    windows = []
    for leg in legs if isinstance(legs, list) else []:
        started, ended = leg.get("started_at"), leg.get("ended_at")
        if leg.get("phase") == phase and started is not None and ended is not None:
            windows.append((float(started), float(ended)))
    return sorted(windows)


def _abandoned_closes(
    records: Sequence[dict[str, Any]],
    sealed: set[str],
    leg_windows: Sequence[tuple[float, float]],
) -> dict[str, float]:
    """When each lease with no result row can be shown to have ended.

    Two records reach it, and the earlier of the two is the one to take. Its own session ending
    is one: a lease cannot outlive the process holding it. Its queue position being dispensed
    again is the other, and it is the stronger of the two because it needs no leg file at all.
    Only a reopened stream replays a position, ``resume`` skipping the ones that already have a
    settled row, and a reopened stream begins with an empty registry, so the earlier lease was
    over before the replay was handed out.

    A lease neither record reaches keeps no close here, which leaves it open for the rest of the
    timeline rather than closed on a guess.
    """
    closes: dict[str, float] = {}
    replayed_at: dict[int, float] = {}
    for record in reversed(records):
        position = int(record["position"])
        lease, at = str(record["lease"]), float(record["dispensed_at"])
        if lease not in sealed:
            bounds = [ended for started, ended in leg_windows if started <= at <= ended]
            if position in replayed_at:
                bounds.append(replayed_at[position])
            if bounds:
                closes[lease] = min(bounds)
        replayed_at[position] = at
    return closes


def stream_concurrency(
    prov_dir: Path,
    *,
    label: str,
    max_in_flight: int | None = None,
    leg_windows: Sequence[tuple[float, float]] = (),
) -> StreamConcurrency:
    """Read one provenance directory and floor its concurrency. Reads only."""
    from shogym.serve import read_dispenses, read_results

    records = read_dispenses(prov_dir)
    rows = read_results(prov_dir)
    opens = [(str(record["lease"]), float(record["dispensed_at"])) for record in records]
    sealed = {row.lease for row in rows}

    events = _floor_timeline(
        opens,
        [row.lease for row in rows],
        _abandoned_closes(records, sealed, leg_windows),
    )
    open_now = 0
    max_open = 0
    at_pull: list[int] = []
    for _, delta in events:
        if delta > 0:
            at_pull.append(open_now)
        open_now += delta
        max_open = max(max_open, open_now)

    drained, at_least, at_most = _drain_split([row.closure for row in rows], max_in_flight)
    # One slot serves one lease, and one queue position is dispensed once per stream, so a
    # directory holding replays of a single position holds leases from streams that never
    # overlapped. Either way no second lease existed for a first to be open beside.
    positions = {int(record["position"]) for record in records}
    sequential: bool | None = None
    if max_open > 1:
        sequential = False
    elif max_in_flight == 1 or len(positions) <= 1:
        sequential = True
    return StreamConcurrency(
        prov_dir=label,
        dispensed=len(opens),
        sealed=len(rows),
        max_open=max_open,
        mean_open_at_pull=(sum(at_pull) / len(at_pull)) if at_pull else 0.0,
        drained=drained,
        displaced_at_least=at_least,
        displaced_at_most=at_most,
        never_sealed=sum(1 for lease, _ in opens if lease not in sealed),
        strictly_sequential=sequential,
    )


def _roll_up(sequential: Sequence[bool | None]) -> bool | None:
    if any(value is False for value in sequential):
        return False
    return True if all(value is True for value in sequential) else None


def phase_streams(run_dir: Path) -> list[tuple[str, list[Path]]]:
    """Each phase under a run directory and the provenance directories it ran.

    A rollout keeps its records in the phase directory itself; an eval phase gives every task
    its own, because it runs one session per task.
    """
    phases = []
    for entry in sorted(run_dir.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / DISPENSES_FILE).exists():
            phases.append((entry.name, [entry]))
            continue
        nested = [
            child
            for child in sorted(entry.iterdir())
            if child.is_dir() and (child / DISPENSES_FILE).exists()
        ]
        if nested:
            phases.append((entry.name, nested))
    return phases


def run_concurrency(run_dir: Path) -> list[PhaseConcurrency]:
    """Every phase of one run, floored. Reads only; nothing under ``run_dir`` is written."""
    manifest: dict[str, Any] = {}
    manifest_path = run_dir / MANIFEST_FILE
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ceiling = manifest.get("cell", {}).get("max_in_flight")
    run_id = str(manifest.get("run_id") or run_dir.name)

    out = []
    for phase, prov_dirs in phase_streams(run_dir):
        windows = _leg_windows(run_dir, phase)
        streams = [
            stream_concurrency(
                prov_dir,
                label=str(prov_dir.relative_to(run_dir)),
                max_in_flight=ceiling,
                leg_windows=windows,
            )
            for prov_dir in prov_dirs
        ]
        pulls = sum(stream.dispensed for stream in streams)
        weighted = sum(stream.mean_open_at_pull * stream.dispensed for stream in streams)
        out.append(
            PhaseConcurrency(
                run_id=run_id,
                phase=phase,
                streams=len(streams),
                dispensed=pulls,
                max_in_flight=ceiling,
                max_open=max(stream.max_open for stream in streams),
                mean_open_at_pull=(weighted / pulls) if pulls else 0.0,
                drained=sum(stream.drained for stream in streams),
                displaced_at_least=sum(stream.displaced_at_least for stream in streams),
                displaced_at_most=sum(stream.displaced_at_most for stream in streams),
                never_sealed=sum(stream.never_sealed for stream in streams),
                strictly_sequential=_roll_up([s.strictly_sequential for s in streams]),
            )
        )
    return out


def find_runs(target: Path) -> list[Path]:
    """A run directory, or the run directories one level under a directory of them."""
    if (target / MANIFEST_FILE).exists():
        return [target]
    return [
        child
        for child in sorted(target.iterdir())
        if child.is_dir() and (child / MANIFEST_FILE).exists()
    ]


def _sequential(value: bool | None) -> str:
    return "unknown" if value is None else ("yes" if value else "no")


def _bracket(at_least: int, at_most: int) -> str:
    return str(at_least) if at_least == at_most else f"{at_least}..{at_most}"


def render_table(rows: Sequence[PhaseConcurrency]) -> str:
    header = (
        "run",
        "phase",
        "streams",
        "tasks",
        "ceiling",
        "max open",
        "open@pull",
        "drained",
        "displaced",
        "unsealed",
        "sequential",
    )
    body = [
        (
            row.run_id,
            row.phase,
            str(row.streams),
            str(row.dispensed),
            "?" if row.max_in_flight is None else str(row.max_in_flight),
            f">={row.max_open}",
            f">={row.mean_open_at_pull:.2f}",
            str(row.drained),
            _bracket(row.displaced_at_least, row.displaced_at_most),
            str(row.never_sealed),
            _sequential(row.strictly_sequential),
        )
        for row in rows
    ]
    widths = [max(len(cell) for cell in column) for column in zip(header, *body, strict=True)]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(header, widths, strict=True)),
        "  ".join("-" * width for width in widths),
    ]
    lines += [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in body
    ]
    lines += [
        "",
        "No result row carries the time its task sealed, and no harness trace supplies one for "
        "every harness, so 'max open' and 'open@pull' are floors: the least concurrency any "
        "history the record admits could have had. 'sequential' says no where that floor rules "
        "sequential out and unknown where the record cannot settle it; only a stream that could "
        "not have held two leases reads yes.",
        "",
        "'drained' counts the tasks the stream ended rather than the agent. A pull arriving at "
        "capacity and the closing drain are recorded under that one closure, so 'displaced' is "
        "a range wherever the record cannot separate them: its low end is what the trailing "
        "rows and the ceiling force, its high end is every drained row. 'unsealed' counts "
        "dispensed tasks with no result row at all.",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args(argv)

    run_dirs = [run_dir for target in args.runs for run_dir in find_runs(target)]
    if not run_dirs:
        print(f"no run directory found under {', '.join(str(t) for t in args.runs)}")
        return 1
    rows = [row for run_dir in run_dirs for row in run_concurrency(run_dir)]
    if args.format == "json":
        print(json.dumps({"phases": [row.to_json() for row in rows]}, indent=2))
    else:
        print(render_table(rows))
    return 0


__all__ = [
    "PhaseConcurrency",
    "StreamConcurrency",
    "find_runs",
    "phase_streams",
    "render_table",
    "run_concurrency",
    "stream_concurrency",
]
