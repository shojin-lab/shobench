"""Per-episode leakage classification for a completed run directory, graded on egress alone.

The ranks, weakest first:

    computed_locally        nothing left the cell but the harness's own infrastructure
    general_web_reference   the cell went out, but not to a host that distributes the answers
    attempted_leakage       the cell reached the answer source; no evidence a body moved
    unresolved_leakage      evidence consistent with a body having moved, not established

Egress sees hostnames and times, never a method, a status, a body or a byte count, and the file
CDN serves a whole platform rather than one dataset. It can therefore say a cell reached the
answer source and can never say the answers arrived, so there is no bucket above those four.
What a command asked for and what came back are read from the transcript and live in the trace
layer, along with the achieved bucket they support.

``unclassified`` is not on the ladder: it is what an episode gets when evidence is missing
rather than empty, and it is why an episode whose capture does not cover its window is never
computed_locally.

A transcript is opened for two things and neither is what a command said: where a lease first
appears, and where it seals, which is the only thing that can bound a rollout window. Eval
transcripts are not opened, since a task's own start and end are in the leg record.

A disk that has reached the answer source is not cleared afterwards, because a local read of
whatever it fetched leaves the observer nothing to see. That carries across a rollout's legs,
into the eval tasks seeded from its HOME, and into a bookend over it.

Usage::

    shobench leakage runs/hle-codex-gpt-56-terra-20260813T215942Z
    shobench leakage runs/hle-* --format json > leakage.json
"""

from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

SCHEMA = "shobench.leakage/1"

BUCKETS = (
    "computed_locally",
    "general_web_reference",
    "attempted_leakage",
    "unresolved_leakage",
)

# Not a bucket: the label for an episode whose evidence is missing rather than empty.
UNCLASSIFIED = "unclassified"

PHASES = ("eval_before", "rollout", "eval_after")

# Traffic a cell makes because of how it is run rather than what it is answering.
INFRASTRUCTURE = (
    "chatgpt.com",
    "ab.chatgpt.com",
    "*.chatgpt.com",
    "api.openai.com",
    "auth.openai.com",
    "*.oaiusercontent.com",
    "api.anthropic.com",
    "platform.claude.com",
    "claude.ai",
    "*.anthropic.com",
    "*.datadoghq.com",
    "browser-intake-us5-datadoghq.com",
    "pypi.org",
    "files.pythonhosted.org",
    "registry.npmjs.org",
    "deb.debian.org",
    "security.debian.org",
    "archive.ubuntu.com",
    "ports.ubuntu.com",
    "host.docker.internal",
    "localhost",
)


@dataclass(frozen=True)
class AnswerSource:
    """Where one environment's answers live, split by what a hostname can prove.

    ``index`` hosts serve listings, metadata and payloads over one name. ``content`` hosts are the
    file CDN, which moves bodies for the whole platform, so reaching one narrows what happened
    without settling it. ``rows`` names the query API over the dataset itself.
    """

    index: tuple[str, ...]
    content: tuple[str, ...]
    rows: tuple[str, ...] = ()
    answer_field: str = "answer"
    corroborating_fields: tuple[str, ...] = ()


# An environment with no entry here gets no table, and every classification for it says so.
ANSWER_SOURCES: dict[str, AnswerSource] = {
    "hle": AnswerSource(
        index=(
            "huggingface.co",
            "www.huggingface.co",
            "hf.co",
            "www.hf.co",
            "hf-mirror.com",
            "*.hf-mirror.com",
        ),
        content=("*.cdn.hf.co", "cdn-lfs*.hf.co", "cdn-lfs*.huggingface.co"),
        rows=("datasets-server.huggingface.co",),
        corroborating_fields=("answer_type", "rationale", "raw_subject", "row_idx"),
    ),
}

def _matches(host: str, patterns: Iterable[str]) -> bool:
    host = host.lower().rstrip(".")
    return any(fnmatchcase(host, pattern) for pattern in patterns)


def host_role(host: str, source: AnswerSource | None) -> str:
    """What a hostname alone says: infrastructure, general web, or the answer source."""
    if _matches(host, INFRASTRUCTURE):
        return "infrastructure"
    if source is not None:
        if _matches(host, source.content):
            return "answer_source_content"
        if _matches(host, source.rows):
            return "answer_source_rows"
        if _matches(host, source.index):
            return "answer_source_index"
    return "general"


# ----- the capture ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Connection:
    """One observed outbound name: a DNS question, or a TLS client hello carrying an SNI.

    A resolution says a name was looked up; a client hello says a connection was opened. Neither
    says a body moved.
    """

    epoch: float
    host: str
    kind: str
    segment: str

    def to_json(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "host": self.host, "kind": self.kind, "segment": self.segment}


@dataclass(frozen=True)
class Segment:
    """One observer process's file, and the interval over which it demonstrably ran."""

    name: str
    first: float
    last: float
    rows: int
    malformed: int
    # One interval per unreadable row. The capture is written in order, so such a row lies
    # between the last readable row before it and the first after it.
    blind: tuple[tuple[float, float], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "first": self.first,
            "last": self.last,
            "rows": self.rows,
            "malformed": self.malformed,
            # ``Infinity`` is not a JSON number, so an open bound is published as null.
            "blind": [
                [None if low == float("-inf") else low, None if high == float("inf") else high]
                for low, high in self.blind
            ],
        }


@dataclass(frozen=True)
class Capture:
    """A run's egress record, and the intervals over which its observers demonstrably ran.

    Coverage is per segment, between its first and last observed row. Outside every interval nothing
    is established: a window in the gap between two segments could be a quiet cell or an observer
    that stopped. A row that could not be read puts a hole inside an otherwise covered interval, and
    a window overlapping that hole is not covered either.
    """

    connections: tuple[Connection, ...]
    segments: tuple[Segment, ...]

    @property
    def available(self) -> bool:
        return any(segment.rows for segment in self.segments)

    @property
    def malformed(self) -> int:
        return sum(segment.malformed for segment in self.segments)

    def blinded(self, start: float | None, end: float | None) -> bool:
        """Does an unreadable row sit anywhere this window could have seen traffic?"""
        if start is None:
            return False
        finish = start if end is None else end
        return any(
            low <= finish and high >= start
            for segment in self.segments
            for low, high in segment.blind
        )

    def covers(self, start: float | None, end: float | None) -> bool:
        # Nothing finite contains a window with no end.
        if start is None or end is None:
            return False
        if self.blinded(start, end):
            return False
        finish = start if end is None else end
        return any(
            segment.rows and segment.first <= start and finish <= segment.last
            for segment in self.segments
        )


def egress_segments(run_dir: Path) -> list[Path]:
    """Every file the capture was written into, base first."""
    first = run_dir / "egress.tsv"
    numbered = sorted(
        (p for p in run_dir.glob("egress.*.tsv") if p.stem.split(".")[-1].isdigit()),
        key=lambda p: int(p.stem.split(".")[-1]),
    )
    return ([first] if first.exists() else []) + numbered


def capture_segments(run_dir: Path) -> list[tuple[str, list[str]]]:
    """One entry per observer process, with each stretch of the capture appearing once.

    A continuation writes its own file and the runner appends it into ``egress.tsv`` when that
    observer stops, leaving the numbered file behind. Reading both counts the stretch twice; reading
    only the base makes one observer out of two and hides the interruption between them. The folded
    stretch is taken back out of the base and handed to the file it came from.
    """
    first = run_dir / "egress.tsv"
    numbered = [p for p in egress_segments(run_dir) if p != first]
    if not first.exists():
        return [(p.name, p.read_text(encoding="utf-8", errors="ignore").splitlines())
                for p in numbered]
    base = first.read_text(encoding="utf-8", errors="ignore").splitlines()
    tail = [(p, p.read_text(encoding="utf-8", errors="ignore").splitlines()) for p in numbered]
    # Backwards: the runner appends in order, so the last folded is the last stretch of the base.
    for _, rows in reversed(tail):
        if rows and len(rows) <= len(base) and base[-len(rows):] == rows:
            base = base[: -len(rows)]
    return [(first.name, base)] + [(p.name, rows) for p, rows in tail]


def read_capture(run_dir: Path) -> Capture:
    """Read every observer's stretch, counting the rows that could not be read."""
    connections: list[Connection] = []
    segments: list[Segment] = []
    for name, rows in capture_segments(run_dir):
        readable = malformed = 0
        first = last = None
        blind: list[list[float]] = []
        pending_blind = 0
        for line in rows:
            if not line.strip():
                continue
            fields = line.split("\t")
            fields += [""] * (6 - len(fields))
            hosts = []
            for column, kind in ((4, "dns"), (5, "tls")):
                for raw in fields[column].split(","):
                    host = raw.strip().rstrip(".").lower()
                    if host:
                        hosts.append((host, kind))
            try:
                epoch = float(fields[0])
            except ValueError:
                epoch = None
            # The display filter emits a row only for a DNS question or a TLS client hello, so
            # a row carrying neither name is torn however well its timestamp parses.
            if epoch is None or not hosts:
                malformed += 1
                pending_blind += 1
                continue
            while pending_blind:
                blind.append([last if last is not None else float("-inf"), epoch])
                pending_blind -= 1
            readable += 1
            first = epoch if first is None else min(first, epoch)
            last = epoch if last is None else max(last, epoch)
            connections.extend(Connection(epoch, host, kind, name) for host, kind in hosts)
        while pending_blind:
            blind.append([last if last is not None else float("-inf"), float("inf")])
            pending_blind -= 1
        segments.append(
            Segment(
                name,
                first or 0.0,
                last or 0.0,
                readable,
                malformed,
                tuple((low, high) for low, high in blind),
            )
        )
    connections.sort(key=lambda c: (c.epoch, c.host, c.kind))
    return Capture(tuple(connections), tuple(segments))


# ----- the trace ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Trace:
    """A transcript read for two things: where each episode starts, and where it seals."""

    path: Path
    first_seen: dict[str, int] = field(default_factory=dict)
    sealed_at: dict[str, int] = field(default_factory=dict)


def _stream_terminated(text: str) -> bool:
    """Did the stream answer this call by ending the episode?

    The corroboration is the stream's own reply, which the transcript's author cannot write beside a
    call that never ran.
    """
    if not text:
        return False
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        payload = None
    if isinstance(payload, dict) and payload.get("terminated") is True:
        return True
    return '"terminated": true' in text or '"terminated":true' in text


def _text_of(blocks: Any) -> str:
    if isinstance(blocks, str):
        return blocks
    if isinstance(blocks, list):
        return "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
    if isinstance(blocks, dict):
        return _text_of(blocks.get("content"))
    return ""


def read_trace(path: Path, leases: Iterable[str]) -> Trace:
    """Read one transcript for where each lease first appears and where it seals.

    A seal is a call that ran and was accepted. Naming the terminal call is not enough: a comment, a
    string, a branch that never ran and a cell that raised before reaching it all name it. So every
    seal needs the stream's own reply saying the task is over, and a call answered with an error is
    not a seal.

    codex and claude_code invoke the call as a tool and name the lease in its arguments;
    prime-agent runs it inside an ipython cell, so the lease is in the code.
    """
    wanted = set(leases)
    first_seen: dict[str, int] = {}
    sealed_at: dict[str, int] = {}
    # claude answers a tool_use in a later event, and prime splits a cell's code from its
    # result across two, so both wait here.
    pending: dict[str, tuple[str, int]] = {}
    cells: dict[str, str] = {}

    def seal(lease: object, offset: int, result: str) -> None:
        if not isinstance(lease, str) or lease not in wanted or lease in sealed_at:
            return
        if _stream_terminated(result):
            sealed_at[lease] = offset

    for offset, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        for lease in wanted:
            if lease not in first_seen and lease in line:
                first_seen[lease] = offset
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")

        # codex.
        if kind == "item.completed" and isinstance(event.get("item"), dict):
            item = event["item"]
            if item.get("type") == "mcp_tool_call" and item.get("tool") == "submit_answer":
                if item.get("error") is None and item.get("status") != "failed":
                    seal(
                        (item.get("arguments") or {}).get("lease"),
                        offset,
                        _text_of(item.get("result")),
                    )

        # claude_code: the call on the assistant side, its reply on the user side.
        elif kind == "assistant" and isinstance(event.get("message"), dict):
            for block in event["message"].get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    if str(block.get("name")).endswith("submit_answer"):
                        lease = (block.get("input") or {}).get("lease")
                        if isinstance(lease, str):
                            pending[str(block.get("id"))] = (lease, offset)
        elif kind == "user" and isinstance(event.get("message"), dict):
            for block in event["message"].get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    waiting = pending.pop(str(block.get("tool_use_id")), None)
                    if waiting is not None and not block.get("is_error"):
                        seal(waiting[0], waiting[1], _text_of(block.get("content")))

        # prime-agent: the code says which lease, the cell's result says whether it ran.
        elif kind == "tool_execution_start":
            cells[str(event.get("toolCallId"))] = json.dumps(event.get("args") or {})
        elif kind == "tool_execution_end":
            code = cells.pop(str(event.get("toolCallId")), "") or json.dumps(
                event.get("args") or {}
            )
            result = event.get("result")
            failed = event.get("isError") or (isinstance(result, dict) and result.get("isError"))
            if "submit_answer" in code and not failed:
                text = _text_of(result)
                for lease in wanted:
                    if lease in code:
                        seal(lease, offset, text)

    return Trace(path, first_seen, sealed_at)


def _trace_files(run_dir: Path, phase: str) -> list[Path]:
    traces = run_dir / phase / "traces"
    return sorted(traces.glob("*.stream.jsonl")) if traces.is_dir() else []


# ----- episodes -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Episode:
    """One dispensed task, and the wall-clock window a connection can be charged to it."""

    phase: str
    task_idx: int
    seq: int | None
    lease: str
    leg: str
    started_at: float | None
    ended_at: float | None
    window_kind: str
    correct: bool | None
    success: bool | None
    reward: float | None

    @property
    def label(self) -> str:
        seq = "" if self.seq is None else f"seq {self.seq} "
        return f"{self.phase} {seq}task {self.task_idx}"

    @property
    def domain(self) -> str:
        """The disk this episode's container reads and writes, which is what carries.

        A rollout continuation is a new container over the same mounted HOME and ``/work``, so
        a rollout is one domain however many legs it took. An eval task gets a private copy of
        HOME and a fresh ``/work``, both discarded when it ends. A leg number is neither: legs
        are numbered per run and reused across phases.
        """
        if self.phase == "rollout":
            return "rollout"
        return f"{self.phase}:{self.task_idx}"

    def identity(self) -> dict[str, Any]:
        return {"phase": self.phase, "seq": self.seq, "task_idx": self.task_idx}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a provenance file, skipping the lines that cannot be read.

    A partial write is missing evidence rather than a crash, which is the shape a record killed
    mid-line has. How many lines were lost is counted separately.
    """
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    return rows


def unreadable_provenance(run_dir: Path) -> int:
    """How many provenance lines in this run could not be read.

    Dispenses and results only. A transcript is not provenance and is read with a parser that
    already tolerates a line it cannot decode.
    """
    damaged = 0
    files = [
        path
        for pattern in ("*/dispenses.jsonl", "*/results.jsonl",
                        "*/task-*/dispenses.jsonl", "*/task-*/results.jsonl")
        for path in run_dir.glob(pattern)
    ]
    for path in sorted(files):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except (json.JSONDecodeError, ValueError):
                damaged += 1
    return damaged


def _feedback(record: dict[str, Any], name: str) -> Any:
    for item in (record.get("score") or {}).get("feedback") or []:
        if item.get("name") == name:
            return item.get("value")
    return None


def _outcome(record: dict[str, Any] | None) -> tuple[bool | None, bool | None, float | None]:
    if record is None:
        return None, None, None
    score = record.get("score") or {}
    correct = _feedback(record, "correct")
    return (
        bool(correct) if isinstance(correct, bool) else None,
        score.get("success") if isinstance(score.get("success"), bool) else None,
        score.get("reward"),
    )


def _legs(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "legs.json"
    if not path.exists():
        return []
    legs = json.loads(path.read_text(encoding="utf-8"))
    return legs if isinstance(legs, list) else []


def _rollout_episodes(
    run_dir: Path,
    legs: list[dict[str, Any]],
    traces: dict[str, Trace],
    run_end: float | None = None,
) -> list[Episode]:
    """Windows running from a dispense to a bound on the seal, not to the next dispense.

    Only the transcript can say when a lease ended: the seal happened no later than the dispense of
    the first task pulled at or after it. With no seal in the transcript the bound is the leg.

    Capacity is deliberately not a bound. ``get_task`` force-drains only when a pull finds every
    slot occupied, so a newer lease that submits frees a slot and lets the next dispense through
    with an older lease still live, and ending it at ``index + max_in_flight`` would invent a seal
    the stream never performed.
    """
    phase_dir = run_dir / "rollout"
    dispenses = sorted(_read_jsonl(phase_dir / "dispenses.jsonl"), key=lambda d: d["dispensed_at"])
    results = {r["lease"]: r for r in _read_jsonl(phase_dir / "results.jsonl")}
    spans = [
        (leg.get("started_at"), leg.get("ended_at"), f"leg-{leg.get('leg')}")
        for leg in legs
        if leg.get("phase") == "rollout" and leg.get("started_at") is not None
    ]

    def leg_of(when: float) -> tuple[str, float | None]:
        for started, ended, name in spans:
            if started <= when and (ended is None or when <= ended):
                return name, ended
        # No leg record covers this dispense. The run's own end is the last moment anything in
        # it could have happened; without that the episode has no upper bound.
        return "rollout", run_end

    first_seen: dict[str, tuple[str, int]] = {}
    sealed_at: dict[str, tuple[str, int]] = {}
    for name, trace in traces.items():
        for lease, offset in trace.first_seen.items():
            first_seen.setdefault(lease, (name, offset))
        for lease, offset in trace.sealed_at.items():
            sealed_at.setdefault(lease, (name, offset))

    times = {str(d["lease"]): float(d["dispensed_at"]) for d in dispenses}
    episodes = []
    for dispense in dispenses:
        lease = str(dispense["lease"])
        started = float(dispense["dispensed_at"])
        leg, leg_end = leg_of(started)

        bounds = []
        kind = "leg_bound"

        # At or after, not after: a harness that submits and pulls again inside one action puts
        # both on the same line, and the lease it pulls there was dispensed after this one sealed.
        seal = sealed_at.get(lease)
        if seal is not None:
            trace_name, offset = seal
            after = [
                times[other]
                for other, (name, where) in first_seen.items()
                if name == trace_name and where >= offset and other != lease and other in times
            ]
            if after:
                bounds.append(min(after))
                kind = "trace_seal_bound"
        if leg_end is not None:
            bounds.append(leg_end)
        ended = min(bounds) if bounds else None

        correct, success, reward = _outcome(results.get(lease))
        episodes.append(
            Episode(
                phase="rollout",
                task_idx=int(dispense["task_idx"]),
                seq=int(dispense["seq"]),
                lease=lease,
                leg=leg,
                started_at=started,
                ended_at=ended,
                window_kind=kind,
                correct=correct,
                success=success,
                reward=reward,
            )
        )
    return episodes


def _eval_episodes(
    run_dir: Path, phase: str, legs: list[dict[str, Any]], task_timeout_s: float
) -> list[Episode]:
    """Windows straight off the leg record, because an eval task is its own container.

    With no leg record the window is the dispense plus the configured per-task timeout: an upper
    bound, labelled as one.
    """
    phase_dir = run_dir / phase
    by_task = {
        int(leg["task_idx"]): leg
        for leg in legs
        if leg.get("phase") == phase and leg.get("task_idx") is not None
    }
    episodes = []
    for task_dir in sorted(phase_dir.glob("task-*")):
        if not task_dir.is_dir():
            continue
        dispenses = _read_jsonl(task_dir / "dispenses.jsonl")
        if not dispenses:
            continue
        dispense = dispenses[-1]
        task_idx = int(dispense["task_idx"])
        results = _read_jsonl(task_dir / "results.jsonl")
        correct, success, reward = _outcome(results[-1] if results else None)
        leg = by_task.get(task_idx)
        if leg is not None and leg.get("started_at") is not None:
            started, ended = float(leg["started_at"]), leg.get("ended_at")
            kind, name = "leg", f"leg-{leg.get('leg')}"
        else:
            started = float(dispense["dispensed_at"])
            ended = started + task_timeout_s
            kind, name = "dispense_timeout_bound", f"{phase}-task-{task_idx}"
        episodes.append(
            Episode(
                phase=phase,
                task_idx=task_idx,
                seq=int(dispense["seq"]) if dispense.get("seq") is not None else None,
                lease=str(dispense["lease"]),
                leg=name,
                started_at=started,
                ended_at=float(ended) if ended is not None else None,
                window_kind=kind,
                correct=correct,
                success=success,
                reward=reward,
            )
        )
    return episodes


# ----- classification ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeLeakage:
    """One episode's bucket, and every fact that put it there."""

    episode: Episode
    bucket: str
    reasons: tuple[str, ...]
    evidence: tuple[Connection, ...]
    covered: bool
    # Whether the whole window was watched, which is what every HOME-inheritance question reads.
    # Separate from the bucket, because evidence raises a bucket and does not close a gap.
    observed: bool
    shared_with: tuple[dict[str, Any], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "phase": self.episode.phase,
            "task_idx": self.episode.task_idx,
            "seq": self.episode.seq,
            "lease": self.episode.lease,
            "leg": self.episode.leg,
            "window": {
                "started_at": self.episode.started_at,
                "ended_at": self.episode.ended_at,
                "kind": self.episode.window_kind,
                "capture_covers": self.covered,
                "observed": self.observed,
                "shared_with": list(self.shared_with),
            },
            "bucket": self.bucket,
            "reasons": list(self.reasons),
            "evidence": [c.to_json() for c in self.evidence],
            "correct": self.episode.correct,
            "success": self.episode.success,
            "reward": self.episode.reward,
        }


LIMITS = [
    "egress observes hostnames and times, never a method, a status or a body, so nothing here "
    "can establish that answer content arrived; this half stops at unresolved leakage and the "
    "content evidence that would settle it is a separate change",
    "the file CDN serves the whole platform, so a client hello to it is a connection to a CDN "
    "and not a download of an answer key",
    "hostname granularity cannot separate a listing endpoint from a data endpoint on a host "
    "that serves both",
    "a local read of an already-downloaded file is invisible to the observer, so once a leg has "
    "reached the answer source none of its later episodes can be cleared",
    "egress cannot show intent, so a dataset host reached for a tokenizer and one reached for "
    "an answer key are the same observation",
    "episodes that overlap in time share one network namespace, so a connection inside more "
    "than one open window is charged to all of them and each record names its rivals",
    "a trace carries what the agent ran, so a harness whose transcript summarises rather than "
    "quotes contributes no request evidence and its episodes rest on egress alone",
]


@dataclass(frozen=True)
class RunLeakage:
    """One run directory's episodes, the counts a reader wants, and the caveats they need."""

    run_dir: Path
    run_id: str
    cell: str
    env: str
    harness: str
    model: str
    finished: bool
    capture: Capture
    answer_source_configured: bool
    episodes: tuple[EpisodeLeakage, ...]
    notes: tuple[str, ...]

    @property
    def label(self) -> str:
        """The directory, not the run id: a repair carries the original's manifest."""
        return self.run_dir.name

    def phases(self) -> list[str]:
        seen = [e.episode.phase for e in self.episodes]
        return [p for p in PHASES if p in seen] + sorted({p for p in seen if p not in PHASES})

    def counts(self, phase: str | None = None) -> dict[str, int]:
        counts = {bucket: 0 for bucket in (*BUCKETS, UNCLASSIFIED)}
        for row in self.episodes:
            if phase is None or row.episode.phase == phase:
                counts[row.bucket] += 1
        return counts

    def correct_rate(self, bucket: str, phase: str | None = None) -> tuple[int, int]:
        """Graded episodes in this bucket, and how many were correct."""
        rows = [
            e
            for e in self.episodes
            if e.bucket == bucket
            and (phase is None or e.episode.phase == phase)
            and e.episode.correct is not None
        ]
        return sum(1 for e in rows if e.episode.correct), len(rows)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "run_dir": str(self.run_dir),
            "run_id": self.run_id,
            "cell": self.cell,
            "env": self.env,
            "harness": self.harness,
            "model": self.model,
            "finished": self.finished,
            "egress": {
                "available": self.capture.available,
                "malformed_rows": self.capture.malformed,
                "segments": [s.to_json() for s in self.capture.segments],
                "observations": len(self.capture.connections),
            },
            "answer_source_configured": self.answer_source_configured,
            "buckets": self.counts(),
            "phases": [
                {
                    "phase": phase,
                    "buckets": self.counts(phase),
                    "correct": {
                        bucket: dict(
                            zip(
                                ("correct", "graded"),
                                self.correct_rate(bucket, phase),
                                strict=True,
                            )
                        )
                        for bucket in (*BUCKETS, UNCLASSIFIED)
                    },
                }
                for phase in self.phases()
            ],
            "notes": list(self.notes),
            "limits": LIMITS,
            "episodes": [e.to_json() for e in self.episodes],
        }


def _rank(bucket: str) -> int:
    return BUCKETS.index(bucket) if bucket in BUCKETS else -1


def _raise_to(current: str, candidate: str) -> str:
    return candidate if _rank(candidate) > _rank(current) else current


def _note(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _window_evidence(
    episode: Episode, connections: Sequence[Connection], starts: Sequence[float]
) -> list[Connection]:
    """Every connection inside this episode's window, which is half-open at the end."""
    if episode.started_at is None:
        return []
    end = episode.ended_at if episode.ended_at is not None else float("inf")
    out = []
    for connection in connections[bisect_left(starts, episode.started_at) :]:
        if connection.epoch >= end:
            break
        out.append(connection)
    return out


def _accounts_for_its_home(source: RunLeakage) -> tuple[bool, str]:
    """Can this run say the HOME a bookend would inherit holds no answer file?

    Only its rollout matters, since that is the phase whose writes land in the mounted HOME, and
    the answer is yes only when that rollout was observed end to end and never reached the answer
    source. Where a fetched file landed is content evidence this half does not carry.
    """
    rollout = [e for e in source.episodes if e.episode.phase == "rollout"]
    if not rollout:
        return False, "has no rollout record to account for that HOME"
    # Off the coverage bit, never the bucket: an unwatched episode that showed one connection is
    # bucketed for it and is still unwatched.
    blind = sum(1 for e in rollout if not e.observed)
    if blind:
        return False, f"could not observe {blind} of its {len(rollout)} rollout episodes end to end"
    contacted = sum(1 for e in rollout if _rank(e.bucket) >= _rank("attempted_leakage"))
    if contacted:
        return False, (
            f"reached the answer source in {contacted} of its {len(rollout)} rollout episodes, "
            "and where anything it fetched landed is not on this record"
        )
    return True, ""


def _inherited_artifacts(run_dir: Path, manifest: dict[str, Any]) -> tuple[list[str], bool]:
    """Whether a bookend can be cleared against the HOME it inherited.

    The runner seeds a bookend from its source's accumulated HOME, which every eval task then gets a
    copy of. This can say the source was fully observed and never reached the answer source; it
    cannot say what a source that did reach it put where.
    """
    rebookend = manifest.get("rebookend") or {}
    source_id = rebookend.get("rebookend_of")
    if not source_id:
        return [], False
    source_dir = run_dir.parent / str(source_id)
    if not source_dir.is_dir():
        return [
            f"this run is a bookend of {source_id}, whose directory is not beside it, so "
            "whether the source left an answer file in the HOME this run inherited cannot be "
            "checked; no episode here is cleared"
        ], True
    accounted, why = _accounts_for_its_home(classify_run(source_dir, _inherit=False))
    if accounted:
        return [], False
    return [
        f"this run is a bookend of {source_id}, which {why}. Whether the HOME it inherited "
        "holds an answer file cannot be established, so no episode here is cleared"
    ], True


def classify_run(run_dir: Path, *, _inherit: bool = True) -> RunLeakage:
    """Grade every episode in a run directory. Reads only; writes nothing back."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cell = manifest.get("cell") or {}
    env = str(cell.get("env") or "")
    source = ANSWER_SOURCES.get(env)
    budget = cell.get("budget") or {}
    timeout = float(budget.get("eval_task_timeout_s") or 900)
    run_end = manifest.get("ended_at")
    run_end = float(run_end) if isinstance(run_end, (int, float)) else None
    finished = run_end is not None

    capture = read_capture(run_dir)
    legs = _legs(run_dir)
    episodes: list[Episode] = []
    traces: dict[str, dict[str, Trace]] = {}

    if (run_dir / "rollout" / "dispenses.jsonl").exists():
        leases = [str(d["lease"]) for d in _read_jsonl(run_dir / "rollout" / "dispenses.jsonl")]
        traces["rollout"] = {
            str(p): read_trace(p, leases) for p in _trace_files(run_dir, "rollout")
        }
        episodes += _rollout_episodes(run_dir, legs, traces["rollout"], run_end)
    for phase in ("eval_before", "eval_after"):
        if (run_dir / phase).is_dir():
            # No transcript for an eval phase: its windows come from the leg record.
            episodes += _eval_episodes(run_dir, phase, legs, timeout)

    inheritance_notes: list[str] = []
    unresolved_inheritance = False
    if _inherit:
        inheritance_notes, unresolved_inheritance = _inherited_artifacts(run_dir, manifest)

    notes = _capture_notes(capture, finished) + _coverage_notes(legs, episodes)
    damaged = unreadable_provenance(run_dir)
    if damaged:
        word, it = ("line", "it") if damaged == 1 else ("lines", "they")
        notes.append(
            f"{damaged} provenance {word} could not be read, so what {it} recorded is not in "
            "this report and the counts here are over what the record could give up"
        )
    notes += inheritance_notes
    if source is None:
        notes.append(
            f"no answer-source hosts are configured for env {env!r}, so leakage cannot be "
            "distinguished from general web reference in this run"
        )

    # Over the whole life of the rollout's legs, not only inside an episode's window: a container
    # is up before its first task and between legs, and what it fetched then is on the same disk.
    disk_contact, orphans = _disk_contact(capture, legs, source, episodes)
    if disk_contact is not None:
        notes.append(
            f"the rollout's containers reached the answer source at {disk_contact:.3f}, "
            f"{orphans} of those observations outside any episode's window; the disk carries "
            "that from then on and no episode reading it is cleared"
        )

    starts = [c.epoch for c in capture.connections]
    ordered = sorted(episodes, key=lambda e: (e.started_at if e.started_at is not None else 0.0))
    graded: list[EpisodeLeakage] = []
    # When each persistence domain first reached the answer source. Keyed by domain rather than
    # leg, and timed from the observed connection rather than from the owning episode's window,
    # since a window is not when its traffic happened.
    contacted: dict[str, float] = {}
    seeded_home_contact = False
    seeded_home_blind = False
    if disk_contact is not None:
        contacted["rollout"] = disk_contact
        seeded_home_contact = True

    for episode in ordered:
        evidence = _window_evidence(episode, capture.connections, starts)
        covered = capture.covers(episode.started_at, episode.ended_at)
        reasons: list[str] = []

        observed = False
        if not capture.available:
            bucket = UNCLASSIFIED
            _note(reasons, "capture_unavailable")
        elif episode.started_at is None:
            bucket = UNCLASSIFIED
            _note(reasons, "no_window")
        elif not finished:
            bucket = UNCLASSIFIED
            _note(reasons, "run_unfinished")
        elif not covered:
            bucket = UNCLASSIFIED
            _note(reasons, "capture_not_covering_window")
        elif unresolved_inheritance:
            bucket = UNCLASSIFIED
            _note(reasons, "inherited_home_unchecked")
        else:
            bucket = "computed_locally"
            observed = True

        # Tops out at unresolved: the observer sees a connection, never a body.
        for connection in evidence:
            role = host_role(connection.host, source)
            if role == "infrastructure":
                continue
            if role == "general":
                bucket = _raise_to(bucket, "general_web_reference")
                _note(reasons, "general_web_host")
                continue
            bucket = _raise_to(bucket, "attempted_leakage")
            _note(reasons, "answer_source_host")
            if role == "answer_source_content" and connection.kind == "tls":
                bucket = _raise_to(bucket, "unresolved_leakage")
                _note(reasons, "content_cdn_handshake")

        # A local read of whatever reached this disk is invisible to the observer, so a later
        # episode reading it is not cleared.
        contact = contacted.get(episode.domain)
        if (
            contact is not None
            and episode.started_at is not None
            and episode.started_at >= contact
            and bucket != UNCLASSIFIED
        ):
            bucket = _raise_to(bucket, "unresolved_leakage")
            _note(reasons, "answer_source_contact_earlier_on_this_disk")

        # An eval_after task runs against a copy of the HOME the rollout accumulated.
        if episode.phase == "eval_after" and bucket != UNCLASSIFIED:
            if seeded_home_blind:
                bucket = UNCLASSIFIED
                _note(reasons, "rollout_home_unaccounted")
            elif seeded_home_contact:
                bucket = _raise_to(bucket, "unresolved_leakage")
                _note(reasons, "rollout_reached_the_answer_source")

        reached = _rank(bucket) >= _rank("attempted_leakage")
        if reached and episode.started_at is not None:
            when = min(
                (
                    c.epoch
                    for c in evidence
                    if host_role(c.host, source).startswith("answer_source")
                ),
                default=episode.started_at,
            )
            contacted[episode.domain] = min(contacted.get(episode.domain, when), when)
        if episode.phase == "rollout":
            if not observed:
                seeded_home_blind = True
            elif reached:
                seeded_home_contact = True

        graded.append(
            EpisodeLeakage(
                episode=episode,
                bucket=bucket,
                reasons=tuple(reasons),
                evidence=tuple(evidence),
                covered=covered,
                observed=observed,
                shared_with=(),
            )
        )

    graded = _mark_shared_windows(graded)
    graded.sort(
        key=lambda g: (
            PHASES.index(g.episode.phase) if g.episode.phase in PHASES else 9,
            g.episode.seq if g.episode.seq is not None else g.episode.task_idx,
        )
    )
    return RunLeakage(
        run_dir=run_dir,
        run_id=str(manifest.get("run_id") or run_dir.name),
        cell=str(cell.get("name") or ""),
        env=env,
        harness=str(cell.get("harness") or ""),
        model=str(cell.get("model") or ""),
        finished=finished,
        capture=capture,
        answer_source_configured=source is not None,
        episodes=tuple(graded),
        notes=tuple(notes),
    )


def _disk_contact(
    capture: Capture,
    legs: list[dict[str, Any]],
    source: AnswerSource | None,
    episodes: Sequence[Episode] = (),
) -> tuple[float | None, int]:
    """When the rollout's disk first reached the answer source, over the whole life of its legs.

    A container is up before its first task is handed out and between one leg and the next, and a
    connection made then reached the same mounted HOME every later episode reads. Such an
    observation is deliberately given no episode of its own, since the stream never dispensed one;
    it sets the disk's contact time and is reported in a note.
    """
    spans = [
        (float(leg["started_at"]), float(leg["ended_at"]))
        for leg in legs
        if leg.get("phase") == "rollout"
        and leg.get("started_at") is not None
        and leg.get("ended_at") is not None
    ]
    if not spans:
        return None, 0
    windows = [
        (e.started_at, e.ended_at if e.ended_at is not None else float("inf"))
        for e in episodes
        if e.phase == "rollout" and e.started_at is not None
    ]
    earliest: float | None = None
    orphans = 0
    for connection in capture.connections:
        if not host_role(connection.host, source).startswith("answer_source"):
            continue
        if not any(start <= connection.epoch <= end for start, end in spans):
            continue
        earliest = connection.epoch if earliest is None else min(earliest, connection.epoch)
        if not any(start <= connection.epoch < end for start, end in windows):
            orphans += 1
    return earliest, orphans


def _capture_notes(capture: Capture, finished: bool) -> list[str]:
    notes = []
    if not finished:
        notes.append(
            "this run has no ended_at, so it was still going when the capture was read: the "
            "observer's record cannot be complete and no episode here is cleared"
        )
    if not capture.available:
        notes.append(
            "no readable egress record in this run directory, so no episode can be cleared; "
            "every episode is unclassified rather than clean"
        )
    if capture.malformed:
        notes.append(
            f"{capture.malformed} capture rows could not be read and are not evidence either "
            "way; the windows around them are still cleared only where a segment covers them"
        )
    return notes


def _coverage_notes(legs: list[dict[str, Any]], episodes: Sequence[Episode]) -> list[str]:
    """What the run ran that this classification has no episode for, said out loud."""
    notes = []
    seen: dict[str, set[int]] = {}
    bounded: dict[str, int] = {}
    for episode in episodes:
        seen.setdefault(episode.phase, set()).add(episode.task_idx)
        if episode.window_kind == "dispense_timeout_bound":
            bounded[episode.phase] = bounded.get(episode.phase, 0) + 1
    for phase in sorted({str(leg.get("phase")) for leg in legs if leg.get("task_idx") is not None}):
        missing = sum(
            1
            for leg in legs
            if leg.get("phase") == phase
            and leg.get("task_idx") is not None
            and int(leg["task_idx"]) not in seen.get(phase, set())
        )
        if missing:
            word = "leg" if missing == 1 else "legs"
            notes.append(
                f"{phase} has {missing} {word} whose task was never dispensed, so they have no "
                "episode here and this phase's counts are over what it did dispense"
            )
    for phase, count in sorted(bounded.items()):
        notes.append(
            f"{phase} has {count} episodes with no leg record, so their windows are the dispense "
            "plus the configured task timeout: an upper bound, which over-attributes traffic"
        )
    return notes


def _mark_shared_windows(graded: Sequence[EpisodeLeakage]) -> list[EpisodeLeakage]:
    """Name, per episode, the other episodes that could equally own its evidence.

    Overlapping episodes share one network namespace, so a connection inside two open windows
    belongs to both. Rivals are named per connection rather than per window: two windows that
    overlap where the traffic is not create no ambiguity about that traffic.
    """
    windows = [
        (
            g.episode.started_at,
            g.episode.ended_at if g.episode.ended_at is not None else float("inf"),
        )
        for g in graded
    ]
    out = []
    for index, row in enumerate(graded):
        start, _ = windows[index]
        if start is None or not row.evidence:
            out.append(row)
            continue
        rivals = tuple(
            other.episode.identity()
            for position, other in enumerate(graded)
            if position != index
            and windows[position][0] is not None
            and any(
                windows[position][0] <= c.epoch < windows[position][1] for c in row.evidence
            )
        )
        out.append(
            EpisodeLeakage(
                episode=row.episode,
                bucket=row.bucket,
                reasons=row.reasons,
                evidence=row.evidence,
                covered=row.covered,
                observed=row.observed,
                shared_with=rivals,
            )
        )
    return out


# ----- output -----------------------------------------------------------------------------------


def render_table(runs: Sequence[RunLeakage]) -> str:
    """One row per run, phase and bucket, because a blended row is the thing being avoided."""
    header = ("run", "phase", "bucket", "episodes", "correct", "rate")
    rows: list[tuple[str, ...]] = []
    for run in runs:
        for phase in run.phases():
            counts = run.counts(phase)
            for bucket in (*BUCKETS, UNCLASSIFIED):
                if not counts[bucket]:
                    continue
                correct, graded = run.correct_rate(bucket, phase)
                rows.append(
                    (
                        run.label,
                        phase,
                        bucket,
                        str(counts[bucket]),
                        f"{correct}/{graded}",
                        f"{correct / graded:.3f}" if graded else "-",
                    )
                )
        if not run.episodes:
            rows.append((run.label, "-", "no episodes recorded", "0", "-", "-"))
    widths = [max(len(str(c)) for c in column) for column in zip(header, *rows, strict=True)]
    lines = [
        "  ".join(str(c).ljust(w) for c, w in zip(header, widths, strict=True)),
        "  ".join("-" * w for w in widths),
    ]
    lines += ["  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=True)) for row in rows]

    for run in runs:
        for note in run.notes:
            lines += ["", f"{run.label}: {note}"]
    lines += ["", "what this cannot establish"]
    lines += [f"  - {limit}" for limit in LIMITS]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="completed run directories")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    parser.add_argument("--out", type=Path, default=None, help="write instead of printing")
    parser.add_argument(
        "--allow-unfinished",
        action="store_true",
        help=(
            "grade runs whose manifest has no ended_at; their capture cannot be complete, so "
            "every episode in them is unclassified unless positive evidence raises it"
        ),
    )
    args = parser.parse_args(argv)

    missing = [d for d in args.run_dirs if not d.is_dir()]
    if missing:
        # The batch, not the rest: a report silently missing a run cannot show you that.
        for run_dir in missing:
            print(f"no run directory at {run_dir}", file=sys.stderr)
        return 1
    targets = list(args.run_dirs)
    if not targets:
        print("no run directories given", file=sys.stderr)
        return 1
    inside = _inside_a_run(args.out, targets)
    if inside is not None:
        # A report landing on a manifest, a capture or a provenance file destroys the evidence
        # it was made from.
        print(
            f"refusing to write {args.out} inside the run directory {inside}: this command "
            "reads a run's record and never writes to it. Choose a path outside every run "
            "directory it was given.",
            file=sys.stderr,
        )
        return 1
    refused = [d for d in targets if not _finished(d)]
    if refused and not args.allow_unfinished:
        for run_dir in refused:
            # On stderr: stdout is the document this command advertises.
            print(
                f"refusing {run_dir}: its manifest has no ended_at, so the run was still going "
                "and the egress record cannot be complete. Pass --allow-unfinished to grade it "
                "anyway, where every episode is unclassified rather than clean.",
                file=sys.stderr,
            )
        targets = [d for d in targets if d not in refused]
        if not targets:
            return 1

    runs = [classify_run(d) for d in targets]
    runs.sort(key=lambda r: r.label)
    if args.format == "json":
        text = json.dumps({"schema": SCHEMA, "runs": [r.to_json() for r in runs]}, indent=2)
    else:
        text = render_table(runs)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"leakage: {args.out}")
    else:
        print(text)
    return 1 if refused and not args.allow_unfinished else 0


def runs_read(targets: Sequence[Path]) -> list[Path]:
    """Every run directory this command will open, not only the ones it was handed.

    A bookend names the run it was made from and classifying it opens that run's record too, so the
    walk follows ``manifest.rebookend.rebookend_of`` transitively with each directory visited once.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    queue = list(targets)
    while queue:
        run_dir = queue.pop()
        try:
            key = run_dir.expanduser().resolve()
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        found.append(run_dir)
        manifest = run_dir / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError, OSError):
            continue
        source_id = (record.get("rebookend") or {}).get("rebookend_of")
        if source_id:
            queue.append(run_dir.parent / str(source_id))
    return found


def _inside_a_run(out: Path | None, targets: Sequence[Path]) -> Path | None:
    """The run directory an output path would land in, if it would land in one.

    Checked against every run the command will read, not only those named on the command line.
    Symlinks are resolved on both sides.
    """
    if out is None:
        return None
    destination = out.expanduser().resolve()
    for run_dir in runs_read(targets):
        try:
            root = run_dir.expanduser().resolve()
        except OSError:
            continue
        if destination == root or root in destination.parents:
            return run_dir
    return None


def _finished(run_dir: Path) -> bool:
    path = run_dir / "manifest.json"
    if not path.exists():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("ended_at") is not None


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANSWER_SOURCES",
    "BUCKETS",
    "INFRASTRUCTURE",
    "LIMITS",
    "SCHEMA",
    "UNCLASSIFIED",
    "AnswerSource",
    "Capture",
    "Connection",
    "Episode",
    "EpisodeLeakage",
    "RunLeakage",
    "Segment",
    "Trace",
    "classify_run",
    "host_role",
    "main",
    "read_capture",
    "read_trace",
    "runs_read",
    "render_table",
]
