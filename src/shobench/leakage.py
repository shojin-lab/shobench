"""Per-episode leakage classification for a completed run directory.

Some v0 cells answer questions whose answers are published. The agent has an unsandboxed shell
and open egress by design, and the runner observes rather than gates, so an episode can be won
by reasoning or by reading the answer key, and the reward is the same number either way. A
blended correct-rate over both is not a measurement of anything. This grades every episode so
the rate can be reported per bucket instead.

The ranks, weakest first:

    computed_locally        nothing left the cell but the harness's own infrastructure
    general_web_reference   the cell went out, but not to a host that distributes the answers
    attempted_leakage       the cell reached the answer source; no evidence a body moved
    unresolved_leakage      evidence consistent with a body having moved, not established
    achieved_leakage        answer content demonstrably reached the episode

``unclassified`` is not on that ladder. It is what an episode gets when the evidence is missing
rather than empty, and it is the reason this command cannot quietly clear anything: an episode
whose capture does not cover its window is unclassified, never computed_locally.

**Egress is the floor and cannot reach achieved.** The record is ``egress.tsv``, one row per
outbound DNS question or TLS client hello observed in the cell's network namespace, written
outside the container by an observer the agent holds no mount of. It sees hostnames and times.
It never sees a method, a status, a body or a byte count, and the Hub's file CDN serves the
whole platform rather than one dataset, so a client hello to it establishes a connection to a
CDN and not a download of an answer key. The highest an egress observation can carry an episode
is ``unresolved_leakage``.

**Achieved needs content, and content is in the transcript.** The trace records what the agent
ran and what came back, and both halves are durable. Two things reach achieved:

- a result carrying the answer source's own answer fields, which is what a dataset row API
  returns when it is not refusing;
- a download whose destination the filesystem then answered for, and any later episode that
  reads that destination.

Refinement reads requests, not prose. Only the text of a command or tool call counts as a
request, so a URL the agent merely mentioned in its reasoning cannot raise an episode, and a
result is only read as answer content when it carries the corroborating fields the dataset ships
alongside its answers, so the agent's own submission cannot be mistaken for a leak.

Nothing the agent wrote about itself lowers a bucket, and no combination of trace text alone
reaches achieved without content in a result.

Usage::

    shobench leakage runs/hle-codex-gpt-56-terra-20260813T215942Z
    shobench leakage runs/hle-* --format json > leakage.json
"""

from __future__ import annotations

import argparse
import json
import re
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "shobench.leakage/1"

BUCKETS = (
    "computed_locally",
    "general_web_reference",
    "attempted_leakage",
    "unresolved_leakage",
    "achieved_leakage",
)

# Not a bucket: the label for an episode whose evidence is missing rather than empty.
UNCLASSIFIED = "unclassified"

PHASES = ("eval_before", "rollout", "eval_after")

# The only two paths inside an agent container that are not thrown away with it. The runner
# mounts the cell's HOME at ``/root`` and a working directory at ``/work``, and starts the
# harness in the second, so a command with a relative destination writes to ``/work`` and not to
# HOME. Everything outside those two mounts is the container's own layer, which is removed with
# the container, which is why there is no "unknown" here: an unrecognised path is scratch.
AGENT_HOME = "/root"
AGENT_WORK = "/work"

# Hosts a cell talks to because of how it is run rather than because of what it is answering:
# the harness's model API and telemetry, and the package registries a tool install goes through.
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
    """Where one environment's answers live, and what its answers look like when they arrive.

    ``index`` hosts serve listings, metadata and payloads over one name. ``content`` hosts are
    the file CDN: they move bodies, but for the whole platform, so reaching one narrows what
    happened without settling it. ``rows`` names the query API over the dataset itself.

    ``answer_field`` and ``corroborating_fields`` are the dataset's own columns. A result that
    carries the answer field beside one of its neighbours is a row of the answer key; the
    agent's own submission carries the answer field alone, which is why the corroboration is
    required rather than assumed.
    """

    index: tuple[str, ...]
    content: tuple[str, ...]
    rows: tuple[str, ...] = ()
    answer_field: str = "answer"
    corroborating_fields: tuple[str, ...] = ()


# Keyed by environment, because "the host that distributes the answers" is a fact about the
# dataset, not about the runner. An environment with no entry gets no table at all and every
# classification for it says so, rather than quietly reporting a clean run.
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

# Endpoints on the row API that return rows rather than a description of rows.
_ROW_ENDPOINTS = ("/rows", "/search", "/filter", "/first-rows")
# Extensions a dataset ships as. A URL ending in one is a file body whatever route served it.
_ARTIFACT_SUFFIXES = (".parquet", ".csv", ".tsv", ".jsonl", ".arrow", ".zip", ".gz")
# Library calls whose only purpose is pulling a dataset or a repo file off the Hub.
_HUB_CALLS = ("load_dataset(", "hf_hub_download(", "snapshot_download(")

_URL = re.compile(r"https?://[^\s\"'\\<>)\]}]+")
# Where a shell command says to put what it fetches.
_DESTINATION = re.compile(
    r"(?:(?<=\s)|^)(?:-o|-O|--output|--output-document)[=\s]+([^\s'\";|&)]+)"
    r"|>\s*([^\s'\";|&)]+)"
)


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


def content_url_kind(url: str, source: AnswerSource | None) -> str | None:
    """What a URL asks the answer source for, when it asks for data rather than a listing.

    ``file_download`` is the Hub's ``resolve`` route or any URL ending in a dataset artifact
    extension. ``row_query`` is the row API's data endpoints, which return rows themselves
    rather than a description of them.

    A blob route renders a file inside a page rather than serving it, so a data extension under
    one is a link someone was reading. Query strings are not required: ``curl -G`` puts the
    query in ``--data-urlencode`` parameters and leaves the URL bare.
    """
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    host = (parts.hostname or "").lower()
    if "/resolve/" in parts.path:
        return "file_download"
    if "/blob/" not in parts.path and path.lower().endswith(_ARTIFACT_SUFFIXES):
        return "file_download"
    if source is not None and _matches(host, source.rows) and path in _ROW_ENDPOINTS:
        return "row_query"
    return None


def carries_answer_content(text: str, source: AnswerSource | None) -> bool:
    """Does this result carry rows of the answer key, as opposed to naming one?

    The answer field beside one of the columns the dataset ships next to it. The pairing is what
    separates a fetched row from the agent's own ``submit_answer`` arguments, which carry an
    answer and nothing that travels with one.
    """
    if source is None or not source.corroborating_fields:
        return False
    if not re.search(rf'"{re.escape(source.answer_field)}"\s*:', text):
        return False
    alternatives = "|".join(re.escape(f) for f in source.corroborating_fields)
    return bool(re.search(rf'"(?:{alternatives})"\s*:', text))


def download_destinations(command: str) -> list[str]:
    """Where a command says to save what it fetches.

    Only paths: a bare word after ``-o`` is as likely to be ``find``'s or operator as a file, so
    a destination has to look like one.
    """
    out = []
    for match in _DESTINATION.finditer(command):
        candidate = (match.group(1) or match.group(2) or "").strip()
        if not candidate or candidate.startswith("-"):
            continue
        if "/" in candidate or candidate.lower().endswith(_ARTIFACT_SUFFIXES):
            out.append(candidate)
    return list(dict.fromkeys(out))


def destination_persistence(path: str) -> str:
    """Which of the container's three storage classes a saved file lands in.

    ``home`` is the ``/root`` mount, ``work`` is the ``/work`` mount, and ``container`` is
    everything else, which the container layer holds and ``docker run --rm`` deletes.

    A relative path is resolved against ``/work``, because that is the working directory the
    harness is started in. Reading a bare ``curl -o key.parquet`` as a HOME file would claim a
    durability the file does not have.
    """
    expanded = path.strip()
    if expanded == "~" or expanded.startswith("~/"):
        expanded = AGENT_HOME + expanded[1:]
    elif expanded.startswith("$HOME"):
        expanded = AGENT_HOME + expanded[len("$HOME") :]
    if not expanded.startswith("/"):
        expanded = f"{AGENT_WORK}/{expanded.lstrip('./')}"
    for root, label in ((AGENT_HOME, "home"), (AGENT_WORK, "work")):
        if expanded == root or expanded.startswith(f"{root}/"):
            return label
    return "container"


# How far a saved file can still be read from, given where it landed and which phase saved it.
#
# The rollout runs in one container per leg against the cell's mounted HOME and one mounted
# ``/work``, and the runner copies that HOME into every eval task and into a bookend's tasks. So
# a rollout HOME file reaches the whole run and its bookends, and a rollout ``/work`` file
# reaches the rest of the rollout but no eval task, which is given a fresh empty one.
#
# An eval task is its own container with a private copy of HOME and its own ``/work``, both
# discarded when the task ends. Nothing it saves anywhere reaches any other episode, which is
# what makes ``episode`` the only honest reach for an acquisition made during an eval.
def reach_of(persistence: str, phase: str) -> str:
    if phase != "rollout":
        return "episode"
    return {"home": "run", "work": "phase"}.get(persistence, "leg")


# ----- the capture ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Connection:
    """One observed outbound name: a DNS question, or a TLS client hello carrying an SNI.

    The kinds are not equivalent evidence. A resolution says a name was looked up; a client
    hello says a connection to that name was opened. Neither says a body moved.
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
    # One interval per row that could not be read, bounding where in time it sat. The capture is
    # written in order, so an unreadable row lies between the last readable row before it and
    # the first after it. That is where the observer saw something this cannot account for.
    blind: tuple[tuple[float, float], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "first": self.first,
            "last": self.last,
            "rows": self.rows,
            "malformed": self.malformed,
            "blind": [list(interval) for interval in self.blind],
        }


@dataclass(frozen=True)
class Capture:
    """A run's whole egress record, and how far it can be trusted to have been watching.

    Coverage is per segment and is the interval between its first and last observed row. Inside
    that interval the observer was demonstrably running, so silence there is real silence.
    Outside every interval nothing is established: a window in the gap between two segments, or
    past the last row of the last one, could be a quiet cell or an observer that stopped, and
    those are not the same finding.

    A row the reader could not parse puts a hole inside an otherwise covered interval. The
    observer saw something there and this cannot say what, so a window overlapping that hole is
    not covered either. Counting the unreadable row and then clearing the window around it would
    be the same silent pass this refuses everywhere else.
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
        if start is None:
            return False
        if self.blinded(start, end):
            return False
        finish = start if end is None else end
        return any(
            segment.rows and segment.first <= start and finish <= segment.last
            for segment in self.segments
        )


def egress_segments(run_dir: Path) -> list[Path]:
    """The capture's segment files in the order they were written.

    A continuation gets a segment of its own rather than truncating the first, so the record of
    a run that was suspended and resumed is several files and reading only the first would drop
    everything after the interruption.
    """
    first = run_dir / "egress.tsv"
    numbered = sorted(
        (p for p in run_dir.glob("egress.*.tsv") if p.stem.split(".")[-1].isdigit()),
        key=lambda p: int(p.stem.split(".")[-1]),
    )
    return ([first] if first.exists() else []) + numbered


def read_capture(run_dir: Path) -> Capture:
    """Read every segment, counting the rows that could not be read rather than dropping them."""
    connections: list[Connection] = []
    segments: list[Segment] = []
    for path in egress_segments(run_dir):
        rows = malformed = 0
        first = last = None
        blind: list[list[float]] = []
        pending_blind = 0
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            fields = line.split("\t")
            fields += [""] * (6 - len(fields))
            try:
                epoch = float(fields[0])
            except ValueError:
                malformed += 1
                pending_blind += 1
                continue
            while pending_blind:
                # The unreadable row sat between the last readable one and this one. With no
                # readable row before it, its lower bound is the start of time.
                blind.append([last if last is not None else float("-inf"), epoch])
                pending_blind -= 1
            rows += 1
            first = epoch if first is None else min(first, epoch)
            last = epoch if last is None else max(last, epoch)
            for column, kind in ((4, "dns"), (5, "tls")):
                for host in fields[column].split(","):
                    host = host.strip().rstrip(".").lower()
                    if host:
                        connections.append(Connection(epoch, host, kind, path.name))
        while pending_blind:
            # Nothing readable followed it, so its upper bound is the end of time.
            blind.append([last if last is not None else float("-inf"), float("inf")])
            pending_blind -= 1
        segments.append(
            Segment(
                path.name,
                first or 0.0,
                last or 0.0,
                rows,
                malformed,
                tuple((low, high) for low, high in blind),
            )
        )
    connections.sort(key=lambda c: (c.epoch, c.host, c.kind))
    return Capture(tuple(connections), tuple(segments))


# ----- the trace ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """One thing the agent ran, what came back, and whether it worked.

    ``offset`` is the line the action completed on, which is the transcript's own order and the
    only ordering a trace reliably carries; several harnesses timestamp nothing.

    ``ok`` is the harness's own verdict, not a reading of the output: codex records a status and
    an exit code, claude_code marks a tool result ``is_error``, prime-agent marks it ``isError``.
    It matters because a command that failed did not read a file, and inferring that from the
    text of a traceback is guesswork where the harness has already said so. A harness that says
    nothing is taken at its word that nothing went wrong, which is the direction that leaves a
    failed read able to confirm a download; the confirmations that matter here come from
    harnesses that do report.
    """

    offset: int
    kind: str
    request: str
    result: str
    ok: bool = True
    trace: str = ""


@dataclass(frozen=True)
class Trace:
    """A transcript read for three things: what ran, where each episode starts, where it seals.

    A lease is the join key because the stream hands one to the agent with the task, so its id
    appears at the moment an episode starts and again in the ``submit_answer`` that ends it. The
    three harness shapes this reads are the three the runner launches, and a shape it does not
    recognise contributes nothing rather than contributing guesses.
    """

    path: Path
    actions: tuple[Action, ...]
    first_seen: dict[str, int] = field(default_factory=dict)
    sealed_at: dict[str, int] = field(default_factory=dict)


def _blocks_text(blocks: Any) -> str:
    if isinstance(blocks, str):
        return blocks
    if isinstance(blocks, list):
        return "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
    if isinstance(blocks, dict):
        return _blocks_text(blocks.get("content"))
    return ""


def read_trace(path: Path, leases: Iterable[str]) -> Trace:
    """Read one transcript into actions and lease marks."""
    wanted = set(leases)
    # The transcript's identity, in a name of its own. It travels on every action and is what
    # keeps one eval task's evidence out of another's, since those are separate containers with
    # separate filesystems, so nothing else in this loop may reuse the variable.
    transcript = str(path)
    actions: list[Action] = []
    first_seen: dict[str, int] = {}
    sealed_at: dict[str, int] = {}
    pending: dict[str, tuple[str, str]] = {}
    prime_args: dict[str, str] = {}

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

        # codex: completed items carry the command and its aggregated output, or an MCP call
        # with its arguments and its result.
        if kind == "item.completed" and isinstance(event.get("item"), dict):
            item = event["item"]
            if item.get("type") == "command_execution":
                exit_code = item.get("exit_code")
                actions.append(
                    Action(
                        offset,
                        "command",
                        item.get("command") or "",
                        item.get("aggregated_output") or "",
                        ok=item.get("status") != "failed" and exit_code in (0, None),
                        trace=transcript,
                    )
                )
            elif item.get("type") == "mcp_tool_call":
                arguments = item.get("arguments") or {}
                request = json.dumps(arguments)
                actions.append(
                    Action(
                        offset,
                        f"mcp:{item.get('tool')}",
                        request,
                        _blocks_text(item.get("result")),
                        ok=item.get("error") is None,
                        trace=transcript,
                    )
                )
                if item.get("tool") == "submit_answer" and arguments.get("lease") in wanted:
                    sealed_at[str(arguments["lease"])] = offset

        # claude_code: a tool_use on the assistant side, its tool_result on the user side.
        elif kind == "assistant" and isinstance(event.get("message"), dict):
            for block in event["message"].get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = str(block.get("name"))
                    arguments = block.get("input") or {}
                    pending[str(block.get("id"))] = (name, json.dumps(arguments))
                    if name.endswith("submit_answer") and arguments.get("lease") in wanted:
                        sealed_at[str(arguments["lease"])] = offset
        elif kind == "user" and isinstance(event.get("message"), dict):
            for block in event["message"].get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool, request = pending.pop(str(block.get("tool_use_id")), ("tool", ""))
                    actions.append(
                        Action(
                            offset,
                            f"tool:{tool}",
                            request,
                            _blocks_text(block.get("content")),
                            ok=not block.get("is_error"),
                            trace=transcript,
                        )
                    )

        # prime-agent: the arguments arrive when execution starts and the result when it ends.
        elif kind == "tool_execution_start":
            prime_args[str(event.get("toolCallId"))] = json.dumps(event.get("args") or {})
        elif kind == "tool_execution_end":
            request = prime_args.pop(str(event.get("toolCallId")), "") or json.dumps(
                event.get("args") or {}
            )
            result = event.get("result")
            failed = event.get("isError") or (
                isinstance(result, dict) and result.get("isError")
            )
            actions.append(
                Action(
                    offset,
                    f"tool:{event.get('toolName')}",
                    request,
                    _blocks_text(result),
                    ok=not failed,
                    trace=transcript,
                )
            )

    return Trace(path, tuple(actions), first_seen, sealed_at)


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

    def identity(self) -> dict[str, Any]:
        return {"phase": self.phase, "seq": self.seq, "task_idx": self.task_idx}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


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
    max_in_flight: int,
    traces: dict[str, Trace],
) -> list[Episode]:
    """Windows that run from a dispense to a bound on the seal, not to the next dispense.

    The stream records when a task was handed out and not when it was sealed, and above one
    lease in flight the agent can still be working an older task when the next is pulled. Two
    things bound the seal, and the tighter one wins.

    The stream itself bounds it. At capacity ``get_task`` force-seals the oldest live episode
    before dispensing, so a task is certainly sealed by the time ``max_in_flight`` further tasks
    have been handed out. That bound needs nothing but the dispense record and holds whatever
    the agent did.

    The transcript bounds it better when it can be read. The ``submit_answer`` that ends an
    episode appears in the trace at a definite place in the order, so the seal happened before
    the dispense of the first task pulled after it. For a strictly sequential agent that lands
    exactly on the next dispense; for one that interleaves it lands later, and the windows
    overlap, which is the point.
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
        return "rollout", None

    # Where each lease starts and seals in the transcripts, merged across this phase's legs.
    first_seen: dict[str, tuple[str, int]] = {}
    sealed_at: dict[str, tuple[str, int]] = {}
    for name, trace in traces.items():
        for lease, offset in trace.first_seen.items():
            first_seen.setdefault(lease, (name, offset))
        for lease, offset in trace.sealed_at.items():
            sealed_at.setdefault(lease, (name, offset))

    times = {str(d["lease"]): float(d["dispensed_at"]) for d in dispenses}
    episodes = []
    for index, dispense in enumerate(dispenses):
        lease = str(dispense["lease"])
        started = float(dispense["dispensed_at"])
        leg, leg_end = leg_of(started)

        # The stream's own bound, from the capacity rule.
        ahead = index + max(int(max_in_flight), 1)
        bounds = [dispenses[ahead]["dispensed_at"]] if ahead < len(dispenses) else []
        kind = "capacity_bound"

        # The transcript's bound, when this lease's seal can be placed in the order.
        seal = sealed_at.get(lease)
        if seal is not None:
            trace_name, offset = seal
            after = [
                times[other]
                for other, (name, where) in first_seen.items()
                if name == trace_name and where > offset and other in times
            ]
            if after:
                bounds.append(min(after))
                kind = "trace_seal_bound"
        if leg_end is not None:
            bounds.append(leg_end)
        ended = min(bounds) if bounds else None
        if ended is not None and leg_end is not None and ended >= leg_end:
            kind = kind if kind == "trace_seal_bound" else "leg_bound"

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

    When the leg record is missing, which is what an unfinished run looks like, the window falls
    back to the dispense plus the configured per-task timeout: an upper bound, labelled as one.
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
    requested: tuple[str, ...]
    covered: bool
    shared_with: tuple[dict[str, Any], ...]
    acquisition: dict[str, Any] | None
    inherited_from: dict[str, Any] | None

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
                "shared_with": list(self.shared_with),
            },
            "bucket": self.bucket,
            "reasons": list(self.reasons),
            "evidence": [c.to_json() for c in self.evidence],
            "requested": list(self.requested),
            "acquisition": self.acquisition,
            "inherited_from": self.inherited_from,
            "correct": self.episode.correct,
            "success": self.episode.success,
            "reward": self.episode.reward,
        }


LIMITS = [
    "egress observes hostnames and times, never a method, a status or a body, so no egress "
    "observation on its own can reach achieved leakage",
    "the file CDN serves the whole platform, so a client hello to it is a connection to a CDN "
    "and not a download of an answer key",
    "hostname granularity cannot separate a listing endpoint from a data endpoint on a host "
    "that serves both",
    "a local read of an already-downloaded file is invisible to the observer, so a resident "
    "answer artifact is only charged to an episode whose own commands name it",
    "egress cannot show intent, so a dataset host reached for a tokenizer and one reached for "
    "an answer key are the same observation",
    "episodes that overlap in time share one network namespace, so a connection inside more "
    "than one open window is charged to all of them and each record names its rivals",
    "a trace carries what the agent ran, so a harness whose transcript summarises rather than "
    "quotes contributes no request evidence and its episodes rest on egress alone",
    "answer content is recognised by the dataset's own columns arriving in a result, so an "
    "episode that fetched a mirror and printed a projection of it rather than its rows reads "
    "as unresolved; observed in this corpus and left for the judge rather than guessed at",
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

    def acquisitions(self) -> list[EpisodeLeakage]:
        return [e for e in self.episodes if e.acquisition is not None]

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
            "acquisitions": [e.to_json() for e in self.acquisitions()],
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


def _actions_for(
    episode: Episode, traces: dict[str, Trace], phase_dir_name: str
) -> list[Action]:
    """The actions this episode ran, cut out of its phase's transcripts by lease.

    An eval task's trace is named for its task, so the whole file is one episode. A rollout is
    one transcript for hundreds, and the lease ids cut it: an episode's actions are the ones
    between where its lease first appears and where the next lease does.
    """
    out: list[Action] = []
    for name, trace in traces.items():
        named = _TASK_TRACE.match(Path(name).name)
        if named is not None:
            if int(named.group(1)) == episode.task_idx:
                out.extend(trace.actions)
            continue
        start = trace.first_seen.get(episode.lease)
        if start is None:
            continue
        later = [o for o in trace.first_seen.values() if o > start]
        end = min(later) if later else None
        out.extend(
            a for a in trace.actions if a.offset >= start and (end is None or a.offset < end)
        )
    return out


_TASK_TRACE = re.compile(r"^task-(\d+)-leg-")


def _requested_urls(actions: Sequence[Action]) -> list[str]:
    """URLs the agent asked for, taken from command text and never from prose or output."""
    urls: list[str] = []
    for action in actions:
        if action.kind.startswith("mcp:"):
            continue
        for match in _URL.finditer(action.request):
            urls.append(_tidy_url(match.group(0)))
    return list(dict.fromkeys(urls))


def _answers_read_from_a_download(
    actions: Sequence[Action], source: AnswerSource | None
) -> list[dict[str, Any]]:
    """Files fetched from anywhere, then read back as rows of the answer key.

    The host table can never be complete. The answers are mirrored on the Hub, and they are also
    mirrored in git repositories, on personal sites, and inside PDFs, so an episode that fetched
    one of those and read the answers out of it would be invisible to a rule that starts from a
    hostname. This one starts from the content instead: something was fetched from the network
    to a named path, a later command read that path, and what came back carries the dataset's
    own answer columns. The fetch is what separates this from an agent printing its own
    reasoning as JSON, which has an answer and a rationale in it like any row does.
    """
    fetched: list[dict[str, Any]] = []
    for action in actions:
        if action.kind.startswith("mcp:") or not _URL.search(action.request):
            continue
        url = _tidy_url(_URL.search(action.request).group(0))
        for destination in download_destinations(action.request):
            fetched.append({"destination": destination, "url": url, "offset": action.offset})
    found = []
    seen: set[str] = set()
    for download in fetched:
        destination = download["destination"]
        if destination in seen:
            continue
        for action in actions:
            if (
                action.offset >= download["offset"]
                and action.ok
                and _reads_the_file(action, destination)
                and carries_answer_content(action.result, source)
            ):
                seen.add(destination)
                found.append(
                    {**download, "persistence": destination_persistence(destination)}
                )
                break
    return found


def _still_reachable(artifact: dict[str, Any], episode: Episode) -> bool:
    """Can this episode's container still open that file?

    The reach was fixed when the file landed, by where it went and which phase put it there.
    ``run`` is the rollout's HOME, which every later episode of the run runs against a copy of;
    ``phase`` is the rollout's ``/work``, shared by the rollout's legs and by nothing else;
    ``leg`` is that container's own scratch; and ``episode`` is anything an eval task saved,
    since its HOME copy and its ``/work`` are thrown away when the task ends.
    """
    reach = artifact.get("reach", "leg")
    if reach == "run":
        return True
    if reach == "phase":
        return artifact.get("phase") == episode.phase
    if reach == "leg":
        return artifact.get("phase") == episode.phase and artifact.get("leg") == episode.leg
    return False


def _answer_source_urls(actions: Sequence[Action], source: AnswerSource | None) -> list[str]:
    """The answer-source URLs these actions asked for, in the order they were asked."""
    urls = []
    for action in actions:
        for match in _URL.finditer(action.request):
            url = _tidy_url(match.group(0))
            host = (urlsplit(url).hostname or "").lower()
            if host_role(host, source).startswith("answer_source"):
                urls.append(url)
    return list(dict.fromkeys(urls))


def _asks_the_answer_source(action: Action, source: AnswerSource | None) -> bool:
    """Did this action's own command go to the answer source?

    A result is only the answer key arriving if something went and asked the answer source for
    it. The question is asked of the request, never of the result, so text that came back from
    somewhere else cannot make an episode look like it fetched a dataset.
    """
    if _answer_source_urls([action], source):
        return True
    return any(call in action.request for call in _HUB_CALLS)


def _tidy_url(url: str) -> str:
    """Cut a URL out of the shell it was quoted in, so a template still names its route."""
    for marker in ("${", "$(", "`"):
        cut = url.find(marker)
        if cut > 0:
            url = url[:cut]
    return url.rstrip(".,;:!?")


def _inherited_artifacts(run_dir: Path, manifest: dict[str, Any]) -> tuple[list[dict], list[str]]:
    """What a bookend starts with, because it starts with its source's HOME.

    A rebookend runs a new eval against the run it names, and the runner seeds it from that
    run's accumulated HOME, which every eval task then gets a copy of. So an answer file the
    source saved under HOME is on disk in this run before its first episode begins. An artifact
    the source left somewhere ephemeral is not: that path died with the source's containers.

    When the source directory is not beside this one the question cannot be answered, and the
    honest answer to a question about durable answer files is not "there were none". The caller
    floors the run at unclassified and says so.
    """
    rebookend = manifest.get("rebookend") or {}
    source_id = rebookend.get("rebookend_of")
    if not source_id:
        return [], []
    source_dir = run_dir.parent / str(source_id)
    if not source_dir.is_dir():
        return [], [
            f"this run is a bookend of {source_id}, whose directory is not beside it, so "
            "whether the source left an answer file in the HOME this run inherited cannot be "
            "checked; no episode here is cleared"
        ]
    source = classify_run(source_dir, _inherit=False)
    # Only what the source's ROLLOUT put in HOME crosses. Its ``/work`` is not copied, its
    # scratch died with its containers, and a file one of its own eval tasks saved lived in that
    # task's private copy of HOME, which the runner discards; none of those are in the HOME this
    # bookend was seeded from.
    carried = [
        {
            "destination": e.acquisition["destination"],
            "persistence": e.acquisition["persistence"],
            "reach": "run",
            "phase": None,
            "leg": None,
            "acquisition": {
                **e.acquisition["episode"],
                "run_id": source.run_id,
                "destination": e.acquisition["destination"],
                "persistence": e.acquisition["persistence"],
                "reach": "run",
            },
        }
        for e in source.acquisitions()
        if e.acquisition.get("destination")
        and e.acquisition.get("persistence") == "home"
        and e.episode.phase == "rollout"
    ]
    note = (
        [
            f"inherited {len(carried)} durable answer artifacts from {source_id}'s rollout HOME, "
            "which this run's eval tasks are copies of"
        ]
        if carried
        else []
    )
    return carried, note


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
    max_in_flight = int(cell.get("max_in_flight") or 1)
    finished = manifest.get("ended_at") is not None

    capture = read_capture(run_dir)
    legs = _legs(run_dir)
    episodes: list[Episode] = []
    traces: dict[str, dict[str, Trace]] = {}

    if (run_dir / "rollout" / "dispenses.jsonl").exists():
        leases = [str(d["lease"]) for d in _read_jsonl(run_dir / "rollout" / "dispenses.jsonl")]
        traces["rollout"] = {
            str(p): read_trace(p, leases) for p in _trace_files(run_dir, "rollout")
        }
        episodes += _rollout_episodes(run_dir, legs, max_in_flight, traces["rollout"])
    for phase in ("eval_before", "eval_after"):
        if (run_dir / phase).is_dir():
            phase_episodes = _eval_episodes(run_dir, phase, legs, timeout)
            leases = [e.lease for e in phase_episodes]
            traces[phase] = {
                str(p): read_trace(p, leases) for p in _trace_files(run_dir, phase)
            }
            episodes += phase_episodes

    inherited_artifacts: list[dict[str, Any]] = []
    inheritance_notes: list[str] = []
    unresolved_inheritance = False
    if _inherit:
        inherited_artifacts, inheritance_notes = _inherited_artifacts(run_dir, manifest)
        unresolved_inheritance = bool(inheritance_notes) and not inherited_artifacts

    notes = _capture_notes(capture, finished) + _coverage_notes(legs, episodes)
    notes += inheritance_notes
    if source is None:
        notes.append(
            f"no answer-source hosts are configured for env {env!r}, so leakage cannot be "
            "distinguished from general web reference in this run"
        )

    starts = [c.epoch for c in capture.connections]
    ordered = sorted(episodes, key=lambda e: (e.started_at if e.started_at is not None else 0.0))
    graded: list[EpisodeLeakage] = []
    # Artifacts already on disk, and where each can still be read from: an ephemeral path only
    # inside the leg that fetched it, a HOME path anywhere later in the run because the runner
    # copies HOME into every eval task, and an unknown path anywhere later because unknown is
    # not the same as gone.
    resident: list[dict[str, Any]] = list(inherited_artifacts)

    for episode in ordered:
        evidence = _window_evidence(episode, capture.connections, starts)
        actions = _actions_for(episode, traces.get(episode.phase, {}), episode.phase)
        requested = _requested_urls(actions)
        covered = capture.covers(episode.started_at, episode.ended_at)
        reasons: list[str] = []
        acquisition: dict[str, Any] | None = None
        inherited: dict[str, Any] | None = None

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

        # The egress floor. It tops out at unresolved: the observer sees a connection, never a
        # body, and the file CDN is shared by the whole platform.
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

        # What the agent asked for, read off its commands.
        for url in requested:
            host = (urlsplit(url).hostname or "").lower()
            if not host_role(host, source).startswith("answer_source"):
                continue
            bucket = _raise_to(bucket, "attempted_leakage")
            _note(reasons, "answer_source_request")
            kind = content_url_kind(url, source)
            if kind == "row_query":
                _note(reasons, "answer_rows_requested")
            elif kind == "file_download":
                _note(reasons, "file_download_requested")
        if any(call in a.request for a in actions for call in _HUB_CALLS):
            bucket = _raise_to(bucket, "attempted_leakage")
            _note(reasons, "hub_download_call")

        # What came back, from the command that went and asked for it. The answer columns have
        # to arrive in the result of an action whose own request reached the answer source:
        # otherwise an agent printing its own reasoning as JSON, which has an answer and a
        # rationale in it like any row does, would read as the answer key arriving.
        fetched_answers = [
            a
            for a in actions
            if not a.kind.startswith("mcp:")
            and _asks_the_answer_source(a, source)
            and carries_answer_content(a.result, source)
        ]
        if fetched_answers:
            bucket = _raise_to(bucket, "achieved_leakage")
            _note(reasons, "answer_content_in_result")
            acquisition = {
                "kind": "answer_content_in_result",
                "episode": episode.identity(),
                "epoch": evidence[0].epoch if evidence else episode.started_at,
                "destination": None,
                "persistence": "response",
                "requested": _answer_source_urls(fetched_answers, source),
            }

        # A download the filesystem then answered for. The destination has to come back in a
        # result, because a command naming a path is a request and not a file.
        session = [
            a for trace in traces.get(episode.phase, {}).values() for a in trace.actions
        ]
        landed = _completed_downloads(actions, session, source)
        if landed:
            bucket = _raise_to(bucket, "achieved_leakage")
            _note(reasons, "file_download_landed")
            if acquisition is None:
                acquisition = {
                    "kind": "file_download_landed",
                    "episode": episode.identity(),
                    "epoch": evidence[0].epoch if evidence else episode.started_at,
                    "destination": landed[0]["destination"],
                    "persistence": landed[0]["persistence"],
                    "reach": reach_of(landed[0]["persistence"], episode.phase),
                    "requested": [d["url"] for d in landed],
                }
        elif "file_download_requested" in reasons:
            bucket = _raise_to(bucket, "unresolved_leakage")
            _note(reasons, "file_download_unconfirmed")

        # A file fetched from anywhere at all, read back, and carrying the answer columns. This
        # is the one rule that does not begin with the host table, because the answers are
        # mirrored in more places than a table can name.
        read_back = _answers_read_from_a_download(actions, source)
        if read_back:
            bucket = _raise_to(bucket, "achieved_leakage")
            _note(reasons, "answer_content_read_from_download")
            if acquisition is None:
                persistence = destination_persistence(read_back[0]["destination"])
                acquisition = {
                    "kind": "answer_content_read_from_download",
                    "episode": episode.identity(),
                    "epoch": evidence[0].epoch if evidence else episode.started_at,
                    "destination": read_back[0]["destination"],
                    "persistence": persistence,
                    "reach": reach_of(persistence, episode.phase),
                    "requested": [d["url"] for d in read_back],
                }
            landed = landed or read_back

        # An artifact fetched earlier and still reachable here. Reading it is achieved; having
        # it within reach and no command naming it is unresolved, because a local read leaves
        # no trace the observer can see and this instrument will not guess either way.
        reachable = [r for r in resident if _still_reachable(r, episode)]
        if reachable and acquisition is None:
            read = [
                r
                for r in reachable
                if any(
                    a.ok and (r["destination"] in a.request or r["destination"] in a.result)
                    for a in actions
                )
            ]
            if read:
                bucket = _raise_to(bucket, "achieved_leakage")
                _note(reasons, "resident_artifact_read")
                inherited = read[0]["acquisition"]
            else:
                bucket = _raise_to(bucket, "unresolved_leakage")
                _note(reasons, "resident_artifact_available")
                inherited = reachable[0]["acquisition"]

        for landing in landed:
            reach = reach_of(landing["persistence"], episode.phase)
            resident.append(
                {
                    "destination": landing["destination"],
                    "persistence": landing["persistence"],
                    "reach": reach,
                    "phase": episode.phase,
                    "leg": episode.leg,
                    "acquisition": {
                        **episode.identity(),
                        "destination": landing["destination"],
                        "persistence": landing["persistence"],
                        "reach": reach,
                    },
                }
            )

        graded.append(
            EpisodeLeakage(
                episode=episode,
                bucket=bucket,
                reasons=tuple(reasons),
                evidence=tuple(evidence),
                requested=tuple(requested),
                covered=covered,
                shared_with=(),
                acquisition=acquisition,
                inherited_from=inherited,
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


# Phrases a shell prints when the file is not there. They only ever withhold a confirmation,
# never grant one, so an unrecognised failure leaves the download resting on the later-read rule
# rather than being waved through by this list.
_MISSING_FILE = ("no such file", "not found", "cannot access", "cannot open", "does not exist")


def _missing_file(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _MISSING_FILE)


# Commands that answer for a file by reading it or by asking the filesystem about it. The list
# grants confirmation and nothing else does, so an operation nobody thought of leaves a download
# unresolved rather than promoting it. That is the direction to be wrong in: the alternative is
# a blacklist, under which ``rm -f`` and ``echo`` both prove a file exists by naming it.
_READ_COMMANDS = (
    "cat", "head", "tail", "less", "more", "od", "xxd", "strings", "file", "du", "ls", "stat",
    "wc", "md5sum", "sha1sum", "sha256sum", "cksum", "grep", "rg", "zgrep", "zcat", "gunzip",
    "unzip", "tar", "jq", "awk", "sed", "sort", "uniq", "python", "python3", "duckdb", "sqlite3",
)
# The same, for calls inside a program rather than words in a shell.
_READ_CALLS = (
    "read_parquet", "read_table", "read_csv", "read_json", "ParquetFile", "load_dataset",
    "np.load", "json.load", "open(", "Path(", "readlines", "getsize", "st_size",
)
_READ_WORD = re.compile(r"\b(?:" + "|".join(_READ_COMMANDS) + r")\b")
# A size beside a path is the filesystem answering: du, ls -l, stat and wc all print one.
_SIZE = re.compile(r"(?:^|\s)\d+(?:[.,]\d+)?\s?[KMGT]?i?B?(?=\s|$)")


def _reads_the_file(action: Action, destination: str) -> bool:
    """Does this action ask the filesystem for that file, rather than merely mention it?"""
    if destination not in action.request:
        return False
    return bool(_READ_WORD.search(action.request)) or any(
        call in action.request for call in _READ_CALLS
    )


def _filesystem_answered(text: str, destination: str) -> bool:
    """Does this output carry the file's own size beside its path, as ``du`` and ``ls`` do?"""
    if _missing_file(text):
        return False
    return any(destination in line and _SIZE.search(line) for line in text.splitlines())


def _completed_downloads(
    actions: Sequence[Action], later: Sequence[Action], source: AnswerSource | None
) -> list[dict[str, Any]]:
    """Downloads of an answer-source file that the filesystem afterwards answered for.

    A command naming a destination says where the agent meant to put a body; it does not say one
    arrived. What says so is a later action that succeeded and names that path: a size, a
    listing, a checksum, or a parser reading it. Success is the harness's own verdict, so a read
    that raised is not a read, and a download whose file never appeared cannot borrow the
    traceback that proves it never appeared.

    ``later`` is the whole transcript this episode's actions came from, because the read that
    confirms a download can land in a following episode of the same session.
    """
    downloads = []
    for action in actions:
        if action.kind.startswith("mcp:"):
            continue
        urls = [
            url
            for url in (_tidy_url(m.group(0)) for m in _URL.finditer(action.request))
            if content_url_kind(url, source) == "file_download"
            and host_role((urlsplit(url).hostname or "").lower(), source).startswith(
                "answer_source"
            )
        ]
        if not urls:
            continue
        for destination in download_destinations(action.request):
            downloads.append(
                {
                    "destination": destination,
                    "url": urls[0],
                    "offset": action.offset,
                    "trace": action.trace,
                }
            )

    landed = []
    seen: set[str] = set()
    for download in downloads:
        destination = download["destination"]
        if destination in seen:
            continue
        # Same transcript only: offsets are per file, and one eval task's read says nothing
        # about another task's container.
        session = [a for a in later if a.trace == download["trace"]]
        # The filesystem reporting the file's size, from any action including the download's
        # own compound command, which is where a ``du`` on the next line lands.
        confirmed = any(
            a.offset >= download["offset"] and _filesystem_answered(a.result, destination)
            for a in session
        )
        # Or a later command that reads the file and got something back. Succeeding while
        # naming the path is not enough: ``rm -f`` and ``echo`` both do that, and a file that
        # was never written is exactly what a cleanup names.
        confirmed = confirmed or any(
            a.ok
            and a.offset > download["offset"]
            and a.result.strip()
            and _reads_the_file(a, destination)
            and not _missing_file(a.result)
            for a in session
        )
        if confirmed:
            seen.add(destination)
            landed.append({**download, "persistence": destination_persistence(destination)})
    return landed


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
    """Name, per episode, the other episodes that could equally own this episode's evidence.

    Overlapping episodes share one network namespace, so a connection inside two open windows
    belongs to both as far as the observer is concerned. Rather than pick one, the connection is
    charged to every window containing it and the rivals travel with the record by identity, so
    a reader can go and look at them.

    Rivals are named per connection rather than per window: two windows that overlap somewhere
    the traffic is not create no ambiguity about that traffic, and an episode that lists rivals
    it does not really have is as misleading as one that hides the rivals it does.
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
                requested=row.requested,
                covered=row.covered,
                shared_with=rivals,
                acquisition=row.acquisition,
                inherited_from=row.inherited_from,
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

    citations = [(run, e) for run in runs for e in run.acquisitions()]
    if citations:
        lines += ["", "achieved-leakage acquisitions, all of them"]
        for run, row in citations:
            acquisition = row.acquisition or {}
            rivals = len(row.shared_with)
            lines.append(
                f"  {run.label}  {row.episode.label}  {acquisition.get('kind')}  "
                f"dest={acquisition.get('destination')} "
                f"({acquisition.get('persistence')})"
                + (f"  window shared with {rivals}" if rivals else "")
            )
            for url in acquisition.get("requested") or []:
                lines.append(f"      {url}")

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

    targets = [d for d in args.run_dirs if d.is_dir()]
    if not targets:
        print("no run directories given")
        return 1
    refused = [d for d in targets if not _finished(d)]
    if refused and not args.allow_unfinished:
        for run_dir in refused:
            print(
                f"refusing {run_dir}: its manifest has no ended_at, so the run was still going "
                "and the egress record cannot be complete. Pass --allow-unfinished to grade it "
                "anyway, where every episode is unclassified rather than clean."
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
    "Action",
    "AnswerSource",
    "Capture",
    "Connection",
    "Episode",
    "EpisodeLeakage",
    "RunLeakage",
    "Segment",
    "Trace",
    "carries_answer_content",
    "classify_run",
    "content_url_kind",
    "destination_persistence",
    "download_destinations",
    "host_role",
    "main",
    "read_capture",
    "read_trace",
    "render_table",
]
