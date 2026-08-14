"""Per-episode leakage classification, derived from the run's egress record.

Some v0 cells answer questions whose answers are published. The agent has an unsandboxed shell
and open egress by design, and the runner observes rather than gates, so an episode can be won
by reasoning or by downloading the answer key, and the reward is the same number either way. A
blended correct-rate over both is not a measurement of anything. This grades every episode into
one of four buckets so the rate can be reported per bucket instead:

    computed_locally        nothing left the cell but the harness's own infrastructure
    general_web_reference   the cell went out, but not to a host that distributes the answers
    attempted_leakage       the cell reached the answer source and content is not established
    achieved_leakage        answer content was obtained

The evidence of record is ``egress.tsv``: one row per outbound DNS question or TLS client hello
observed in the cell's network namespace, written outside the container by an observer the agent
holds no mount of and cannot edit. That is the floor, and nothing the agent wrote about itself
moves it down.

What egress can and cannot establish, stated plainly because the buckets are only as good as
this:

- It sees hostnames and times. It never sees payloads, because TLS is not decrypted. "Connected
  to a host that distributes the answers" is therefore not by itself proof that answers were
  obtained, which is exactly why ``attempted_leakage`` exists as a bucket rather than being
  rounded up.
- Hostname granularity blurs endpoints. ``huggingface.co`` serves listings, metadata and file
  redirects over one name, and ``datasets-server.huggingface.co`` serves both ``/splits``
  metadata and ``/rows`` payloads. Only the file CDN (``*.cdn.hf.co``, ``cdn-lfs*``) is a
  hostname that exists to move file content and nothing else, so only that hostname carries
  achieved leakage on egress alone.
- It cannot see a local read. Once a dataset file is on the container's disk, every later
  episode in that same container can consult it silently, so achieved leakage propagates
  forward within a leg (see ``answer_source_resident``) rather than being charged to one
  episode.
- It cannot show intent. Reaching a dataset host to pull a tokenizer and reaching it to pull an
  answer key look identical. The bucket says what the reward can be trusted to mean, not what
  the agent meant.

The trace refines but never overrides. An agent's own transcript shows the endpoint it asked
for, which egress cannot; the two together are stronger than either. So a trace URL can raise an
episode's bucket and can never lower it, and the one refinement that reaches ``achieved_leakage``
requires both halves: the trace naming a content endpoint, and egress showing a TLS connection to
that endpoint's host inside the episode's window.

Usage::

    shobench leakage runs/hle-codex-gpt-56-terra-20260813T215942Z
    shobench leakage runs/hle-* --format json > leakage.json
"""

from __future__ import annotations

import argparse
import json
import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "shobench.leakage/1"

# The four buckets, weakest evidence first. Refinement only ever moves an episode later in this
# tuple, which is what "the egress floor is a floor" means in code.
BUCKETS = (
    "computed_locally",
    "general_web_reference",
    "attempted_leakage",
    "achieved_leakage",
)

# Not a bucket: the label for an episode whose evidence is missing rather than empty. An episode
# with no egress record and no window is not a clean episode, and calling it computed_locally
# would be the same silent pass a fail-closed judge refuses to give.
UNCLASSIFIED = "unclassified"

PHASES = ("eval_before", "rollout", "eval_after")

# Hosts a cell talks to because of how it is run rather than because of what it is answering:
# the harness's model API and telemetry, and the package registries a tool install goes through.
# Traffic here is evidence of neither honesty nor leakage, so it is recorded and set aside.
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

    ``index`` hosts serve listings, metadata and payloads over the same name, so reaching one
    proves the cell went to the answer source and proves nothing about content. ``content``
    hosts exist to move file bodies and serve nothing else, so a TLS connection to one is the
    hard signal. ``rows`` names the hosts running a query API over the dataset itself, whose
    answers-bearing endpoints share a hostname with its metadata endpoints and whose refusals
    look exactly like its successes from outside the TLS session.
    """

    index: tuple[str, ...]
    content: tuple[str, ...]
    rows: tuple[str, ...] = ()


# Keyed by environment, because "the host that distributes the answers" is a fact about the
# dataset, not about the runner. An environment with no entry gets no answer-source table at all
# and every classification for it says so, rather than quietly reporting a clean run.
#
# hle's questions and answers are the `cais/hle` HuggingFace dataset and the unofficial mirrors
# of it that live on the same Hub, so the Hub's families are the table.
ANSWER_SOURCES: dict[str, AnswerSource] = {
    "hle": AnswerSource(
        index=(
            "huggingface.co",
            "www.huggingface.co",
            "hf.co",
            "www.hf.co",
            "datasets-server.huggingface.co",
            "hf-mirror.com",
            "*.hf-mirror.com",
        ),
        content=(
            "*.cdn.hf.co",
            "cdn-lfs*.hf.co",
            "cdn-lfs*.huggingface.co",
        ),
        rows=("datasets-server.huggingface.co",),
    ),
}

# Endpoints on the dataset host that return rows rather than a description of rows.
_ROW_ENDPOINTS = ("/rows", "/search", "/filter", "/first-rows")
# Extensions a dataset is shipped as. A URL ending in one of these is a file body by definition.
_ARTIFACT_SUFFIXES = (".parquet", ".csv", ".tsv", ".jsonl", ".arrow", ".zip", ".gz")

_URL = re.compile(r"https?://[^\s\"'\\<>)\]}]+")


def _tidy_url(url: str) -> str:
    """Cut a URL out of the prose and the shell it was quoted in.

    A transcript quotes URLs inside sentences and inside heredocs, so what the pattern matches
    can end in a sentence's punctuation or carry an unexpanded shell variable. Both are trimmed:
    the trimmed form still names the host and the route, which is all the classification reads.
    """
    for marker in ("${", "$(", "`"):
        cut = url.find(marker)
        if cut > 0:
            url = url[:cut]
    return url.rstrip(".,;:!?")


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
        if _matches(host, source.index):
            return "answer_source_index"
    return "general"


def content_url_kind(url: str, source: AnswerSource | None) -> str | None:
    """What a URL asks the answer source for, when it asks for the data rather than a listing.

    ``file_download`` is the Hub's ``resolve`` route or any URL ending in a dataset artifact
    extension: a request for a file body, which on this Hub is served by a redirect to a CDN
    hostname that does nothing else, so the observer can corroborate whether the body moved.

    ``row_query`` is the dataset server's row endpoints, which return the rows themselves rather
    than a description of them. It gets a name of its own because it cannot be corroborated: the
    request and the refusal ride the same hostname and the same TLS session, and the run that
    prompted all this queried the gated ``cais/hle`` this way and was turned down. Confirming a
    row query means reading the response, which is the model-judge half of the problem and not
    this half.
    """
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    host = (parts.hostname or "").lower()
    if "/resolve/" in parts.path:
        return "file_download"
    # A blob route renders the file in a page rather than serving it, so a data extension under
    # one is a link someone was reading, not a download. Every other route that ends in a data
    # extension is the file itself.
    if "/blob/" not in parts.path and path.lower().endswith(_ARTIFACT_SUFFIXES):
        return "file_download"
    rows = source.rows if source is not None else ()
    if _matches(host, rows) and path in _ROW_ENDPOINTS and parts.query:
        return "row_query"
    return None


@dataclass(frozen=True)
class Connection:
    """One observed outbound name: a DNS question, or a TLS client hello carrying an SNI.

    The kinds are not equivalent evidence. A resolution says a name was looked up; a client
    hello says a connection to that name was opened and bytes moved. Achieved leakage needs the
    second.
    """

    epoch: float
    host: str
    kind: str
    segment: str

    def to_json(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "host": self.host, "kind": self.kind, "segment": self.segment}


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
        return f"{seq}task {self.task_idx}"


@dataclass(frozen=True)
class EpisodeLeakage:
    """One episode's bucket, and every fact that put it there."""

    episode: Episode
    bucket: str
    reasons: tuple[str, ...]
    evidence: tuple[Connection, ...]
    trace_urls: tuple[str, ...]
    shared_window_with: int
    acquisition: dict[str, Any] | None

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
                "shared_with": self.shared_window_with,
            },
            "bucket": self.bucket,
            "reasons": list(self.reasons),
            "evidence": [c.to_json() for c in self.evidence],
            "trace_urls": list(self.trace_urls),
            "acquisition": self.acquisition,
            "correct": self.episode.correct,
            "success": self.episode.success,
            "reward": self.episode.reward,
        }


@dataclass(frozen=True)
class RunLeakage:
    """One run directory's episodes, the counts a reader wants, and the caveats they need."""

    run_dir: Path
    run_id: str
    cell: str
    env: str
    harness: str
    model: str
    egress_available: bool
    egress_segments: tuple[str, ...]
    observations: int
    answer_source_configured: bool
    episodes: tuple[EpisodeLeakage, ...]
    notes: tuple[str, ...]

    @property
    def label(self) -> str:
        """The directory, not the run id.

        A repair or a hand copy of a run directory carries the original's manifest, so two
        directories can answer to one run id and a table keyed on the id prints them as one row
        twice. The path is what the reader passed in and what they can go back to.
        """
        return self.run_dir.name

    def phases(self) -> list[str]:
        seen = [e.episode.phase for e in self.episodes]
        return [p for p in PHASES if p in seen] + sorted({p for p in seen if p not in PHASES})

    def counts(self, phase: str | None = None) -> dict[str, int]:
        rows = [e for e in self.episodes if phase is None or e.episode.phase == phase]
        counts = {bucket: 0 for bucket in (*BUCKETS, UNCLASSIFIED)}
        for row in rows:
            counts[row.bucket] += 1
        return counts

    def correct_rate(self, bucket: str, phase: str | None = None) -> tuple[int, int]:
        """Graded episodes in this bucket, and how many were correct.

        Two numbers rather than a rate, because a bucket with three episodes in it and a bucket
        with three hundred should not print the same way.
        """
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

    def unresolved(self, phase: str | None = None) -> list[EpisodeLeakage]:
        """Episodes this half cannot finish, which the deferred model judge would take.

        An episode that queried the dataset's row API asked for answers over a hostname whose
        refusals are indistinguishable from its successes. Only the response settles it, and
        reading responses is the judge's job, so these are named rather than guessed at.
        """
        return [
            e
            for e in self.episodes
            if "answer_rows_requested" in e.reasons
            and e.bucket != "achieved_leakage"
            and (phase is None or e.episode.phase == phase)
        ]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "run_dir": str(self.run_dir),
            "run_id": self.run_id,
            "cell": self.cell,
            "env": self.env,
            "harness": self.harness,
            "model": self.model,
            "egress": {
                "available": self.egress_available,
                "segments": list(self.egress_segments),
                "observations": self.observations,
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
                    "unresolved_by_egress": len(self.unresolved(phase)),
                }
                for phase in self.phases()
            ],
            "acquisitions": [e.to_json() for e in self.acquisitions()],
            "notes": list(self.notes),
            "limits": LIMITS,
            "episodes": [e.to_json() for e in self.episodes],
        }


LIMITS = [
    "egress observes hostnames and times, never payloads, so a connection to a host that "
    "distributes the answers is not by itself proof that answers were obtained",
    "hostname granularity cannot separate a listing endpoint from a content endpoint on a host "
    "that serves both, so only the file CDN carries achieved leakage on egress alone",
    "a read of an already-downloaded file is invisible to egress, so achieved leakage carries "
    "forward to later episodes in the same container rather than being charged to one episode",
    "egress cannot show intent, so a dataset host reached for a tokenizer and one reached for "
    "an answer key are the same observation",
    "episodes that overlap in time share one network namespace, so a connection inside more "
    "than one open window is charged to all of them and flagged as shared",
    "a URL in a transcript is a mention, which may be a plan the agent never ran, so trace "
    "evidence alone never reaches achieved leakage and only ever raises an episode to attempted",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def egress_segments(run_dir: Path) -> list[Path]:
    """The capture's segments in the order they were written.

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


def read_egress(run_dir: Path) -> list[Connection]:
    """Fold every segment into one time-ordered list of observed names."""
    out: list[Connection] = []
    for path in egress_segments(run_dir):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            fields = line.split("\t")
            fields += [""] * (6 - len(fields))
            try:
                epoch = float(fields[0])
            except ValueError:
                continue
            for column, kind in ((4, "dns"), (5, "tls")):
                for host in fields[column].split(","):
                    host = host.strip().rstrip(".").lower()
                    if host:
                        out.append(Connection(epoch, host, kind, path.name))
    out.sort(key=lambda c: (c.epoch, c.host, c.kind))
    return out


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


def _rollout_episodes(run_dir: Path, legs: list[dict[str, Any]]) -> list[Episode]:
    """Windows from the dispense record, since a rollout's episodes are not separately timed.

    The stream records when each task was handed out and not when it was sealed, so an episode's
    window runs from its own dispense to the next one, which charges a connection to the most
    recently pulled task. A harness holding several leases at once can still be working an older
    task when the next is pulled, and the window model will charge that traffic to the newer
    one; the run's ``max_in_flight`` is the size of that error and the shared-window count is
    where it shows up in the output.
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

    episodes = []
    for index, dispense in enumerate(dispenses):
        started = float(dispense["dispensed_at"])
        leg, leg_end = leg_of(started)
        following = dispenses[index + 1]["dispensed_at"] if index + 1 < len(dispenses) else None
        candidates = [x for x in (following, leg_end) if x is not None]
        ended = min(candidates) if candidates else None
        correct, success, reward = _outcome(results.get(dispense["lease"]))
        episodes.append(
            Episode(
                phase="rollout",
                task_idx=int(dispense["task_idx"]),
                seq=int(dispense["seq"]),
                lease=str(dispense["lease"]),
                leg=leg,
                started_at=started,
                ended_at=ended,
                window_kind="dispense_interval",
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

    Each held-out task runs in a leg of its own with a recorded start and end, so the window is
    exact. When the leg record is missing, which is what an unfinished run looks like, the
    window falls back to the dispense plus the configured per-task timeout: an upper bound, and
    labelled as one.
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


def read_episodes(run_dir: Path, task_timeout_s: float = 900.0) -> list[Episode]:
    legs = _legs(run_dir)
    episodes: list[Episode] = []
    if (run_dir / "rollout" / "dispenses.jsonl").exists():
        episodes += _rollout_episodes(run_dir, legs)
    for phase in ("eval_before", "eval_after"):
        if (run_dir / phase).is_dir():
            episodes += _eval_episodes(run_dir, phase, legs, task_timeout_s)
    return episodes


def _trace_files(run_dir: Path, phase: str) -> list[Path]:
    traces = run_dir / phase / "traces"
    return sorted(traces.glob("*.stream.jsonl")) if traces.is_dir() else []


_TASK_TRACE = re.compile(r"^task-(\d+)-leg-")


def trace_urls_by_lease(run_dir: Path, episodes: Sequence[Episode]) -> dict[str, list[str]]:
    """Collect the URLs each episode's slice of the transcript names, keyed by lease.

    An eval task runs in a leg of its own and its trace file says which task in its name, so
    that whole file is one episode and no splitting is needed. A rollout is one transcript for
    hundreds of episodes, and the lease is what splits it: the stream hands the agent a lease
    with the task, so the id appears in the transcript at the moment that episode starts and the
    next id marks where it ends. That is ordering rather than timing, and it is exact, which is
    what a trace is good for. A harness that summarises its transcript instead of quoting it
    will not carry every lease, and the leases that cannot be found contribute no trace evidence
    rather than borrowing another episode's.
    """
    found: dict[str, list[str]] = {}
    by_phase: dict[str, list[Episode]] = {}
    for episode in episodes:
        by_phase.setdefault(episode.phase, []).append(episode)
    for phase, phase_episodes in by_phase.items():
        leases = {e.lease for e in phase_episodes}
        by_task = {e.task_idx: e.lease for e in phase_episodes}
        for path in _trace_files(run_dir, phase):
            text = path.read_text(encoding="utf-8", errors="ignore")
            named = _TASK_TRACE.match(path.name)
            if named is not None and int(named.group(1)) in by_task:
                lease = by_task[int(named.group(1))]
                found.setdefault(lease, []).extend(
                    _tidy_url(m.group(0)) for m in _URL.finditer(text)
                )
                continue
            offsets = sorted(
                (text.find(lease), lease) for lease in leases if text.find(lease) >= 0
            )
            if not offsets:
                continue
            starts = [offset for offset, _ in offsets]
            for match in _URL.finditer(text):
                index = bisect_right(starts, match.start()) - 1
                if index < 0:
                    continue
                found.setdefault(offsets[index][1], []).append(_tidy_url(match.group(0)))
    return {lease: sorted(dict.fromkeys(urls)) for lease, urls in found.items()}


def _coverage_notes(legs: list[dict[str, Any]], episodes: Sequence[Episode]) -> list[str]:
    """What the run ran that this classification has no episode for, said out loud.

    An eval leg that started but never got a task dispensed leaves an empty task directory, and
    an episode list built from the dispense records will not mention it at all. That is the
    silent gap a classifier should not have: a phase whose legs outnumber its episodes is
    reported as unaccounted for rather than as a phase with fewer episodes.
    """
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
            legs_word = "leg" if missing == 1 else "legs"
            notes.append(
                f"{phase} has {missing} {legs_word} whose task was never dispensed, so they have "
                "no episode here and this phase's counts are over what it did dispense"
            )
    for phase, count in sorted(bounded.items()):
        notes.append(
            f"{phase} has {count} episodes with no leg record, so their windows are the dispense "
            "plus the configured task timeout: an upper bound, which over-attributes traffic"
        )
    return notes


def _note(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _rank(bucket: str) -> int:
    return BUCKETS.index(bucket) if bucket in BUCKETS else -1


def _raise_to(current: str, candidate: str) -> str:
    return candidate if _rank(candidate) > _rank(current) else current


def _window_evidence(
    episode: Episode, connections: Sequence[Connection], starts: Sequence[float]
) -> list[Connection]:
    """Every connection observed inside this episode's window, which is half-open.

    Half-open because consecutive rollout windows meet at a dispense: a connection landing on
    that instant belongs to the episode being pulled, not to the one just finished.
    """
    if episode.started_at is None:
        return []
    end = episode.ended_at if episode.ended_at is not None else float("inf")
    out = []
    for connection in connections[bisect_left(starts, episode.started_at) :]:
        if connection.epoch >= end:
            break
        out.append(connection)
    return out


def classify_run(run_dir: Path) -> RunLeakage:
    """Grade every episode in a completed run directory."""
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

    connections = read_egress(run_dir)
    segments = tuple(p.name for p in egress_segments(run_dir))
    episodes = read_episodes(run_dir, task_timeout_s=timeout)
    urls = trace_urls_by_lease(run_dir, episodes)

    notes: list[str] = []
    if not segments:
        notes.append(
            "no egress record in this run directory, so no episode can be cleared; every "
            "episode is unclassified rather than clean"
        )
    if source is None:
        notes.append(
            f"no answer-source hosts are configured for env {env!r}, so leakage cannot be "
            "distinguished from general web reference in this run"
        )
    notes += _coverage_notes(_legs(run_dir), episodes)

    starts = [c.epoch for c in connections]
    graded: list[EpisodeLeakage] = []
    # Acquisitions carry forward inside a container and no further. A rollout is one container
    # for all its episodes, so a file fetched at task 80 is on disk for tasks 81 onward; every
    # eval task is a fresh container, so nothing carries between them.
    acquired_at: dict[str, tuple[float, dict[str, Any]]] = {}
    ordered = sorted(episodes, key=lambda e: (e.started_at if e.started_at is not None else 0.0))

    for episode in ordered:
        evidence = _window_evidence(episode, connections, starts)
        episode_urls = tuple(urls.get(episode.lease, ()))
        reasons: list[str] = []
        acquisition: dict[str, Any] | None = None

        if not segments or episode.started_at is None:
            bucket = UNCLASSIFIED
            reasons.append("no_egress_record" if not segments else "no_window")
        else:
            bucket = "computed_locally"
            roles = [(c, host_role(c.host, source)) for c in evidence]
            if any(role == "general" for _, role in roles):
                bucket = _raise_to(bucket, "general_web_reference")
                reasons.append("general_web_host")
            if any(role.startswith("answer_source") for _, role in roles):
                bucket = _raise_to(bucket, "attempted_leakage")
                reasons.append("answer_source_host")
            content = [
                c for c, role in roles if role == "answer_source_content" and c.kind == "tls"
            ]
            if content:
                bucket = _raise_to(bucket, "achieved_leakage")
                reasons.append("answer_source_content_host")
                acquisition = {
                    "kind": "answer_source_content_host",
                    "host": content[0].host,
                    "epoch": content[0].epoch,
                    "requested": [],
                }

        # The trace, which can only raise. An answer-source URL is a reach, and that alone is an
        # attempt. A file-download URL beside a content-host connection in the same window is the
        # two halves of an obtainment, the file named by the command and the transfer seen by the
        # observer, and it is also where the citation for an acquisition comes from.
        fetched = [
            c
            for c in evidence
            if c.kind == "tls" and host_role(c.host, source) == "answer_source_content"
        ]
        for url in episode_urls:
            host = (urlsplit(url).hostname or "").lower()
            if not host_role(host, source).startswith("answer_source"):
                continue
            bucket = _raise_to(bucket, "attempted_leakage")
            _note(reasons, "answer_source_url")
            kind = content_url_kind(url, source)
            if kind == "row_query":
                _note(reasons, "answer_rows_requested")
            elif kind == "file_download" and fetched:
                bucket = _raise_to(bucket, "achieved_leakage")
                _note(reasons, "file_download_confirmed_by_egress")
                if acquisition is None:
                    acquisition = {
                        "kind": "file_download_confirmed_by_egress",
                        "host": fetched[0].host,
                        "epoch": fetched[0].epoch,
                        "requested": [],
                    }
                acquisition["kind"] = "file_download_confirmed_by_egress"
                # Every file this episode asked the answer source for, not one of them. The
                # observer saw a body move and cannot say which request carried it, and an
                # episode that asked for a gated file and an ungated mirror of it asked twice.
                requested = acquisition["requested"]
                if isinstance(requested, list) and url not in requested:
                    requested.append(url)
            elif kind == "file_download":
                _note(reasons, "file_download_unconfirmed")

        prior = acquired_at.get(episode.leg)
        if (
            acquisition is None
            and prior is not None
            and episode.started_at is not None
            and episode.started_at >= prior[0]
            and bucket != UNCLASSIFIED
        ):
            bucket = _raise_to(bucket, "achieved_leakage")
            reasons.append("answer_source_resident")
        if acquisition is not None:
            existing = acquired_at.get(episode.leg)
            if existing is None or acquisition["epoch"] < existing[0]:
                acquired_at[episode.leg] = (float(acquisition["epoch"]), acquisition)

        graded.append(
            EpisodeLeakage(
                episode=episode,
                bucket=bucket,
                reasons=tuple(reasons),
                evidence=tuple(evidence),
                trace_urls=episode_urls,
                shared_window_with=0,
                acquisition=acquisition,
            )
        )

    graded = _mark_shared_windows(graded)
    graded.sort(key=lambda g: (PHASES.index(g.episode.phase) if g.episode.phase in PHASES else 9,
                               g.episode.seq if g.episode.seq is not None else g.episode.task_idx))
    return RunLeakage(
        run_dir=run_dir,
        run_id=str(manifest.get("run_id") or run_dir.name),
        cell=str(cell.get("name") or ""),
        env=env,
        harness=str(cell.get("harness") or ""),
        model=str(cell.get("model") or ""),
        egress_available=bool(segments),
        egress_segments=segments,
        observations=len(connections),
        answer_source_configured=source is not None,
        episodes=tuple(graded),
        notes=tuple(notes),
    )


def _mark_shared_windows(graded: Sequence[EpisodeLeakage]) -> list[EpisodeLeakage]:
    """Count, per episode, how many other episodes were open at the same time.

    Overlapping episodes share one network namespace, so a connection inside two open windows
    belongs to both as far as the observer is concerned. Rather than pick one, the connection is
    charged to every window that contains it and the count of rivals travels with the record.
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
        start, end = windows[index]
        if start is None or not row.evidence:
            out.append(row)
            continue
        rivals = sum(
            1
            for other, (s, e) in enumerate(windows)
            if other != index and s is not None and s < end and e > start
        )
        out.append(
            EpisodeLeakage(
                episode=row.episode,
                bucket=row.bucket,
                reasons=row.reasons,
                evidence=row.evidence,
                trace_urls=row.trace_urls,
                shared_window_with=rivals,
                acquisition=row.acquisition,
            )
        )
    return out


MAX_CITATIONS = 10


def render_table(runs: Sequence[RunLeakage]) -> str:
    """One row per run, phase and bucket, because a blended row is the thing being avoided."""
    header = ("run", "phase", "bucket", "episodes", "correct", "rate", "judge")
    rows: list[tuple[str, ...]] = []
    for run in runs:
        for phase in run.phases():
            counts = run.counts(phase)
            unresolved = {id(e) for e in run.unresolved(phase)}
            for bucket in (*BUCKETS, UNCLASSIFIED):
                if not counts[bucket]:
                    continue
                correct, graded = run.correct_rate(bucket, phase)
                waiting = sum(
                    1
                    for e in run.episodes
                    if e.episode.phase == phase and e.bucket == bucket and id(e) in unresolved
                )
                rows.append(
                    (
                        run.label,
                        phase,
                        bucket,
                        str(counts[bucket]),
                        f"{correct}/{graded}",
                        f"{correct / graded:.3f}" if graded else "-",
                        str(waiting) if waiting else "-",
                    )
                )
        if not run.episodes:
            rows.append((run.label, "-", "no episodes recorded", "0", "-", "-", "-"))
    widths = [max(len(str(c)) for c in column) for column in zip(header, *rows, strict=True)]
    lines = [
        "  ".join(str(c).ljust(w) for c, w in zip(header, widths, strict=True)),
        "  ".join("-" * w for w in widths),
    ]
    lines += [
        "  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=True)) for row in rows
    ]

    citations = [(run, e) for run in runs for e in run.acquisitions()]
    if citations:
        lines += ["", "achieved-leakage acquisitions"]
        for run, row in citations[:MAX_CITATIONS]:
            acquisition = row.acquisition or {}
            rivals = row.shared_window_with
            lines.append(
                f"  {run.label}  {row.episode.phase}  {row.episode.label}  "
                f"{acquisition.get('epoch')}  {acquisition.get('host')}  "
                f"{acquisition.get('kind')}"
                + (f"  (window shared with {rivals})" if rivals else "")
            )
            for url in acquisition.get("requested") or []:
                lines.append(f"      {url}")
        if len(citations) > MAX_CITATIONS:
            lines.append(
                f"  ... and {len(citations) - MAX_CITATIONS} more; --format json lists them all"
            )

    if any(run.unresolved() for run in runs):
        lines += [
            "",
            "JUDGE: the judge column counts episodes that queried the answer source's row API. "
            "That endpoint returns rows over the same hostname it returns refusals over, so "
            "whether content came back is in the response and not in the egress record. They "
            "are held at attempted rather than rounded either way.",
        ]

    shared = [run.label for run in runs if any(e.shared_window_with for e in run.episodes)]
    if shared:
        lines += [
            "",
            "SHARED WINDOWS: episodes in "
            + ", ".join(sorted(set(shared)))
            + " overlap in time and share one network namespace, so a connection inside more "
            "than one open window is charged to all of them. Per-episode evidence in these runs "
            "is a set the episode belongs to, not a fact about it alone.",
        ]
    for run in runs:
        for note in run.notes:
            lines += ["", f"{run.label}: {note}"]
    lines += ["", "what egress cannot establish"]
    lines += [f"  - {limit}" for limit in LIMITS]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="completed run directories")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    parser.add_argument("--out", type=Path, default=None, help="write instead of printing")
    args = parser.parse_args(argv)

    targets = [d for d in args.run_dirs if (d / "manifest.json").exists() or d.is_dir()]
    if not targets:
        print("no run directories given")
        return 1
    runs = [classify_run(d) for d in targets]
    runs.sort(key=lambda r: r.run_id)
    if args.format == "json":
        text = json.dumps({"schema": SCHEMA, "runs": [r.to_json() for r in runs]}, indent=2)
    else:
        text = render_table(runs)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"leakage: {args.out}")
    else:
        print(text)
    return 0


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
    "Connection",
    "Episode",
    "EpisodeLeakage",
    "RunLeakage",
    "classify_run",
    "content_url_kind",
    "host_role",
    "main",
    "read_egress",
    "read_episodes",
    "render_table",
    "trace_urls_by_lease",
]
