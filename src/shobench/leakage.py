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

There is no bucket above those, and that is the point of this half. Egress sees hostnames and
times, never a method, a status or a body, so it can say a cell reached the answer source and it
can never say the answers arrived. Establishing that needs the transcript read for content, which
is a separate change; here the ceiling is unresolved.

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

**The transcript is read for two things, and neither of them is content.** A lease's terminal
call says where an episode ended, which is what bounds its window; and the text of a command
says what the agent asked for, which can raise an episode to attempted or unresolved. A URL the
agent only mentioned in prose is not a request, and no reading of the transcript here reaches
past unresolved.

**A leg that has reached the answer source cannot be cleared afterwards.** Whatever a container
fetched may be on its disk from that moment, and a local read leaves nothing for the observer to
see, so every later episode in that container is unresolved at best. What such an episode
actually did with such a file is content evidence, which this half does not carry.

Nothing the agent wrote about itself lowers a bucket.

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
)

# Not a bucket: the label for an episode whose evidence is missing rather than empty.
UNCLASSIFIED = "unclassified"

PHASES = ("eval_before", "rollout", "eval_after")

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

_SUBMIT = re.compile(r"submit_answer")
_URL = re.compile(r"https?://[^\s\"'\\<>)\]}]+")
# A comment runs to the end of its line. The ``#`` has to follow whitespace or start the text,
# so a URL fragment stays part of its URL. What an agent wrote a note about is not what it ran.
_COMMENT = re.compile(r"(?:(?<=\s)|^)#[^\n]*")
# Where one command ends and the next begins, so a URL is read against the command that used it.
# Command substitution counts as a boundary: what is inside ``$(...)`` or backticks is run.
_SEPARATORS = re.compile(r"\|\||&&|\$\(|[;|\n&`]")
# Programs that go out and get something, and the calls that do it from inside a program.
_FETCH_COMMANDS = ("curl", "wget", "aria2c", "axel", "lftp", "scp", "rsync")
_FETCH_CALLS = (
    "urlretrieve", "urlopen", "urllib.request", "requests.get", "requests.post", "httpx.get",
    "wget.download", *_HUB_CALLS,
)
# Words that stand in front of the command actually being run, rather than being it.
_WRAPPERS = frozenset(
    {
        "bash", "sh", "zsh", "dash", "ksh", "env", "timeout", "nohup", "sudo", "nice", "time",
        "command", "exec", "xargs", "stdbuf", "setsid", "do", "then", "else", "elif", "while",
        "until", "if", "eval",
    }
)
# Interpreters, because a fetch written as a library call is only a fetch when one is running.
_INTERPRETERS = ("python", "ipython", "node", "ruby", "perl", "php", "deno", "bun")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_DURATION = re.compile(r"^\d+(?:\.\d+)?[smhd]?$")
# Tool names whose input is the request rather than a shell line.
_FETCH_TOOLS = frozenset({"webfetch", "websearch"})
_SHELL_TOOLS = frozenset({"bash", "shell", "sh", "run_command", "execute"})


def invoked_command(invocation: str) -> str | None:
    """The program this fragment actually runs, or None if it runs nothing recognisable.

    A fetch is a command that was invoked, not a word that appears. ``printf "%s" "curl <url>"``
    contains ``curl`` and runs ``printf``, and reading the first as a request would let an agent
    quoting a snippet look like an agent downloading a dataset. So the tokens are walked from
    the left, past environment assignments and past the wrappers that stand in front of a real
    command, and what is left at the front is what ran.
    """
    tokens = invocation.split()
    index = 0
    while index < len(tokens):
        token = tokens[index].strip("\"'`(){}").lstrip("$")
        if not token:
            index += 1
            continue
        if _ASSIGNMENT.match(token):
            index += 1
            continue
        base = token.rsplit("/", 1)[-1]
        if base in _WRAPPERS:
            index += 1
            while index < len(tokens):
                following = tokens[index].strip("\"'`")
                if following.startswith("-") or (base == "timeout" and _DURATION.match(following)):
                    index += 1
                    continue
                break
            continue
        return base
    return None


def _invokes_a_fetch(invocation: str) -> bool:
    command = invoked_command(invocation)
    if command is None:
        return False
    if command in _FETCH_COMMANDS:
        return True
    return command.startswith(_INTERPRETERS) and any(c in invocation for c in _FETCH_CALLS)


def _fetches(action: Action, invocation: str) -> bool:
    """Did this fragment of this action ask a remote host for something?

    A shell line is read for the command at its front. A tool whose whole job is fetching is a
    fetch whatever its input looks like. And a tool whose input is code rather than a command
    line has no command position to read, so there the call itself is the signal.
    """
    tool = action.kind.split(":", 1)[-1].lower()
    if tool in _FETCH_TOOLS:
        return True
    if action.kind == "command" or tool in _SHELL_TOOLS:
        return _invokes_a_fetch(invocation)
    return any(call in invocation for call in _FETCH_CALLS)


def _invocations(command: str) -> list[str]:
    """One command per entry, with comments removed."""
    return [part.strip() for part in _SEPARATORS.split(_COMMENT.sub("", command)) if part.strip()]


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
            # An open bound has no JSON number, and ``Infinity`` is not one: a bound that
            # reaches past every readable row is published as null rather than as a token a
            # strict reader refuses.
            "blind": [
                [None if low == float("-inf") else low, None if high == float("inf") else high]
                for low, high in self.blind
            ],
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
    """Every file the capture was written into, base first."""
    first = run_dir / "egress.tsv"
    numbered = sorted(
        (p for p in run_dir.glob("egress.*.tsv") if p.stem.split(".")[-1].isdigit()),
        key=lambda p: int(p.stem.split(".")[-1]),
    )
    return ([first] if first.exists() else []) + numbered


def capture_segments(run_dir: Path) -> list[tuple[str, list[str]]]:
    """One entry per observer process, with each stretch of the capture appearing once.

    A continuation gets a file of its own, because the capture command truncates whatever file it
    is pointed at. When that continuation's observer stops, the runner appends its file into
    ``egress.tsv`` so the published record covers the whole cell, and leaves the numbered file
    behind. Reading both counts that stretch twice.

    Skipping the numbered file is not the fix either: the base then reads as one observer running
    from its first row to its last, which papers over the interruption in the middle. The gap
    between one observer stopping and the next starting is exactly where this cannot say the cell
    was quiet, so the folded stretch is taken back out of the base and handed to the file it came
    from, and the two intervals stay apart.
    """
    first = run_dir / "egress.tsv"
    numbered = [p for p in egress_segments(run_dir) if p != first]
    if not first.exists():
        return [(p.name, p.read_text(encoding="utf-8", errors="ignore").splitlines())
                for p in numbered]
    base = first.read_text(encoding="utf-8", errors="ignore").splitlines()
    tail = [(p, p.read_text(encoding="utf-8", errors="ignore").splitlines()) for p in numbered]
    # Backwards, because the runner appends them in order, so the last one folded is the last
    # stretch of the base.
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
            # The display filter emits a row only for an outbound DNS question or a TLS client
            # hello, so a row carrying neither name is a torn one however well its timestamp
            # parses. Counting it as a readable observation would let a truncated line extend
            # the stretch this claims to have been watching.
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


def _request_text(arguments: Any) -> str:
    """A tool's input as the text that was run, where there is one.

    A shell tool carries its command in a field, and the command is what has a program at its
    front. Handing the JSON envelope to a tokeniser looking for that program finds the field
    name instead, so the field is unwrapped and everything else keeps its envelope.
    """
    if isinstance(arguments, dict):
        for field in ("command", "cmd", "script"):
            value = arguments.get(field)
            if isinstance(value, str):
                return value
    return json.dumps(arguments)


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
    pending: dict[str, tuple[str, str, int]] = {}
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

        # The terminal call, wherever a harness puts it. codex and claude_code call the tool
        # directly and their arguments are parsed below; prime-agent writes
        # ``await shogym_stream.submit_answer(..., lease='...')`` inside an ipython cell, so the
        # lease is in the code rather than in a field. A line that names the terminal call and a
        # lease this run dispensed is that lease sealing, in any of the three.
        if _SUBMIT.search(line):
            for lease in wanted:
                if lease not in sealed_at and lease in line:
                    sealed_at[lease] = offset

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
                    pending[str(block.get("id"))] = (name, _request_text(arguments), offset)
                    if name.endswith("submit_answer") and arguments.get("lease") in wanted:
                        sealed_at[str(arguments["lease"])] = offset
        elif kind == "user" and isinstance(event.get("message"), dict):
            for block in event["message"].get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool, request, _ = pending.pop(
                        str(block.get("tool_use_id")), ("tool", "", offset)
                    )
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

    # An invocation whose result never arrived: a timeout, a kill, or a transcript that ends
    # mid-turn. What it asked for is the evidence this phase reads, and dropping the record
    # because nothing answered would lose a known request and report the episode lower than the
    # ceiling. It is kept with no result and marked failed, since nothing says it succeeded.
    for tool, request, offset in pending.values():
        actions.append(
            Action(offset, f"tool:{tool}", request, "", ok=False, trace=transcript)
        )
    actions.sort(key=lambda a: a.offset)
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
    traces: dict[str, Trace],
) -> list[Episode]:
    """Windows that run from a dispense to a bound on the seal, not to the next dispense.

    The stream records when a task was handed out and not when it was sealed, and above one
    lease in flight the agent can still be working an older task when the next is pulled. Only
    the transcript can say when a lease ended: the ``submit_answer`` that ends an episode sits
    at a definite place in the order, so the seal happened no later than the dispense of the
    first task pulled at or after it. For a strictly sequential agent that lands on the next
    dispense; for one that interleaves it lands later and the windows overlap, which is the
    point.

    Capacity is deliberately not a bound. ``get_task`` force-drains only when a pull finds every
    slot occupied, so a newer lease that submits frees a slot and lets the next dispense through
    with an older lease still live: at capacity three, dispense A B C, submit B, dispense D,
    submit C, dispense E leaves A open past two later dispenses. Ending A at
    ``index + max_in_flight`` would invent a seal the stream never performed and hand A's later
    traffic to somebody else. With no seal in the transcript the only sound bound is the leg,
    which is where the container that could have opened the connection stops existing.
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
    for dispense in dispenses:
        lease = str(dispense["lease"])
        started = float(dispense["dispensed_at"])
        leg, leg_end = leg_of(started)

        bounds = []
        kind = "leg_bound"

        # The transcript's bound, when this lease's seal can be placed in the order. The seal
        # happened no later than the dispense of the first task pulled at or after it. At or
        # after, because a harness that submits and pulls again inside one action puts both on
        # the same line, and the lease it pulls there was dispensed after this one sealed.
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
    # Leases whose live region overlaps this episode's, so its commands are not exclusively its.
    action_rivals: tuple[str, ...]

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
            "action_rivals": list(self.action_rivals),
            "bucket": self.bucket,
            "reasons": list(self.reasons),
            "evidence": [c.to_json() for c in self.evidence],
            "requested": list(self.requested),
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


def _lease_regions(trace: Trace) -> dict[str, tuple[int, int]]:
    """Where each lease is live in one transcript: first appearance to seal.

    A lease with no seal in the transcript is live to the end of it. That is the honest end for
    a harness whose terminal call cannot be found, and it is why a missing seal shows up as
    wide, shared ownership rather than as a confident slice of somebody's commands.
    """
    end_of_trace = max((a.offset for a in trace.actions), default=0) + 1
    return {
        lease: (start, trace.sealed_at.get(lease, end_of_trace))
        for lease, start in trace.first_seen.items()
    }


def _actions_for(episode: Episode, traces: dict[str, Trace]) -> tuple[list[Action], list[str]]:
    """The actions this episode could have run, and the leases that could equally own them.

    An eval task's trace is named for its task, so the whole file is one episode. A rollout is
    one transcript for hundreds, and the lease ids cut it: an episode's actions run from where
    its lease first appears to where that lease seals.

    Those cuts overlap when the agent holds more than one lease, and an action inside an overlap
    has no owner the transcript can name. Giving it to whichever lease was pulled most recently
    is a guess that goes wrong in both directions at once: it clears the lease that really ran
    the command and charges the one that did not. So an overlapping action belongs to every
    lease live at that point and the rivals travel with the record, which is what the egress
    side already does with a connection inside two open windows.
    """
    out: list[Action] = []
    rivals: set[str] = set()
    for name, trace in traces.items():
        named = _TASK_TRACE.match(Path(name).name)
        if named is not None:
            if int(named.group(1)) == episode.task_idx:
                out.extend(trace.actions)
            continue
        regions = _lease_regions(trace)
        if episode.lease not in regions:
            continue
        for action in trace.actions:
            live = [
                lease for lease, (start, end) in regions.items() if start <= action.offset <= end
            ]
            if episode.lease not in live:
                continue
            out.append(action)
            rivals.update(lease for lease in live if lease != episode.lease)
    return out, sorted(rivals)


_TASK_TRACE = re.compile(r"^task-(\d+)-leg-")


def _requested_urls(actions: Sequence[Action]) -> list[str]:
    """URLs the agent asked for, taken from command text and never from prose or output."""
    urls: list[str] = []
    for action in actions:
        if action.kind.startswith("mcp:"):
            continue
        # Only from an invocation that fetches. A URL the agent printed, grepped for or wrote
        # into a file is data it handled, and nothing there asked the remote host for a body.
        for invocation in _invocations(action.request):
            if not _fetches(action, invocation):
                continue
            for match in _URL.finditer(invocation):
                urls.append(_tidy_url(match.group(0)))
    return list(dict.fromkeys(urls))


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


def _tidy_url(url: str) -> str:
    """Cut a URL out of the shell it was quoted in, so a template still names its route."""
    for marker in ("${", "$(", "`"):
        cut = url.find(marker)
        if cut > 0:
            url = url[:cut]
    return url.rstrip(".,;:!?")


def _accounts_for_its_home(source: RunLeakage) -> tuple[bool, str]:
    """Can this run say the HOME a bookend would inherit holds no answer file?

    Only its rollout matters, since that is the phase whose writes land in the mounted HOME. On
    the egress floor the answer is yes only when that rollout was fully observed and never
    reached the answer source at all. An episode this could not classify is one where anything
    may have happened; an episode that did reach the answer source may have saved what it found,
    and where a fetched file landed is content evidence this half does not carry.
    """
    rollout = [e for e in source.episodes if e.episode.phase == "rollout"]
    if not rollout:
        return False, "has no rollout record to account for that HOME"
    blind = sum(1 for e in rollout if e.bucket == UNCLASSIFIED)
    if blind:
        return False, f"could not classify {blind} of its {len(rollout)} rollout episodes"
    contacted = sum(1 for e in rollout if _rank(e.bucket) >= _rank("attempted_leakage"))
    if contacted:
        return False, (
            f"reached the answer source in {contacted} of its {len(rollout)} rollout episodes, "
            "and where anything it fetched landed is not on this record"
        )
    return True, ""


def _inherited_artifacts(run_dir: Path, manifest: dict[str, Any]) -> tuple[list[str], bool]:
    """Whether a bookend can be cleared against the HOME it inherited.

    A rebookend runs a new eval against the run it names, and the runner seeds it from that
    run's accumulated HOME, which every eval task then gets a copy of. So a file the source's
    rollout saved under HOME is on disk in this run before its first episode begins.

    The floor can only answer this one way. It can say the source was fully observed and never
    reached the answer source, in which case there is nothing to have inherited; it cannot say
    what a source that did reach the answer source put where. Tracking that file is content
    evidence and lives in the follow-up, so short of a clean source this run is not cleared.
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
        episodes += _rollout_episodes(run_dir, legs, traces["rollout"])
    for phase in ("eval_before", "eval_after"):
        if (run_dir / phase).is_dir():
            phase_episodes = _eval_episodes(run_dir, phase, legs, timeout)
            leases = [e.lease for e in phase_episodes]
            traces[phase] = {
                str(p): read_trace(p, leases) for p in _trace_files(run_dir, phase)
            }
            episodes += phase_episodes

    inheritance_notes: list[str] = []
    unresolved_inheritance = False
    if _inherit:
        inheritance_notes, unresolved_inheritance = _inherited_artifacts(run_dir, manifest)

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
    # When each leg first reached the answer source, so a later episode in the same container is
    # not cleared over a file that may have been sitting on its disk since. The time comes from
    # the observed connection where there is one, because an episode's window is not when its
    # traffic happened: charging contact to the window's start taints episodes that had already
    # ended when the connection was made. A request seen only in the transcript has no epoch, so
    # that one falls back to the window's start, which taints the most and claims the least.
    contacted: dict[str, float] = {}
    # What the rollout leaves in the HOME its eval_after tasks are copies of.
    seeded_home_contact = False
    seeded_home_blind = False

    for episode in ordered:
        evidence = _window_evidence(episode, capture.connections, starts)
        actions, action_rivals = _actions_for(episode, traces.get(episode.phase, {}))
        requested = _requested_urls(actions)
        covered = capture.covers(episode.started_at, episode.ended_at)
        reasons: list[str] = []

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
                # A command that asked the answer source for a file body. Whether one came back
                # is in the response, which this half does not read, so the episode sits at the
                # ceiling rather than at either answer.
                bucket = _raise_to(bucket, "unresolved_leakage")
                _note(reasons, "file_download_requested")
        # Only where the environment has an answer source at all. Without one this cannot tell a
        # dataset pull from any other, and bucketing it as leakage while the run's own note says
        # the two cannot be distinguished would be the metadata contradicting the number.
        if source is not None and any(
            call in invocation
            for a in actions
            for invocation in _invocations(a.request)
            for call in _HUB_CALLS
        ):
            bucket = _raise_to(bucket, "attempted_leakage")
            _note(reasons, "hub_download_call")

        # Answer-source contact earlier in this leg. The container that made it may have a
        # copy of the answers on its disk from that moment on, and a local read is invisible to
        # the observer, so a later episode in the same container cannot be cleared. What it
        # actually did with such a file is content evidence, which this half does not carry.
        contact = contacted.get(episode.leg)
        if (
            contact is not None
            and episode.started_at is not None
            and episode.started_at >= contact
            and bucket != UNCLASSIFIED
        ):
            bucket = _raise_to(bucket, "unresolved_leakage")
            _note(reasons, "answer_source_contact_earlier_in_leg")

        # An eval_after task runs against a copy of the HOME the rollout accumulated, so a
        # rollout that reached the answer source hands every one of them a disk this cannot
        # clear, and a rollout this could not classify hands them one it cannot describe.
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
            contacted[episode.leg] = min(contacted.get(episode.leg, when), when)
        if episode.phase == "rollout":
            if bucket == UNCLASSIFIED:
                seeded_home_blind = True
            elif reached:
                seeded_home_contact = True

        graded.append(
            EpisodeLeakage(
                episode=episode,
                bucket=bucket,
                reasons=tuple(reasons),
                evidence=tuple(evidence),
                requested=tuple(requested),
                covered=covered,
                shared_with=(),
                action_rivals=tuple(action_rivals),
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
                action_rivals=row.action_rivals,
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
    "classify_run",
    "content_url_kind",
    "host_role",
    "main",
    "read_capture",
    "read_trace",
    "render_table",
]
