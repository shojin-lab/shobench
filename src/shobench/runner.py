"""The cell runner: three phases, one manifest, one results JSON.

One cell is one call to :func:`run_cell`. It stands the cell's sandbox up, records the
manifest before anything spends, runs the held-out eval, the improvement rollout, and the
held-out eval again, then writes the results and tears the sandbox down.

Two structural choices are worth stating, because both are what the scope's protocol requires
rather than conveniences:

**One session per eval task, enforced by the server.** Each eval task gets its own
single-task ``EvalStream``, so a session that ignores its instruction and pulls a second task
is told the stream is done rather than quietly consuming the next task's measurement. Whether
that session starts cold or forks the rollout's terminal conversation is the cell's
``eval_context`` axis; the one-task rule holds either way. The serving process is this one, so
the dataset loads once per phase rather than once per task.

**The rollout is one honest run of the harness against one live stream.** A single harness
invocation is driven against the pool for the cell's wall clock, and the runner does not
relaunch it: whether a harness sustains autonomous operation is one of the things being
measured. A run that ends on the agent's own terms while the queue still had tasks is the stop
the charter asks about, and the runner does not prompt the agent onward.

**A provider usage limit suspends the cell instead of ending it.** That stop is not the
agent's, so the cell keeps everything it has, writes what a continuation needs, and leaves the
process without unwinding the stream. :func:`resume_cell` picks it up when the window resets
and finishes through the same ending an uninterrupted run reaches, so the eval that follows the
rollout still follows a real rollout terminus.

**Every ending a person or the runner imposes goes through the leg's normal ending.** Killing the
process ends the run before it can write ``legs.json`` and ``rollout_stopping.json``, and a run
without those has no terminus, so its rollout can never be bookended and the cell produces no
paired delta at all. So an operator asks through a file the run polls
(:const:`STOP_REQUEST_FILE`) and the runner ends the current leg the way a budget does, and a
rollout leg that has shown no evidence of progress anywhere for the cell's own bound is ended the
same way. Each records its own stop kind.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from shobench import egress
from shobench.config import Cell, Instruction, load_cell_by_name, load_instruction
from shobench.containers import (
    AGENT_IMAGE,
    HOST_ALIAS,
    CellSandbox,
    docker,
    home_digest,
    home_inventory,
    image_digest,
    run_relative,
    run_stem,
    write_json,
)
from shobench.credentials import (
    CredentialSpec,
    preflight_seeded_credential,
    refresh_seeded_credential,
    seed_home,
    spec_for,
)
from shobench.credentials import effective_mode as credential_effective_mode
from shobench.harness import Harness, LaunchSpec, StopKind, StopVerdict
from shobench.harnesses import harness_for
from shobench.pins import SHOGYM_REPO, SHOGYM_REV, shobench_revision
from shobench.redact import MARKER as redact_marker
from shobench.redact import Redactor, redactor_for, secrets_in_file
from shobench.results import (
    INCOMPLETE_SUFFIX,
    MISSING_CLOSURE,
    TaskResult,
    dispensed_positions,
    fill_missing,
    heldout_accounting,
    read_phase,
    write_results,
)
from shobench.serving import DEFAULT_PORT, SERVER_NAME, build_stream, side_for_phase, warm_env
from shobench.splits import Split, load_split_by_name

# What does not count as the agent's durable self.
#
# This list decides what the headline home digest means. A digest that includes session
# transcripts and caches changes on every run whether or not the agent changed itself, so it
# answers "did a session happen", which is always yes. Excluding them makes it answer "did the
# rollout write something a later session can use", which is the benchmark's question. What an
# eval session inherits beyond that channel is the cell's recorded eval_context axis (a resumed
# eval_after carries the rollout conversation itself), so the digest stays a statement about the
# durable channel rather than about everything a session saw.

# Directories that hold a session's byproducts rather than the agent's own writing. Matched on
# any path component, since the three harnesses nest them differently.
NOISE_DIRS = frozenset(
    {
        ".cache",
        "backups",
        "daemon-workers",
        "history",
        "kernel-venv",
        "logs",
        "node_modules",
        "sessions",
        "session-artifacts",
        "session-leases",
        "shell-snapshots",
        "shell_snapshots",
        "statsig",
        "telemetry",
        "tmp",
        "__pycache__",
    }
)

# Bookkeeping files a harness rewrites on its own schedule.
NOISE_FILES = frozenset(
    {
        ".last-cleanup",
        "history.jsonl",
        "installation_id",
        "models_cache.json",
        "policy-limits.json",
        "remote-settings.json",
        "session_index.jsonl",
        "version.json",
    }
)

# Credential material. Excluded from the digest and from the inventory both, so no record this
# runner writes carries anything about a credential beyond the fact that a mode was used.
# `.claude.json` is included here because it holds the OAuth session alongside per-project
# trust state, so it is both credential material and harness bookkeeping rather than anything
# the agent wrote about itself.
CREDENTIAL_FILES = frozenset({".claude.json", ".credentials.json", "auth.json"})

# How often the runner re-reads a running leg's credential files. The HOME a leg runs against is
# a host directory bind-mounted into the container, so the runner can watch what the harness
# writes into it while the leg is still going. A leg is one invocation of up to eight hours and a
# file-backed OAuth client refreshes inside it repeatedly, so reading only when the leg ends
# learns the last generation and no other. A second is far below any refresh interval any of the
# three harnesses uses and costs one read of a kilobyte file per leg per second.
CREDENTIAL_POLL_S = 1.0


def is_noise(rel_path: str) -> bool:
    """Is this file a session byproduct rather than part of the durable self?"""
    parts = rel_path.split("/")
    name = parts[-1]
    if name in NOISE_FILES or name in CREDENTIAL_FILES:
        return True
    if any(part in NOISE_DIRS for part in parts):
        return True
    if name.endswith((".sqlite", ".sqlite-shm", ".sqlite-wal", ".lock", ".log")):
        return True
    # A session transcript sits directly under a per-project directory as <uuid>.jsonl, while
    # what the agent wrote about itself sits in a named subdirectory beside it (memory/,
    # skills/). Excluding the flat jsonl keeps the transcript out and the memory in.
    if "projects" in parts and name.endswith(".jsonl"):
        index = parts.index("projects")
        if len(parts) - index == 3:
            return True
    return False


def durable_filter(harness: Harness) -> Callable[[str], bool]:
    """What a cell of this harness leaves out of its durable self: noise, plus what the runner owns.

    Two different reasons to exclude, and both have to hold or the measurement is wrong in a way
    that reads as a result. Noise is a session byproduct: it changes whether or not the agent
    changed itself, so counting it answers "did a session happen". Runner-owned files are the
    opposite, files only the runner ever writes, rewritten on every launch because the served
    endpoint moves between phases and between concurrent eval tasks. Counting those made a
    prime_agent cell that wrote nothing at all publish a changed durable self, which is a false
    positive on the one question the cell exists to answer.
    """
    owned = frozenset(harness.runner_owned_home_files)
    return lambda rel_path: rel_path in owned or is_noise(rel_path)


# The order every ending was decided in, across every handle in this process. One counter under
# one lock rather than a lock per handle: a leg asks which of ITS handles fired FIRST, and its
# handles are built in different places (a drain or stall handle per leg, the operator's once per
# run). Per-handle locks left that comparison to the order a caller happened to list them in.
_ENDING_ORDER = threading.Lock()
_ENDING_SEQUENCE = itertools.count()


@dataclass
class EarlyEnding:
    """A leg's handle for being ended before its budget, and the verdict that ending carries.

    Three endings share this seam: a drained eval leg, a leg an operator asked to end, and a
    stalled rollout. ``fired`` is set from wherever the decision is made (an event loop watching a
    stream, a thread watching a directory) and read by the thread supervising the container.

    The verdict travels WITH the decision rather than being re-derived by the supervisor, because
    the leg has to say which of the three happened. First writer wins, and ``order`` is what makes
    that true BETWEEN handles and not only within one.
    """

    fired: threading.Event = field(default_factory=threading.Event)
    verdict: StopVerdict | None = None
    # Where this fire sits in the process-wide order of endings, unset until it fires.
    order: int | None = None

    def fire(self, verdict: StopVerdict) -> bool:
        """End the leg with this verdict, unless another watcher already ended it.

        The event is set INSIDE the ordering lock, with the verdict and the order, because the
        three are one publication: setting it afterwards leaves a gap in which a handle holds an
        order and no event, and a reader that skips it for having no event takes the later
        decision instead.
        """
        with _ENDING_ORDER:
            if self.verdict is not None:
                return False
            self.verdict = verdict
            self.order = next(_ENDING_SEQUENCE)
            self.fired.set()
        return True


@dataclass
class LegRecord:
    """One harness invocation inside a phase.

    ``trace_path`` is the absolute path to this leg's trace, kept absolute in-process because
    the rollout reads the leg's real session id back off it before resuming. The durable record
    the runner writes carries the run-dir-relative form instead, computed in :meth:`to_json`, so
    no results artifact leaks the operator's absolute layout.
    """

    leg: int
    phase: str
    task_idx: int | None
    started_at: float
    ended_at: float
    returncode: int
    verdict: StopVerdict
    tasks_consumed_before: int
    tasks_consumed_after: int
    trace_path: str
    run_dir: Path
    observed_models: list[str] = field(default_factory=list)
    # The session this leg actually ran under, read back off its own trace. A rollout a usage
    # limit suspended is continued by naming this id to the harness, so it belongs in the
    # durable record rather than only in a live object the suspension takes with it.
    session_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "leg": self.leg,
            "observed_models": list(self.observed_models),
            "phase": self.phase,
            "session_id": self.session_id,
            "task_idx": self.task_idx,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "wall_clock_s": round(self.ended_at - self.started_at, 3),
            "returncode": self.returncode,
            "verdict": self.verdict.to_json(),
            "tasks_consumed_before": self.tasks_consumed_before,
            "tasks_consumed_after": self.tasks_consumed_after,
            "trace_path": run_relative(self.trace_path, self.run_dir),
        }


@dataclass
class RunContext:
    """Everything one cell run needs, resolved once."""

    cell: Cell
    split: Split
    instruction: Instruction
    harness: Harness
    run_id: str
    run_dir: Path
    sandbox: CellSandbox
    port: int = DEFAULT_PORT
    # What docker is actually given: the image's content id once it resolves, so every probe and
    # every leg of one run uses the same bytes even if the tag is moved under it mid-run. The tag
    # it was asked for and the id it resolved to are carried beside it for the record.
    agent_image: str = AGENT_IMAGE
    image_tag: str = ""
    image_digest: str | None = None
    credentials: dict[str, str] = field(default_factory=dict)
    legs: list[LegRecord] = field(default_factory=list)
    # Legs this process did not run: the record a suspended run left behind. A continuation is
    # a second process writing the same run directory, so it carries the earlier legs forward
    # rather than replacing them with its own.
    prior_legs: list[dict[str, Any]] = field(default_factory=list)
    # What to stop before a suspension leaves the process. The caller owns the pieces that
    # outlive a phase (the sandbox, an egress capture), and a suspension has to stop them
    # without unwinding to where they are held, so it is handed a way to do it.
    teardown: Callable[[], None] | None = None
    # The exact credential values this cell provisioned, replaced by a marker in everything
    # durable this runner writes. Empty is valid and makes every call below a no-op. It grows
    # over the run: see :meth:`watch_credentials`.
    redactor: Redactor = field(default_factory=Redactor)
    # Where this harness keeps a credential inside a HOME, relative to it. Read from the
    # credential spec rather than guessed, and read again after every leg, because the file is
    # the harness's to rewrite.
    credential_home_paths: tuple[str, ...] = ()
    # Eval tasks run their legs from several threads at once, and each of them may learn a new
    # credential value at the same moment. The swap below is what they serialize on.
    redactor_lock: Any = field(default_factory=threading.Lock, repr=False)
    # The run's operator-stop handle, shared by every leg this process runs, because the ask is
    # about the RUN: a stop requested while four eval legs are in flight ends all four, and one
    # requested between legs is still set when the next starts. :func:`owning_run` fires it.
    operator_stop: EarlyEnding = field(default_factory=EarlyEnding)

    def watch_credentials(self, home: Path) -> None:
        """Fold whatever the credential files hold *now* into the redactor, keeping what it had.

        The values a cell provisions are not the only values it ever holds. A file-backed OAuth
        client refreshes an expired token during a run and persists the new access, refresh, and
        expiry back to the same auth file, and the pinned prime-agent and codex both do exactly
        that. A redactor built once at seeding time cannot match what a refresh minted, so an
        ordinary shell or config inspection later in the run would put a live token through
        every redaction untouched and into a published artifact.

        Called with the HOME a leg actually ran against, since an eval task's HOME is a private
        copy that is deleted moments after the leg ends: what it minted can be learned then or
        never. Nothing here reads a value back out or reports one; only the count changes.
        """
        values: set[str] = set()
        for rel in self.credential_home_paths:
            values |= secrets_in_file(home / rel)
        if not values:
            return
        with self.redactor_lock:
            self.redactor = self.redactor.extended(values)

    def publish_json(self, path: Path, body: Any) -> Path:
        """Write one of this run's durable JSON artifacts, redacted.

        Every JSON file the runner writes into the run directory goes through here rather than
        through ``write_json`` directly. Routing them all through one method is what makes the
        guarantee checkable: the boundary is a single call site per artifact, so a new artifact
        that skips it is visible in a diff instead of being a silent hole.
        """
        return write_json(path, self.redactor.json(body))

    def leg_records(self) -> list[dict[str, Any]]:
        """Every leg of this run, in a deterministic order, inherited ones included.

        Two things decide that order. Eval legs finish in whatever order concurrency produced
        them, so they are sorted by phase, then task, then leg rather than left in completion
        order. And a continuation's legs follow the ones a suspension left behind, because the
        run is one record written by two processes.
        """

        def key(leg: LegRecord) -> tuple[str, int, int]:
            return (leg.phase, -1 if leg.task_idx is None else leg.task_idx, leg.leg)

        return [*self.prior_legs, *(leg.to_json() for leg in sorted(self.legs, key=key))]

    @property
    def mcp_url(self) -> str:
        return f"http://{HOST_ALIAS}:{self.port}/mcp"

    @property
    def cfg_dir(self) -> Path:
        return self.run_dir / "cfg"

    @property
    def durable(self) -> Callable[[str], bool]:
        """This cell's durable-self filter, which depends on which harness is running."""
        return durable_filter(self.harness)


# ----- the manifest ------------------------------------------------------------------------


def _probe(
    argv: list[str],
    *,
    image: str,
    sandbox: CellSandbox,
    env: dict[str, str],
) -> str:
    """Run a short command in the agent image and return its raw output, for the manifest.

    ``env`` is what this particular probe needs and nothing more. A version probe needs no
    credential at all, so it is not given one: a token in a child's environment is a token an
    inherited process can read and a crash report can echo, and the manifest is a published
    artifact. The output comes back unredacted and stays in memory, because a probe that
    authenticates can itself refresh a file-backed credential: the caller folds in whatever the
    file holds afterwards and only then redacts, so a value the probe just minted is covered by
    the redaction of the probe's own output rather than published in the manifest it feeds.
    """
    args = sandbox.docker_args(env=env, mounts={})
    result = subprocess.run(
        ["docker", *args, image, *argv], capture_output=True, text=True, timeout=180
    )
    return (result.stdout + result.stderr).strip()


def substrate_block() -> dict[str, Any]:
    """What a row is produced ON, in the shape the manifest records and a comparison reads.

    shogym serves and scores the task; this package decides how it is launched and supervised,
    which is the other half, so its revision sits beside shogym's. Built here rather than inline
    because the same block is what a later run compares itself against: one function means the
    recorded fact and the checked fact cannot drift apart.
    """
    revision, dirty = shobench_revision()
    return {
        "shogym_repo": SHOGYM_REPO,
        "shogym_rev": SHOGYM_REV,
        "mcp_server_name": SERVER_NAME,
        "shobench_rev": revision,
        "shobench_dirty": dirty,
    }


def effort_axis(cell: Cell, harness: Harness) -> dict[str, Any]:
    """What the cell asked of the harness, and what the harness will do with it.

    ``requested`` is the ask and ``applied``/``how`` are whether it reaches the CLI at all, which
    is a property of the harness rather than of the cell: every prime_agent cell requests an
    effort no prime_agent build can apply. Shared with the comparison for the same reason as the
    substrate block.
    """
    return {
        "requested": cell.effort,
        "applied": bool(cell.effort and harness.effort_flag),
        "how": (
            harness.effort_flag
            or f"{harness.name} exposes no reasoning-effort control; it is ignored"
        ),
    }


def pinned_image(agent_image: str) -> tuple[str, str, str | None]:
    """The image a run pins itself to: what to run, the tag asked for, and the content id.

    A tag is mutable and a run is long. Resolving it once, here, and handing the ID to every
    probe and every leg is what makes the recorded digest a statement about the bytes that ran
    rather than about what the tag happened to point at when the manifest was written; a
    concurrent rebuild or retag then cannot slide a different image under the run. Resolution
    that fails leaves the tag in place and records no digest, which is the honest absence: a
    host without docker is about to fail for better reasons anyway.
    """
    digest = image_digest(agent_image)
    return digest or agent_image, agent_image, digest


def build_manifest(ctx: RunContext, *, probes: dict[str, str]) -> dict[str, Any]:
    """The record of what this cell was, written before anything spends.

    It carries the substrate pin, the split digest, the instruction digests, the resolved
    harness version and model, and both persistent channels' digests at the start. Everything a
    reader needs to know whether two cells were the same experiment.

    The baseline is taken after the runner has placed everything it owns, which is the state the
    rollout actually begins from. Taken before, the vendored skill package the runner seeds
    would land on the far side of it and every prime_agent cell would publish those bytes as
    something the rollout wrote.
    """
    exclude = ctx.durable
    return {
        "schema": "shobench.manifest/1",
        "run_id": ctx.run_id,
        "started_at": time.time(),
        "cell": ctx.cell.to_manifest(),
        "split": ctx.split.to_manifest(),
        "instruction": {
            **ctx.instruction.to_manifest(),
            # Which standing instruction eval_after launches with, so the artifact says it
            # rather than leaving a reader to derive it from the eval_context axis: a resumed
            # after carries the rollout instruction (the conversation already holds the
            # objective; swapping it mid-conversation would measure an agent that never
            # existed), a cold one the blind eval instruction. eval_before is always the
            # eval instruction regardless.
            "eval_prompt_used": (
                "rollout_system" if ctx.cell.eval_context == "resumed" else "eval_system"
            ),
        },
        # What the row was produced ON. shogym serves and scores the task; this package decides
        # how the task is launched and supervised, which is the other half, so its revision is
        # recorded beside shogym's rather than left for a reader to infer from a timestamp. The
        # dirty flag says whether that revision identifies the code: a modified tree's commit
        # does not, and a comparison has to be able to tell.
        "substrate": substrate_block(),
        "harness_probes": probes,
        # What the cell asked for, and what the harness will actually do with it. These used to
        # be one field each, copied out of the cell file, which made the manifest a restatement
        # of the config rather than a record of the run: every prime_agent cell published effort
        # xhigh for a harness with no effort control, and both codex and prime published an empty
        # observed_models that reads as "nothing answered" rather than "nothing reports it".
        # ``observed`` is filled from the traces when the run ends; the rest is knowable now.
        "axes": {
            "model": {
                "requested": ctx.cell.model,
                "observed": [],
                "observable": ctx.harness.reports_observed_models,
                "source": (
                    "read off the harness trace"
                    if ctx.harness.reports_observed_models
                    else f"{ctx.harness.name} names no model in its trace; nothing observes it"
                ),
            },
            "effort": effort_axis(ctx.cell, ctx.harness),
        },
        "container": {
            "agent_image": ctx.image_tag or ctx.agent_image,
            # The tag names the image; this identifies it. Rebuilding one tag on a newer base
            # or a different runtime leaves the tag and the harness version probe unchanged
            # while changing the agent, so the content id is what a later comparison rests on.
            # Resolved ONCE, before the first container, and reused for every probe and leg, so
            # what is recorded is what ran rather than what the tag pointed at when asked. Null
            # where docker could not answer, which reads as the absence it is.
            "image_digest": ctx.image_digest,
            "network": ctx.sandbox.network,
            "netns_container": ctx.sandbox.netns_container,
            "home": run_relative(ctx.sandbox.home, ctx.run_dir),
        },
        # That redaction ran, and over how many distinct forms, never over which. A reader of an
        # artifact that carries the marker needs to know it came from here; a reader of one that
        # does not needs to know whether that means clean or unwatched. Zero is the honest
        # answer for a cell whose credential this runner could not name, and it is visible.
        "redaction": {
            "marker": redact_marker,
            "forms_watched": ctx.redactor.count,
            "applied_to": ["traces", "legs.json", "manifest.json", "results", "suspension"],
        },
        # The two channels that persist across a leg and that the agent can write. HOME is the
        # durable self the benchmark measures. /work is the writable cwd every harness runs in;
        # it is not the durable self (an eval session gets a fresh empty one, so nothing written
        # there reaches a later session) but it is persistent and agent-visible for the whole
        # rollout, and a cell that left a CLAUDE.md or a pile of scripts there used to publish a
        # manifest that mentioned none of it. Recorded, not scored.
        "home": {
            "digest_before": home_digest(ctx.sandbox.home, exclude=exclude),
            "digest_after": None,
            "inventory_after": [],
        },
        "work": {
            "digest_before": home_digest(ctx.sandbox.workdir, exclude=exclude),
            "digest_after": None,
            "inventory_after": [],
            "note": (
                "the rollout's writable cwd, persistent for the cell and visible to the agent; "
                "an eval task gets its own empty one, so this reaches no later session"
            ),
        },
    }


# ----- running one harness leg -------------------------------------------------------------


def write_home_files(home: Path, spec: LaunchSpec) -> None:
    """Put a leg's HOME files in place: the per-leg ones always, the seeds only when absent.

    A harness that reads its configuration only from HOME gets it written there once per leg,
    because the endpoint changes between phases and between concurrent eval tasks and the runner
    is the only party that knows it. Rewriting also repairs an entry the agent edited, and the
    home inventory in the manifest still shows that it did.

    Seeds are the opposite and need the opposite rule. They are assets the agent may improve
    (a skill package is the case that forced this), and the rollout exists to measure exactly
    such improvements: the eval-after home is a copy of the accumulated rollout home, so a leg
    that rewrote a seed would restore the vendored bytes over the agent's version in the moment
    before the session meant to read it starts. Absent means never installed; present means the
    agent's, whether it wrote those bytes or merely kept them.
    """
    for name, body in spec.home_files.items():
        target = home / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    for name, body in spec.home_seed_files.items():
        target = home / name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")


@contextlib.contextmanager
def _watching_credentials(ctx: RunContext, home: Path):
    """Fold every generation of this HOME's credential files into the redactor as they appear.

    A read taken when the harness has already exited can only learn the generation that survived
    it. Anything the harness minted and then overwrote inside the same invocation was never in
    the redactor and is no longer in the file, so nothing downstream can name it: that is a
    credential the runner provisioned and cannot protect. Reading while the harness runs is what
    turns "the last generation" into "every generation this saw", and the runner is in a position
    to do it because the HOME is a host directory it bind-mounts into the container.

    The first read is taken here on the calling thread rather than by the poller, so the state
    the invocation starts from is covered before it has run at all; a copied eval HOME is the
    case that needs it, since its file arrives from outside this leg entirely.

    What this does not close is the gap between two reads: a generation written and overwritten
    inside one interval is missed exactly as before. That is a race against a token lifetime
    measured in hours with an interval measured in a second, and it is a narrowing rather than a
    guarantee. The guarantee is elsewhere, in what a published artifact is allowed to carry at
    all (see :func:`shobench.harness.stderr_evidence`); this keeps the operator's own local trace
    clean as well.
    """
    ctx.watch_credentials(home)
    done = threading.Event()

    def poll() -> None:
        while not done.wait(CREDENTIAL_POLL_S):
            ctx.watch_credentials(home)

    watcher = threading.Thread(target=poll, name="shobench-credentials", daemon=True)
    watcher.start()
    try:
        yield
    finally:
        done.set()
        # Bounded, and a daemon thread besides: a poller that somehow wedged on a read must not
        # be what keeps a finished leg from being recorded.
        watcher.join(timeout=CREDENTIAL_POLL_S * 5)


# How often the leg supervisor looks up from waiting on the container. Every leg carries at least
# the run's operator-stop handle, so every leg polls.
LEG_POLL_S = 1.0


def _feed_stdin(proc: subprocess.Popen, data: str) -> None:
    """Write a harness's prompt into its stdin from a thread, then close it.

    Threaded because the supervisor cannot block here: a prompt larger than the pipe buffer would
    otherwise wait for a reader that is not reading yet, with nothing left to notice a watchdog or
    a budget. A harness that exits before it drains the pipe closes it under this write, which is
    that harness's own ending and not an error here.
    """
    if proc.stdin is None:
        return
    with contextlib.suppress(OSError, ValueError):
        proc.stdin.write(data)
        proc.stdin.close()


def _supervise(
    argv: list[str],
    *,
    out: Any,
    err: Any,
    stdin_data: str | None,
    timeout_s: int,
    container: str,
    endings: Sequence[EarlyEnding] = (),
) -> tuple[int, bool, bool]:
    """Run one container to its end and say how that end came about.

    Returns ``(returncode, timed_out, ended_early)``, at most one of the two flags set. A leg that
    ended by itself sets neither, and its return code is the harness's own. WHICH early ending
    fired is not this function's to say: the caller reads it off the handle that fired.

    Every ending the runner imposes is the same two steps, and the order is what makes them work:
    the docker client is killed first, then the container is removed by name. Killing the client
    alone leaves the container running, and a container that outlives its leg keeps spending and
    keeps holding the task it was handed.
    """
    proc = subprocess.Popen(
        argv,
        stdout=out,
        stderr=err,
        stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
        text=stdin_data is not None,
    )
    if stdin_data is not None:
        threading.Thread(
            target=_feed_stdin, args=(proc, stdin_data), name="shobench-stdin", daemon=True
        ).start()

    def end_it() -> int:
        proc.kill()
        proc.wait()
        docker("rm", "-f", container, check=False)
        # The runner ended this, so the client's own exit status describes the kill rather than
        # the run. -1 is what the timeout path has always recorded, and both endings keep it.
        return -1

    deadline = time.monotonic() + timeout_s
    # Nothing can end this leg early, so it waits once rather than spinning. No runner caller
    # passes an empty tuple.
    if not endings:
        try:
            return proc.wait(timeout=timeout_s), False, False
        except subprocess.TimeoutExpired:
            return end_it(), True, False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return end_it(), True, False
        try:
            return proc.wait(timeout=min(LEG_POLL_S, remaining)), False, False
        except subprocess.TimeoutExpired:
            pass
        if any(ending.fired.is_set() for ending in endings):
            return end_it(), False, True


def _fired_verdict(endings: Sequence[EarlyEnding]) -> StopVerdict | None:
    """The verdict of whichever handle fired FIRST, by the decisions' own order.

    Never by the order the caller listed them in: a leg carries its own handle and the run's
    operator handle, and both can fire inside one supervisor poll. The decision is the verdict and
    its order, read under the lock that publishes them; the event is not consulted, because a
    handle that has decided but whose event a reader has not yet seen has still decided.

    ``None`` where a leg ended early and no handle carries a verdict, which a caller reads as
    "classify it normally".
    """
    with _ENDING_ORDER:
        fired = [(e.order, e.verdict) for e in endings if e.order is not None and e.verdict]
    if not fired:
        return None
    return min(fired, key=lambda decision: decision[0] or 0)[1]


def leg_stem(leg: int, task_idx: int | None) -> str:
    """What one leg's artifacts are named after: its number, and its task where it has one."""
    return f"leg-{leg:04d}" if task_idx is None else f"task-{task_idx:05d}-leg-{leg:04d}"


def leg_trace_path(run_dir: Path, phase: str, leg: int, task_idx: int | None = None) -> Path:
    """Where a leg's structured trace is written, computed rather than remembered.

    :func:`run_leg` opens this file and the rollout's progress watcher stats it while the leg is
    still running, so the two agree by construction.
    """
    return run_dir / phase / "traces" / f"{leg_stem(leg, task_idx)}.stream.jsonl"


def run_leg(
    ctx: RunContext,
    *,
    phase: str,
    leg: int,
    system_prompt: str,
    user_prompt: str,
    session_id: str | None,
    resume: bool,
    timeout_s: int,
    task_idx: int | None,
    consumed_before: int,
    mcp_url: str | None = None,
    cfg_dir: Path | None = None,
    home: Path | None = None,
    workdir: Path | None = None,
    container_name: str | None = None,
    endings: Sequence[EarlyEnding] = (),
) -> LegRecord:
    """Run one harness invocation to completion and classify how it ended.

    ``mcp_url``, ``cfg_dir``, ``home``, ``workdir`` and ``container_name`` default to the
    cell-wide values on the context, which is what the single rollout leg uses. An eval task
    overrides all five so it reaches its own stream on its own port, writes its own config,
    mounts its own throwaway home and its own throwaway ``/work``, and names its own container,
    which is what lets many tasks run at once without sharing any of that mutable state.

    ``endings`` are the handles a caller may end this leg through (see :class:`EarlyEnding`). The
    run's own operator-stop handle is appended here rather than passed by each caller, so no
    caller can leave a phase an operator cannot end.

    What no caller may hand the rollout is a drain handle. A rollout leg with an empty queue in
    front of it is the charter's own question, and a runner that ended it would answer that
    question for the agent.
    """
    if resume and not session_id:
        raise RuntimeError("cannot resume without a session id; the previous leg wrote none")
    mcp_url = ctx.mcp_url if mcp_url is None else mcp_url
    cfg_dir = ctx.cfg_dir if cfg_dir is None else cfg_dir
    home = ctx.sandbox.home if home is None else home
    trace_dir = ctx.run_dir / phase / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    stem = leg_stem(leg, task_idx)
    stdout_path = trace_dir / f"{stem}.stream.jsonl"
    stderr_path = trace_dir / f"{stem}.err.txt"
    endings = (*endings, ctx.operator_stop)

    spec = ctx.harness.launch(
        mcp_url=mcp_url,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=ctx.cell.model,
        trace_path=stdout_path,
        session_id=session_id,
        resume=resume,
        leg_timeout_s=timeout_s,
        effort=ctx.cell.effort,
    )
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for name, body in spec.config_files.items():
        target = cfg_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    write_home_files(home, spec)

    env = dict(spec.env)
    env.update(ctx.credentials)
    # Named so a leg the runner ends at its budget can actually be removed. Killing the docker
    # client leaves the container running, and a container that outlives its leg keeps
    # spending and keeps pulling tasks the runner has stopped watching.
    container = (
        container_name
        if container_name is not None
        else f"{ctx.sandbox.netns_container}-{phase[:4]}-{leg:04d}"[:63]
    )
    args = ctx.sandbox.docker_args(
        env=env, mounts={cfg_dir: "/cfg:ro"}, name=container, home=home, workdir=workdir
    )
    argv = ["docker", *args, ctx.agent_image, *spec.argv]

    started = time.time()
    # Watched for the whole invocation rather than after it, because a leg is where a harness
    # refreshes its credential and a refresh that is itself overwritten before the leg ends
    # leaves nothing on disk to learn from afterwards.
    with (
        _watching_credentials(ctx, home),
        stdout_path.open("a", encoding="utf-8") as out,
        stderr_path.open("a") as err,
    ):
        # Every harness in v0 either hangs on an open stdin or reads its prompt from it, so
        # stdin is always explicit and never inherited.
        returncode, timed_out, ended_early = _supervise(
            argv,
            out=out,
            err=err,
            stdin_data=spec.stdin,
            timeout_s=timeout_s,
            container=container,
            endings=endings,
        )

    # The generation the harness left behind, taken before a byte of its output is read. The
    # watcher above covered the ones it wrote and replaced; this is the last one, and it is also
    # the last moment an eval task's values can be learned at all, since its HOME is a private
    # copy discarded the moment this returns.
    ctx.watch_credentials(home)

    # The credential values out of the leg's own output, before anything reads it. Order is the
    # whole point: `observed_models`, the session id and the classification are all read off
    # these files, so redacting here means every downstream copy is taken from bytes that no
    # longer hold a value this cell can name, rather than each of them having to remember to.
    #
    # What this cannot promise is that it named every value the leg saw. A generation written and
    # overwritten between two of the watcher's reads is in these files and in nothing the runner
    # can still read, so these two calls are a best effort over the operator's own local trace.
    # The guarantee is elsewhere: a published artifact quotes none of a leg's raw output. The
    # verdict describes its stderr rather than quoting it (`shobench.harness.stderr_evidence`),
    # and everything else it carries is a field the harness's own structured stream named.
    ctx.redactor.file(stdout_path)
    ctx.redactor.file(stderr_path)

    # A leg the runner ended early is classified by the runner rather than by the harness's own
    # rules, exactly as a timeout is: what ended it was a decision here, and the trace holds
    # whatever the harness happened to be saying when the container was removed. A usage limit in
    # that trace is not read either, and must not be: a drained eval leg's held-out row is already
    # sealed and scored, and an operator stop or a stalled rollout is over by a decision no window
    # reopening would undo.
    early = _fired_verdict(endings) if ended_early else None
    if early is not None:
        verdict = early
    else:
        verdict = ctx.harness.classify(
            returncode=returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timed_out=timed_out,
        )
    record = LegRecord(
        leg=leg,
        phase=phase,
        task_idx=task_idx,
        started_at=started,
        ended_at=time.time(),
        returncode=returncode,
        verdict=verdict,
        tasks_consumed_before=consumed_before,
        tasks_consumed_after=consumed_before,
        trace_path=str(stdout_path),
        run_dir=ctx.run_dir,
        observed_models=ctx.harness.observed_models(stdout_path),
        # The id the harness really used: the one the runner pinned when it may pin one, and
        # otherwise the one the harness minted and wrote into its own trace.
        session_id=ctx.harness.session_id_from_trace(stdout_path) or session_id,
    )
    ctx.legs.append(record)
    return record


# ----- the phases --------------------------------------------------------------------------


def free_port() -> int:
    """Claim a free ephemeral port by binding it and letting go.

    A fixed port is not safe here. A stale server from an unrelated session was found
    listening on the conventional port during development, and the agent connected to it and
    was told the queue was exhausted, which would have been recorded as an agent that chose to
    stop immediately. Asking the kernel for a port it believes is free, and then proving our
    own server is the one answering, is what makes that impossible rather than unlikely.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.asynccontextmanager
async def _served(stream, port: int, host: str = "0.0.0.0"):  # noqa: S104 (container-local)
    """Run the stream's HTTP server for the body of the block, then shut it down.

    The server task is cancelled rather than asked to stop because FastMCP owns the transport
    and exposes no stop handle; cancelling it unwinds uvicorn's own shutdown, and the stream's
    own context manager is what seals anything still in flight.
    """
    from shogym.serve import build_stream_server

    server = build_stream_server(stream, name=SERVER_NAME)
    task = asyncio.create_task(server.run_async(transport="http", host=host, port=port))
    try:
        await _wait_ready(task, host, port)
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, SystemExit, Exception):
            await task


async def _wait_ready(task, host: str, port: int, *, timeout: float = 60.0) -> None:
    """Wait until OUR server accepts a connection, and raise if it died instead.

    Two conditions, not one. Waiting only on the socket would accept a foreign listener and
    hand the agent someone else's queue; waiting only on the task would race the transport.
    Checking that the task is still alive on every poll is what turns a failed bind into an
    exception here rather than a silently empty phase later.
    """
    target = "127.0.0.1" if host in ("0.0.0.0", "") else host  # noqa: S104
    deadline = time.time() + timeout
    while time.time() < deadline:
        if task.done():
            # Re-raising the transport's own failure is the diagnosis; uvicorn exits with
            # SystemExit(3) when it cannot bind, which is the common case.
            task.result()
            raise RuntimeError(f"the stream server exited before serving on {target}:{port}")
        try:
            _, writer = await asyncio.open_connection(target, port)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.25)
    raise TimeoutError(f"stream server did not accept connections on {target}:{port}")


def _eval_container_name(netns_container: str, phase: str, idx: int) -> str:
    """A unique, id-preserving container name for one concurrent eval task.

    Docker names are capped at 63 characters. The task id is what distinguishes the containers
    that are alive at once, so it must survive the cap: this truncates the run-id prefix rather
    than the suffix, because a truncated id would let two live containers collide and a timeout's
    ``docker rm -f`` could then remove the wrong task's container.
    """
    suffix = f"-{phase[:4]}-t{idx:05d}"
    return f"{netns_container[: 63 - len(suffix)]}{suffix}"


def _copy_task_home(base: Path, dst: Path, *, keep: tuple[str, ...] = ()) -> None:
    """Copy one eval task's working HOME off the phase's base home.

    The task reads the accumulated durable self the phase is measuring (whatever the rollout
    wrote: memory, skills, notes) plus the harness credential, and writes only into this copy,
    which is discarded when the task ends. Two properties make that the isolation the eval needs:
    a task's writes never reach the base home or a sibling's copy, and nothing but the durable
    channel and the credential crosses into the fresh session.

    What is left behind is exactly the noise the durability filter already names: caches,
    session transcripts, logs, and per-harness bookkeeping. Leaving it keeps the copy small
    (memory and skills are kilobytes; a ``.cache`` or ``node_modules`` is not) and keeps the
    session fresh, since a task that wants a cache rebuilds its own. Credential files are noise
    for the digest but the harness cannot authenticate without them, so those alone cross even
    though the rest of the noise does not.

    ``keep`` names HOME subtrees that cross despite being noise. A resumed eval_after is the
    case that needs it: each task forks the rollout's terminal session, and the harness can only
    reopen a session whose transcript is in the HOME it runs with, so the harness's recorded
    conversations ride into the copy. They stay out of every digest either way, since the
    durability filter is not this copy.
    """
    dst.mkdir(parents=True, exist_ok=True)
    if not base.exists():
        return
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(base).as_posix()
        kept = any(rel == prefix or rel.startswith(prefix + "/") for prefix in keep)
        if not kept and is_noise(rel) and path.name not in CREDENTIAL_FILES:
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _eval_task_valid_row(prov_dir: Path, idx: int) -> TaskResult | None:
    """The one valid completed row for held-out id ``idx``, or ``None`` when the task is not done.

    Completion is read off the task's own provenance, the same way the standalone held-out
    evaluator this descends from reads it: a held-out id is done when its directory holds exactly
    one scored, non-``drained`` row for that id. Each of those conditions rules out a way a pass
    can look finished without being it.

    A ``drained`` row is the case this exists to catch. It is what an orderly stream close records
    for a task a provider usage limit cut off in flight: scored, reward zero, which reads as the
    model failing the held-out task when what failed was the subscription window. So a drained row
    is not-done, and a resume re-runs the id in a fresh session rather than publishing it.

    A ``broker_abort`` or an absent row (a dispense that never sealed) is not scored, so it is
    not-done too. More than one row means something dispensed twice and the outcome is ambiguous,
    so it is not-done and the id is cleared and re-run. Only an ``aborted`` or ``sealed`` row is a
    real outcome the agent reached, and that id is done: a resume must never re-run it, because a
    completed held-out task is expensive and re-running it would also risk a second row.
    """
    return _settled_row(read_phase(prov_dir), idx)


def _settled_row(rows: list[TaskResult], idx: int) -> TaskResult | None:
    """The rule of :func:`_eval_task_valid_row`, applied to rows already read.

    The baseline carry asks the same question of rows it holds in hand rather than of a
    provenance directory, and one rule answering both is what keeps a carried row and a
    pending id from disagreeing about whether an id is done.
    """
    if len(rows) != 1:
        return None
    row = rows[0]
    if row.task_idx == idx and row.scored and row.closure != "drained":
        return row
    return None


def _eval_pending_ids(phase_dir: Path, task_ids: Sequence[str]) -> list[str]:
    """The held-out ids that still lack a valid completed row, in the split's order.

    A fresh phase has none done, so this is every id; a resume has the ids the first process
    finished already done, so this is only the interrupted one and any that never started. It is
    what makes a resume re-run only the incomplete work rather than the whole held-out set.
    """
    return [
        task_id
        for task_id in task_ids
        if _eval_task_valid_row(phase_dir / f"task-{int(task_id):05d}", int(task_id)) is None
    ]


# prime_agent is why the watchdog exists. Launched autonomous with no quality gate, its
# continuation check has nothing to evaluate and returns "keep going" unconditionally, so the leg
# has no terminal condition of its own and runs until the task timeout kills it: a median
# time-to-terminal of about two minutes stretched into 25 to 30 minutes of wall clock, roughly 90%
# of it after the measurement was already complete. The score is unaffected either way (the row
# seals at the task's completion call and the per-task home is discarded), so what this recovers
# is time and spend, and the ending is recorded as its own kind so the finding stays visible. How
# long a leg is given after its task is finished, if anything at all, is the harness's own
# ``eval_drain_grace_s``.
#
# How often the condition is re-read. The condition is monotone in practice, so this only decides
# how promptly the grace timer starts; it costs one queue read and one small file read. For a
# harness given no grace it is the whole of the delay, so the ending lands within one interval of
# the seal.
EVAL_DRAIN_POLL_S = 5.0


def _eval_task_is_finished(stream: Any, prov_dir: Path, idx: int) -> bool:
    """Has this eval task's leg got anything left to do?

    Two independent readings have to agree, because either alone is wrong. The stream's own
    queue says nothing was dispensed that is still live and nothing is left to dispense; the
    provenance directory says a valid completed row for this id is on disk. The queue alone would
    read the same before the first ``get_task`` if the harness never asked for one (an empty
    queue would be a leg to leave running, since it has not started); the row alone would not
    notice a task still in flight behind it.
    """
    info = stream.queue_info()
    if info.consumed < 1 or info.remaining or info.in_flight:
        return False
    return _eval_task_valid_row(prov_dir, idx) is not None


async def _watch_for_drain(
    stream: Any,
    prov_dir: Path,
    idx: int,
    ending: EarlyEnding,
    *,
    grace_s: float | None,
    harness: Harness,
    poll_s: float = EVAL_DRAIN_POLL_S,
) -> bool:
    """Fire ``ending`` once this task has been finished for the harness's whole grace period, or
    on the first reading that finds it finished when the harness declares no grace.

    Runs on the phase's event loop, which is where the stream mutates, so reading the queue here
    cannot catch it mid-change. The leg itself is on a worker thread and is told through the
    handle rather than by this coroutine, which touches no container.

    ``grace_s`` is the harness's own and goes into the verdict, so the record says how long the
    leg was given after there was nothing left for it to do. ``None`` means no grace: the first
    finished reading ends the leg.

    The grace timer restarts if the condition stops holding. That cannot happen to a single-task
    eval stream, and the reset is here anyway: it makes this a statement about the condition
    rather than about the first time the condition was seen.
    """
    finished_since: float | None = None
    while True:
        await asyncio.sleep(poll_s)
        try:
            finished = _eval_task_is_finished(stream, prov_dir, idx)
        except Exception:
            # A watcher that cannot read the state fails safe and never fails the task. It is
            # reading a stream and a file that another party is writing, and the cost of guessing
            # wrong in this direction is a leg that ends at its budget exactly as it did before
            # this existed; the cost in the other direction would be a raise landing in the task's
            # cleanup and turning a finished measurement into an unscored one.
            finished = False
        if not finished:
            finished_since = None
            continue
        if grace_s is None:
            # No grace: this reading is the ending. Written as its own branch rather than as a
            # zero-length timer, because a zero would still be compared on the NEXT poll and the
            # leg would live a poll interval longer, which is the whole of what this is for.
            ending.fire(harness.drained_verdict(grace_s=grace_s))
            return True
        if finished_since is None:
            finished_since = time.monotonic()
        elif time.monotonic() - finished_since >= grace_s:
            ending.fire(harness.drained_verdict(grace_s=grace_s))
            return True


def terminal_session_of(stopping: dict[str, Any]) -> str | None:
    """The rollout's terminal session id as a stopping record names it, or ``None``.

    The record's ``session_id`` is the terminal one (it is updated to the id the last leg really
    ran under); a record without it falls back to the last rollout leg that names one. Taken from
    the dict rather than from the file, so the runner can answer for a record it is still holding.
    """
    session_id = str(stopping.get("session_id") or "") or None
    if session_id is None:
        for leg in reversed(stopping.get("legs", [])):
            if leg.get("session_id"):
                return str(leg["session_id"])
    return session_id


def terminal_session_in(run_dir: Path) -> str | None:
    """The rollout's terminal session id as the run's own record on disk names it, or ``None``.

    Shared between the resumed eval_after preflight and the rebookend entry, which reads it off
    the SOURCE run it bookends, so the two cannot drift on what "the terminal session" means.
    """
    stopping_path = run_dir / ROLLOUT_STOPPING_FILE
    if not stopping_path.is_file():
        return None
    return terminal_session_of(json.loads(stopping_path.read_text(encoding="utf-8")))


def terminus_is_rebookendable(ctx: RunContext, stopping: dict[str, Any]) -> tuple[bool, str]:
    """Can a rebookend actually resume from this rollout's ending? Says so, and why not.

    The same two facts ``rebookend`` blocks on, checked so a run STATES them at the moment it
    ends. A record naming a session is not enough: claude runs under an id the RUNNER pinned
    before launch, so the id is in the record whatever the leg managed to write, while codex and
    prime mint their own and a leg ended early enough never got one into its trace. Read against
    the cell's own home, which is what a bookend would copy, before any eval phase has run.
    """
    session_id = terminal_session_of(stopping)
    if session_id is None:
        return False, "the rollout record names no terminal session to fork"
    if ctx.harness.session_transcript(ctx.sandbox.home, session_id) is None:
        return False, (
            f"the recorded terminal session {session_id} has no resumable transcript under the "
            f"run's home, so there is no conversation for a bookend to reopen"
        )
    return True, ""


def _rollout_terminal_session(ctx: RunContext) -> str:
    """The session the rollout ended in, proven forkable from the cell's base home.

    Read from the run's own record rather than from live objects, because eval_after does not
    always run in the process that ran the rollout: a fresh cell has just written
    ``rollout_stopping.json``, while a resume or a reopened run has only the disk. The id alone
    is not enough: the transcript must be in the base home the forks copy from, since each
    per-task launch would otherwise discover the absence only after its copy, its stream, and
    its container were already paid for, one task at a time.

    Both absences raise. The axis is a recorded claim about what the measurement was, and the
    one wrong recovery is to run cold under the resumed label; an operator who wants the cold
    measurement has a cell axis that says so.
    """
    stopping_path = ctx.run_dir / ROLLOUT_STOPPING_FILE
    session_id = terminal_session_in(ctx.run_dir)
    if session_id is None:
        raise RuntimeError(
            f"{ctx.cell.name}: eval_context is 'resumed' but the rollout record at "
            f"{stopping_path.name} names no session to fork, so eval_after cannot carry the "
            "rollout's context. Running it cold instead would publish a mislabeled "
            "measurement; use a cell with eval_context = 'cold' if cold is the intent."
        )
    if ctx.harness.session_transcript(ctx.sandbox.home, session_id) is None:
        raise RuntimeError(
            f"{ctx.cell.name}: eval_context is 'resumed' but the rollout session "
            f"{session_id} has no resumable transcript under the cell home's "
            f"{', '.join(ctx.harness.session_state_dirs) or 'session state'}: nothing there "
            f"both names that session and carries what {ctx.harness.name} itself requires to "
            "reopen it. Nothing has been spent; restore the home or run a cell with "
            "eval_context = 'cold' if cold is the intent."
        )
    return session_id


# How long the runner waits between the launches of an eval phase's FIRST wave.
#
# After the first wave the launches space themselves out, because a slot opens only when a task
# finishes. The first wave is the one moment N containers start at the same instant, each with
# its own copy of one file-backed OAuth credential, each presenting the same token and each free
# to refresh it. That is the shape of the failure this addresses: one prime eval phase lost 119
# of 120 legs to "No API key for provider: anthropic", in bursts, at ~19s a leg, with the file's
# own expiry hours in the future, which rules out plain expiry and leaves the simultaneity.
#
# Generic rather than prime-specific: codex seeds a refreshable auth.json the same way. It is
# skipped for a mode that seeds no file, because a token handed to the container as an
# environment variable is not refreshed by the harness and so has nothing to race.
EVAL_LAUNCH_STAGGER_S = 2.0


class _StaggeredAdmission:
    """Hands the first launches of an eval phase out one at a time, a fixed gap apart.

    It sits AFTER the concurrency gate on purpose. That costs a held slot for the length of the
    spacing and buys the only property worth having: the first ``first_n`` LAUNCHES are spaced,
    in the order the gate admitted them, whatever order the coroutines were scheduled in.
    Spacing ahead of the gate spaces only the coroutines that sleep, and a task that skips the
    sleep takes a free slot the sleeping ones have not claimed yet: eight tasks at a concurrency
    of four launched in the order 5, 6, 7, 8, 1, 2, 3, 4, which is an unstaggered first wave
    wearing reversed membership. Spacing where admission happens cannot be bypassed, because
    there is no path to a launch that does not pass through it.

    Once ``first_n`` launches have gone through this is a no-op forever. Past the first wave a
    slot opens only when a task finishes, so the launches are already spread by the work itself.
    """

    def __init__(self, gap_s: float, first_n: int) -> None:
        self._gap_s = gap_s
        self._left = first_n
        self._lock = asyncio.Lock()
        # Monotonic, and zero until the first launch, so the first one is never delayed.
        self._next_at = 0.0

    async def wait(self) -> None:
        if self._gap_s <= 0 or self._left <= 0:
            return
        # The lock is held across the sleep, which is what serializes the wave: each waiter takes
        # its turn in arrival order, and the one behind it starts its own gap from there.
        async with self._lock:
            if self._left <= 0:
                return
            delay = self._next_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._left -= 1
            self._next_at = time.monotonic() + self._gap_s


def _preflight_eval_credential(ctx: RunContext) -> None:
    """Refuse the phase, before anything is spent, when the credential every task copies is dead.

    Two steps, both free, and both judged on the ONE credential this cell will present. A
    multi-provider file is the ordinary case rather than the exception: a real gpt-terra home
    carries a live openai-codex login beside a long-expired anthropic one, and a check that
    judged every entry would refuse that cell over a credential it never reaches for. Which
    entry it reaches for is the harness's own resolution
    (:meth:`shobench.harness.Harness.credential_provider`, which prime derives from the cell's
    model exactly as ``launch`` derives the ``--provider`` it passes), so the two cannot drift.

    The refresh is prime-specific and replaces one provider's entry rather than the file, so no
    other provider's credential can be regressed by it; see
    :func:`shobench.credentials.refresh_seeded_credential`. The check is generic by credential
    schema, and a mode that seeds no file has nothing to check and passes.

    Raising is the point rather than a fallback. Fanning out anyway costs a container, a home
    copy, a stream and a port per held-out id before the first leg says the credential is no
    good, and the phase's rows come back unscored, which reads as an agent that failed the
    held-out set rather than as a phase that never authenticated.
    """
    spec = spec_for(ctx.harness.name, ctx.cell.credential_mode)
    provider = ctx.harness.credential_provider(ctx.cell.model)
    note = refresh_seeded_credential(spec, ctx.sandbox.home, provider=provider)
    if note:
        # A credential this runner just placed is one it can name, and naming it here rather than
        # waiting for the first leg's watcher keeps the window at zero.
        ctx.watch_credentials(ctx.sandbox.home)
        print(f"[shobench] {ctx.cell.name}: {note}", file=sys.stderr)
    ok, why_not = preflight_seeded_credential(spec, ctx.sandbox.home, provider=provider)
    if ok:
        return
    raise RuntimeError(
        f"{ctx.cell.name}: the held-out phase will not start because {why_not}. Every task "
        "copies that one file into a home of its own and launches within seconds of its "
        "siblings, so an unusable credential costs the whole phase rather than one leg. "
        "Nothing has been spent; renew the login on the host and rerun."
    )


async def run_eval_phase(ctx: RunContext, phase: str) -> list[TaskResult]:
    """Serve the held-out split, one session per task, up to N tasks at once.

    Each task gets its own single-task stream on its own port, its own session, its own
    throwaway copy of the phase's home, and its own throwaway ``/work``, so the one-session-per-task
    rule is enforced by the server and nothing a task does can reach another task or the base home.
    Concurrency is bounded by ``budget.eval_concurrency``; the tasks are independent, so a task that
    fails to run lands unscored (``reconcile`` records the dispense-without-seal) rather than
    sinking the batch, and the reported rows are sorted by task id regardless of finish order.

    What the session starts from is the ``eval_context`` axis, and it decides what the phase
    measures. Under "resumed", eval_after forks the rollout's terminal session into every task:
    the transcript rides in the task's own home copy, so the forks are independent by the same
    isolation that keeps their writes apart, and each one carries what the rollout still held in
    context, compaction summaries included. Under "cold" (and always for eval_before, which has
    no conversation to resume) the session is fresh and only the durable channels cross.

    Two things happen before the first container starts, and both exist because a fan-out
    multiplies a single fault by the size of the held-out set. The credential every task will
    copy is refreshed where that is free and the phase is refused where it is already unusable
    (:func:`_preflight_eval_credential`), and the first wave's launches are spaced by
    ``EVAL_LAUNCH_STAGGER_S`` so N homes do not present one token at the same instant.

    Only the ids that lack a valid completed row are run, which is what makes this both a fresh
    phase and a resume: a fresh phase has none done and runs all of them, a resumed one runs only
    the interrupted and unstarted ids and leaves the finished rows untouched. If a leg ends on a
    provider usage limit, the cell suspends here rather than publishing the drained row that an
    orderly close would score as a failure: no more tasks are admitted, the interrupted id is left
    for the resume to re-run, and the process hard-exits through the same guaranteed path the
    rollout uses. :func:`resume_cell` picks the phase back up and finishes the ids still pending.
    """
    side = side_for_phase(ctx.split, phase)
    phase_dir = ctx.run_dir / phase
    # Before anything is copied, served or launched: the one credential every task in this phase
    # will carry, refreshed where that is possible and refused where it is already unusable.
    _preflight_eval_credential(ctx)
    # The home every task copies from: pristine-plus-credential for eval_before, the rollout's
    # accumulated home for eval_after. It is read only from here on, since tasks write to copies.
    base_home = ctx.sandbox.home
    # The session every task forks, when this phase carries the rollout's context. Resolved and
    # proven present before anything is copied, served, or launched: a missing session must fail
    # the phase here, loudly, because the axis is a recorded claim about what was measured and a
    # fallback to cold would publish a mislabeled measurement. There is no such fallback.
    resume_session: str | None = None
    if phase == "eval_after" and ctx.cell.eval_context == "resumed":
        resume_session = _rollout_terminal_session(ctx)
    # Provision the env's upstream once, before the fan-out, so the first wave of streams reuses
    # the on-disk cache instead of racing to fetch it. Read-only serving-side data, safe to share.
    await asyncio.to_thread(warm_env, ctx.cell)
    limit = max(1, ctx.cell.budget.eval_concurrency)
    gate = asyncio.Semaphore(limit)
    # Set by the first eval leg a provider usage limit ends, with the evidence for the record.
    # Once set, no further task is admitted: the window is closed for every task, not just this
    # one, so admitting more only spends doomed legs and drains more positions to re-run.
    usage_limit: dict[str, Any] = {}

    # Whether this phase's launches need spacing at all, decided by the credential rather than by
    # the harness: the race is between N homes holding copies of one refreshable file.
    seeds_a_credential_file = bool(spec_for(ctx.harness.name, ctx.cell.credential_mode).seed_to)
    admission = _StaggeredAdmission(
        EVAL_LAUNCH_STAGGER_S if seeds_a_credential_file else 0.0, limit
    )

    async def one_task(task_id: str) -> None:
        idx = int(task_id)
        prov_dir = phase_dir / f"task-{idx:05d}"
        task_home = phase_dir / "homes" / f"task-{idx:05d}"
        task_work = phase_dir / "work" / f"task-{idx:05d}"
        task_cfg = phase_dir / "cfg" / f"task-{idx:05d}"
        async with gate:
            # Spaced inside the gate, where a launch actually becomes a launch. Nothing about the
            # phase's ordering or its accounting rides on it: the ids are gathered, the rows are
            # read back by task id, and a task held here runs the same leg against the same
            # one-task stream a moment later.
            await admission.wait()
            if usage_limit:
                return  # a usage limit closed the window; this task waits for the resume
            if ctx.operator_stop.fired.is_set():
                # An operator ended the run while this task waited on the gate. Admitting it would
                # copy a home and start a container for the supervisor to kill a second later,
                # once per remaining held-out id.
                return
            # Clear any stale attempt (a drained row from an earlier suspension, a half-written
            # dispense) so a run leaves exactly one row. A fresh phase's directory is empty and
            # this is a no-op; a resume's incomplete id starts clean, which is what keeps the
            # published phase at one valid row per id with no drained leftover.
            shutil.rmtree(prov_dir, ignore_errors=True)
            prov_dir.mkdir(parents=True, exist_ok=True)
            try:
                # A resumed fork's copy also carries the harness's recorded conversations,
                # which are noise everywhere else; the transcript has to be in the HOME the
                # harness runs with or there is nothing to reopen.
                _copy_task_home(
                    base_home,
                    task_home,
                    keep=ctx.harness.session_state_dirs if resume_session else (),
                )
                # A fresh empty /work of its own, discarded with the home. /work is the writable
                # cwd every harness runs in and is not part of the measured self, so sharing it
                # would leak one task's files into another with nothing in the HOME digest to show
                # it; a private empty directory is the isolation the concurrency needs.
                task_work.mkdir(parents=True, exist_ok=True)
                # A fresh port per task, so a socket still in TIME_WAIT from a finished task
                # cannot make another look like a server that refused to start.
                port = free_port()
                mcp_url = f"http://{HOST_ALIAS}:{port}/mcp"
                container = _eval_container_name(ctx.sandbox.netns_container, phase, idx)
                stream = build_stream(
                    ctx.cell,
                    _single_task_split(ctx.split, phase, task_id),
                    phase,
                    prov_dir,
                    deadline=float(ctx.cell.budget.eval_task_timeout_s),
                )
                # The drain watcher and the leg run side by side: the leg holds a worker thread,
                # and the loop is free to keep reading this task's stream while it does. Cancelled
                # in the `finally` so a leg that ended on its own leaves nothing watching a stream
                # that is about to close. The grace it waits out belongs to the harness, since
                # what the grace covers is that harness's own wrap-up after the seal.
                drained = EarlyEnding()
                async with stream, _served(stream, port):
                    watching = asyncio.create_task(
                        _watch_for_drain(
                            stream,
                            prov_dir,
                            idx,
                            drained,
                            grace_s=ctx.harness.eval_drain_grace_s,
                            harness=ctx.harness,
                        )
                    )
                    try:
                        record = await asyncio.to_thread(
                            run_leg,
                            ctx,
                            phase=phase,
                            leg=idx,
                            # A resumed fork carries the ROLLOUT's standing instruction, not
                            # the eval one. The rule that the eval instruction never carries
                            # the improvement objective was designed for cold measurement; a
                            # resumed conversation already carries the objective in its history
                            # and its compaction summaries, and swapping the standing
                            # instruction mid-conversation would measure an agent that never
                            # existed. The resumed after measures the agent as it lived,
                            # objective included; every cold session (eval_before always) keeps
                            # the blind eval instruction.
                            system_prompt=(
                                ctx.instruction.rollout_system
                                if resume_session
                                else ctx.instruction.eval_system
                            ),
                            user_prompt=ctx.instruction.kickoff,
                            # A resumed fork names the rollout's terminal session; every task
                            # names the same one, and the per-task home copies are what keep the
                            # forks independent. A cold task pins a fresh id instead.
                            session_id=resume_session or str(uuid.uuid4()),
                            resume=resume_session is not None,
                            timeout_s=ctx.cell.budget.eval_task_timeout_s,
                            task_idx=idx,
                            consumed_before=0,
                            mcp_url=mcp_url,
                            cfg_dir=task_cfg,
                            home=task_home,
                            workdir=task_work,
                            container_name=container,
                            endings=(drained,),
                        )
                    finally:
                        watching.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await watching
                if record.verdict.kind is StopKind.USAGE_LIMIT and not usage_limit:
                    # First usage limit wins: it is the evidence the suspension records, and it
                    # closes admission for every task still waiting on the gate.
                    usage_limit["verdict"] = record.verdict
            except Exception as exc:  # one task's failure is unscored, never fatal to the batch
                # Redacted like every other durable file: a docker failure's message carries the
                # command it ran, and that command carries the leg's `-e` arguments.
                (prov_dir / "runner-error.txt").write_text(
                    ctx.redactor.text(f"{type(exc).__name__}: {exc}\n"), encoding="utf-8"
                )
            finally:
                # Discard the task's home and /work the moment it is done, so N concurrent copies
                # is the ceiling on disk and nothing a task wrote survives to be read by anything.
                shutil.rmtree(task_home, ignore_errors=True)
                shutil.rmtree(task_work, ignore_errors=True)

    pending = _eval_pending_ids(phase_dir, side.task_ids)
    await asyncio.gather(*(one_task(task_id) for task_id in pending))
    if usage_limit:
        # A provider stopped a leg, not the agent. The cell suspends where it stands: the finished
        # ids keep their rows, the interrupted one is left row-less-equivalent for the resume, and
        # this does not return (it hard-exits), so nothing publishes the interrupted phase.
        _suspend_eval_and_exit(ctx, phase=phase, verdict=usage_limit["verdict"])
    # Every task's rows, the finished-earlier ones included, gathered across the per-task
    # directories in task-id order, which is the list an uninterrupted phase would publish.
    return read_eval_phase(phase_dir, side.task_ids)


def _no_row_diagnostic(phase_dir: Path, idx: int) -> str:
    """Why a held-out id has no row, in whatever words the runner managed to record.

    The runner writes ``runner-error.txt`` beside the task when its own attempt to run it raised,
    which is the common case and the informative one. The other case has nothing to read: a
    harness that exited before it ever called ``get_task`` raised nowhere and left the broker no
    dispense to reconcile, so the honest diagnostic says exactly that.
    """
    error = phase_dir / f"task-{idx:05d}" / "runner-error.txt"
    if error.is_file():
        # Redacted where it was written, so it can be read back verbatim here.
        body = error.read_text(encoding="utf-8", errors="replace").strip()
        return f"the task did not run: {body}"
    return "no row: nothing recorded an outcome for this held-out id"


def read_eval_phase(phase_dir: Path, task_ids: Sequence[str] | Sequence[int]) -> list[TaskResult]:
    """Every requested held-out id of an eval phase, read from its per-task provenance directories.

    An eval phase gives each task its own stream under ``<phase>/task-<id>/``, because one fresh
    session per task is what serving a single task per stream enforces. So the phase's rows are
    spread across those subdirectories, and ``read_phase`` cannot find them at the phase root:
    it reads one directory's ``results.jsonl`` and does not recurse, and neither do the shogym
    readers under it. This walks the task directories and concatenates in task-id order, which
    is the same list ``run_eval_phase`` returns for a phase it ran in one process. It is how a
    continuation reads back the eval that finished before the interruption, so the published
    result carries the before side of the pair rather than an empty one.

    ``task_ids`` is the committed held-out set, and the phase is filled out against it. A task
    can leave the record holding nothing at all: the runner caught its exception and wrote only a
    breadcrumb, or the harness exited before calling ``get_task`` and nothing raised anywhere.
    Read by directory alone, that id is simply not in the list, and an id in neither phase's list
    disappears from the measurement while the manifest goes on saying it was requested. So an id
    with no row of its own gets an explicit unscored one carrying why it has none.
    """
    rows: list[TaskResult] = []
    for task_dir in sorted(p for p in phase_dir.glob("task-*") if p.is_dir()):
        rows.extend(read_phase(task_dir))
    return fill_missing(
        rows,
        task_ids=[int(task_id) for task_id in task_ids],
        diagnostic=lambda idx: _no_row_diagnostic(phase_dir, idx),
    )


def _single_task_split(split: Split, phase: str, task_id: str) -> Split:
    """A one-task view of the split, so the phase's stream can only dispense that task."""
    from shobench.splits import Side

    side = side_for_phase(split, phase)
    one = Side(task_ids=(task_id,), env_kwargs=dict(side.env_kwargs))
    if phase == "rollout":
        return replace(split, pool=one)
    return replace(split, heldout=one)


SUSPENSION_FILE = "suspended.json"
# The baseline's eval_before rows, carried INTO a bookend run directory at creation and
# published as the bookend artifact's own before block. The baseline's result artifact lives
# under the shared cell stem and is routinely evicted by later same-cell publications, so an
# artifact that depended on it could stop assembling the day another run of the cell
# published; carrying the rows makes the bookend artifact self-contained, and the block is
# labeled with the run id it came from so no reader mistakes it for rows this run measured.
BASELINE_BEFORE_FILE = "baseline_before.json"
# The rollout's stop classification, persisted beside its provenance the moment the rollout ends.
# Only the runner sees how a leg ended, so this is the one piece of the rollout that cannot be
# re-read from the shogym record. An eval_after suspension leaves the rollout finished but the
# results file unwritten, and its resume republishes the rollout from provenance plus this file,
# so the stop reason survives an interruption that happens after the rollout is already paid for.
ROLLOUT_STOPPING_FILE = "rollout_stopping.json"
# An operator's ask that this run end through its normal path, written into the live run
# directory by `shobench stop` and read by the runner that owns it.
#
# A file the runner polls rather than a signal it handles: the run directory is the run's identity
# and a process is not, since a suspended rollout is continued by a DIFFERENT process against the
# same directory. The only pid a run records is in run.lock, which is never unlinked, so a
# finished run names a pid the operating system may since have handed to something else.
STOP_REQUEST_FILE = "stop.request.json"
# Seconds between a run's looks for that ask.
STOP_POLL_S = 2.0
# The run's whole egress capture. One process writes it and any continuation appends to it, so
# the published summary covers the cell rather than only its last stretch.
EGRESS_LOG = "egress.tsv"
# The exit status a suspended cell leaves. `run` cannot return it, because a suspension is the
# one path that must not unwind, so the status is how a shell or a supervising script tells a
# cell that is waiting for a window from one that failed. 75 is the conventional "temporary
# failure, try the same thing again later", which is exactly what this is.
SUSPENDED_EXIT_CODE = 75


def write_stop_request(run_dir: Path, *, reason: str = "") -> dict[str, Any]:
    """Record an operator's ask that this run end, and return what was written.

    Written by the CLI rather than by a runner, so nothing here assumes a live process. The reason
    is free text and reaches a published artifact through the leg verdict, so it is redacted where
    it is published rather than here.
    """
    request = {
        "schema": "shobench.stop_request/1",
        "requested_at": time.time(),
        "reason": reason,
    }
    write_json(run_dir / STOP_REQUEST_FILE, request)
    return request


def read_stop_request(run_dir: Path) -> dict[str, Any] | None:
    """The operator's ask, or ``None`` where there is none or it cannot be read.

    An unreadable request reads as no request, which is the fail-safe direction: a half-written
    file is one a writer is still finishing, and the next poll reads the finished one.
    """
    path = run_dir / STOP_REQUEST_FILE
    try:
        request = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return request if isinstance(request, dict) else None


def _honor_stop_request(run_dir: Path, stop: EarlyEnding) -> bool:
    """Latch an operator's ask onto a run's handle and consume it. Says whether there was one.

    The unlink is the ACKNOWLEDGMENT, which is why it happens here and nowhere else: the CLI waits
    on exactly it, and it is what keeps the ask one-shot, since a request left on disk would end
    the next process to open the directory.

    Firing the handle is all this does; ending the container is the supervisor's. The verdict is
    built off the base :class:`Harness` because the ownership that watches for an ask exists
    before a cell's harness is known.
    """
    request = read_stop_request(run_dir)
    if request is None:
        return False
    stop.fire(Harness().operator_verdict(request=request))
    with contextlib.suppress(OSError):
        (run_dir / STOP_REQUEST_FILE).unlink()
    return True


@dataclass(frozen=True)
class Suspension:
    """A rollout stopped by a provider limit, with everything a continuation needs.

    Everything here is read back from disk by ``shobench resume``, so it holds inputs rather
    than objects: the session to reattach to, how much of the one rollout wall clock is already
    spent, and where the record stood. The provenance directory and the harness are not in it
    twice, since the manifest beside it already names them.
    """

    run_id: str
    session_id: str | None
    elapsed_rollout_s: float
    tasks_dispensed: int
    pool_queued: int
    legs_before: int
    suspended_at: float
    # The clock the interrupted rollout was actually given. It rides in the record instead of
    # being re-read from the cell file, because a cell file is a working document and a running
    # experiment is not: a budget edited while the run waited for its window would otherwise
    # change the length of a rollout that is already half spent.
    rollout_wall_clock_s: int

    @property
    def remaining_rollout_s(self) -> int:
        """What is left of the one wall clock the rollout was given, never a fresh one.

        A continuation inherits the run's remaining time. Handing it a budget again would make
        the measured rollout as long as its interruptions were frequent, and a cell suspended
        twice would quietly be a longer experiment than one that ran straight through. Repeated
        suspensions accumulate into ``elapsed_rollout_s``, so this shrinks monotonically to zero
        and the operator is told when there is nothing left.
        """
        return int(max(0.0, self.rollout_wall_clock_s - self.elapsed_rollout_s))

    @classmethod
    def read(cls, run_dir: Path) -> Suspension:
        record = json.loads((run_dir / SUSPENSION_FILE).read_text(encoding="utf-8"))
        return cls(
            run_id=record["run_id"],
            session_id=record["session_id"],
            elapsed_rollout_s=float(record["elapsed_rollout_s"]),
            tasks_dispensed=int(record["tasks_dispensed"]),
            pool_queued=int(record["pool_queued"]),
            legs_before=int(record["legs_before"]),
            suspended_at=float(record["suspended_at"]),
            rollout_wall_clock_s=int(record["rollout_wall_clock_s"]),
        )


@dataclass(frozen=True)
class EvalSuspension:
    """An eval phase stopped by a provider limit, with what a continuation needs to finish it.

    An eval phase suspends differently from the rollout, because it is not one session on one
    clock: it is one fresh session per held-out task, and completion is a property of each task's
    own provenance rather than of a shared wall clock. So this holds no session and no clock. It
    names the phase that was interrupted (a resume runs a different set of phases for an
    eval_before than for an eval_after), and it records which ids were finished and which were
    not, for an operator reading the plan. The resume itself re-derives both from provenance, the
    ground truth, so a row written between the suspension and the continuation is not missed.
    """

    run_id: str
    phase: str
    completed_task_ids: tuple[int, ...]
    pending_task_ids: tuple[int, ...]
    suspended_at: float

    @classmethod
    def read(cls, run_dir: Path) -> EvalSuspension:
        record = json.loads((run_dir / SUSPENSION_FILE).read_text(encoding="utf-8"))
        return cls(
            run_id=record["run_id"],
            phase=record["phase"],
            completed_task_ids=tuple(record.get("completed_task_ids", ())),
            pending_task_ids=tuple(record.get("pending_task_ids", ())),
            suspended_at=float(record["suspended_at"]),
        )


def recorded_rollout_feedback(manifest: dict[str, Any]) -> str:
    """The feedback arm the recorded run actually served.

    A manifest written before the axis existed carries no key, and that absence is
    unambiguous: never was the only rollout posture before the axis, so absence reads as
    never rather than as whatever the checkout's default is today.
    """
    return str(manifest.get("cell", {}).get("rollout_feedback") or "never")


def recorded_eval_context(manifest: dict[str, Any]) -> str:
    """The eval context the recorded run actually measured under.

    Same rule as the feedback arm: a manifest written before the axis existed carries no key,
    and cold was the only eval posture then, so absence reads as cold rather than as the
    checkout's default today. A continuation that read the default instead would fork a
    conversation into a measurement whose before-bookend never had one.
    """
    return str(manifest.get("cell", {}).get("eval_context") or "cold")


# Which definitional edits a reopening refuses, named by what the reopening produces.
#
# A CONTINUATION writes more of a measurement that already exists, under the run id that already
# names it: ``resume`` finishes the phases a usage limit interrupted, and ``rerun-eval`` fills
# rows infrastructure lost. Both refuse every definitional edit there is, because one run id must
# not describe two experiments, and rows measured under a second definition spliced into one
# artifact are exactly that.
#
# A BOOKEND is a new run. ``rebookend`` reuses only the source's finished home and its terminal
# session, runs eval_after alone, and publishes under an id of its own; the source's rollout is
# complete and immutable, so an edit made after it ended cannot reach it and cannot turn the
# bookend into a second description of it. What it must not do is measure its after side by a
# different rule than the before side it will be paired against, so nothing is allowed to drift:
# the arm and the eval runtime are RECOVERED from the record and run under, and every field the
# comparison covers refuses. What the bookend does not inherit is the cell file's digest, which
# moves for a comment as readily as for a swapped model and so answers the wrong question.
DRIFT_CONTINUATION = "continuation"
DRIFT_BOOKEND = "bookend"
DRIFT_SCOPES = (DRIFT_CONTINUATION, DRIFT_BOOKEND)

# The eval runtime a bookend takes from the record rather than from the cell file, because both
# sides of a paired measurement have to be scored under one stopping rule.
#
# eval_task_timeout_s is the deadline the held-out task stream is built with AND the leg's hard
# timeout, so a task that seals between the recorded bound and a shorter current one is scoreable
# on the before side and force-stopped on the after side. The paired delta would then be two
# measurements under different rules, which no amount of recording in the artifact repairs.
#
# eval_concurrency is not score-neutral either. Per-task homes and sessions isolate task STATE,
# not the resources a timed leg needs: concurrent legs share the host, the network and the
# provider account while their per-task clocks run, so contention and throttling turn into
# timeouts and unscored rows. It comes from the record for the same reason.
BOOKEND_EVAL_RUNTIME_FIELDS = ("eval_task_timeout_s", "eval_concurrency")

# The cell fields a bookend does not compare at all, each for its own reason.
#
# eval_context is the one axis a rebookend changes by definition: the runner pins resumed
# whatever the file says, and every source this entry exists for measured its after cold, so
# comparing it would refuse every bookend there is. The bookend's axes block states what it ran.
#
# split and instruction_arm are lookup keys rather than identities. The held-out ids are
# identified by the split's id digest and the prompts by their system-prompt digests, and both
# are compared against the record separately below; a file renamed with its content unchanged is
# not a different measurement, and one edited in place is caught by the digest.
#
# config_sha256 is the check this scope exists to replace. It covers every byte of the file,
# comments and runtime fields alike, so it cannot tell a retuned timeout from a swapped model.
# It is still reported into the bookend's record, where "the file itself changed" is worth a
# reader knowing.
#
# config_path is where the file sits in the checkout and note is free text carried to a reader.
# Neither reaches a session.
BOOKEND_UNCOMPARED_CELL_FIELDS = (
    "eval_context",
    "split",
    "instruction_arm",
    "config_sha256",
    "config_path",
    "note",
)

# Every other field in the block refuses on any difference, including a field added after this
# comment was written. Named in full, because a field nobody judged is a field nobody can rely on.
#
# env, harness, model, effort, credential_mode and env_kwargs are what the bookend's own eval
# sessions are made of: the environment, the CLI, the model behind it, the effort it thinks at,
# the account that serves it, and how the environment is constructed (tau2's task split, hle's
# judge).
#
# rollout_feedback is the source's arm and eval_task_timeout_s and eval_concurrency are the eval
# runtime. All three are recovered from the record onto the cell before the comparison runs, so
# the checkout's values cannot leak into the bookend; the comparison holds those recoveries to
# their word, and refuses where the record carried no value to recover.
#
# max_in_flight, rollout_wall_clock_s and pool_ceiling define the rollout the bookend inherits a
# home from. The bookend runs none of it, so they cannot change its eval, but a bookend that
# published them as today's numbers would label the source's arm with a rollout nobody ran.
#
# required_env is a precondition rather than a measurement, and it refuses anyway: a cell that
# needs a key it did not need before is a cell whose legs are produced differently, and the
# operator is the one who can say whether the edit was meant for this bookend.
#
# name cannot differ, since the cell is loaded by the name the record carries. It refuses rather
# than resting on that.

# The cell fields that shape the ROLLOUT and reach nothing else, which is what makes them the one
# group a PAIRING does not compare. A deferred baseline runs eval_before alone: ``EvalStream``
# pins the blind feedback posture whatever the cell's arm says and refuses a provenance directory
# recorded under any other, the eval fan-out is one session per task whatever max_in_flight says,
# and neither the rollout's wall clock nor its serving ceiling is read by an eval phase. Both v0
# pairs really do differ here, their sources having run the immediate arm and their deferred
# baselines the never arm, so comparing these would refuse every pairing there is over what
# provably cannot reach a before row. The source-to-checkout comparison still refuses them, for a
# different reason: there they would relabel the arm the bookend publishes.
ROLLOUT_ONLY_CELL_FIELDS = (
    "rollout_feedback",
    "max_in_flight",
    "budget.rollout_wall_clock_s",
    "budget.pool_ceiling",
    "budget.rollout_no_progress_s",
)

# The digests a bookend rests on once it has given up the whole-cell one: the only remaining proof
# that its eval measures the run's held-out ids under the run's prompts.
IDENTITY_DIGESTS = (
    ("split ids", "split", "id_digest"),
    ("rollout instruction", "instruction", "rollout_system_sha256"),
    ("eval instruction", "instruction", "eval_system_sha256"),
)

# WHEN each fact can be compared, which is a property of the fact rather than of the caller. Most
# are knowable from the checkout and docker before anything runs. Two are not: a harness probe
# exists only once a container has run one, and the effective credential mode only once a
# credential has been seeded, so those are checked at the point they become known, before any row
# is filled, rather than being dropped for being awkward.
IDENTITY_PRE_SPEND = "pre_spend"
IDENTITY_AFTER_SETUP = "after_setup"

# A row's EXECUTION IDENTITY outside the cell block: what it was measured over and under, and
# what ran it. The kickoff is here because no cell digest covers it, the instructions living
# outside cells/, and the image tag because it names the CLI that ran, with its content id below.
#
# The effective credential mode says which KIND of account served the legs (subscription, api_key
# or unknown) and no more: two different subscription accounts record the same value and compare
# equal, so account identity is NOT established here. It is not recorded either, deliberately.
# The auth files carry rotating tokens, so hashing one fingerprints a credential rather than an
# account and would refuse a pairing across an ordinary refresh, and their only stable
# identifiers are the account email and id, which must never enter a published artifact.
PAIRING_IDENTITY_FIELDS = (
    ("split ids", "split.id_digest", IDENTITY_PRE_SPEND),
    ("eval instruction", "instruction.eval_system_sha256", IDENTITY_PRE_SPEND),
    ("eval kickoff", "instruction.kickoff", IDENTITY_PRE_SPEND),
    ("agent image tag", "container.agent_image", IDENTITY_PRE_SPEND),
    ("credential mode", "axes.credential_mode.effective", IDENTITY_AFTER_SETUP),
)

# Blocks compared whole, key by key, because pinning what produced a row is their entire purpose.
#
# substrate is the code the row ran on: the shogym revision that serves and scores every task, the
# repo it comes from, the MCP server name the agent's tools appear under, and this runner's own
# revision, whose absence is handled below.
# harness_probes is what the harness reported from inside the image.
# axes.effort is not a restatement of the cell's effort: ``requested`` is the cell's ask, and
# ``applied`` and ``how`` are whether that ask reached the harness at all. A before side that
# applied an effort the source's did not is a different measurement, and only this block records
# the difference.
# split.provenance is what the held-out POSITIONS resolve against. id_digest hashes the env name,
# the ids and the env kwargs, which is positions rather than content: tau2 resolves those
# positions against a byte-verified upstream tree whose sha lives here, so two archives can share
# every id and score different task bytes. Recording is what makes this checkable, and it is only
# as good as what an env records: hle carries no immutable dataset revision today, so for hle this
# proves the split file and not the dataset behind it. That gap is real and is not closed here.
#
# A key added to any of these is eval-defining until someone judges otherwise, which is the
# fail-closed direction and the reason they are compared whole rather than field by named field.
PAIRING_IDENTITY_BLOCKS = (
    ("substrate", IDENTITY_PRE_SPEND),
    ("split.provenance", IDENTITY_PRE_SPEND),
    ("axes.effort", IDENTITY_PRE_SPEND),
    ("harness_probes", IDENTITY_AFTER_SETUP),
)

# The identities no archive carries yet, and so the only ones whose absence is tolerated rather
# than refused: every other fact here is in every real archive, and requiring these would refuse
# every pairing that exists. Recording them starts now; ``_identity_agreement`` holds the rule.
# A dirty runner tree reads as an absence too, its commit not identifying the code that ran.
PAIRING_VERSIONED_IDENTITY = (
    ("agent image digest", "container.image_digest", IDENTITY_PRE_SPEND),
    ("runner revision", "substrate.shobench_rev", IDENTITY_PRE_SPEND),
)

# Keys inside the compared blocks that identify nothing on their own. ``shobench_dirty`` says
# whether the revision beside it identifies anything, and it is read exactly there; comparing it
# separately would refuse a pair of edited checkouts for agreeing that they were edited.
PAIRING_UNCOMPARED_IDENTITY_PATHS = ("substrate.shobench_dirty",)

# Deliberately NOT compared, each for a reason a reader can check.
#
# instruction.continuation is the cue that reopens a ROLLOUT; no eval leg is ever sent it.
# instruction.rollout_system_sha256 is the standing prompt of a rollout a deferred baseline never
# ran, and the bookend's own resumed after side carries it, where the source-to-checkout
# comparison guards it.
# instruction.arm and split.path are lookup keys whose identities are the digests above.
# axes.model.observed and observed_models are OUTCOMES read off the traces, not definitions, and
# the two sides' come from different phases: in the real terra pair the source recorded
# ['gpt-5.6-terra'] from its rollout and the baseline [] from its before legs, so comparing them
# would refuse a pairing for having measured something.
# axes.model.requested restates a cell field already compared. axes.effort does NOT, and is
# compared as a block above.
# substrate.shobench_dirty is not compared for its own sake: it says whether the revision beside
# it identifies anything, and that is read where the revision is compared.
# container.network, container.netns_container, container.home, home, work, redaction,
# credential_seed, run_id, started_at, ended_at, resumptions, eval_reruns and operator_stop are
# run-local bookkeeping: they name this run's resources and history, not what its rows were
# produced by.
# schema names the record's shape rather than the measurement, and every field the pairing rests
# on is compared by name, so a purely additive bump must not refuse every archive that predates
# it.

# What a pairing compares between the two RECORDED runs, as everything the cell block carries
# except these. The bookend's uncompared bookkeeping is uncompared here for the same reasons; the
# rollout-only fields for the reason above; and the eval runtime because it is compared under the
# stricter rule below, where both sides must state it rather than merely agree.
PAIRING_UNCOMPARED_CELL_FIELDS = (
    *BOOKEND_UNCOMPARED_CELL_FIELDS,
    *ROLLOUT_ONLY_CELL_FIELDS,
    *(f"budget.{field}" for field in BOOKEND_EVAL_RUNTIME_FIELDS),
)

# How a missing field is SHOWN, in a refusal line and in the record a bookend publishes.
CELL_FIELD_ABSENT = "<absent>"

# How a missing field is COMPARED. Absence has to be an identity rather than a spelling, since
# model and effort take arbitrary text and reach harness session construction, so a field whose
# value spells like the marker must not compare equal to a missing one. None cannot serve either,
# being a legitimate recorded value (a cell with no pool_ceiling records null). This object never
# leaves the comparison: it becomes the display form only once the two sides are known to differ.
_MISSING = object()

# The only absences read as values rather than as differences, because each is a versioned axis
# whose pre-axis meaning is known: never was the only rollout posture before the feedback arm
# existed, cold the only eval posture before the eval context did, and a rollout ran under no
# no-progress bound at all before that bound existed, which is what 0 spells. Nothing else is
# normalized. A field this list does not name is missing, not defaulted, and missing refuses.
LEGACY_AXIS_DEFAULTS: dict[str, Any] = {
    "rollout_feedback": "never",
    "eval_context": "cold",
    "budget.rollout_no_progress_s": 0,
}


def _flat_cell(cell_manifest: dict[str, Any]) -> dict[str, Any]:
    """The cell manifest block as one level of dotted names, so a field is comparable by name.

    Only the budget table nests, and its fields are judged one at a time rather than as a block,
    so it is the only thing flattened. ``env_kwargs`` stays whole deliberately: it is one
    statement of how the environment is constructed, and half of it agreeing means nothing.
    """
    flat: dict[str, Any] = {}
    for key, value in cell_manifest.items():
        if key == "budget" and isinstance(value, dict):
            flat.update({f"budget.{name}": item for name, item in value.items()})
        else:
            flat[key] = value
    return {**LEGACY_AXIS_DEFAULTS, **flat}


def _cell_differences(
    recorded_cell: dict[str, Any], current_cell: dict[str, Any]
) -> list[tuple[str, Any, Any]]:
    """Every field the two cell blocks state differently, over the UNION of their fields.

    Field by field rather than by the file's digest, because the digest answers only whether the
    bytes moved: a caller that has to decide whether the difference matters, or that has to tell
    a reader which value ran, gets nothing from it.

    The union is what makes it fail closed. A field added to the cell after a run was recorded
    exists on one side only, and so does one since removed, so presence is compared before values
    and a field missing on one side differs from one present with any value at all. The versioned
    axes are the exception, and only because their pre-axis meaning is known.

    ``_MISSING`` is kept in the result rather than rendered, so each caller can say absence its
    own way: unambiguously in a refusal line, and as the published string in the record.
    """
    recorded = _flat_cell(recorded_cell)
    current = _flat_cell(current_cell)
    differences = []
    for name in sorted(set(recorded) | set(current)):
        was = recorded.get(name, _MISSING)
        now = current.get(name, _MISSING)
        present = (was is not _MISSING, now is not _MISSING)
        if present[0] != present[1] or (all(present) and was != now):
            differences.append((name, was, now))
    return differences


def _shown(value: Any) -> str:
    """A side of a difference as a refusal line says it: no such field, or the value itself.

    The two forms are deliberately not interchangeable: a field missing from the record needs a
    value added or a record repaired, and a field whose value merely SPELLS like the absence
    marker needs neither. The published record keeps one string for both.

    Long values are cut, because a harness probe is a page of CLI output and a refusal nobody can
    read is a refusal nobody reads. Both manifests carry the whole value either way.
    """
    if value is _MISSING:
        return "no such field"
    shown = repr(value)
    return shown if len(shown) <= 120 else f"{shown[:117]}..."


def cell_field_drift(
    recorded_cell: dict[str, Any], current_cell: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """The differences as a record carries them: dotted field name to both values.

    The same computation the refusal reads, so what an operator is refused for and what an
    artifact reports can never be two different answers.
    """
    return {
        name: {
            "recorded": CELL_FIELD_ABSENT if was is _MISSING else was,
            "checkout": CELL_FIELD_ABSENT if now is _MISSING else now,
        }
        for name, was, now in _cell_differences(recorded_cell, current_cell)
    }


def recorded_no_progress_bound(manifest: dict[str, Any]) -> int:
    """The no-progress bound the recorded rollout actually ran under.

    Absence is a value here rather than a gap, unlike the eval runtime below: every rollout before
    the bound existed ran with none, so a record without the field states an unbounded rollout as
    surely as a record with a zero does.
    """
    budget = manifest.get("cell", {}).get("budget", {})
    return int(budget.get("rollout_no_progress_s") or 0)


def bookend_cell(cell: Cell, source_manifest: dict[str, Any]) -> Cell:
    """The cell a rebookend actually runs: the source's recorded arm and eval runtime, resumed.

    Four fields come from the record rather than from the checkout, each for its own reason. The
    feedback arm is what the source's rollout served, so a never-arm source publishes honestly as
    never plus resumed. The eval runtime is the stopping rule the before side was scored under,
    and a paired delta whose two sides ran under different rules is two measurements rather than
    one. The eval context is the single axis a rebookend exists to change, so it is pinned rather
    than inherited. The rollout's no-progress bound is a fact about the rollout the bookend
    inherits a home from and never re-runs, so the checkout's current value would relabel it.

    The runner and the plan both build the cell here, so the plan's verdict is the runner's.
    Where the record carries no eval runtime to inherit, the checkout's value stands and the
    drift check refuses it: an absence is not a value, and only the operator can say what a run
    recorded before the field existed was measured under.
    """
    return replace(
        cell,
        rollout_feedback=recorded_rollout_feedback(source_manifest),
        eval_context="resumed",
        budget=replace(
            cell.budget,
            rollout_no_progress_s=recorded_no_progress_bound(source_manifest),
            **_inheritable_eval_runtime(source_manifest),
        ),
    )


def reopened_cell(cell: Cell, manifest: dict[str, Any]) -> Cell:
    """The cell a reopening runs, rebuilt from the run's own record rather than off the checkout.

    Two recoveries apply to every run. The feedback arm and the eval context are the axes the run
    was measured under, and the checkout's defaults may have moved since; shogym would refuse to
    reopen the provenance directory under the other regime anyway, so recovering them makes this
    an explicit reconstruction rather than a refusal an operator has to decode.

    The third applies to a BOOKEND, and it is the one the whole-config check cannot catch: a
    bookend's eval runtime came from the run it bookends, so its record legitimately differs from
    its cell file while its recorded config_sha256 IS that unchanged file's digest. The digest
    check therefore passes the reopening, and a reopening that read the budget off the checkout
    would finish the remaining ids under a rule the finished ones never saw. The recorded cell
    block is the authority, so a bookend published before the runtime was inherited also
    reconstructs to what it ran.
    """
    cell = replace(
        cell,
        rollout_feedback=recorded_rollout_feedback(manifest),
        eval_context=recorded_eval_context(manifest),
        # A reopening runs no rollout, so this changes nothing it does; it keeps the
        # reconstructed cell from describing the finished rollout by today's number.
        budget=replace(
            cell.budget, rollout_no_progress_s=recorded_no_progress_bound(manifest)
        ),
    )
    if "rebookend" not in manifest:
        # Every other run recorded the checkout's own budget, and the digest check that follows
        # refuses if the file moved since, so there is nothing more here to reconstruct.
        return cell
    return replace(cell, budget=replace(cell.budget, **_inheritable_eval_runtime(manifest)))


def _inheritable_eval_runtime(manifest: dict[str, Any]) -> dict[str, Any]:
    """The eval runtime fields a record can hand over, absences left out.

    A field the record does not carry is not inherited and not defaulted: the caller keeps what
    it had and the drift check refuses the difference, because only an operator can say what a
    run recorded before the field existed was measured under.
    """
    budget = manifest.get("cell", {}).get("budget", {}) or {}
    return {
        field: budget[field]
        for field in BOOKEND_EVAL_RUNTIME_FIELDS
        if budget.get(field) is not None
    }


def recorded_eval_runtime(manifest: dict[str, Any]) -> dict[str, Any]:
    """The eval runtime a run recorded, as a reader is shown it, absences included."""
    budget = manifest.get("cell", {}).get("budget", {}) or {}
    return {field: budget.get(field, CELL_FIELD_ABSENT) for field in BOOKEND_EVAL_RUNTIME_FIELDS}


def _recorded_value(manifest: dict[str, Any], block: str, key: str) -> Any:
    """One recorded field, with a null read as the absence it is rather than as a value."""
    value = (manifest.get(block) or {}).get(key)
    return _MISSING if value is None else value


def _recorded_path(manifest: dict[str, Any], path: str) -> Any:
    """One recorded field named by its dotted path, absent or null reading as absence."""
    value: Any = manifest
    for step in path.split("."):
        if not isinstance(value, dict) or step not in value:
            return _MISSING
        value = value[step]
    return _MISSING if value is None else value


def _pairing_identity_lines(label: str, source_value: Any, baseline_value: Any) -> list[str]:
    """One identity compared between two archives: stated on both sides, and the same.

    Absence refuses rather than passing, the whole point of an identity being that a record
    ASSERTS what produced its rows; two silences agree about nothing.
    """
    if source_value is _MISSING or baseline_value is _MISSING:
        return [
            f"{label} is not recorded on both sides (source {_shown(source_value)}, baseline "
            f"{_shown(baseline_value)}), so nothing proves the two archives measured the same way"
        ]
    if source_value != baseline_value:
        return [
            f"{label} differs (source {_shown(source_value)}, baseline {_shown(baseline_value)})"
        ]
    return []


def _versioned_identity(manifest: dict[str, Any], path: str) -> Any:
    """A versioned identity as recorded, with a runner revision from a dirty tree read as absent.

    A commit is an identity only when the tree that ran was that commit. A modified one shares
    the sha and not the code, so two edited checkouts must not prove anything about each other.
    """
    value = _recorded_path(manifest, path)
    if path == "substrate.shobench_rev" and _recorded_path(manifest, "substrate.shobench_dirty"):
        return _MISSING
    return value


def identity_facts(stage: str | None = None) -> tuple[tuple[str, str, bool], ...]:
    """The enumerated identity as (label, path, versioned) triples, optionally for one stage.

    One list, read by all three comparisons, so archive-to-archive, archive-to-checkout and
    archive-to-execution can never be checking different things. Block members expand to their
    keys at comparison time, since a block's keys are whatever the two records carry.
    """
    facts: list[tuple[str, str, bool]] = []
    for label, path, fact_stage in PAIRING_IDENTITY_FIELDS:
        if stage in (None, fact_stage):
            facts.append((f"{label} ({path})", path, False))
    for label, path, fact_stage in PAIRING_VERSIONED_IDENTITY:
        if stage in (None, fact_stage):
            facts.append((f"{label} ({path})", path, True))
    return tuple(facts)


def identity_blocks(stage: str | None = None) -> tuple[str, ...]:
    """The blocks compared key by key, optionally for one stage."""
    return tuple(
        block for block, fact_stage in PAIRING_IDENTITY_BLOCKS if stage in (None, fact_stage)
    )


def _identity_agreement(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_name: str,
    right_name: str,
    stage: str | None,
    archives: bool,
) -> tuple[list[str], list[str]]:
    """Two identities compared fact by fact: what disagrees, and what neither could state.

    THE absence discipline, stated once here because every comparison in this module routes
    through it. ``archives`` picks which of the two applies, and they differ because the sides
    differ.

    Between two ARCHIVES, both are finished records of finished runs, so:

        both state it   -> compared, and a difference refuses;
        one states it   -> refuses, one record proving what the other cannot;
        neither         -> refuses, unless the fact is versioned (no archive written before it
                           can carry it), in which case it is named as unproven instead.

    Against a CURRENT identity, which is not a finished record (docker may not answer for a
    digest, a path may take no probe, and the archive may predate the field):

        both state it   -> compared, and a difference refuses;
        either silent   -> named as unproven.

    Both disciplines put the same fact in the same list, which is what makes one published
    unproven list mean one thing.
    """
    lines: list[str] = []
    unproven: list[str] = []
    for label, path, versioned in identity_facts(stage):
        left_value = _versioned_identity(left, path) if versioned else _recorded_path(left, path)
        right_value = (
            _versioned_identity(right, path) if versioned else _recorded_path(right, path)
        )
        missing = left_value is _MISSING or right_value is _MISSING
        if missing and (not archives or (versioned and left_value is right_value)):
            unproven.append(path)
            continue
        lines += _identity_lines(label, left_value, right_value, left_name, right_name)
    versioned_paths = {path for _, path, _stage in PAIRING_VERSIONED_IDENTITY}
    for block in identity_blocks(stage):
        left_block = _recorded_path(left, block)
        right_block = _recorded_path(right, block)
        left_keys = set(left_block) if isinstance(left_block, dict) else set()
        right_keys = set(right_block) if isinstance(right_block, dict) else set()
        if archives and not (left_keys and right_keys):
            # An absent block in an archive is not an empty difference: it is a record that names
            # none of what its rows were produced by, leaving nothing to compare.
            lines.append(
                f"{block} is not recorded on both sides ({left_name} "
                f"{'recorded' if left_keys else 'no such block'}, {right_name} "
                f"{'recorded' if right_keys else 'no such block'}), so nothing proves the two "
                "were produced the same way"
            )
            continue
        for key in sorted(left_keys | right_keys):
            path = f"{block}.{key}"
            if path in versioned_paths or path in PAIRING_UNCOMPARED_IDENTITY_PATHS:
                # Owned by the versioned rule, or judged to identify nothing on its own.
                continue
            left_value = _recorded_path(left, path)
            right_value = _recorded_path(right, path)
            if left_value is _MISSING or right_value is _MISSING:
                if not archives:
                    unproven.append(path)
                    continue
            lines += _identity_lines(path, left_value, right_value, left_name, right_name)
    return lines, sorted(set(unproven))


def _identity_lines(
    label: str, left_value: Any, right_value: Any, left_name: str, right_name: str
) -> list[str]:
    """One fact's verdict as a reader sees it, absence and difference worded differently."""
    if left_value is _MISSING or right_value is _MISSING:
        return [
            f"{label} is not recorded on both sides ({left_name} {_shown(left_value)}, "
            f"{right_name} {_shown(right_value)}), so nothing proves the two measured the same way"
        ]
    if left_value != right_value:
        return [
            f"{label} differs ({left_name} {_shown(left_value)}, "
            f"{right_name} {_shown(right_value)})"
        ]
    return []


def current_identity(
    *,
    cell: Cell,
    split: Split,
    instruction: Instruction,
    harness: Harness,
    image_tag: str,
    image_digest_value: str | None,
) -> dict[str, Any]:
    """The identity of the run about to happen, in manifest shape, for what is knowable pre-spend.

    Manifest shape rather than a bespoke structure, so the same paths name the same facts on both
    sides of the comparison and a field added to the manifest is a field the comparison can be
    taught in one place. Everything here is knowable before a container exists: the checkout's
    split and prompts, the resolved image, the substrate, and the effort the harness will or will
    not apply. What is not knowable yet is in ``after_setup_identity``.
    """
    return {
        "split": {
            "id_digest": split.to_manifest().get("id_digest"),
            "provenance": dict(split.provenance),
        },
        "instruction": {
            "eval_system_sha256": instruction.eval_system_sha256,
            "kickoff": instruction.kickoff,
        },
        "container": {"agent_image": image_tag, "image_digest": image_digest_value},
        "substrate": substrate_block(),
        "axes": {"effort": effort_axis(cell, harness)},
    }


def after_setup_identity(
    *, probes: dict[str, str] | None, credential_mode: dict[str, Any] | None
) -> dict[str, Any]:
    """The identity facts that exist only once a container has run and a credential is placed.

    A harness probe is output from inside the image and the effective credential mode is read off
    the seeded home, so neither can be compared before setup. A path that takes no probe passes
    none, and the facts it cannot state are named as unproven rather than assumed: the reopen
    paths are that case, and the image content id, compared pre-spend, is a stricter statement
    about the same image than a version string would be.
    """
    identity: dict[str, Any] = {"harness_probes": dict(probes or {})}
    if credential_mode is not None:
        identity["axes"] = {"credential_mode": credential_mode}
    return identity


def execution_drift(
    recorded: dict[str, Any], current: dict[str, Any], *, stage: str
) -> tuple[list[str], list[str]]:
    """The third side: what the run ABOUT TO HAPPEN does not share with the record it continues.

    A pairing proves two archives agree with each other and a drift check proves the cell file
    has not moved, and neither says anything about the image, the substrate, the prompts as sent
    or the effort as applied that the new rows will actually be produced under. Recording those
    in the new manifest documents a mismatch rather than preventing it, so they are compared
    here, before ``_run_phases``, and a stated disagreement refuses.

    ``stage`` says which facts are knowable yet: see ``IDENTITY_AFTER_SETUP``. The caller
    publishes the returned unproven names, so the artifact says what it could not establish.
    """
    return _identity_agreement(
        recorded,
        current,
        left_name="recorded",
        right_name="current",
        stage=stage,
        archives=False,
    )


def _refuse_execution_drift(
    recorded: dict[str, Any], current: dict[str, Any], *, stage: str, what: str
) -> list[str]:
    """Refuse a stated disagreement with the record, and hand back what could not be stated.

    Raised rather than returned, unlike the plan-facing checks, because by the time a caller has
    a current identity in hand it has decided to spend: the only useful thing to do with a
    disagreement here is to stop before a row exists.
    """
    lines, unproven = execution_drift(recorded, current, stage=stage)
    if lines:
        raise RuntimeError(
            f"the execution identity no longer matches the run {what}: "
            + "; ".join(lines)
            + ". Rows produced here would not belong to the measurement they would be published "
            "as."
        )
    return unproven


def pairing_unproven(
    source_manifest: dict[str, Any], baseline_manifest: dict[str, Any]
) -> list[str]:
    """The identities NEITHER archive states, named so the artifact can say what it did not prove.

    Published rather than swallowed, so a reader of the delta sees exactly which facts about the
    two sides were never established. ``_identity_agreement`` decides what silence means.
    """
    return _identity_agreement(
        source_manifest,
        baseline_manifest,
        left_name="source",
        right_name="baseline",
        stage=None,
        archives=True,
    )[1]


def pairing_drift(
    source_manifest: dict[str, Any], baseline_manifest: dict[str, Any]
) -> list[str]:
    """What the baseline's before rows were measured by that the source's after rows are not.

    A rebookend publishes ONE measurement out of two archives: the after rows it runs against the
    source's definition, and the before rows it carries from the baseline's. The delta is only a
    measurement if both sides were produced the same way, and matching the cell NAME is no
    evidence of that: two archived runs of one name can sit either side of any edit to the file.
    So the comparison is over what the two RECORDS state, field by field, and everything refuses
    except what provably cannot reach a before row (``PAIRING_UNCOMPARED_CELL_FIELDS``).

    The ROLLOUT instruction is deliberately not compared. A deferred baseline ran no rollout, and
    its before rows were produced under the eval prompt; the rollout prompt the bookend's own
    resumed after side carries is checked against the source instead, where it belongs.

    Returned as lines rather than a bool so the refusal can name every difference at once, and
    shared by the plan and the spending path so a dry run cannot say something the ``--go`` will
    not. Empty means the two archives are the same measurement seen twice.
    """
    lines = [
        f"cell {name} differs (source {_shown(was)}, baseline {_shown(now)})"
        for name, was, now in _cell_differences(
            source_manifest.get("cell", {}), baseline_manifest.get("cell", {})
        )
        if name not in PAIRING_UNCOMPARED_CELL_FIELDS
    ]
    # Every stage, because both sides are finished records: a probe and a credential mode are as
    # available in an archive as a split digest is. The stages only matter to the third
    # comparison, where the current side is still becoming knowable.
    lines += _identity_agreement(
        source_manifest,
        baseline_manifest,
        left_name="source",
        right_name="baseline",
        stage=None,
        archives=True,
    )[0]
    source_runtime = _inheritable_eval_runtime(source_manifest)
    baseline_runtime = _inheritable_eval_runtime(baseline_manifest)
    for bound in BOOKEND_EVAL_RUNTIME_FIELDS:
        was, now = source_runtime.get(bound, _MISSING), baseline_runtime.get(bound, _MISSING)
        if was is _MISSING or now is _MISSING:
            lines.append(
                f"eval runtime {bound} is not recorded on both sides (source {_shown(was)}, "
                f"baseline {_shown(now)}), so the two sides are not known to stop by one rule"
            )
        elif was != now:
            lines.append(
                f"eval runtime {bound} differs (source {was!r}, baseline {now!r}): the before "
                "side and the after side would not be scored under one stopping rule"
            )
    return lines


def experiment_drift(
    manifest: dict[str, Any],
    *,
    cell: Cell,
    split: Split,
    instruction: Instruction,
    scope: str = DRIFT_CONTINUATION,
) -> list[str]:
    """What the checkout now says that the recorded run does not, as human-readable lines.

    A suspended cell can wait hours, and a repository is edited in hours. A continuation runs
    against whatever the files say today, so anything that moved between the two is a different
    experiment wearing the first one's run id: a re-tuned budget, a regenerated split whose
    positions no longer mean what the record says they mean, a reworded instruction the second
    half of the rollout would be run under. The digests to compare are already in the manifest,
    written before anything spent, so this is a comparison rather than a new mechanism.

    ``scope`` names which comparison applies. A continuation refuses on the cell file's digest,
    the strongest statement available: nothing about that run may change. A bookend compares the
    cell field by field instead, because the digest cannot tell a rewritten comment from a
    swapped model; every field it covers still refuses, and the fields a bookend inherits rather
    than reads are recovered onto the cell before this runs, so the comparison is what proves the
    inheritance happened. The split and instruction digests are compared under both scopes.

    Returned rather than raised, so a caller can report every difference at once. An operator
    told about the budget, only to be told about the split on the next attempt, learns to stop
    reading.
    """
    if scope not in DRIFT_SCOPES:
        raise ValueError(f"unknown drift scope {scope!r}; expected one of {DRIFT_SCOPES}")
    recorded_cell = manifest.get("cell", {})
    now_by_label = {
        "split ids": split.to_manifest().get("id_digest"),
        "rollout instruction": instruction.rollout_system_sha256,
        "eval instruction": instruction.eval_system_sha256,
    }
    checks = [
        (label, _recorded_value(manifest, block, key), now_by_label[label])
        for label, block, key in IDENTITY_DIGESTS
    ]
    if scope == DRIFT_CONTINUATION:
        # The whole-cell digest is this scope's proof, and it is stronger than the three below,
        # so a record that predates one of them is not refused for lacking it: the file it
        # hashed is the file this process reads. Absence keeps its old meaning here.
        checks.insert(
            0,
            (
                "cell config",
                _recorded_value(manifest, "cell", "config_sha256"),
                cell.to_manifest().get("config_sha256"),
            ),
        )
        return [
            f"{what} changed since the run started (recorded {was}, now {now})"
            for what, was, now in checks
            if was is not _MISSING and now is not None and was != now
        ]
    # A bookend gives up the whole-cell digest, so these three are the only proof left that its
    # eval measures the source's held-out ids under the source's prompts. An absent one is not
    # agreement, it is a record that cannot say what it measured, and it fails closed the way an
    # absent eval runtime does. Every archived run states all three.
    lines = [
        f"cell {field} changed since the run started "
        f"(recorded {_shown(was)}, checkout {_shown(now)})"
        for field, was, now in _cell_differences(recorded_cell, cell.to_manifest())
        if field not in BOOKEND_UNCOMPARED_CELL_FIELDS
    ]
    for what, was, now in checks:
        if was is _MISSING or now is None:
            lines.append(
                f"{what} is not recorded (recorded {_shown(was)}), so nothing proves this "
                "bookend measures what the run it follows measured"
            )
        elif was != now:
            lines.append(f"{what} changed since the run started (recorded {was}, now {now})")
    return lines


def _fresh_session_id(ctx: RunContext) -> str | None:
    """The id a fresh rollout runs under: pinned when the harness lets the runner choose one.

    Pinning matters more than it looks. A harness that mints its own id writes it into its
    trace, and a leg that dies before writing anything leaves nothing to resume; one the runner
    pinned is resumable from the moment it launches.
    """
    return str(uuid.uuid4()) if ctx.harness.pins_session_id else None


def _finalize_suspension(ctx: RunContext, suspension: dict[str, Any], *, message: str) -> None:
    """Write the suspension, stop the containers, and leave without unwinding. Never returns.

    The one guaranteed hard exit both a rollout and an eval suspension reach, factored out so
    they share it byte for byte. Three things have to be true at once, and only this order gets
    all three. The record has to be on disk before anything else, because it is what a
    continuation reads. The cell's containers have to stop, because a suspended cell may wait
    hours for a window and a running container is a running bill. And the process has to leave
    without unwinding the stream: an orderly close drains whatever task is in flight into a scored
    row, and shogym replays only positions with no row, so the tidy exit is the one that would
    cost the agent the task it was working on. Exiting hard leaves the position row-less, which is
    precisely the state a resume recovers.

    Nothing else in the runner exits the process. It is confined here because this is the one
    place where the correct behavior and the tidy behavior are not the same.
    """
    run_dir = ctx.run_dir
    # The record first, and outside the guard below. Everything after it is best-effort, but
    # this is not: a suspension nobody can read is not a suspension, and if it cannot be written
    # then failing through the normal path, which at least publishes, beats exiting into silence.
    ctx.publish_json(run_dir / SUSPENSION_FILE, suspension)
    try:
        ctx.publish_json(run_dir / "legs.json", ctx.leg_records())
        # The agent's container is already gone, its process being what ended, so this stops the
        # network namespace and the egress observer.
        if ctx.teardown is not None:
            ctx.teardown()
        print(message, file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
    finally:
        # Reporting and cleanup are courtesies; the exit is the correctness property, so it sits
        # in a finally where nothing above it can prevent it. A closed output pipe or a docker
        # daemon that will not answer must not propagate from here: the exception would unwind
        # the stream this was called from, shogym would drain the task in flight into a scored
        # row, and the position the suspension exists to preserve would be spent. Leaving on the
        # way out of a failure is also what discards it, since os._exit runs no handlers.
        os._exit(SUSPENDED_EXIT_CODE)


def _suspend_and_exit(
    ctx: RunContext,
    *,
    record: LegRecord,
    session_id: str | None,
    elapsed_rollout_s: float,
    tasks_dispensed: int,
    pool_queued: int,
    rollout_wall_clock_s: int,
) -> None:
    """Suspend the rollout a provider usage limit stopped, and hard-exit. Never returns."""
    run_dir = ctx.run_dir
    suspension = {
        "schema": "shobench.suspension/1",
        "run_id": ctx.run_id,
        "cell": ctx.cell.name,
        "harness": ctx.cell.harness,
        "phase": "rollout",
        "session_id": session_id,
        "legs_before": len(ctx.leg_records()),
        "tasks_dispensed": tasks_dispensed,
        "pool_queued": pool_queued,
        "elapsed_rollout_s": round(elapsed_rollout_s, 3),
        # The clock this rollout was given, carried forward rather than re-read: a second
        # suspension has to hand on the same budget the first one was measured against.
        "rollout_wall_clock_s": rollout_wall_clock_s,
        "remaining_rollout_s": round(max(0.0, rollout_wall_clock_s - elapsed_rollout_s), 3),
        "stop_evidence": record.verdict.to_json(),
        "suspended_at": time.time(),
        "resume_with": f"uv run shobench resume --run {run_dir} --go",
    }
    _finalize_suspension(
        ctx,
        suspension,
        message=(
            f"[shobench] {ctx.cell.name}: suspended on a usage limit after "
            f"{tasks_dispensed} task(s); resume with: {suspension['resume_with']}"
        ),
    )


def _suspend_eval_and_exit(ctx: RunContext, *, phase: str, verdict: StopVerdict) -> None:
    """Suspend an eval phase a provider usage limit stopped, and hard-exit. Never returns.

    The eval counterpart of :func:`_suspend_and_exit`, reaching the same guaranteed exit. What it
    records is different, because an eval phase is not one session on one clock: it is a fresh
    session per held-out task, and completion lives in each task's own provenance. So the record
    carries no session and no clock, only the phase that was interrupted and, for an operator
    reading the plan, which ids were finished and which still need running. The resume re-derives
    both from provenance rather than trusting the record, so nothing is missed if a row landed
    between the last read and the exit. The finished ids keep their rows; the interrupted one was
    drained into a row this run will not publish, and the resume clears and re-runs it.
    """
    side = side_for_phase(ctx.split, phase)
    phase_dir = ctx.run_dir / phase
    pending = _eval_pending_ids(phase_dir, side.task_ids)
    pending_ints = [int(task_id) for task_id in pending]
    pending_set = set(pending_ints)
    completed_ints = [int(t) for t in side.task_ids if int(t) not in pending_set]
    suspension = {
        "schema": "shobench.suspension/1",
        "run_id": ctx.run_id,
        "cell": ctx.cell.name,
        "harness": ctx.cell.harness,
        "phase": phase,
        "legs_before": len(ctx.leg_records()),
        "completed_task_ids": completed_ints,
        "pending_task_ids": pending_ints,
        "stop_evidence": verdict.to_json(),
        "suspended_at": time.time(),
        "resume_with": f"uv run shobench resume --run {ctx.run_dir} --go",
    }
    _finalize_suspension(
        ctx,
        suspension,
        message=(
            f"[shobench] {ctx.cell.name}: suspended on a usage limit during {phase} with "
            f"{len(pending_ints)} held-out task(s) still to run; resume with: "
            f"{suspension['resume_with']}"
        ),
    )


# How many entries of a leg's tree the progress reading walks before it gives up on answering.
# Hitting the cap reads as PROGRESS rather than as silence: the worst that produces is a rollout
# with no stall detection, which is what every rollout had before this existed, while the other
# direction would end a working leg for having a big working directory.
PROGRESS_WALK_LIMIT = 20_000

# How often the rollout's progress reading is taken, as a share of the bound it is measuring
# against, and the ceiling on that in seconds. The fraction bounds the lag between the real moment
# of silence and the reading that notices.
PROGRESS_POLL_FRACTION = 0.1
PROGRESS_POLL_CEILING_S = 60.0


class _PulseUnreadable(Exception):
    """This tree could not be read to the bottom, so no reading of it may be compared."""


def _unreadable_pulse() -> tuple[int, int, int, int]:
    """A reading that can never equal another one, so the caller records progress and resets.

    The fail-safe direction: a tree the host cannot read is a tree the container may be writing
    to, and reading it as a constant would end a working leg for being unreadable.
    """
    return (-1, -1, time.monotonic_ns(), 0)


# What the fold over entry identities is taken modulo: one machine word.
_MARK_MODULUS = 1 << 64


def _entry_mark(rel: str, name: str, path: Path) -> int:
    """A number that moves if this directory entry stops being the same entry.

    The counters beside it are about the CONTENT a walk reaches, and following links makes them
    about the targets alone: a link retargeted between two stat-identical trees moves no count, no
    byte and no mtime, and an empty directory link appearing moves nothing at all. So the entry is
    folded in BESIDE its target's stats: where it sits, what it is called, and what it points at.
    ``readlink`` is the whole test for the last of those, since it fails on anything else.
    """
    try:
        target = os.readlink(path)
    except OSError:
        # Not a link, or gone between the listing and here. Either way it contributes its name.
        target = ""
    body = "\0".join((rel, name, target)).encode("utf-8", "surrogateescape")
    return int.from_bytes(hashlib.blake2b(body, digest_size=8).digest(), "big")


def _tree_pulse(root: Path, *, limit: int | None = None) -> tuple[int, int, int, int]:
    """A cheap fingerprint of a directory tree: its files, their bytes, the newest write, and a
    fold over the entries themselves. Asked only whether the tree CHANGED between two readings.

    An absent tree reads as empty, and has to: prime's ``session-artifacts`` appears only once a
    child session runs. A tree not read to the bottom (a permission error, a directory that
    vanished mid-walk, an unreachable link target, a tree past the walk limit) reads as
    unreadable, and the caller records progress.

    Two ``os.walk`` defaults are against "to the bottom". ``onerror`` SWALLOWS a directory whose
    listing fails, so a mode-000 subtree reads as a stable constant while the container writes
    inside it; the callback here raises. ``followlinks`` skips a symlinked directory, and the
    matching ``lstat`` fingerprints the LINK rather than the target whose size and mtime move, so
    links are followed and ``os.stat`` is used: ``/work`` is the agent's own writable cwd. That
    makes the walk unbounded in principle, so the limit counts DIRECTORIES as well as files, which
    is what terminates a symlink cycle, and it makes the counters describe the TARGETS, which is
    why each entry is folded in beside them (see :func:`_entry_mark`). A tree reached twice by two
    links is counted twice, which is deterministic.
    """
    limit = PROGRESS_WALK_LIMIT if limit is None else limit
    files = 0
    entries = 0
    total = 0
    newest = 0
    mark = 0

    def cannot_read(error: OSError) -> None:
        raise _PulseUnreadable(str(error))

    try:
        os.stat(root)
    except FileNotFoundError:
        return (0, 0, 0, 0)
    except OSError:
        # There, and not listable from here: a statement about this reading, not about the tree.
        return _unreadable_pulse()
    try:
        for parent, dirs, names in os.walk(root, onerror=cannot_read, followlinks=True):
            entries += 1
            rel = os.path.relpath(parent, root)
            # Directories are marked but not counted here: every one of them arrives as a
            # `parent` of its own, which is where the limit counts it and what bounds a cycle.
            for name in dirs:
                mark = (mark + _entry_mark(rel, name, Path(parent) / name)) % _MARK_MODULUS
            for name in names:
                files += 1
                entries += 1
                if entries > limit:
                    raise _PulseUnreadable("tree past the walk limit")
                mark = (mark + _entry_mark(rel, name, Path(parent) / name)) % _MARK_MODULUS
                # Through the link, not at it. A dangling one raises and reads as unreadable,
                # which makes the check inert rather than blind.
                info = os.stat(Path(parent) / name)
                total += info.st_size
                newest = max(newest, info.st_mtime_ns)
            if entries > limit:
                raise _PulseUnreadable("tree past the walk limit")
    except (OSError, _PulseUnreadable):
        return _unreadable_pulse()
    return (files, total, newest, mark)


def _file_pulse(path: Path) -> tuple[int, int]:
    """The same fingerprint for one file: its size and its last write.

    Size alone would miss a rewrite of equal length, and mtime alone can be coarse on a filesystem
    that stores it in seconds; together they move on any append, which is all a trace ever does.
    """
    try:
        stat = path.stat()
    except OSError:
        return (-1, -1)
    return (stat.st_size, stat.st_mtime_ns)


def rollout_progress(ctx: RunContext, *, trace_path: Path, prov_dir: Path) -> tuple[Any, ...]:
    """Every source that says the rollout leg is getting somewhere, read as one value.

    Silence in the TRACE is not evidence of a stall: a task can legitimately take an hour inside
    one tool call, and an agent that delegates goes quiet in its own trace by design while its
    children work. So four sources are read, and any one of them moving resets the clock:

    - the leg's own trace, which grows on every record the harness emits;
    - the rollout's provenance, which gains a file on every dispense and every sealed row;
    - the harness's session state under the cell HOME, which is where a delegating agent's
      children are written (prime keeps its RLM children under ``session-artifacts/<id>/sub-*``,
      claude and codex append their sidechains into the session transcripts);
    - the cell's ``/work``, which is the writable directory every harness runs in, so an agent
      building something has to touch it.

    A stall is the ABSENCE of all four while the container is alive.
    """
    return (
        _file_pulse(trace_path),
        _tree_pulse(prov_dir),
        tuple(_tree_pulse(ctx.sandbox.home / rel) for rel in ctx.harness.session_state_dirs),
        _tree_pulse(ctx.sandbox.workdir),
    )


def _progress_poll_s(bound_s: float) -> float:
    """How often to take the reading, given the bound it is measuring against."""
    return max(1.0, min(PROGRESS_POLL_CEILING_S, bound_s * PROGRESS_POLL_FRACTION))


async def _watch_for_no_progress(
    ctx: RunContext,
    ending: EarlyEnding,
    *,
    trace_path: Path,
    prov_dir: Path,
    bound_s: float,
    poll_s: float | None = None,
) -> bool:
    """Fire ``ending`` once nothing in :func:`rollout_progress` has moved for ``bound_s``.

    Runs on the rollout's event loop beside the leg and is cancelled when the leg ends, which is
    what scopes the check to a LIVE container. Scoped to the rollout alone: every eval leg is
    already bounded per task by ``eval_task_timeout_s``, while the rollout is one long session
    with nothing bounding a single tool call.

    The clock is over the CONDITION rather than over the first time it was seen: any reading that
    differs from the last resets it.
    """
    poll_s = _progress_poll_s(bound_s) if poll_s is None else poll_s

    def read() -> tuple[Any, ...]:
        return rollout_progress(ctx, trace_path=trace_path, prov_dir=prov_dir)

    # Off the loop: the reading walks directories the agent is writing, and a loop parked on a
    # filesystem walk is not serving the stream the leg is talking to.
    pulse = await asyncio.to_thread(read)
    since = time.monotonic()
    while True:
        await asyncio.sleep(poll_s)
        try:
            reading = await asyncio.to_thread(read)
        except Exception:
            # A watcher that cannot read the state never ends a leg on that failure: it reads
            # files another party is writing, and the cost of guessing wrong here is a rollout
            # bounded only by its wall clock.
            pulse, since = None, time.monotonic()
            continue
        if reading != pulse:
            pulse, since = reading, time.monotonic()
            continue
        silent = time.monotonic() - since
        if silent >= bound_s:
            ending.fire(ctx.harness.no_progress_verdict(bound_s=bound_s, silent_s=round(silent, 3)))
            return True


async def run_rollout_phase(
    ctx: RunContext, *, suspended: Suspension | None = None
) -> tuple[list[TaskResult], dict[str, Any]]:
    """Serve the improvement pool as one honest run of the harness. No automatic restart.

    Whether a harness sustains autonomous operation is one of the things this measures, so the
    runner does not relaunch a harness that stops. A single invocation is driven against the
    pool; where it ends is the outcome:

    - it exhausts the pool: sustained autonomy, the success case;
    - it ends its run with tasks still available: it stopped on its own, and that is the
      finding, recorded with how far it got;
    - it wedges past the run's time cap: timed out, a different finding;
    - it hits a provider usage limit: the environment stopped it, not the agent, so the cell
      suspends where it stands and an operator continues it when the window resets (windows can
      run to hours, and nothing here waits or auto-retries).

    Restarting the harness to push it further would launder "gave up after N tasks" into "ran
    the whole pool", which is the one thing this must not do. Continuing the same session after
    a provider cut it off is the opposite: without it, the pool ends where the subscription
    window did.

    ``suspended`` is the record of an earlier usage limit, and passing one continues that run:
    the stream reopens on the same provenance directory, the harness resumes the same session,
    and the clock carries the time already spent.
    """
    prov_dir = ctx.run_dir / "rollout"
    prov_dir.mkdir(parents=True, exist_ok=True)
    budget = ctx.cell.budget
    resuming = suspended is not None

    # Claude Code accepts a pinned id, so a run that is later resumed by hand has one to name.
    session_id = suspended.session_id if resuming else _fresh_session_id(ctx)
    stopping: dict[str, Any] = {
        "stop_reason": "unrecorded",
        "legs": [leg for leg in ctx.prior_legs if leg.get("phase") == "rollout"],
        "session_id": session_id,
    }

    # A resumed rollout reopens the record it already wrote. That is the only way shogym will
    # point a stream at a directory holding rows, and it is what replays the position the
    # suspended run held: resume walks queue positions with no result row, and the suspension
    # left the in-flight one row-less on purpose.
    #
    # The resume semantics, stated plainly, because the continuation cue is written to them and
    # the result shaping below depends on them. Pinned shogym reopens with an EMPTY live registry
    # and empty settled-lease set, so the old lease the resumed harness session still holds for
    # its interrupted task denotes nothing here: its next task-tool call under that lease is
    # refused as `unknown_lease`, and above max_in_flight 1 (every v0 cell) there is no other
    # routing to fall back on. The runner cannot make the agent's tool calls, so nothing here can
    # re-mint or re-attach that lease. The honest model is therefore that the in-flight task is
    # abandoned and replayed as a fresh dispense: shogym re-offers the row-less position on the
    # next `get_task`, minting a new lease, and the continuation cue is what drives the resumed
    # session to pull it instead of retrying its dead one, so no position is lost or stalled. The
    # abandoned dispense stays in the record as a `broker_abort`, which the result shaping below
    # collapses by position (see `dispensed_positions` and `collapse_replays`) so a resumed cell
    # publishes the one attempt per position an uninterrupted run does.
    stream = build_stream(ctx.cell, ctx.split, "rollout", prov_dir, resume=resuming)
    queued = suspended.pool_queued if resuming else stream.queue_info().remaining
    spent_before = suspended.elapsed_rollout_s if resuming else 0.0
    # A continuation's clock comes off the record, which carries the budget the interrupted
    # rollout was given. Subtracting from today's cell file instead would let a budget edited
    # while the run waited change the length of a rollout that is already half spent.
    clock_s = suspended.rollout_wall_clock_s if resuming else budget.rollout_wall_clock_s
    remaining_s = suspended.remaining_rollout_s if resuming else clock_s
    ctx.port = free_port()
    leg = suspended.legs_before if resuming else 0
    # The rollout's own early ending, and the one leg in the run that can carry it. A leg with an
    # empty queue in front of it is the charter's question and gets no drain handle; a leg that
    # has stopped producing anything at all is not that. A bound of zero asks for no such ending.
    stalled = EarlyEnding()
    bound_s = float(ctx.cell.budget.rollout_no_progress_s)
    async with stream, _served(stream, ctx.port):
        consumed_before = stream.queue_info().consumed
        watching = (
            asyncio.create_task(
                _watch_for_no_progress(
                    ctx,
                    stalled,
                    trace_path=leg_trace_path(ctx.run_dir, "rollout", leg),
                    prov_dir=prov_dir,
                    bound_s=bound_s,
                )
            )
            if bound_s > 0
            else None
        )
        try:
            record = await asyncio.to_thread(
                run_leg,
                ctx,
                phase="rollout",
                leg=leg,
                system_prompt=ctx.instruction.rollout_system,
                # A resumed rollout is continued, not begun. The arm defines two user turns for
                # exactly this: kickoff opens a fresh run, continuation resumes one. Sending the
                # opener again would run a suspended cell under a different intervention than the
                # one its record names.
                user_prompt=ctx.instruction.continuation if resuming else ctx.instruction.kickoff,
                session_id=session_id,
                resume=resuming,
                timeout_s=remaining_s,
                task_idx=None,
                consumed_before=consumed_before,
                endings=(stalled,),
            )
        finally:
            if watching is not None:
                watching.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watching
        record.tasks_consumed_after = stream.queue_info().consumed
        stopping["legs"].append(record.to_json())
        if record.session_id:
            stopping["session_id"] = record.session_id

        info = stream.queue_info()
        if record.verdict.kind is StopKind.USAGE_LIMIT:
            # The provider stopped it, not the agent, so the cell suspends here rather than
            # finishing. This call does not return: the record is written, the containers are
            # stopped, and the process exits without unwinding the stream, because an orderly
            # close would drain the task in flight into a scored row and shogym's resume skips
            # a position that already has one. Crash semantics are what keep the position
            # replayable, and they are only available before this block exits.
            _suspend_and_exit(
                ctx,
                record=record,
                session_id=stopping["session_id"],
                elapsed_rollout_s=spent_before + (record.ended_at - record.started_at),
                # Distinct positions dispensed across the whole record, not this process's share
                # plus the suspension's counter: the position this suspension is abandoning may be
                # a replay of one an earlier suspension already counted, and adding the counters
                # would count it twice. The in-flight dispense is durable before the task is
                # handed out, so it is already on disk and in this count.
                tasks_dispensed=dispensed_positions(prov_dir),
                pool_queued=queued,
                rollout_wall_clock_s=clock_s,
            )
        elif info.remaining == 0:
            stopping["stop_reason"] = "pool_exhausted"
        elif record.verdict.kind is StopKind.CHOSEN:
            # It ended its run with work still on the queue: it stopped on its own.
            stopping["stop_reason"] = "agent_stopped_early"
        elif record.verdict.kind is StopKind.LEG_TIMEOUT:
            stopping["stop_reason"] = "timed_out"
        elif record.verdict.kind is StopKind.OPERATOR:
            # A person ended it, so the treatment is shorter than the cell intended. It is still
            # a terminus, and an eval_after belongs on the far side of it.
            stopping["stop_reason"] = "operator_stopped"
        elif record.verdict.kind is StopKind.STALLED:
            stopping["stop_reason"] = "stalled"
        else:
            stopping["stop_reason"] = "harness_error"
        stopping["stop_evidence"] = record.verdict.to_json()
        # Dispensed counts distinct queue positions across the whole rollout, not this process's
        # share of it. A resumed stream numbers its own dispenses from zero while the record it
        # continues already holds the earlier ones, and it redispenses the position the suspension
        # abandoned, so summing each process's counter would count that position twice (the exact
        # overcount a two-position pool published as three). Counting distinct positions makes a
        # resumed run report the total an uninterrupted one does.
        stopping["tasks_dispensed"] = dispensed_positions(prov_dir)
        stopping["tasks_remaining_in_pool"] = info.remaining
        stopping["pool_queued"] = queued
        stopping["rollout_wall_clock_s"] = round(
            spent_before + (record.ended_at - record.started_at), 3
        )
        if resuming:
            stopping["resumed_from_suspension_at"] = suspended.suspended_at
        # The charter's question is whether it stopped with work still available.
        stopping["stopped_with_tasks_available"] = (
            stopping["stop_reason"] == "agent_stopped_early" and info.remaining > 0
        )
        # Answered at the terminus, against the home a bookend would copy and before any eval
        # phase has run, and for every ending rather than only the ones the runner imposed: a leg
        # that died on its own can leave an unforkable terminus too.
        rebookendable, why_not = terminus_is_rebookendable(ctx, stopping)
        stopping["terminus_rebookendable"] = rebookendable
        stopping["terminus_not_rebookendable_because"] = why_not

    return read_phase(prov_dir), stopping


# ----- the cell ----------------------------------------------------------------------------


def _run_id(cell: Cell) -> str:
    return f"{cell.name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


class _Egress:
    """One cell's egress observer, stopped exactly once from whichever path gets there first.

    A cell now has two endings. The ordinary one stops the observer on its way out; a suspension
    stops it from inside the rollout, because a cell waiting hours for a provider window must
    not leave containers running. Both call this, and only the first call does anything, so the
    summary is written once no matter which ending happened.
    """

    def __init__(self, capture: egress.EgressCapture | None, run_dir: Path) -> None:
        self._capture = capture
        self._run_dir = run_dir
        self.summary: dict[str, Any] = {}

    def stop(self) -> dict[str, Any]:
        if self._capture is None:
            return self.summary
        capture, self._capture = self._capture, None
        egress.stop(capture)
        record = self._run_dir / EGRESS_LOG
        if capture.log_path != record:
            # A continuation's observer writes its own segment, because the capture command
            # truncates whatever file it is pointed at. Its traffic is appended to the run's
            # record here, so the published summary covers the whole cell rather than only the
            # last continuation: the eval that ran before the interruption is evidence too.
            with record.open("a", encoding="utf-8") as whole:
                whole.write(capture.log_path.read_text(encoding="utf-8", errors="ignore"))
        self.summary = dict(egress.write_summary(record, self._run_dir / "egress.json"))
        return self.summary


async def _run_phases(
    ctx: RunContext,
    *,
    manifest: dict[str, Any],
    phases: tuple[str, ...],
    results_dir: Path,
    observer: _Egress,
    suspended: Suspension | None = None,
    recorded_phases: tuple[str, ...] = (),
    artifact: str | None = None,
) -> Path:
    """Run this cell's phases, then finalize the manifest and publish the results.

    Shared by a fresh cell and a resumed one, and that sharing is the point rather than a
    convenience: a continuation has to end exactly the way an uninterrupted run ends. eval_after
    is the measurement the rollout is supposed to precede, so it belongs on the far side of a
    real rollout terminus and nowhere else. A suspension never reaches this code at all, since it
    leaves the process from inside the phase that hit the limit, which is what keeps a
    half-finished phase from publishing an ending and from spending an exhausted window.

    ``recorded_phases`` are the phases an earlier process already finished and this one carries
    forward: eval_before for a rollout suspension, and both eval_before and the rollout for an
    eval_after suspension, since an eval_after limit falls after the rollout is already paid for
    and must not lose it.

    ``artifact`` overrides the published result's stem. The default, the cell name, is right
    for every run that IS the cell's measurement, because a rerun replacing the cell's last
    artifact is the intended one-artifact-per-cell rule. A rebookend is not that run: it
    pairs WITH the cell's artifact, and publishing under the same stem destroyed the very
    result it must pair with, so it publishes under its own run id instead.

    An operator's stop is the one ending that stops the LOOP rather than a leg. It reaches a
    running leg through the supervisor, which ends the container the way a budget does, and it
    reaches this loop by keeping any phase that has not started from starting. Everything below
    then runs unchanged, which is the whole of what stopping this way buys.

    Nothing here starts the watcher that fires that handle: it belongs to the run's OWNERSHIP and
    lives as long as the exclusive lock does (see :func:`owning_run`), because an ask can arrive
    at any moment a process holds the directory, including the setup before this function and the
    publication and teardown after it.
    """

    def teardown() -> None:
        # Independent best-effort steps, with the sandbox in a finally. This runs on the way out
        # of a suspension, where an observer that will not stop must not leave a namespace
        # container and a network running for the hours a cell waits for its window.
        try:
            observer.stop()
        finally:
            ctx.sandbox.down()

    ctx.teardown = teardown
    # The ids this cell committed to measuring. Both eval phases serve this one side, and every
    # published count is against it rather than against whatever arrived.
    heldout_ids = [int(task_id) for task_id in side_for_phase(ctx.split, "eval_before").task_ids]
    # A continuation starts from the phases the interrupted run already recorded, not from
    # nothing. A recorded eval phase (eval_before) is read with the eval-phase reader, because
    # its rows live in per-task subdirectories a single-directory read would miss; a recorded
    # rollout is read flat. Omitting a recorded phase publishes a file missing half the
    # measurement: no requested eval tasks, no deltas, every after row unpaired.
    phase_rows: dict[str, list[TaskResult]] = {
        phase: (
            read_phase(ctx.run_dir / phase)
            if phase == "rollout"
            else read_eval_phase(ctx.run_dir / phase, heldout_ids)
        )
        for phase in recorded_phases
    }
    stopping: dict[str, Any] = {}
    # Phases an operator's stop kept from starting at all, named for the record.
    stopped_before: list[str] = []
    if "rollout" in recorded_phases:
        # The rollout's stop classification is not in provenance, so it was persisted when the
        # rollout ended and is read back here. This is the path an eval_after resume takes: it
        # republishes the rollout it did not re-run, stop reason and all, rather than a blank one.
        stopping = json.loads(
            (ctx.run_dir / ROLLOUT_STOPPING_FILE).read_text(encoding="utf-8")
        )
    for phase in phases:
        if ctx.operator_stop.fired.is_set():
            # Asked for between phases. Nothing is running to end, so the ending is that no
            # further phase begins; the run publishes what it has below.
            stopped_before.append(phase)
            continue
        if phase == "rollout":
            phase_rows[phase], stopping = await run_rollout_phase(ctx, suspended=suspended)
            # Persist the stop classification the moment the rollout ends, so an eval_after
            # suspension that follows can republish it: only the runner saw how the leg ended.
            ctx.publish_json(ctx.run_dir / ROLLOUT_STOPPING_FILE, stopping)
            # The durable measurement is taken here, at the rollout's terminus, and written
            # into the manifest before eval_after runs. That is the whole boundary: what the
            # rollout left, read at the moment the rollout ended, and never again afterwards.
            # Taken at the end of the cell instead, anything an eval phase managed to write
            # into the base home would be attributed to the rollout, and the reader would
            # have no way to tell.
            _snapshot_durable_state(ctx, manifest)
        else:
            phase_rows[phase] = await run_eval_phase(ctx, phase)
        ctx.publish_json(ctx.run_dir / "legs.json", ctx.leg_records())
    if ctx.operator_stop.fired.is_set():
        # Said in the manifest as well as in the leg that carries the verdict, so a reader knows
        # the treatment was cut short by a person before they read anything else about it.
        #
        # This says an operator ASKED, which is not the same claim as any leg's verdict: a leg
        # whose stall watcher got there first is recorded as `no_progress` in
        # `rollout_stopping.json` and this block is still here, because both are true.
        request = ctx.operator_stop.verdict.evidence if ctx.operator_stop.verdict else {}
        manifest["operator_stop"] = {
            **request,
            "phases_not_run": stopped_before,
            # Answered by the rollout that ran, never assumed: a stop can end a leg before a
            # resumable transcript exists.
            "terminus_rebookendable": bool(stopping.get("terminus_rebookendable")),
        }

    # How many times a provider limit suspended and resumed this cell, counted off the one
    # place that record lives: a resumption entry is appended per continuation. Set whenever the
    # published result carries a rollout, whether this process ran it or carried it forward, so an
    # eval_after resume updates the count for the resume it is itself performing.
    if "rollout" in phases or "rollout" in recorded_phases:
        stopping["usage_limit_resumes"] = len(manifest.get("resumptions", []))

    # Which model answered, read off the traces rather than assumed from the config, and off
    # every leg of the run rather than the ones this process happened to launch.
    manifest["observed_models"] = sorted(
        {model for leg in ctx.leg_records() for model in leg.get("observed_models", [])}
    )
    manifest.setdefault("axes", {}).setdefault("model", {})["observed"] = manifest[
        "observed_models"
    ]
    if manifest["home"]["digest_after"] is None:
        # No rollout ran in this process and none was carried forward, which happens when an
        # operator asked for a subset of the phases, and when one stopped the run before its
        # rollout began. There is no rollout terminus to have measured, so the state is read now
        # and said to have been.
        _snapshot_durable_state(ctx, manifest, measured_at="publish")
    else:
        _check_evals_left_the_snapshot_alone(ctx, manifest)
    # One last read of the live credential file, immediately before anything is published. Every
    # leg already folded in what it minted, so this covers the rest: a probe, a container this
    # process did not launch, a refresh that landed after the final leg. The count is republished
    # with it, because a manifest written before the phases would otherwise report how many forms
    # were watched at the start rather than how many were watched over the run.
    ctx.watch_credentials(ctx.sandbox.home)
    manifest.setdefault("redaction", {})["forms_watched"] = ctx.redactor.count
    manifest["ended_at"] = time.time()
    ctx.publish_json(ctx.run_dir / "manifest.json", manifest)

    # A bookend publishes the BASELINE's before rows as its own block, labeled with the run
    # they came from, so the artifact carries its whole pairing and depends on no other file:
    # the baseline's own artifact lives under the shared cell stem and is routinely evicted
    # by later same-cell publications. The carried rows were persisted at creation; a
    # marker-bearing run whose carry is gone cannot publish self-contained and must not
    # publish an empty before under the resumed label, so it refuses.
    before_source_run_id: str | None = None
    baseline_payload_path = ctx.run_dir / BASELINE_BEFORE_FILE
    if "rebookend" in manifest:
        if not baseline_payload_path.is_file():
            raise RuntimeError(
                f"{ctx.run_dir} is a rebookend but its carried baseline rows "
                f"({BASELINE_BEFORE_FILE}) cannot be read, so the artifact cannot be "
                "published self-contained. The file is written at creation; restore it or "
                "recreate the bookend."
            )
        payload = json.loads(baseline_payload_path.read_text(encoding="utf-8"))
        phase_rows["eval_before"] = [TaskResult(**row) for row in payload["rows"]]
        before_source_run_id = str(payload.get("source_run_id") or "") or None
    egress_summary = observer.stop()
    results_path = write_results(
        results_dir / f"{artifact or ctx.cell.name}.json",
        manifest=manifest,
        phases=phase_rows,
        stopping=stopping,
        heldout_ids=heldout_ids,
        egress=egress_summary,
        redact=ctx.redactor.json,
        before_source_run_id=before_source_run_id,
    )
    # Said out loud as well as written down. A cell that cannot account for every held-out id is
    # not this cell's result, and the operator who ran it is the one who can decide what to do
    # about the ids it lost; the path alone is easy to read past.
    for phase in ("eval_before", "eval_after"):
        # Both phases, whether or not this process produced one, because that is the rule the
        # published file is judged by: a phase that recorded nothing accounts for nothing either.
        entry = heldout_accounting(phase_rows.get(phase, []), task_ids=heldout_ids)
        if not entry["complete"]:
            print(
                f"[shobench] {ctx.cell.name}: INCOMPLETE. {phase} cannot account for "
                f"{len(entry['missing_task_ids'])} of {len(heldout_ids)} held-out task(s) "
                f"{entry['missing_task_ids']}; published as {results_path.name}, which is not "
                "a finished measurement.",
                file=sys.stderr,
            )
    if "operator_stop" in manifest:
        # What an operator who stopped a run needs to be told: the records they stopped for are
        # written, and what can be done next with the ending they got. Which ending that is comes
        # from the rollout's own record, because saying otherwise sends them to a refusal.
        if manifest["operator_stop"]["terminus_rebookendable"]:
            recovery = (
                "Its rollout terminus is resumable, so "
                f"`uv run shobench rebookend --run {ctx.run_dir}` can give it an eval_after."
            )
        elif "rollout" in phases or "rollout" in recorded_phases:
            recovery = (
                "Its rollout terminus is NOT resumable, so it cannot be bookended: "
                f"{stopping.get('terminus_not_rebookendable_because') or 'no reason recorded'}."
            )
        else:
            recovery = "No rollout ran, so there is no terminus to bookend."
        print(
            f"[shobench] {ctx.cell.name}: STOPPED by an operator. The run's records are written "
            f"({', '.join(sorted(stopped_before)) or 'no phase was skipped'} did not run); "
            f"published as {results_path.name}. {recovery}",
            file=sys.stderr,
        )
    return results_path


def _snapshot_durable_state(
    ctx: RunContext, manifest: dict[str, Any], *, measured_at: str = "rollout_end"
) -> None:
    """Record what the rollout left in each persistent channel, at the rollout's terminus.

    The boundary this cell publishes, stated once:

    - the **baseline** is the state after the runner has placed everything it owns and before
      any phase runs, so the seeds a cell starts with are on the same side as the rollout that
      may improve them;
    - **eval_before** reads that baseline through a throwaway copy per task and writes only into
      the copy, so it moves neither channel;
    - the **rollout** is the only thing that runs against the cell's own HOME and ``/work``;
    - **this snapshot** is taken the moment the rollout ends, which makes the recorded delta the
      rollout's and nothing else's;
    - **eval_after** reads that same post-rollout state through its own throwaway copies.

    Both channels are recorded, but they answer different questions. The HOME digest is the
    durable self, the thing a later fresh session inherits. ``/work`` is persistent and
    agent-visible but reaches no later session, so it is inventoried as evidence rather than
    scored: a rollout that spent its time writing scripts and notes into its cwd should be
    legible in the record instead of showing up as an agent that did nothing.
    """
    exclude = ctx.durable
    for key, root in (("home", ctx.sandbox.home), ("work", ctx.sandbox.workdir)):
        channel = manifest.setdefault(key, {"digest_before": None})
        channel["digest_after"] = home_digest(root, exclude=exclude)
        channel["inventory_after"] = home_inventory(root, exclude=exclude)
        channel["changed"] = channel["digest_after"] != channel["digest_before"]
        channel["measured_at"] = measured_at


def _check_evals_left_the_snapshot_alone(ctx: RunContext, manifest: dict[str, Any]) -> None:
    """Confirm at publish time that no eval phase moved either channel after the rollout.

    An eval task runs against its own copy of the HOME and its own empty ``/work``, so this is
    an invariant rather than a measurement. Checking it is what turns a future change that
    quietly shares one of them into a recorded fact instead of a rollout credited with writes it
    did not make. It never rewrites the snapshot: the rollout's terminus is the measurement, and
    a mismatch is reported beside it rather than folded into it.
    """
    exclude = ctx.durable
    for key, root in (("home", ctx.sandbox.home), ("work", ctx.sandbox.workdir)):
        channel = manifest.get(key)
        if channel is None or channel.get("digest_after") is None:
            continue
        channel["unchanged_by_evals"] = (
            home_digest(root, exclude=exclude) == channel["digest_after"]
        )


def _place_runner_files(ctx: RunContext) -> list[str]:
    """Put the harness assets the cell starts with into its HOME, before the baseline is taken.

    These are the agent's from here on: the rollout may improve them, and the eval that follows
    has to read what the rollout left rather than what the checkout shipped. So they are written
    only when absent, and they are inside the baseline rather than beside it. Seeded lazily on
    the rollout's first leg instead, as they were, every one of these files landed after the
    digest that was supposed to precede it, and a prime_agent cell that wrote nothing published
    ten kilobytes of vendored skill as its own durable output.
    """
    placed = []
    for name, body in ctx.harness.home_seed_files().items():
        target = ctx.sandbox.home / name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        placed.append(name)
    return sorted(placed)


def _credential_home_paths(spec: CredentialSpec) -> tuple[str, ...]:
    """Where inside a cell's HOME this harness's credential can live, as the spec declares it.

    The file the runner seeded, plus every path the spec already names for the isolation check.
    Both, because the two are not the same set: a mode whose token arrives in the environment
    still mints a file of its own from it, and that file is one the harness rewrites.
    """
    seeded = (spec.seed_to,) if spec.seed_to else ()
    return tuple(dict.fromkeys([*spec.home_paths, *seeded]))


def _watch_cell_credential(ctx: RunContext, spec: CredentialSpec) -> None:
    """Point this cell's redaction at the credential it provisioned, and at where it can move to.

    Set after the seeding, because the seeded file is half of what is protected. The other half
    is ``ctx.credentials``, which is the token the harness reads from its environment. Neither is
    stored anywhere else and neither is ever printed; what is recorded about this, in the
    manifest, is how many distinct forms are being watched for and not one of them.

    The paths are kept because the file is not a constant: a harness that refreshes an expired
    OAuth token rewrites it mid-run, and :meth:`RunContext.watch_credentials` re-reads them after
    every leg and before anything is published so the values a refresh minted are covered too.
    """
    ctx.credential_home_paths = _credential_home_paths(spec)
    seeded = (ctx.sandbox.home / spec.seed_to,) if spec.seed_to else ()
    ctx.redactor = redactor_for(environment=ctx.credentials, credential_files=seeded)


# The exclusive-ownership marker of a live run directory. Three entry points mutate one (a
# fresh run, a resume, an eval re-run), and two shells can point two of them at the same
# directory: the second would rebuild the namespace holder out from under the first
# (CellSandbox.up is a docker rm -f before the create) and re-run ids the first is mid-way
# through. The lock is kernel-held (flock on a kept-open fd), which makes every hard ending
# safe by construction: a suspension's os._exit, a crash, and a kill all release it in the
# kernel, so there is no stale-lock steal protocol to race and no pid to misidentify. The
# file is never unlinked, because unlinking a path a contender has already opened would put two
# processes behind two different inodes of one name.
#
# One field of its contents is protocol rather than diagnostics: `stop_protocol` is how an owner
# ADVERTISES that it is watching for an operator's stop, written by the one entry that starts
# that watcher, and `shobench stop` refuses an owner that does not carry it. The pid beside it is
# diagnostics only and must stay that way: the file is never unlinked, so a finished run names a
# pid the system may since have reissued.
RUN_LOCK_FILE = "run.lock"

# What this build's owners advertise. A version rather than a flag, so a later change to what an
# ask means can be told apart from this one.
STOP_PROTOCOL = 1


def _acquire_run_lock(run_dir: Path, *, stoppable: bool = False) -> int:
    """Take exclusive kernel ownership of ``run_dir``; returns the fd that holds it.

    ``stoppable`` writes the advertisement, and only :func:`owning_run` passes it, because only
    that entry starts the watcher the advertisement promises.
    """
    import fcntl

    run_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(run_dir / RUN_LOCK_FILE, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        holder = ""
        with contextlib.suppress(OSError, ValueError):
            holder = os.read(fd, 256).decode("utf-8", "replace").strip()
        os.close(fd)
        raise RuntimeError(
            f"{run_dir} is owned by a live process"
            + (f" ({holder})" if holder else "")
            + "; a second owner would tear down its network and re-run ids it is mid-way "
            "through. Wait for it to finish, or stop it first."
        ) from None
    holder: dict[str, Any] = {"pid": os.getpid(), "at": time.time()}
    if stoppable:
        holder["stop_protocol"] = STOP_PROTOCOL
    # Emptied first, then written. The file outlives every owner, so until this payload lands the
    # bytes on disk are the PREVIOUS owner's: an owner that does not watch, briefly inheriting an
    # advertisement, is the one state that produces an ask nobody consumes. Truncating first makes
    # the window read as an owner that advertises nothing, which refuses.
    os.ftruncate(fd, 0)
    os.pwrite(fd, json.dumps(holder).encode("utf-8"), 0)
    return fd


def _release_run_lock(lock_fd: int) -> None:
    """Closing the fd releases the kernel lock; every hard ending already did this implicitly."""
    with contextlib.suppress(OSError):
        os.close(lock_fd)


def read_lock_holder(run_dir: Path) -> dict[str, Any]:
    """What a run's lock file says about its owner, empty where it says nothing readable."""
    try:
        holder = json.loads((run_dir / RUN_LOCK_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return holder if isinstance(holder, dict) else {}


@contextlib.contextmanager
def owning_run(run_dir: Path):
    """Own ``run_dir`` exclusively, watching for an operator's stop throughout. Yields the handle.

    The watcher's lifetime IS the lock's lifetime, and that is the correctness property: an ask
    can arrive at any moment a process holds the directory, and one scoped to the phase loop left
    every ask outside that window unread and on disk for the next resume to latch. The
    advertisement goes into the lock in the same write that takes it, and the watcher's first
    action is to read the file, so an ask landing before the thread starts is not lost.

    It does NOT stop at the first ask it consumes, because an owner goes on holding the directory,
    still advertising, through leg shutdown, publication and teardown. A later ask is acknowledged
    (unlinked) even though the handle it fires is already latched, which is what makes the command
    safe to call twice. It is stopped BEFORE ownership is released and takes one last look on the
    way out; an ask landing after that is the CLI's to take back.
    """
    stop = EarlyEnding()
    fd = _acquire_run_lock(run_dir, stoppable=True)
    done = threading.Event()

    def poll() -> None:
        while True:
            _honor_stop_request(run_dir, stop)
            if done.wait(STOP_POLL_S):
                # Ownership is ending. One last look, so an ask written since the previous poll
                # is consumed rather than left behind.
                _honor_stop_request(run_dir, stop)
                return

    watcher = threading.Thread(target=poll, name="shobench-stop", daemon=True)
    watcher.start()
    try:
        yield stop
    finally:
        done.set()
        # Bounded, and a daemon thread besides: a wedged watcher must not be what keeps a
        # finished run from releasing what it owns.
        watcher.join(timeout=STOP_POLL_S * 2)
        _release_run_lock(fd)


async def run_cell(
    cell: Cell,
    split: Split,
    *,
    runs_dir: Path,
    results_dir: Path,
    port: int = DEFAULT_PORT,
    agent_image: str = AGENT_IMAGE,
    credentials: dict[str, str] | None = None,
    phases: tuple[str, ...] = ("eval_before", "rollout", "eval_after"),
    capture_egress: bool = True,
) -> Path:
    """Run one cell end to end and return the path of its results JSON.

    Ownership is taken before anything else and released after everything, including a
    teardown that raises: the lock has to outlive every fallible setup step it protects, and so
    does the operator-stop watcher that comes with it.
    """
    run_id = _run_id(cell)
    run_dir = runs_dir / run_id
    with owning_run(run_dir) as stop:
        return await _run_cell_owned(
            cell,
            split,
            run_id=run_id,
            run_dir=run_dir,
            results_dir=results_dir,
            port=port,
            agent_image=agent_image,
            credentials=credentials,
            phases=phases,
            capture_egress=capture_egress,
            stop=stop,
        )


async def _run_cell_owned(
    cell: Cell,
    split: Split,
    *,
    run_id: str,
    run_dir: Path,
    results_dir: Path,
    port: int,
    agent_image: str,
    credentials: dict[str, str] | None,
    phases: tuple[str, ...],
    capture_egress: bool,
    stop: EarlyEnding,
) -> Path:
    instruction = load_instruction(cell.instruction_arm)
    sandbox = CellSandbox(run_id=run_id, home=run_dir / "home", workdir=run_dir / "work")
    # A fresh run pins its image exactly as a reopening does, and for the same two reasons: its
    # probes and its legs must be the same bytes, and the archive it becomes has to STATE which
    # bytes those were. A fresh run that skipped this recorded no content id at all, so every
    # pairing it ever took part in would have called the image unproven for the life of the
    # archive, and the promise that recording starts now would have been empty.
    image_ref, image_tag, image_id = pinned_image(agent_image)
    ctx = RunContext(
        cell=cell,
        split=split,
        instruction=instruction,
        harness=harness_for(cell.harness),
        run_id=run_id,
        run_dir=run_dir,
        sandbox=sandbox,
        port=port,
        agent_image=image_ref,
        image_tag=image_tag,
        image_digest=image_id,
        credentials=dict(credentials or {}),
        # The handle the run's ownership is already watching, so every leg this cell launches is
        # stoppable from the moment the directory was claimed rather than from the phase loop.
        operator_stop=stop,
    )

    sandbox.up()
    # A harness whose credential lives in a file gets it placed in this cell's own HOME. The
    # negative control validated the same placement in its own sandbox; without this the cell
    # itself would start with an empty HOME and no credential at all. The durability filter
    # excludes credential files, so seeding one changes no record.
    spec = spec_for(cell.harness, cell.credential_mode)
    seeded = seed_home(spec, sandbox.home)
    _watch_cell_credential(ctx, spec)
    observer = _Egress(_start_egress(sandbox, run_dir) if capture_egress else None, run_dir)
    try:
        probes = {
            # No credential: a version probe reports what the image installed, which no harness
            # needs to authenticate to answer. The model probe is the one that does.
            "version": ctx.redactor.text(
                _probe(
                    ctx.harness.version_probe(),
                    # The pinned reference, never the tag: a probe from one image beside legs
                    # from another describes a run that did not happen, and two builds printing
                    # one version string is precisely the case the content id exists to tell
                    # apart, so the version probe cannot be the thing that notices.
                    image=ctx.agent_image,
                    sandbox=sandbox,
                    env={},
                )
            )
        }
        model_probe = ctx.harness.model_probe()
        if model_probe:
            # The first thing in the run that authenticates, so the first thing that can refresh
            # a file-backed credential. Its raw output goes into the manifest, so it is watched
            # while it runs and once more before that output is redacted: a probe that refreshed
            # twice would otherwise put the first of the two into the manifest it feeds.
            with _watching_credentials(ctx, sandbox.home):
                output = _probe(
                    model_probe, image=ctx.agent_image, sandbox=sandbox, env=ctx.credentials
                )
            ctx.watch_credentials(sandbox.home)
            probes["model"] = ctx.redactor.text(output)
        # Everything the runner owns goes in before the baseline digest is taken, so what the
        # rollout starts from and what the manifest calls the starting point are the same thing.
        seeds = _place_runner_files(ctx)
        manifest = build_manifest(ctx, probes=probes)
        manifest["credential_seed"] = seeded
        manifest["home"]["seeded"] = seeds
        # The mode this cell will actually run under, read from the credential that was just
        # placed rather than copied from the cell file. Subscription billing is why the scope
        # sets no token ceiling, so a cell that quietly ran on an API key is a different
        # experiment and has to be visible as one.
        manifest["axes"]["credential_mode"] = credential_effective_mode(
            spec, sandbox.home, env_names=sorted(ctx.credentials)
        )
        ctx.publish_json(run_dir / "manifest.json", manifest)
        return await _run_phases(
            ctx,
            manifest=manifest,
            phases=phases,
            results_dir=results_dir,
            observer=observer,
        )
    finally:
        with contextlib.suppress(Exception):
            observer.stop()
        sandbox.down()


async def resume_cell(
    run_dir: Path,
    *,
    results_dir: Path,
    agent_image: str = AGENT_IMAGE,
    credentials: dict[str, str] | None = None,
    capture_egress: bool = True,
) -> Path:
    """Continue a cell a provider limit suspended, and finish it. Returns the results path.

    Ownership first: the suspension's own lock was released by its hard exit in the kernel,
    so this acquisition succeeds against a genuinely waiting run and refuses a live one. The
    continuation is as stoppable as the run it continues.
    """
    with owning_run(run_dir) as stop:
        return await _resume_cell_owned(
            run_dir,
            results_dir=results_dir,
            agent_image=agent_image,
            credentials=credentials,
            capture_egress=capture_egress,
            stop=stop,
        )


async def _resume_cell_owned(
    run_dir: Path,
    *,
    results_dir: Path,
    agent_image: str = AGENT_IMAGE,
    credentials: dict[str, str] | None = None,
    capture_egress: bool = True,
    stop: EarlyEnding | None = None,
) -> Path:
    """The suspended cell's continuation, run under an already-held run-directory lock.

    Everything this needs is on disk, because the process that wrote it is gone: the suspension
    record names which phase was interrupted, and the manifest beside it names the cell. The run
    directory is reused rather than copied, so the agent continues in the home it built, against
    the provenance record it already wrote.

    Which phases run depends on where the limit fell, and the shared tail publishes the same shape
    either way:

    - a **rollout** limit reopens the rollout on its remaining clock and then runs eval_after,
      carrying the recorded eval_before forward;
    - an **eval_before** limit finishes the interrupted eval_before (only the ids still pending)
      and then runs the rollout and eval_after, since none of those had run yet;
    - an **eval_after** limit finishes the interrupted eval_after alone, carrying both the recorded
      eval_before and the recorded rollout forward, because an eval_after limit falls after the
      eight-hour rollout is already paid for and must not lose it.

    What it must not do is start a second measurement of a different thing. The cell is the one
    the manifest recorded, checked against it rather than trusted, and the phases already measured
    are carried into the published result rather than re-run.
    """
    record = json.loads((run_dir / SUSPENSION_FILE).read_text(encoding="utf-8"))
    interrupted_phase = record.get("phase", "rollout")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    # The continuation finishes the experiment the record started, so the run's own record wins
    # over the checkout: its axes, and a bookend's inherited eval runtime.
    cell = reopened_cell(load_cell_by_name(manifest["cell"]["name"]), manifest)
    # Backfilled so a later resumption reads explicit values where this one read absence.
    manifest["cell"]["rollout_feedback"] = cell.rollout_feedback
    manifest["cell"]["eval_context"] = cell.eval_context
    # The instruction record stays consistent with the recovered axis, so the artifact keeps
    # naming the prompt its eval_after actually launches with; a pre-axis manifest recovers
    # cold and so names the blind eval instruction.
    manifest.setdefault("instruction", {})["eval_prompt_used"] = (
        "rollout_system" if cell.eval_context == "resumed" else "eval_system"
    )
    split = load_split_by_name(cell.split)
    instruction = load_instruction(cell.instruction_arm)
    drift = experiment_drift(manifest, cell=cell, split=split, instruction=instruction)
    if drift:
        # Refused rather than reconciled. Continuing under an edited definition would publish
        # one run id describing two experiments, and the operator is the only one who can say
        # whether the edit was meant for this run or for the next one.
        raise RuntimeError(
            "the checkout no longer matches the run being continued: "
            + "; ".join(drift)
            + ". Restore the recorded definition, or start a fresh cell."
        )
    harness = harness_for(cell.harness)
    image_ref, image_tag, image_id = pinned_image(agent_image)
    # This process writes rows into a run that already has some, so what produces them has to be
    # what produced the others. Everything knowable without a container is checked here, before
    # the sandbox exists; the rest below, still before any row.
    unproven = _refuse_execution_drift(
        manifest,
        current_identity(
            cell=cell,
            split=split,
            instruction=instruction,
            harness=harness,
            image_tag=image_tag,
            image_digest_value=image_id,
        ),
        stage=IDENTITY_PRE_SPEND,
        what="being continued",
    )
    run_id = record["run_id"]
    sandbox = CellSandbox(run_id=run_id, home=run_dir / "home", workdir=run_dir / "work")
    _migrate_recorded_containers(manifest, sandbox)
    ctx = RunContext(
        cell=cell,
        split=split,
        instruction=instruction,
        harness=harness,
        run_id=run_id,
        run_dir=run_dir,
        sandbox=sandbox,
        agent_image=image_ref,
        image_tag=image_tag,
        image_digest=image_id,
        credentials=dict(credentials or {}),
        # The handle this process's ownership is already watching. The default is honest for a
        # caller that took no ownership, such as a test driving one phase.
        operator_stop=stop if stop is not None else EarlyEnding(),
    )
    # The credential is placed again because the sandbox is new even though the home is not;
    # credential files are excluded from every digest, so re-seeding changes no record. It is
    # placed here rather than after the manifest is rewritten because the redactor is built from
    # it, and a continuation writes durable artifacts from its very first line.
    spec = spec_for(cell.harness, cell.credential_mode)
    seed_home(spec, sandbox.home)
    _watch_cell_credential(ctx, spec)
    # The effective credential mode exists only once a credential is placed, so it is compared
    # here. No probe is taken on this path, so the harness probe is named unproven rather than
    # invented; the image content id, compared above, is the stricter statement about that image.
    unproven += _refuse_execution_drift(
        manifest,
        after_setup_identity(
            probes=None,
            credential_mode=credential_effective_mode(
                spec, sandbox.home, env_names=sorted(ctx.credentials)
            ),
        ),
        stage=IDENTITY_AFTER_SETUP,
        what="being continued",
    )
    manifest["identity_unproven"] = sorted(set(unproven))
    legs_path = run_dir / "legs.json"
    if legs_path.is_file():
        # The legs the suspended run recorded. This process appends to that record rather than
        # replacing it, so a finished cell shows its whole rollout and not just the last stretch.
        ctx.prior_legs = json.loads(legs_path.read_text(encoding="utf-8"))
    # Which phases this process runs, and which it carries forward, keyed to where the limit fell.
    # A rollout suspension reopens the rollout on its remaining clock; an eval suspension has no
    # clock and no session to reattach to, so it passes none and lets the eval phase re-run only
    # the ids its provenance still shows pending.
    suspended: Suspension | None = None
    if interrupted_phase == "rollout":
        suspended = Suspension.read(run_dir)
        phases: tuple[str, ...] = ("rollout", "eval_after")
        recorded_phases: tuple[str, ...] = ("eval_before",)
    elif interrupted_phase == "eval_before":
        phases = ("eval_before", "rollout", "eval_after")
        recorded_phases = ()
    elif interrupted_phase == "eval_after":
        phases = ("eval_after",)
        recorded_phases = ("eval_before", "rollout")
        if "rebookend" in manifest:
            # A bookend never ran an eval_before or a rollout of its own: its stopping file
            # is the SOURCE's terminus, copied in for the resumed preflight, and carrying it
            # as a recorded phase would publish it as this run's rollout. The identical
            # after-measurement must not change artifact shape just because it hit a usage
            # limit, so a resumed bookend records nothing and publishes the same empty
            # rollout an uninterrupted one does.
            recorded_phases = ()
    else:
        raise RuntimeError(f"suspension names an unknown phase {interrupted_phase!r}")
    # Written before anything can suspend again, because a manifest that lives only in memory
    # loses every resumption to the next hard exit, and a cell continued three times would
    # publish as though it had been continued once.
    resumption: dict[str, Any] = {
        "suspended_at": record["suspended_at"],
        "resumed_at": time.time(),
        "phase": interrupted_phase,
    }
    if interrupted_phase == "rollout":
        # Only a rollout suspension carries a session and a spent clock; an eval resume mints a
        # fresh session per task and has no shared clock, so it records neither.
        resumption["elapsed_rollout_s_before"] = record["elapsed_rollout_s"]
        resumption["session_id"] = record["session_id"]
    manifest.setdefault("resumptions", []).append(resumption)
    ctx.publish_json(run_dir / "manifest.json", manifest)
    sandbox.up()
    observer = _Egress(_start_egress(sandbox, run_dir) if capture_egress else None, run_dir)
    try:
        results_path = await _run_phases(
            ctx,
            manifest=manifest,
            phases=phases,
            results_dir=results_dir,
            observer=observer,
            suspended=suspended,
            recorded_phases=recorded_phases,
            # A bookend keeps its own artifact stem across every reopening: the cell-name
            # fallback is the SOURCE's artifact, and a resumed bookend publishing under it
            # would destroy the result it pairs with, exactly the destruction the run-id
            # namespace exists to prevent.
            artifact=run_id if "rebookend" in manifest else None,
        )
        # Only now is the record spent. It is this run's one retry handle, and everything
        # between here and the top of this function can fail: a stream that will not open, a
        # server that will not bind, a leg that dies. Removing it earlier turned any of those
        # into a cell that could neither finish nor be tried again. A second usage limit never
        # reaches this line, having already replaced the record with its own.
        (run_dir / SUSPENSION_FILE).unlink(missing_ok=True)
        return results_path
    finally:
        with contextlib.suppress(Exception):
            observer.stop()
        sandbox.down()


async def rerun_eval(
    run_dir: Path,
    *,
    results_dir: Path,
    phase: str = "eval_after",
    agent_image: str = AGENT_IMAGE,
    credentials: dict[str, str] | None = None,
    capture_egress: bool = True,
    refresh_baseline: bool = False,
    baseline_run_dir: Path | None = None,
) -> Path:
    """Finish an eval phase that lost tasks without a suspension, and republish.

    Ownership first, refusals second: acquiring the lock of a genuinely finished run is free
    and leaves no trace, while a live owner refuses here before anything is read or written.
    """
    if phase not in ("eval_before", "eval_after"):
        raise ValueError(f"rerun_eval repairs eval phases, not {phase!r}")
    with owning_run(run_dir) as stop:
        return await _rerun_eval_owned(
            run_dir,
            results_dir=results_dir,
            phase=phase,
            agent_image=agent_image,
            credentials=credentials,
            capture_egress=capture_egress,
            refresh_baseline=refresh_baseline,
            baseline_run_dir=baseline_run_dir,
            stop=stop,
        )


async def _rerun_eval_owned(
    run_dir: Path,
    *,
    results_dir: Path,
    phase: str = "eval_after",
    agent_image: str = AGENT_IMAGE,
    credentials: dict[str, str] | None = None,
    capture_egress: bool = True,
    refresh_baseline: bool = False,
    baseline_run_dir: Path | None = None,
    stop: EarlyEnding | None = None,
) -> Path:
    """The reopened run's eval_after, run under an already-held run-directory lock.

    A suspension is the runner's own record and ``resume`` spends it; this entry exists for the
    ending no record names. Legs that die on infrastructure (a network that falls over mid
    fan-out) leave row-less held-out ids, the run publishes under the incomplete name, and the
    process exits cleanly, so there is nothing for ``resume`` to hold on to. The eval phase
    runner is already idempotent (only ids lacking a valid completed row are run), so reopening
    the same run directory re-runs exactly the holes, against the home the rollout left, and
    publishes the same shape the uninterrupted run would have.

    Refused while a suspension record exists, because that ending belongs to ``resume`` and
    running here would spend the window the suspension is waiting out. Refused when the rollout
    never reached a terminus, because eval_after belongs on the far side of one and nowhere
    else, which is the same rule ``_run_phases`` states for a fresh cell.

    ``refresh_baseline`` catches a bookend's carried before rows up to the baseline they were
    snapshotted from, for the baseline that finished after the snapshot was taken. It composes
    with the repair rather than replacing it: the pending legs still run, and a run with none
    pending refreshes the carry and republishes.
    """
    if (run_dir / SUSPENSION_FILE).is_file():
        raise RuntimeError(
            f"{run_dir} holds a suspension record; use `shobench resume`, which knows what "
            "the interruption was waiting for. This entry is for a run that ended with no "
            "record and left held-out ids row-less."
        )
    if phase == "eval_after" and not (run_dir / "rollout_stopping.json").is_file():
        raise RuntimeError(
            f"{run_dir} has no rollout terminus, so eval_after must not run: the measurement "
            "belongs on the far side of a real rollout ending."
        )
    if phase == "eval_before" and (
        (run_dir / "rollout_stopping.json").is_file() or (run_dir / "rollout").is_dir()
    ):
        # The one repair that must never happen: eval tasks copy the cell home at task time,
        # and a home a rollout has already touched is not the before state. A baseline that
        # holed out in a run that later rolled out is measured by a FRESH baseline-only run,
        # never by reopening this one.
        raise RuntimeError(
            f"{run_dir} has run its rollout, so its eval_before cannot be re-measured here: "
            "a before taken after the rollout would measure the accumulated home. Run a fresh "
            "--phases eval_before cell instead."
        )
    if phase == "eval_before" and not (run_dir / "eval_before").is_dir():
        raise RuntimeError(
            f"{run_dir} never ran an eval_before, so there is nothing to repair: launch the "
            "phase with `shobench run --phases eval_before` instead."
        )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    # Same reconstruction as a resume: the holes are re-run under the posture and the stopping
    # rule the finished ids were measured under, never today's.
    cell = reopened_cell(load_cell_by_name(manifest["cell"]["name"]), manifest)
    manifest["cell"]["rollout_feedback"] = cell.rollout_feedback
    manifest["cell"]["eval_context"] = cell.eval_context
    # Same consistency rule as a resume: the record names the prompt its eval_after runs with.
    manifest.setdefault("instruction", {})["eval_prompt_used"] = (
        "rollout_system" if cell.eval_context == "resumed" else "eval_system"
    )
    split = load_split_by_name(cell.split)
    instruction = load_instruction(cell.instruction_arm)
    # The whole-file comparison, and a repaired BOOKEND gets it too, though a rebookend does
    # not: a rebookend measures every held-out task under one definition, while a repair splices
    # rows into an eval whose other rows are already measured, and nothing in a row would say
    # which definition produced it. An operator refused here can rebookend the source instead.
    drift = experiment_drift(manifest, cell=cell, split=split, instruction=instruction)
    if drift:
        raise RuntimeError(
            "the checkout no longer matches the run being reopened: "
            + "; ".join(drift)
            + ". Restore the recorded definition, or start a fresh cell."
        )
    harness = harness_for(cell.harness)
    image_ref, image_tag, image_id = pinned_image(agent_image)
    # A repair splices rows in beside rows this run already has, so rows produced by another
    # image or another runner would sit in one artifact with the ones they claim to complete.
    unproven = _refuse_execution_drift(
        manifest,
        current_identity(
            cell=cell,
            split=split,
            instruction=instruction,
            harness=harness,
            image_tag=image_tag,
            image_digest_value=image_id,
        ),
        stage=IDENTITY_PRE_SPEND,
        what="being reopened",
    )
    run_id = str(manifest["run_id"])
    sandbox = CellSandbox(run_id=run_id, home=run_dir / "home", workdir=run_dir / "work")
    _migrate_recorded_containers(manifest, sandbox)
    ctx = RunContext(
        cell=cell,
        split=split,
        instruction=instruction,
        harness=harness,
        run_id=run_id,
        run_dir=run_dir,
        sandbox=sandbox,
        agent_image=image_ref,
        image_tag=image_tag,
        image_digest=image_id,
        credentials=dict(credentials or {}),
        # The handle this process's ownership is already watching. The default is honest for a
        # caller that took no ownership, such as a test driving one phase.
        operator_stop=stop if stop is not None else EarlyEnding(),
    )
    # Re-seeded for the same reason a resume re-seeds: the sandbox is new even though the home
    # is not, credential files are excluded from every digest, and the redactor is built from
    # what was placed.
    spec = spec_for(cell.harness, cell.credential_mode)
    seed_home(spec, sandbox.home)
    _watch_cell_credential(ctx, spec)
    unproven += _refuse_execution_drift(
        manifest,
        after_setup_identity(
            probes=None,
            credential_mode=credential_effective_mode(
                spec, sandbox.home, env_names=sorted(ctx.credentials)
            ),
        ),
        stage=IDENTITY_AFTER_SETUP,
        what="being reopened",
    )
    manifest["identity_unproven"] = sorted(set(unproven))
    legs_path = run_dir / "legs.json"
    if legs_path.is_file():
        ctx.prior_legs = json.loads(legs_path.read_text(encoding="utf-8"))
    # Everything this run measured other than the phase under repair is carried forward, so
    # the republication has the same shape the run's own ending produced: a baseline-only run
    # carries nothing, an eval_after repair carries the rollout and any recorded eval_before.
    recorded: list[str] = []
    if phase != "eval_before" and (run_dir / "eval_before").is_dir():
        recorded.append("eval_before")
    # A bookend's stopping file is the SOURCE's terminus, copied in for the resumed preflight
    # and never this run's own rollout, so a rerun must not republish it as one: the same
    # narrowing the resume applies, or a repaired bookend would change artifact shape.
    if (run_dir / "rollout_stopping.json").is_file() and "rebookend" not in manifest:
        recorded.append("rollout")
    recorded_phases = tuple(recorded)
    if refresh_baseline:
        # Before the sandbox and before any leg, so a carry that cannot be caught up refuses
        # having spent nothing. The refreshed rows go through the run's own publish path, and
        # the republication below reads them back the way every publication of a bookend does.
        refresh, refreshed_rows = baseline_refresh_plan(
            run_dir, manifest, split=split, baseline_run_dir=baseline_run_dir
        )
        if refresh["refused"]:
            raise RuntimeError(
                f"the baseline {refresh['baseline_run_dir']} no longer holds the rows this "
                f"bookend carries for held-out {refresh['refused']}: a refresh adds ids the "
                "carry has none for and upgrades ones that never settled, and never replaces "
                "a measured row. Nothing has been spent."
            )
        ctx.publish_json(
            run_dir / BASELINE_BEFORE_FILE,
            {
                "source_run_id": refresh["baseline_run_id"],
                "rows": [asdict(row) for row in refreshed_rows],
            },
        )
        manifest["rebookend"].setdefault("baseline_refreshes", []).append(
            {
                "refreshed_at": time.time(),
                "tasks_added": refresh["added"],
                "tasks_upgraded": refresh["upgraded"],
            }
        )
    manifest.setdefault("eval_reruns", []).append({"at": time.time(), "phase": phase})
    ctx.publish_json(run_dir / "manifest.json", manifest)
    sandbox.up()
    observer = _Egress(_start_egress(sandbox, run_dir) if capture_egress else None, run_dir)
    try:
        return await _run_phases(
            ctx,
            manifest=manifest,
            phases=(phase,),
            results_dir=results_dir,
            observer=observer,
            recorded_phases=recorded_phases,
            # Same rule as a resume: a reopened bookend republishes under its own run id.
            artifact=run_id if "rebookend" in manifest else None,
        )
    finally:
        with contextlib.suppress(Exception):
            observer.stop()
        sandbox.down()


def _materialize_home(
    source: Path,
    destination: Path,
    *,
    root: Path | None = None,
    active: set[Path] | None = None,
) -> None:
    """Copy ``source`` into ``destination`` with every symlink turned into the bytes it names.

    ``shutil.copytree(symlinks=False, ignore_dangling_symlinks=True)`` was not this: CPython
    tests a link's TEXTUAL target against the process cwd rather than the link's parent, so a
    valid relative link (the shape of every link in the real prime homes: in-home cache links,
    all relative, all resolving) read as dangling and silently vanished from the snapshot.
    Here each link resolves the way the filesystem resolves it, from its own parent:

    - a link to a file inside the source home becomes that file's bytes;
    - a link to a directory inside the source home becomes that tree, links within it
      materialized the same way, with the chain of directories currently being materialized
      tracked so a cycle fails loudly instead of recursing forever (two links to ONE target
      are fine; a link into its own ancestry is not);
    - a genuinely dangling link is dropped: it names no bytes, and the CLIs resolve it to
      nothing as well;
    - a link resolving OUTSIDE the source home is refused loudly. The snapshot must be of the
      source, and importing whatever the operator's host keeps at that path would be
      contamination wearing the archive's name; a container-absolute link (``/root/...``)
      usually names nothing on the host and is dropped as dangling, but one that does resolve
      here names bytes the run never saw, and that is a refusal, not an import;
    - a special file (FIFO, socket) fails loudly, as the copy always did.
    """
    root = root if root is not None else source.resolve()
    root_stat = os.stat(root)
    if active is None:
        active = {(root_stat.st_dev, root_stat.st_ino)}
    destination.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        target_path = destination / entry.name
        if entry.is_symlink():
            try:
                resolved = Path(os.path.realpath(entry, strict=True))
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise RuntimeError(f"cannot materialize the symlink {entry}: {exc}") from exc
            if not _same_or_under(resolved, root_stat):
                raise RuntimeError(
                    f"the symlink {entry} resolves outside the source home ({resolved}); a "
                    "snapshot that imported those bytes would not be of the source. Remove "
                    "the link from the archive copy, or bookend a source without it."
                )
            if resolved.is_dir():
                resolved_stat = os.stat(resolved)
                key = (resolved_stat.st_dev, resolved_stat.st_ino)
                if key in active:
                    raise RuntimeError(
                        f"the symlink {entry} cycles into a directory already being "
                        f"materialized ({resolved})"
                    )
                active.add(key)
                _materialize_home(resolved, target_path, root=root, active=active)
                active.discard(key)
            elif resolved.is_file():
                shutil.copy2(resolved, target_path)
            else:
                raise RuntimeError(
                    f"the symlink {entry} resolves to {resolved}, which is neither a file "
                    "nor a directory; a special file cannot be part of the snapshot"
                )
        elif entry.is_dir():
            _materialize_home(entry, target_path, root=root, active=active)
        elif entry.is_file():
            shutil.copy2(entry, target_path)
        else:
            raise RuntimeError(
                f"{entry} is neither a file, a directory, nor a symlink; a special file "
                "cannot be part of the snapshot"
            )
    # The directory's own metadata, applied after its contents so the content writes cannot
    # re-stamp it. mkdir alone left every directory with the process defaults, which widened
    # the real homes' 0700 directories (session leases and daemon caches among them) to 0755:
    # a loosened mode is not the snapshot, and a resumed CLI can behave differently over it.
    # Every directory passes through here, ordinary ones and materialized link targets alike,
    # because both recurse through this function.
    shutil.copystat(source, destination)


def _same_or_under(resolved: Path, root_stat: os.stat_result) -> bool:
    """Is this resolved path the source home or inside it, by filesystem identity?

    Identity, not spelling: on a case-insensitive volume one directory answers to many
    spellings, ``realpath`` keeps whichever spelling the link used, and a lexical prefix test
    refused valid in-home links over casing alone (observed on APFS). The test that cannot be
    fooled by spelling is ``samestat`` against the root's device and inode, walked up the
    resolved target's ancestry.
    """
    for candidate in (resolved, *resolved.parents):
        try:
            candidate_stat = os.stat(candidate)
        except OSError:
            continue
        if os.path.samestat(candidate_stat, root_stat):
            return True
    return False


def _has_eval_before(run_dir: Path) -> bool:
    """Did this run measure an eval_before of its own: per-task provenance, not rows.

    The eval phase records one directory per held-out task, so presence of any is what
    distinguishes a run that measured a before-side from a rollout-only or after-only run.
    """
    phase_dir = run_dir / "eval_before"
    if not phase_dir.is_dir():
        return False
    return any(p.is_dir() and p.name.startswith("task-") for p in phase_dir.iterdir())


def _refuse_live_source(source_run_dir: Path) -> None:
    """Refuse a source whose run lock a live process still holds, without mutating the source.

    A source can hold a settled-looking terminus while its own eval_after or a later rerun
    still owns the directory, and a snapshot taken under a live writer is not an archived
    state. The probe is a moment of :func:`_holding_source_still`: acquired and released the
    instant it answers, for the fast refusal before anything is created; the snapshot itself
    is taken under the held form.
    """
    with _holding_source_still(source_run_dir):
        pass


@contextlib.contextmanager
def _holding_source_still(source_run_dir: Path):
    """Hold the source still for the body of the block, without ever mutating the source.

    A SHARED flock on the source's EXISTING lock file, opened read-only and never created.
    Shared is exactly sufficient: every mutating owner (a run, a resume, a rerun) takes the
    lock EXCLUSIVE through ``_acquire_run_lock``, so a held shared lock refuses acquisition
    to any would-be mutator for as long as the block runs, a live mutator refuses THIS
    acquisition, and concurrent rebookends of one source, which only read, can hold it
    together. Releasing on the way out is what lets the archive be mutated again the moment
    the snapshot no longer depends on it; the probe form releases immediately, and the
    snapshot form holds across the whole materialization, because a lock released after a
    probe left the copy racing any mutator that acquired in between (a concurrent rerun's
    mid-copy write landed in the published snapshot, in review).

    A missing lock file is a refusal, not an empty hold. Every mutator would CREATE the lock
    on its way in (``_acquire_run_lock`` opens with O_CREAT), so a source without one is not
    quiet, it is unholdable: a resume or rerun starting mid-copy would mint the lock and
    mutate under the advertised hold (reproduced). Creating the lock from here would itself
    write into the archive, so the honest options are refusing or copy-and-revalidate, and
    refusal is chosen because it is simple and the archives this entry exists for all carry
    their locks (every run since the lock landed writes one; only pre-lock-era directories
    lack it). The message names the workaround, which is the operator's deliberate one-file
    write, never this runner's.
    """
    import fcntl

    lock_path = source_run_dir / RUN_LOCK_FILE
    if not lock_path.is_file():
        raise RuntimeError(
            f"{source_run_dir} has no {RUN_LOCK_FILE}, so it cannot be held still while the "
            "snapshot is taken: a mutator would create the lock and write mid-copy. If the "
            "archive is genuinely settled and you accept adding one file to it, create the "
            f"lock yourself (touch {source_run_dir / RUN_LOCK_FILE}) and re-run."
        )
    fd = os.open(lock_path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(
                f"{source_run_dir} is owned by a live process, so its state is still moving: "
                "a snapshot taken now would not be the archived run. Wait for it to finish, "
                "or stop it first."
            ) from None
        yield
    finally:
        os.close(fd)


def read_baseline_before(baseline_dir: Path, task_ids: Sequence[int]) -> list[TaskResult]:
    """The baseline's eval_before rows, read the one way every carry of them is taken.

    Held still when the baseline carries a lock file, so a repair that owns it refuses this
    read rather than being read mid-write; the pre-lock-era baselines have none, and their row
    files are append-only JSONL written by archived runs, so a plain read is honest there.
    Creation and refresh both come through here, which is what lets a refreshed carry be
    compared with the carried one field by field.
    """
    hold = (
        _holding_source_still(baseline_dir)
        if (baseline_dir / RUN_LOCK_FILE).is_file()
        else contextlib.nullcontext()
    )
    with hold:
        return read_eval_phase(baseline_dir / "eval_before", task_ids)


def _rows_by_task(rows: Sequence[TaskResult]) -> dict[int, list[TaskResult]]:
    grouped: dict[int, list[TaskResult]] = {}
    for row in rows:
        grouped.setdefault(row.task_idx, []).append(row)
    return grouped


def baseline_carry_gaps(
    rows: Sequence[TaskResult], *, task_ids: Sequence[int]
) -> dict[str, list[int]]:
    """The held-out ids a baseline's eval_before cannot account for yet, by kind.

    ``missing`` produced no row at all; ``unsealed`` produced rows but no settled outcome (a
    drained row, a dispense that never sealed, a double). Both are ids a
    ``rerun-eval --phase eval_before`` on the baseline would re-run, so both are ids whose
    rows can still change after a carry is taken.
    """
    grouped = _rows_by_task(rows)
    gaps: dict[str, list[int]] = {"missing": [], "unsealed": []}
    for idx in task_ids:
        for_id = grouped.get(idx, [])
        if _settled_row(for_id, idx) is not None:
            continue
        recorded = any(row.closure != MISSING_CLOSURE for row in for_id)
        gaps["unsealed" if recorded else "missing"].append(idx)
    return gaps


def baseline_refresh_delta(
    carried: Sequence[TaskResult], live: Sequence[TaskResult], *, task_ids: Sequence[int]
) -> tuple[dict[str, list[int]], list[TaskResult]]:
    """What catching a carry up to its baseline would change, and the rows it would then hold.

    Two directions are allowed, because neither replaces a measurement: an id the carry has no
    row for takes the live one, and an id whose carried row never settled takes a live settled
    one. Every other difference is refused, since a carried settled row that would change, or
    one whose live row is gone, is exactly what the carry exists to hold still.
    """
    was_by, now_by = _rows_by_task(carried), _rows_by_task(live)
    delta: dict[str, list[int]] = {"added": [], "upgraded": [], "refused": []}
    for idx in task_ids:
        was, now = was_by.get(idx, []), now_by.get(idx, [])
        if [asdict(row) for row in was] == [asdict(row) for row in now]:
            continue
        if _settled_row(was, idx) is None and _settled_row(now, idx) is not None:
            recorded = any(row.closure != MISSING_CLOSURE for row in was)
            delta["upgraded" if recorded else "added"].append(idx)
        else:
            delta["refused"].append(idx)
    catching_up = set(delta["added"]) | set(delta["upgraded"])
    rows: list[TaskResult] = []
    for idx in sorted(set(was_by) | set(now_by)):
        rows.extend(now_by[idx] if idx in catching_up else was_by.get(idx, []))
    return delta, rows


def baseline_refresh_plan(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    split: Split,
    baseline_run_dir: Path | None = None,
) -> tuple[dict[str, Any], list[TaskResult]]:
    """A bookend's carried baseline rows against the baseline's live ones.

    Returns what a refresh would change and the rows it would carry afterwards, so the plan
    and the spending path say the same thing. Raises for everything that makes the comparison
    meaningless: a run that carries no baseline rows at all, and a baseline directory that is
    not the run the carry and the marker both name.
    """
    if "rebookend" not in manifest:
        raise RuntimeError(
            f"{run_dir} is not a rebookend, so it carries no baseline rows: a refresh acts on "
            "the before block a bookend carries, and every other run measured its own."
        )
    carry_path = run_dir / BASELINE_BEFORE_FILE
    if not carry_path.is_file():
        raise RuntimeError(
            f"{run_dir} is a rebookend but its carried baseline rows ({BASELINE_BEFORE_FILE}) "
            "cannot be read, so there is nothing to refresh."
        )
    payload = json.loads(carry_path.read_text(encoding="utf-8"))
    recorded_id = str(manifest["rebookend"].get("baseline_run_id") or "")
    carried_id = str(payload.get("source_run_id") or "")
    if carried_id != recorded_id:
        raise RuntimeError(
            f"{run_dir} carries rows labeled {carried_id!r} while its manifest names "
            f"{recorded_id!r} as the baseline: the two disagree about which run measured the "
            "before side, and a refresh must not decide that."
        )
    baseline_dir = (
        Path(baseline_run_dir) if baseline_run_dir is not None else run_dir.parent / recorded_id
    ).resolve()
    baseline_manifest_path = baseline_dir / "manifest.json"
    if not baseline_manifest_path.is_file():
        raise RuntimeError(
            f"{baseline_dir} has no manifest.json, so the baseline {recorded_id} cannot be "
            "re-read. Name its run directory with --baseline."
        )
    live_id = str(
        json.loads(baseline_manifest_path.read_text(encoding="utf-8")).get("run_id") or ""
    )
    if live_id != recorded_id:
        raise RuntimeError(
            f"{baseline_dir} is run {live_id!r}, not the {recorded_id!r} this bookend carries: "
            "refreshing from another run would splice a different experiment's rows into a "
            "measured before side."
        )
    task_ids = [int(task_id) for task_id in side_for_phase(split, "eval_before").task_ids]
    delta, rows = baseline_refresh_delta(
        [TaskResult(**row) for row in payload["rows"]],
        read_baseline_before(baseline_dir, task_ids),
        task_ids=task_ids,
    )
    return (
        {
            "baseline_run_id": recorded_id,
            "baseline_run_dir": str(baseline_dir),
            **delta,
        },
        rows,
    )


async def rebookend_run(
    source_run_dir: Path,
    *,
    runs_dir: Path,
    results_dir: Path,
    baseline_run_dir: Path | None = None,
    agent_image: str = AGENT_IMAGE,
    credentials: dict[str, str] | None = None,
    capture_egress: bool = True,
    allow_partial_baseline: bool = False,
) -> Path:
    """Give an EXISTING run a resumed eval_after, as a NEW run, and return its results path.

    The source is an archived artifact and stays byte-untouched: it is read, never locked and
    never written. Everything this creates lives in a fresh run directory whose lock is taken
    before anything else, exactly as a fresh cell takes its own.
    """
    source_run_dir = Path(source_run_dir).resolve()
    manifest_path = source_run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"{source_run_dir} has no manifest.json; this is not a run directory.")
    # No output may land at or under the source, whatever the operator typed: the untouched
    # guarantee is over the tree, and `--runs <source>` or `--results <source>` would write the
    # new lock, the run directory, or the published JSON straight into the archive.
    for label, out_dir in (("runs_dir", Path(runs_dir)), ("results_dir", Path(results_dir))):
        resolved = out_dir.resolve()
        if resolved == source_run_dir or resolved.is_relative_to(source_run_dir):
            raise RuntimeError(
                f"{label} {out_dir} is inside the source run directory, and a rebookend never "
                "writes into the archive it bookends. Point it elsewhere."
            )
    if (source_run_dir / SUSPENSION_FILE).is_file():
        raise RuntimeError(
            f"{source_run_dir} holds a suspension record, so the run is not finished: use "
            "`shobench resume`, which knows what the interruption was waiting for. A rebookend "
            "measures the far side of a settled terminus."
        )
    _refuse_live_source(source_run_dir)
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "rebookend" in source_manifest:
        # A bookend of a bookend expresses nothing a direct rebookend does not express
        # better. The bookend's home IS the source's terminal home, copied; its own
        # eval_after advanced no rollout and left no new terminus, so chaining re-measures
        # the same terminal state with an extra hop of provenance for a reader to unwind,
        # and the reporter would have to walk chains to find the one real before-side.
        # Refused, with the original named, because the original is the run to bookend.
        original = source_manifest["rebookend"].get("rebookend_of", "unknown")
        raise RuntimeError(
            f"{source_run_dir} is itself a rebookend (of {original}), and a bookend of a "
            "bookend re-measures the same terminal state as rebookending the original "
            f"directly. Rebookend the original run ({original}) instead."
        )
    cell = load_cell_by_name(source_manifest["cell"]["name"])
    # The baseline is its own identity, not an assumption about the source. The v0 sources
    # are rollout-only or after-only runs whose before-side was measured by a SEPARATE
    # deferred-baseline run, so a marker that named only the rollout source left the
    # assembler pairing against emptiness. A source that measured its own eval_before
    # defaults to itself, which is the self-paired case stated rather than assumed; a source
    # that did not REQUIRES the baseline run to be named, and every named baseline is
    # validated here, before anything spends: it must be a run, must not itself be a bookend
    # (the chain rule, for the same reason), must be the same cell, and, the load-bearing
    # check, must carry the same split id digest, because before rows over different held-out
    # ids would pair task numbers that are not the same tasks.
    if baseline_run_dir is not None:
        baseline_dir = Path(baseline_run_dir).resolve()
        baseline_manifest_path = baseline_dir / "manifest.json"
        if not baseline_manifest_path.is_file():
            raise RuntimeError(
                f"{baseline_dir} has no manifest.json; the baseline is not a run directory."
            )
        baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
        if "rebookend" in baseline_manifest:
            raise RuntimeError(
                f"{baseline_dir} is itself a rebookend, and a bookend has no before-side to "
                "pair against; name the run that measured the baseline."
            )
        if baseline_manifest["cell"]["name"] != source_manifest["cell"]["name"]:
            raise RuntimeError(
                f"the baseline {baseline_dir} measured cell "
                f"{baseline_manifest['cell']['name']!r}, not the source's "
                f"{source_manifest['cell']['name']!r}; a pairing across cells is not a "
                "measurement."
            )
        source_digest = source_manifest.get("split", {}).get("id_digest")
        baseline_digest = baseline_manifest.get("split", {}).get("id_digest")
        if source_digest != baseline_digest:
            raise RuntimeError(
                f"the baseline {baseline_dir} ran split id digest {baseline_digest!r} and "
                f"the source ran {source_digest!r}: the before rows would not measure the "
                "same held-out ids, so the pairing would be task numbers, not tasks."
            )
        if not _has_eval_before(baseline_dir):
            raise RuntimeError(
                f"the baseline {baseline_dir} has no eval_before provenance of its own, so "
                "it holds no before rows to pair with."
            )
        # The bookend runs its after side against the SOURCE's definition and carries THIS
        # run's before rows, so the two definitions have to be one. Refused rather than
        # reconciled, since nothing here can say which of them the pair should have had.
        pairing = pairing_drift(source_manifest, baseline_manifest)
        if pairing:
            raise RuntimeError(
                f"the baseline {baseline_dir} was not measured by the same definition as the "
                f"source: {'; '.join(pairing)}. Name a baseline of this definition, or measure "
                "one."
            )
        baseline_run_id = str(baseline_manifest.get("run_id", ""))
        baseline_dir_resolved = baseline_dir
        pairing_identity_manifest = baseline_manifest
    elif _has_eval_before(source_run_dir):
        baseline_run_id = str(source_manifest.get("run_id", ""))
        baseline_dir_resolved = source_run_dir
        # Self-paired: the before rows are the source's own, so the two sides are one archive
        # and every identity matches itself. What the source does not record about itself is
        # still worth naming, since the bookend's after side runs on today's image and runner.
        pairing_identity_manifest = source_manifest
    else:
        raise RuntimeError(
            f"{source_run_dir} has no eval_before of its own (a rollout-only or after-only "
            "run), so the bookend would have nothing to pair with. Name the run that "
            "measured this cell's baseline with --baseline."
        )
    # Taken against the cell FILE rather than against the cell this run will build from it: the
    # reader's answer to "the file changed, so what governed this run?". The refusal below is a
    # different comparison, against the cell that actually runs.
    checkout_drift = cell_field_drift(source_manifest.get("cell", {}), cell.to_manifest())
    # What the pairing could not establish, computed where both records are in hand and carried
    # into the bookend's own manifest, so the artifact states it rather than a reader assuming.
    unproven_identities = pairing_unproven(source_manifest, pairing_identity_manifest)
    # The record wins over the checkout for everything the new measurement inherits: the arm the
    # source's rollout served, and the eval runtime its before side was scored under.
    cell = bookend_cell(cell, source_manifest)
    split = load_split_by_name(cell.split)
    instruction = load_instruction(cell.instruction_arm)
    # Field by field rather than by the cell file's bytes. The fields this run inherits were
    # just recovered onto the cell, so this holds that recovery to its word.
    drift = experiment_drift(
        source_manifest,
        cell=cell,
        split=split,
        instruction=instruction,
        scope=DRIFT_BOOKEND,
    )
    if drift:
        raise RuntimeError(
            "the checkout no longer matches the run being rebookended: "
            + "; ".join(drift)
            + ". Restore the recorded definition; a bookend under an edited definition would "
            "not pair with the run it claims to follow."
        )
    # The third side, compared here before a byte is copied, and again after setup for the two
    # facts that do not exist yet.
    image_ref, image_tag, image_id = pinned_image(agent_image)
    unproven_execution = _refuse_execution_drift(
        source_manifest,
        current_identity(
            cell=cell,
            split=split,
            instruction=instruction,
            harness=harness_for(cell.harness),
            image_tag=image_tag,
            image_digest_value=image_id,
        ),
        stage=IDENTITY_PRE_SPEND,
        what="being rebookended",
    )
    stopping_path = source_run_dir / ROLLOUT_STOPPING_FILE
    if not stopping_path.is_file():
        raise RuntimeError(
            f"{source_run_dir} has no rollout terminus, so there is no conversation end to "
            "resume from: a rebookend belongs on the far side of a real rollout ending, the "
            "same rule eval_after always follows."
        )
    if terminal_session_in(source_run_dir) is None:
        raise RuntimeError(
            f"{source_run_dir}'s rollout record names no terminal session, so its conversation "
            "cannot be resumed. Running the bookend cold instead would publish a mislabeled "
            "measurement; there is no fallback."
        )
    # A fresh id even for two rebookends of one source in the same second: the timestamped stem
    # has one-second resolution, and a collision would hand the second caller the first one's
    # lock refusal instead of its own new run. The suffix is what makes the destination
    # genuinely fresh rather than fresh-if-nobody-else-was-quick.
    run_id = f"{_run_id(cell)}-rb{uuid.uuid4().hex[:8]}"
    run_dir = Path(runs_dir) / run_id
    # The concrete artifacts too, not only their directories. The bookend publishes under its
    # OWN run id, never the cell name: the cell-name artifact is the SOURCE's measurement, the
    # one this bookend exists to pair with, and sharing the stem destroyed it (write_results
    # keeps one artifact per stem by design). The run-id stem keeps every bookend of every
    # source coexisting beside the source result, and it makes the leaf unpredictable before
    # this moment, so nothing can pre-occupy it; these checks still bound the minted names
    # and the run path before the lock creates anything through them.
    for target in (
        run_dir,
        Path(results_dir) / f"{run_id}.json",
        Path(results_dir) / f"{run_id}{INCOMPLETE_SUFFIX}",
    ):
        resolved_target = target.resolve()
        if resolved_target == source_run_dir or resolved_target.is_relative_to(source_run_dir):
            raise RuntimeError(
                f"{target} resolves into the source run directory, and a rebookend never "
                "writes into the archive it bookends."
            )
    with owning_run(run_dir) as stop:
        return await _rebookend_owned(
            source_run_dir,
            source_manifest,
            cell=cell,
            split=split,
            instruction=instruction,
            baseline_run_id=baseline_run_id,
            baseline_dir=baseline_dir_resolved,
            checkout_drift=checkout_drift,
            unproven_identities=unproven_identities,
            unproven_execution=unproven_execution,
            image_ref=image_ref,
            image_tag=image_tag,
            image_id=image_id,
            run_id=run_id,
            run_dir=run_dir,
            results_dir=results_dir,
            agent_image=agent_image,
            credentials=credentials,
            capture_egress=capture_egress,
            allow_partial_baseline=allow_partial_baseline,
            stop=stop,
        )


async def _rebookend_owned(
    source_run_dir: Path,
    source_manifest: dict[str, Any],
    *,
    cell: Cell,
    split: Split,
    instruction: Instruction,
    baseline_run_id: str,
    baseline_dir: Path,
    checkout_drift: dict[str, dict[str, Any]],
    unproven_identities: list[str],
    unproven_execution: list[str],
    image_ref: str,
    image_tag: str,
    image_id: str | None,
    run_id: str,
    run_dir: Path,
    results_dir: Path,
    agent_image: str,
    credentials: dict[str, str] | None,
    capture_egress: bool,
    allow_partial_baseline: bool,
    stop: EarlyEnding,
) -> Path:
    """The rebookend's own run, under an already-held lock on the NEW directory.

    Three copies make it work, and each has a reason to be a copy rather than a reference.
    The source's accumulated HOME is copied whole, transcripts included, because the resumed
    forks reopen the terminal session out of it and the source must stay the archived artifact
    it is; nothing here ever writes back. The source's stopping record is copied in, because
    the resumed preflight reads the terminus off the run it is part of, and because it makes
    the new run self-contained: a usage limit that suspends this bookend resumes through the
    ordinary `shobench resume`, which re-reads the same copied record. And the manifest is
    built fresh from the recovered cell rather than copied, so its digests describe the home
    this run actually starts from, with a provenance block naming the source it bookends.

    What is deliberately NOT copied: the source's provenance rows and legs. This run measures
    one thing, the after-bookend, and publishes as itself; it pairs with the source post-hoc,
    the way the deferred baselines pair. Its artifact carries the incomplete name, because a
    run with no eval_before cannot account for the before side, and that name is the honest
    one.
    """
    sandbox = CellSandbox(run_id=run_id, home=run_dir / "home", workdir=run_dir / "work")
    ctx = RunContext(
        cell=cell,
        split=split,
        instruction=instruction,
        harness=harness_for(cell.harness),
        run_id=run_id,
        run_dir=run_dir,
        sandbox=sandbox,
        agent_image=image_ref,
        image_tag=image_tag,
        image_digest=image_id,
        credentials=dict(credentials or {}),
        # The handle this process's ownership is already watching. The default is honest for a
        # caller that took no ownership, such as a test driving one phase.
        operator_stop=stop if stop is not None else EarlyEnding(),
    )
    # The post-rollout self, whole: durable channels AND session state, because the fork
    # machinery resolves the terminal transcript inside this copy before any fan-out. Copied
    # before the sandbox comes up so a failure here leaves nothing running.
    #
    # MATERIALIZED, never linked: a symlink in the snapshot is a hole in the untouched
    # guarantee, because everything that runs afterwards writes into this tree (the credential
    # seeding, the runner files, the RW container mount), and a preserved link pointing back
    # into the source turns any of those writes into a write THROUGH the copy into the
    # archive; a credential reseed did exactly that in review. Every link becomes the bytes it
    # names, resolved from the link's own parent (see ``_materialize_home`` for the whole
    # policy: valid links materialize, dangling ones drop, escaping ones and cycles refuse
    # loudly), and the snapshot references nothing beyond itself.
    source_home = source_run_dir / "home"
    if not source_home.is_dir():
        raise RuntimeError(f"{source_run_dir} has no home directory to bookend.")
    # The baseline's before rows, read here and published below as this artifact's own before
    # block. Read before the snapshot rather than after it, because both refusals below are
    # about the baseline alone and a home copy is minutes of a real archive's bytes.
    heldout_ids = [int(t) for t in side_for_phase(split, "eval_before").task_ids]
    baseline_rows = read_baseline_before(baseline_dir, heldout_ids)
    if not any(row.scored for row in baseline_rows):
        raise RuntimeError(
            f"the baseline {baseline_dir} holds no readable scored eval_before rows, so the "
            "bookend would have nothing to pair with. Nothing has been spent."
        )
    # A baseline still short of ids is the race this refuses: the carry freezes whatever the
    # baseline holds at this instant, and a repair that finishes minutes later leaves the
    # bookend's artifact reporting holes forever against a baseline that has none.
    baseline_gaps = baseline_carry_gaps(baseline_rows, task_ids=heldout_ids)
    if (baseline_gaps["missing"] or baseline_gaps["unsealed"]) and not allow_partial_baseline:
        raise RuntimeError(
            f"the baseline {baseline_dir} has not finished its eval_before: no rows for "
            f"held-out {baseline_gaps['missing']}, no settled row for "
            f"{baseline_gaps['unsealed']}. Finish it "
            f"(`uv run shobench rerun-eval --run {baseline_dir} --phase eval_before --go`) "
            "and rebookend, or pass --allow-partial-baseline to carry the rows as they stand. "
            "Nothing has been spent."
        )
    # Held STILL for the whole snapshot, not merely probed: a probe released before the copy
    # left the copy racing any mutator that acquired in between, and a concurrent rerun's
    # mid-copy write landed in the published snapshot. The shared hold refuses a live owner
    # and blocks a would-be one until the copy is whole, and it is released here, before any
    # spend, so the archive is never held during the eval itself.
    with _holding_source_still(source_run_dir):
        # Everything the plan relied on is re-proven under the hold, because a mutator that
        # ran WHOLE between the early checks and this hold left no live lock to refuse. A
        # suspension is the record such a mutator writes WITHOUT touching the manifest
        # (reproduced: an owner took the lock, wrote suspended.json, released, and the old
        # recheck saw an unchanged manifest and copied anyway), so it is rechecked first;
        # the terminus and its terminal session are rechecked the same way; and the manifest
        # compare catches every definitional rewrite, since a snapshot of new bytes under
        # the old cell would be two runs wearing one name.
        if (source_run_dir / SUSPENSION_FILE).is_file():
            raise RuntimeError(
                f"{source_run_dir} was suspended between the plan and the snapshot: the run "
                "is not finished, and its ending belongs to `shobench resume`."
            )
        if terminal_session_in(source_run_dir) is None:
            raise RuntimeError(
                f"{source_run_dir}'s rollout terminus changed between the plan and the "
                "snapshot and no longer names a terminal session. Re-run against the "
                "settled archive."
            )
        if json.loads((source_run_dir / "manifest.json").read_text(encoding="utf-8")) != (
            source_manifest
        ):
            raise RuntimeError(
                f"{source_run_dir}'s manifest changed between the plan and the snapshot; "
                "the source was mutated. Re-run the rebookend against the settled archive."
            )
        _materialize_home(source_home, sandbox.home)
        shutil.copy2(source_run_dir / ROLLOUT_STOPPING_FILE, run_dir / ROLLOUT_STOPPING_FILE)
        source_stopping = json.loads(
            (source_run_dir / ROLLOUT_STOPPING_FILE).read_text(encoding="utf-8")
        )

    # Persisted into the new run directory, so every publication of this run (the first, a
    # resumed one, a repaired one) reads the same carried rows.
    ctx.publish_json(
        run_dir / BASELINE_BEFORE_FILE,
        {"source_run_id": baseline_run_id, "rows": [asdict(row) for row in baseline_rows]},
    )

    sandbox.up()
    spec = spec_for(cell.harness, cell.credential_mode)
    seeded = seed_home(spec, sandbox.home)
    _watch_cell_credential(ctx, spec)
    observer = _Egress(_start_egress(sandbox, run_dir) if capture_egress else None, run_dir)
    try:
        probes = {
            "version": ctx.redactor.text(
                _probe(
                    ctx.harness.version_probe(),
                    # The pinned reference, never the tag: a probe from one image beside legs
                    # from another describes a run that did not happen, and two builds printing
                    # one version string is precisely the case the content id exists to tell
                    # apart, so the version probe cannot be the thing that notices.
                    image=ctx.agent_image,
                    sandbox=sandbox,
                    env={},
                )
            )
        }
        seeds = _place_runner_files(ctx)
        manifest = build_manifest(ctx, probes=probes)
        manifest["credential_seed"] = seeded
        manifest["home"]["seeded"] = seeds
        manifest["axes"]["credential_mode"] = credential_effective_mode(
            spec, sandbox.home, env_names=sorted(ctx.credentials)
        )
        # The facts that did not exist until now: the probe came out of the image a moment ago
        # and the credential mode off the home it was seeded into. Checked here, after the
        # manifest is built and before ``_run_phases``, which is the last moment before a row
        # can exist. The manifest this run publishes is its own; what it is checked against is
        # the SOURCE's record, the run whose measurement these rows will be paired with.
        unproven = unproven_execution + _refuse_execution_drift(
            source_manifest,
            after_setup_identity(
                probes=manifest["harness_probes"],
                credential_mode=manifest["axes"]["credential_mode"],
            ),
            stage=IDENTITY_AFTER_SETUP,
            what="being rebookended",
        )
        # What this run is a bookend OF, by run id rather than by path: a durable artifact
        # carries no operator layout, and the id is what pairs it with the source post-hoc.
        manifest["rebookend"] = {
            "rebookend_of": str(source_manifest.get("run_id", "")),
            # Two identities, deliberately: the source is the terminal-state lineage (whose
            # conversation this run resumes), and the baseline is the pairing identity (whose
            # eval_before rows the report joins this run's after rows with). They coincide
            # only when the source measured its own before-side.
            "baseline_run_id": baseline_run_id,
            "source_rollout_feedback": cell.rollout_feedback,
            "source_stop_reason": str(source_stopping.get("stop_reason", "")),
            # The eval runtime this run took from the source's RECORD rather than from the cell
            # file, with the values that actually bounded its legs. It is the fact a reader
            # needs before comparing any number here with the baseline's: both sides of the
            # pair stopped by the same rule.
            "eval_runtime_from_record": {
                field: getattr(cell.budget, field) for field in BOOKEND_EVAL_RUNTIME_FIELDS
            },
            # The source's cell block verbatim, beside the block this run ran under
            # (``manifest["cell"]``), and every field the checkout's cell file states
            # differently from the record, with both values. The file having moved is neither
            # hidden nor load-bearing: anything that could change the measurement refused
            # before this run spent, and the runtime fields above were inherited rather than
            # read, so a reader sees what the file would have run and what did run without
            # finding two checkouts to diff.
            "source_cell": dict(source_manifest.get("cell", {})),
            "cell_drift": checkout_drift,
            # What the pairing could NOT establish, named rather than left to a reader's
            # assumption: the identities neither archive recorded, which is empty for any pair
            # of runs made after the recording started and never empty for the v0 archives.
            # Every other identity refused before this run spent, so this list is the whole of
            # what the published delta rests on trust for.
            "pairing_identity_unproven": unproven_identities,
            # And what THIS run could not prove about itself against the source's record: the
            # same discipline one comparison further out, so the artifact names every identity
            # it rests on trust for rather than only the ones between the two archives.
            "execution_identity_unproven": sorted(set(unproven)),
        }
        if baseline_gaps["missing"] or baseline_gaps["unsealed"]:
            # Carried over the creation guard by --allow-partial-baseline: the ids the baseline
            # could not account for at snapshot time, so the artifact names the rows an operator
            # chose to freeze rather than leaving the holes to look like the baseline's own.
            manifest["rebookend"]["partial_baseline"] = baseline_gaps
        ctx.publish_json(run_dir / "manifest.json", manifest)
        return await _run_phases(
            ctx,
            manifest=manifest,
            phases=("eval_after",),
            results_dir=results_dir,
            observer=observer,
            # Under the bookend's own name: the cell-name artifact is the source's
            # measurement, and this run pairs with it rather than replacing it.
            artifact=run_id,
        )
    finally:
        with contextlib.suppress(Exception):
            observer.stop()
        sandbox.down()


def _migrate_recorded_containers(manifest: dict[str, Any], sandbox: CellSandbox) -> None:
    """One-time migration for a run recorded under the pre-digest container names.

    A continuation builds its names from the run id, so a run created before the digest fix
    computes DIFFERENT names than its manifest recorded. The manifest's container block is
    rewritten to the names the continuation actually uses, so the record keeps describing the
    run's real resources, and nothing recorded is deleted: the legacy formula is
    collision-prone by definition, so two runs' manifests can legitimately claim the SAME
    strings, and removing them here could tear down a live neighbor still running on a pre-fix
    process, which is the production failure this change exists to end. A legacy holder a
    crashed teardown left behind is inert (nothing joins it by name again) and is for explicit
    operator cleanup.
    """
    recorded = manifest.get("container") or {}
    if recorded and (
        recorded.get("netns_container") != sandbox.netns_container
        or recorded.get("network") != sandbox.network
    ):
        manifest["container"] = {
            **recorded,
            "network": sandbox.network,
            "netns_container": sandbox.netns_container,
        }


def _start_egress(sandbox: CellSandbox, run_dir: Path) -> egress.EgressCapture:
    """Attach the observer, writing to its own segment when the run already has a capture.

    The capture command truncates the file it writes, so pointing a continuation's observer at
    the existing record would erase the traffic from before the interruption. Each process gets
    a segment of its own instead, and the segments are folded into the record as they stop.
    """
    log_path = run_dir / EGRESS_LOG
    if log_path.exists():
        segment = 2
        while (run_dir / f"egress.{segment}.tsv").exists():
            segment += 1
        log_path = run_dir / f"egress.{segment}.tsv"
    return egress.start(
        netns_container=sandbox.netns_container,
        name=f"{sandbox.netns_container}-egress"[:63],
        log_path=log_path,
    )


def cleanup(run_id: str) -> None:
    """Remove a crashed run's containers and network, which are named after its run id."""
    stem = run_stem(run_id)
    for name in (f"{stem}-ns-egress", f"{stem}-ns"):
        docker("rm", "-f", name, check=False)
    docker("network", "rm", f"{stem}-net", check=False)


__all__ = [
    "CREDENTIAL_FILES",
    "NOISE_DIRS",
    "NOISE_FILES",
    "free_port",
    "is_noise",
    "EarlyEnding",
    "LegRecord",
    "RunContext",
    "ROLLOUT_STOPPING_FILE",
    "STOP_REQUEST_FILE",
    "SUSPENDED_EXIT_CODE",
    "SUSPENSION_FILE",
    "EvalSuspension",
    "Suspension",
    "baseline_carry_gaps",
    "baseline_refresh_delta",
    "baseline_refresh_plan",
    "build_manifest",
    "durable_filter",
    "cleanup",
    "read_baseline_before",
    "read_eval_phase",
    "read_stop_request",
    "rebookend_run",
    "resume_cell",
    "rollout_progress",
    "run_cell",
    "terminal_session_in",
    "run_eval_phase",
    "run_leg",
    "run_rollout_phase",
    "write_home_files",
    "write_stop_request",
]
