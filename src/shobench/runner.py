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
"""

from __future__ import annotations

import asyncio
import contextlib
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
from dataclasses import dataclass, field, replace
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
    run_relative,
    run_stem,
    write_json,
)
from shobench.credentials import CredentialSpec, seed_home, spec_for
from shobench.credentials import effective_mode as credential_effective_mode
from shobench.harness import Harness, LaunchSpec, StopKind, StopVerdict
from shobench.harnesses import harness_for
from shobench.pins import SHOGYM_REPO, SHOGYM_REV
from shobench.redact import MARKER as redact_marker
from shobench.redact import Redactor, redactor_for, secrets_in_file
from shobench.results import (
    INCOMPLETE_SUFFIX,
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
    agent_image: str = AGENT_IMAGE
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
        "substrate": {
            "shogym_repo": SHOGYM_REPO,
            "shogym_rev": SHOGYM_REV,
            "mcp_server_name": SERVER_NAME,
        },
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
            "effort": {
                "requested": ctx.cell.effort,
                "applied": bool(ctx.cell.effort and ctx.harness.effort_flag),
                "how": (
                    ctx.harness.effort_flag
                    or f"{ctx.harness.name} exposes no reasoning-effort control; it is ignored"
                ),
            },
        },
        "container": {
            "agent_image": ctx.agent_image,
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
) -> LegRecord:
    """Run one harness invocation to completion and classify how it ended.

    ``mcp_url``, ``cfg_dir``, ``home``, ``workdir`` and ``container_name`` default to the
    cell-wide values on the context, which is what the single rollout leg uses. An eval task
    overrides all five so it reaches its own stream on its own port, writes its own config,
    mounts its own throwaway home and its own throwaway ``/work``, and names its own container,
    which is what lets many tasks run at once without sharing any of that mutable state.
    """
    if resume and not session_id:
        raise RuntimeError("cannot resume without a session id; the previous leg wrote none")
    mcp_url = ctx.mcp_url if mcp_url is None else mcp_url
    cfg_dir = ctx.cfg_dir if cfg_dir is None else cfg_dir
    home = ctx.sandbox.home if home is None else home
    trace_dir = ctx.run_dir / phase / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    stem = f"leg-{leg:04d}" if task_idx is None else f"task-{task_idx:05d}-leg-{leg:04d}"
    stdout_path = trace_dir / f"{stem}.stream.jsonl"
    stderr_path = trace_dir / f"{stem}.err.txt"

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
    timed_out = False
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
        stdin_data = spec.stdin
        try:
            completed = subprocess.run(
                argv,
                stdout=out,
                stderr=err,
                input=stdin_data,
                text=stdin_data is not None,
                stdin=None if stdin_data is not None else subprocess.DEVNULL,
                timeout=timeout_s,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -1
            docker("rm", "-f", container, check=False)

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
    rows = read_phase(prov_dir)
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


def terminal_session_in(run_dir: Path) -> str | None:
    """The rollout's terminal session id as the run's own record names it, or ``None``.

    The stopping record's ``session_id`` is the terminal one (it is updated to the id the last
    leg really ran under); a record without it falls back to the last rollout leg that names
    one. Shared between the resumed eval_after preflight and the rebookend entry, which reads
    it off the SOURCE run it bookends, so the two cannot drift on what "the terminal session"
    means.
    """
    stopping_path = run_dir / ROLLOUT_STOPPING_FILE
    if not stopping_path.is_file():
        return None
    stopping = json.loads(stopping_path.read_text(encoding="utf-8"))
    session_id = str(stopping.get("session_id") or "") or None
    if session_id is None:
        for leg in reversed(stopping.get("legs", [])):
            if leg.get("session_id"):
                return str(leg["session_id"])
    return session_id


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

    async def one_task(task_id: str) -> None:
        idx = int(task_id)
        prov_dir = phase_dir / f"task-{idx:05d}"
        task_home = phase_dir / "homes" / f"task-{idx:05d}"
        task_work = phase_dir / "work" / f"task-{idx:05d}"
        task_cfg = phase_dir / "cfg" / f"task-{idx:05d}"
        async with gate:
            if usage_limit:
                return  # a usage limit closed the window; this task waits for the resume
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
                async with stream, _served(stream, port):
                    record = await asyncio.to_thread(
                        run_leg,
                        ctx,
                        phase=phase,
                        leg=idx,
                        # A resumed fork carries the ROLLOUT's standing instruction, not the
                        # eval one. The rule that the eval instruction never carries the
                        # improvement objective was designed for cold measurement; a resumed
                        # conversation already carries the objective in its history and its
                        # compaction summaries, and swapping the standing instruction
                        # mid-conversation would measure an agent that never existed. The
                        # resumed after measures the agent as it lived, objective included;
                        # every cold session (eval_before always) keeps the blind eval
                        # instruction.
                        system_prompt=(
                            ctx.instruction.rollout_system
                            if resume_session
                            else ctx.instruction.eval_system
                        ),
                        user_prompt=ctx.instruction.kickoff,
                        # A resumed fork names the rollout's terminal session; every task names
                        # the same one, and the per-task home copies are what keep the forks
                        # independent. A cold task pins a fresh id instead.
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
                    )
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
# The rollout's stop classification, persisted beside its provenance the moment the rollout ends.
# Only the runner sees how a leg ended, so this is the one piece of the rollout that cannot be
# re-read from the shogym record. An eval_after suspension leaves the rollout finished but the
# results file unwritten, and its resume republishes the rollout from provenance plus this file,
# so the stop reason survives an interruption that happens after the rollout is already paid for.
ROLLOUT_STOPPING_FILE = "rollout_stopping.json"
# The run's whole egress capture. One process writes it and any continuation appends to it, so
# the published summary covers the cell rather than only its last stretch.
EGRESS_LOG = "egress.tsv"
# The exit status a suspended cell leaves. `run` cannot return it, because a suspension is the
# one path that must not unwind, so the status is how a shell or a supervising script tells a
# cell that is waiting for a window from one that failed. 75 is the conventional "temporary
# failure, try the same thing again later", which is exactly what this is.
SUSPENDED_EXIT_CODE = 75


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


def experiment_drift(
    manifest: dict[str, Any], *, cell: Cell, split: Split, instruction: Instruction
) -> list[str]:
    """What the checkout now says that the recorded run does not, as human-readable lines.

    A suspended cell can wait hours, and a repository is edited in hours. A continuation runs
    against whatever the files say today, so anything that moved between the two is a different
    experiment wearing the first one's run id: a re-tuned budget, a regenerated split whose
    positions no longer mean what the record says they mean, a reworded instruction the second
    half of the rollout would be run under. The digests to compare are already in the manifest,
    written before anything spent, so this is a comparison rather than a new mechanism.

    Returned rather than raised, so a caller can report every difference at once. An operator
    told about the budget, only to be told about the split on the next attempt, learns to stop
    reading.
    """
    recorded_cell = manifest.get("cell", {})
    recorded_split = manifest.get("split", {})
    recorded_instruction = manifest.get("instruction", {})
    checks = (
        (
            "cell config",
            recorded_cell.get("config_sha256"),
            cell.to_manifest().get("config_sha256"),
        ),
        ("split ids", recorded_split.get("id_digest"), split.to_manifest().get("id_digest")),
        (
            "rollout instruction",
            recorded_instruction.get("rollout_system_sha256"),
            instruction.rollout_system_sha256,
        ),
        (
            "eval instruction",
            recorded_instruction.get("eval_system_sha256"),
            instruction.eval_system_sha256,
        ),
    )
    return [
        f"{what} changed since the run started (recorded {was}, now {now})"
        for what, was, now in checks
        if was is not None and now is not None and was != now
    ]


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
    async with stream, _served(stream, ctx.port):
        consumed_before = stream.queue_info().consumed
        record = await asyncio.to_thread(
            run_leg,
            ctx,
            phase="rollout",
            leg=suspended.legs_before if resuming else 0,
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
        )
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
    if "rollout" in recorded_phases:
        # The rollout's stop classification is not in provenance, so it was persisted when the
        # rollout ended and is read back here. This is the path an eval_after resume takes: it
        # republishes the rollout it did not re-run, stop reason and all, rather than a blank one.
        stopping = json.loads(
            (ctx.run_dir / ROLLOUT_STOPPING_FILE).read_text(encoding="utf-8")
        )
    for phase in phases:
        if phase == "rollout":
            phase_rows[phase], stopping = await run_rollout_phase(ctx, suspended=suspended)
            # Persist the stop classification the moment the rollout ends, so an eval_after
            # suspension that follows can republish it: only the runner saw how the leg ended.
            ctx.publish_json(ctx.run_dir / ROLLOUT_STOPPING_FILE, stopping)
            # The durable measurement is taken here, at the rollout's terminus, and written into
            # the manifest before eval_after runs. That is the whole boundary: what the rollout
            # left, read at the moment the rollout ended, and never again afterwards. Taken at
            # the end of the cell instead, anything an eval phase managed to write into the base
            # home would be attributed to the rollout, and the reader would have no way to tell.
            _snapshot_durable_state(ctx, manifest)
        else:
            phase_rows[phase] = await run_eval_phase(ctx, phase)
        ctx.publish_json(ctx.run_dir / "legs.json", ctx.leg_records())

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
        # No rollout ran in this process and none was carried forward, which only happens when
        # an operator asked for a subset of the phases. There is no rollout terminus to have
        # measured, so the state is read now and said to have been.
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

    egress_summary = observer.stop()
    results_path = write_results(
        results_dir / f"{artifact or ctx.cell.name}.json",
        manifest=manifest,
        phases=phase_rows,
        stopping=stopping,
        heldout_ids=heldout_ids,
        egress=egress_summary,
        redact=ctx.redactor.json,
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
# file's contents are diagnostics for a human reading a refusal, never part of the protocol,
# and the file is never unlinked, because unlinking a path a contender has already opened
# would put two processes behind two different inodes of one name.
RUN_LOCK_FILE = "run.lock"


def _acquire_run_lock(run_dir: Path) -> int:
    """Take exclusive kernel ownership of ``run_dir``; returns the fd that holds it."""
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
    os.ftruncate(fd, 0)
    os.write(fd, json.dumps({"pid": os.getpid(), "at": time.time()}).encode("utf-8"))
    return fd


def _release_run_lock(lock_fd: int) -> None:
    """Closing the fd releases the kernel lock; every hard ending already did this implicitly."""
    with contextlib.suppress(OSError):
        os.close(lock_fd)


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
    teardown that raises: the lock has to outlive every fallible setup step it protects.
    """
    run_id = _run_id(cell)
    run_dir = runs_dir / run_id
    lock_fd = _acquire_run_lock(run_dir)
    try:
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
        )
    finally:
        _release_run_lock(lock_fd)


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
) -> Path:
    instruction = load_instruction(cell.instruction_arm)
    sandbox = CellSandbox(run_id=run_id, home=run_dir / "home", workdir=run_dir / "work")
    ctx = RunContext(
        cell=cell,
        split=split,
        instruction=instruction,
        harness=harness_for(cell.harness),
        run_id=run_id,
        run_dir=run_dir,
        sandbox=sandbox,
        port=port,
        agent_image=agent_image,
        credentials=dict(credentials or {}),
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
                _probe(ctx.harness.version_probe(), image=agent_image, sandbox=sandbox, env={})
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
                    model_probe, image=agent_image, sandbox=sandbox, env=ctx.credentials
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
    so this acquisition succeeds against a genuinely waiting run and refuses a live one.
    """
    lock_fd = _acquire_run_lock(run_dir)
    try:
        return await _resume_cell_owned(
            run_dir,
            results_dir=results_dir,
            agent_image=agent_image,
            credentials=credentials,
            capture_egress=capture_egress,
        )
    finally:
        _release_run_lock(lock_fd)


async def _resume_cell_owned(
    run_dir: Path,
    *,
    results_dir: Path,
    agent_image: str = AGENT_IMAGE,
    credentials: dict[str, str] | None = None,
    capture_egress: bool = True,
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
    cell = load_cell_by_name(manifest["cell"]["name"])
    recorded_regime = recorded_rollout_feedback(manifest)
    if cell.rollout_feedback != recorded_regime:
        # The continuation finishes the experiment the record started, so the run's recorded
        # feedback arm wins over the checkout's default. shogym would refuse to reopen the
        # provenance directory under the other regime anyway; this makes the recovery explicit
        # instead of a refusal the operator has to decode. Backfilled into the manifest below,
        # so later resumptions read an explicit value.
        cell = replace(cell, rollout_feedback=recorded_regime)
    manifest["cell"]["rollout_feedback"] = recorded_regime
    recorded_context = recorded_eval_context(manifest)
    if cell.eval_context != recorded_context:
        # Same recovery for the eval context: the run's recorded posture wins, so a pre-axis
        # run's remaining eval tasks run cold, the way the finished ones were measured.
        cell = replace(cell, eval_context=recorded_context)
    manifest["cell"]["eval_context"] = recorded_context
    # The instruction record stays consistent with the recovered axis, so the artifact keeps
    # naming the prompt its eval_after actually launches with; a pre-axis manifest recovers
    # cold and so names the blind eval instruction.
    manifest.setdefault("instruction", {})["eval_prompt_used"] = (
        "rollout_system" if recorded_context == "resumed" else "eval_system"
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
    run_id = record["run_id"]
    sandbox = CellSandbox(run_id=run_id, home=run_dir / "home", workdir=run_dir / "work")
    _migrate_recorded_containers(manifest, sandbox)
    ctx = RunContext(
        cell=cell,
        split=split,
        instruction=instruction,
        harness=harness_for(cell.harness),
        run_id=run_id,
        run_dir=run_dir,
        sandbox=sandbox,
        agent_image=agent_image,
        credentials=dict(credentials or {}),
    )
    # The credential is placed again because the sandbox is new even though the home is not;
    # credential files are excluded from every digest, so re-seeding changes no record. It is
    # placed here rather than after the manifest is rewritten because the redactor is built from
    # it, and a continuation writes durable artifacts from its very first line.
    spec = spec_for(cell.harness, cell.credential_mode)
    seed_home(spec, sandbox.home)
    _watch_cell_credential(ctx, spec)
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
) -> Path:
    """Finish an eval phase that lost tasks without a suspension, and republish.

    Ownership first, refusals second: acquiring the lock of a genuinely finished run is free
    and leaves no trace, while a live owner refuses here before anything is read or written.
    """
    if phase not in ("eval_before", "eval_after"):
        raise ValueError(f"rerun_eval repairs eval phases, not {phase!r}")
    lock_fd = _acquire_run_lock(run_dir)
    try:
        return await _rerun_eval_owned(
            run_dir,
            results_dir=results_dir,
            phase=phase,
            agent_image=agent_image,
            credentials=credentials,
            capture_egress=capture_egress,
        )
    finally:
        _release_run_lock(lock_fd)


async def _rerun_eval_owned(
    run_dir: Path,
    *,
    results_dir: Path,
    phase: str = "eval_after",
    agent_image: str = AGENT_IMAGE,
    credentials: dict[str, str] | None = None,
    capture_egress: bool = True,
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
    cell = load_cell_by_name(manifest["cell"]["name"])
    recorded_regime = recorded_rollout_feedback(manifest)
    if cell.rollout_feedback != recorded_regime:
        # Same recovery as a resume: the run's recorded arm wins over the checkout's default,
        # even though an eval re-run never constructs a rollout stream, so the manifest this
        # process republishes keeps describing the experiment that actually ran.
        cell = replace(cell, rollout_feedback=recorded_regime)
    manifest["cell"]["rollout_feedback"] = recorded_regime
    recorded_context = recorded_eval_context(manifest)
    if cell.eval_context != recorded_context:
        # And for the eval context, which a re-run acts on directly: the holes are re-run
        # under the posture the finished ids were measured under, never today's default.
        cell = replace(cell, eval_context=recorded_context)
    manifest["cell"]["eval_context"] = recorded_context
    # Same consistency rule as a resume: the record names the prompt its eval_after runs with.
    manifest.setdefault("instruction", {})["eval_prompt_used"] = (
        "rollout_system" if recorded_context == "resumed" else "eval_system"
    )
    split = load_split_by_name(cell.split)
    instruction = load_instruction(cell.instruction_arm)
    drift = experiment_drift(manifest, cell=cell, split=split, instruction=instruction)
    if drift:
        raise RuntimeError(
            "the checkout no longer matches the run being reopened: "
            + "; ".join(drift)
            + ". Restore the recorded definition, or start a fresh cell."
        )
    run_id = str(manifest["run_id"])
    sandbox = CellSandbox(run_id=run_id, home=run_dir / "home", workdir=run_dir / "work")
    _migrate_recorded_containers(manifest, sandbox)
    ctx = RunContext(
        cell=cell,
        split=split,
        instruction=instruction,
        harness=harness_for(cell.harness),
        run_id=run_id,
        run_dir=run_dir,
        sandbox=sandbox,
        agent_image=agent_image,
        credentials=dict(credentials or {}),
    )
    # Re-seeded for the same reason a resume re-seeds: the sandbox is new even though the home
    # is not, credential files are excluded from every digest, and the redactor is built from
    # what was placed.
    spec = spec_for(cell.harness, cell.credential_mode)
    seed_home(spec, sandbox.home)
    _watch_cell_credential(ctx, spec)
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


async def rebookend_run(
    source_run_dir: Path,
    *,
    runs_dir: Path,
    results_dir: Path,
    agent_image: str = AGENT_IMAGE,
    credentials: dict[str, str] | None = None,
    capture_egress: bool = True,
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
    cell = load_cell_by_name(source_manifest["cell"]["name"])
    # The recorded arm wins over the checkout's default, exactly as a resume recovers it: the
    # new measurement inherits the SOURCE's axes and changes exactly one of them, so a never-arm
    # source publishes honestly as never + resumed.
    recorded_regime = recorded_rollout_feedback(source_manifest)
    cell = replace(cell, rollout_feedback=recorded_regime, eval_context="resumed")
    split = load_split_by_name(cell.split)
    instruction = load_instruction(cell.instruction_arm)
    drift = experiment_drift(source_manifest, cell=cell, split=split, instruction=instruction)
    if drift:
        raise RuntimeError(
            "the checkout no longer matches the run being rebookended: "
            + "; ".join(drift)
            + ". Restore the recorded definition; a bookend under an edited definition would "
            "not pair with the run it claims to follow."
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
    lock_fd = _acquire_run_lock(run_dir)
    try:
        return await _rebookend_owned(
            source_run_dir,
            source_manifest,
            cell=cell,
            split=split,
            instruction=instruction,
            run_id=run_id,
            run_dir=run_dir,
            results_dir=results_dir,
            agent_image=agent_image,
            credentials=credentials,
            capture_egress=capture_egress,
        )
    finally:
        _release_run_lock(lock_fd)


async def _rebookend_owned(
    source_run_dir: Path,
    source_manifest: dict[str, Any],
    *,
    cell: Cell,
    split: Split,
    instruction: Instruction,
    run_id: str,
    run_dir: Path,
    results_dir: Path,
    agent_image: str,
    credentials: dict[str, str] | None,
    capture_egress: bool,
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
        agent_image=agent_image,
        credentials=dict(credentials or {}),
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

    sandbox.up()
    spec = spec_for(cell.harness, cell.credential_mode)
    seeded = seed_home(spec, sandbox.home)
    _watch_cell_credential(ctx, spec)
    observer = _Egress(_start_egress(sandbox, run_dir) if capture_egress else None, run_dir)
    try:
        probes = {
            "version": ctx.redactor.text(
                _probe(ctx.harness.version_probe(), image=agent_image, sandbox=sandbox, env={})
            )
        }
        seeds = _place_runner_files(ctx)
        manifest = build_manifest(ctx, probes=probes)
        manifest["credential_seed"] = seeded
        manifest["home"]["seeded"] = seeds
        manifest["axes"]["credential_mode"] = credential_effective_mode(
            spec, sandbox.home, env_names=sorted(ctx.credentials)
        )
        # What this run is a bookend OF, by run id rather than by path: a durable artifact
        # carries no operator layout, and the id is what pairs it with the source post-hoc.
        manifest["rebookend"] = {
            "rebookend_of": str(source_manifest.get("run_id", "")),
            "source_rollout_feedback": cell.rollout_feedback,
            "source_stop_reason": str(source_stopping.get("stop_reason", "")),
        }
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
    "LegRecord",
    "RunContext",
    "ROLLOUT_STOPPING_FILE",
    "SUSPENDED_EXIT_CODE",
    "SUSPENSION_FILE",
    "EvalSuspension",
    "Suspension",
    "build_manifest",
    "durable_filter",
    "cleanup",
    "read_eval_phase",
    "rebookend_run",
    "resume_cell",
    "run_cell",
    "terminal_session_in",
    "run_eval_phase",
    "run_leg",
    "run_rollout_phase",
    "write_home_files",
]
