"""The cell runner: three phases, one manifest, one results JSON.

One cell is one call to :func:`run_cell`. It stands the cell's sandbox up, records the
manifest before anything spends, runs the held-out eval, the improvement rollout, and the
held-out eval again, then writes the results and tears the sandbox down.

Two structural choices are worth stating, because both are what the scope's protocol requires
rather than conveniences:

**One fresh session per eval task, enforced by the server.** Each eval task gets its own
single-task ``EvalStream``, so a session that ignores its instruction and pulls a second task
is told the stream is done rather than quietly consuming the next task's measurement. The
serving process is this one, so the dataset loads once per phase rather than once per task.

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
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
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
    write_json,
)
from shobench.credentials import seed_home, spec_for
from shobench.harness import Harness, LaunchSpec, StopKind, StopVerdict
from shobench.harnesses import harness_for
from shobench.pins import SHOGYM_REPO, SHOGYM_REV
from shobench.results import TaskResult, read_phase, write_results
from shobench.serving import DEFAULT_PORT, SERVER_NAME, build_stream, side_for_phase
from shobench.splits import Split, load_split_by_name

# What does not count as the agent's durable self.
#
# This list decides what the headline home digest means. A digest that includes session
# transcripts and caches changes on every run whether or not the agent changed itself, so it
# answers "did a session happen", which is always yes. Excluding them makes it answer "did the
# rollout write something the next fresh session can use", which is the benchmark's question,
# because eval sessions are fresh by design and only the durable channel survives them.

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

    def leg_records(self) -> list[dict[str, Any]]:
        """Every leg of this run, the ones inherited from a suspension included."""
        return [*self.prior_legs, *(leg.to_json() for leg in self.legs)]

    @property
    def mcp_url(self) -> str:
        return f"http://{HOST_ALIAS}:{self.port}/mcp"

    @property
    def cfg_dir(self) -> Path:
        return self.run_dir / "cfg"


# ----- the manifest ------------------------------------------------------------------------


def _probe(argv: list[str], *, image: str, sandbox: CellSandbox, env: dict[str, str]) -> str:
    """Run a short command in the agent image and return its output, for the manifest."""
    args = sandbox.docker_args(env=env, mounts={})
    result = subprocess.run(
        ["docker", *args, image, *argv], capture_output=True, text=True, timeout=180
    )
    return (result.stdout + result.stderr).strip()


def build_manifest(ctx: RunContext, *, probes: dict[str, str]) -> dict[str, Any]:
    """The record of what this cell was, written before anything spends.

    It carries the substrate pin, the split digest, the instruction digests, the resolved
    harness version and model, and the agent home's digest at the start. Everything a reader
    needs to know whether two cells were the same experiment.
    """
    return {
        "schema": "shobench.manifest/1",
        "run_id": ctx.run_id,
        "started_at": time.time(),
        "cell": ctx.cell.to_manifest(),
        "split": ctx.split.to_manifest(),
        "instruction": ctx.instruction.to_manifest(),
        "substrate": {
            "shogym_repo": SHOGYM_REPO,
            "shogym_rev": SHOGYM_REV,
            "mcp_server_name": SERVER_NAME,
        },
        "harness_probes": probes,
        "container": {
            "agent_image": ctx.agent_image,
            "network": ctx.sandbox.network,
            "netns_container": ctx.sandbox.netns_container,
            "home": run_relative(ctx.sandbox.home, ctx.run_dir),
        },
        "home": {
            "digest_before": home_digest(ctx.sandbox.home, exclude=is_noise),
            "digest_after": None,
            "inventory_after": [],
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
) -> LegRecord:
    """Run one harness invocation to completion and classify how it ended."""
    if resume and not session_id:
        raise RuntimeError("cannot resume without a session id; the previous leg wrote none")
    trace_dir = ctx.run_dir / phase / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    stem = f"leg-{leg:04d}" if task_idx is None else f"task-{task_idx:05d}-leg-{leg:04d}"
    stdout_path = trace_dir / f"{stem}.stream.jsonl"
    stderr_path = trace_dir / f"{stem}.err.txt"

    spec = ctx.harness.launch(
        mcp_url=ctx.mcp_url,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=ctx.cell.model,
        trace_path=stdout_path,
        session_id=session_id,
        resume=resume,
        leg_timeout_s=timeout_s,
        effort=ctx.cell.effort,
    )
    ctx.cfg_dir.mkdir(parents=True, exist_ok=True)
    for name, body in spec.config_files.items():
        target = ctx.cfg_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    write_home_files(ctx.sandbox.home, spec)

    env = dict(spec.env)
    env.update(ctx.credentials)
    # Named so a leg the runner ends at its budget can actually be removed. Killing the docker
    # client leaves the container running, and a container that outlives its leg keeps
    # spending and keeps pulling tasks the runner has stopped watching.
    container = f"{ctx.sandbox.netns_container}-{phase[:4]}-{leg:04d}"[:63]
    args = ctx.sandbox.docker_args(env=env, mounts={ctx.cfg_dir: "/cfg:ro"}, name=container)
    argv = ["docker", *args, ctx.agent_image, *spec.argv]

    started = time.time()
    timed_out = False
    with stdout_path.open("a", encoding="utf-8") as out, stderr_path.open("a") as err:
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


async def run_eval_phase(ctx: RunContext, phase: str) -> list[TaskResult]:
    """Serve the held-out split, one fresh session per task.

    Each task gets its own single-task stream, so the one-session-per-task rule is enforced by
    the server rather than requested of the agent.
    """
    side = side_for_phase(ctx.split, phase)
    rows: list[TaskResult] = []
    leg = 0
    for task_id in side.task_ids:
        idx = int(task_id)
        prov_dir = ctx.run_dir / phase / f"task-{idx:05d}"
        prov_dir.mkdir(parents=True, exist_ok=True)
        stream = build_stream(
            ctx.cell,
            _single_task_split(ctx.split, phase, task_id),
            phase,
            prov_dir,
            deadline=float(ctx.cell.budget.eval_task_timeout_s),
        )
        # A fresh port per task, so a socket still in TIME_WAIT from the previous task cannot
        # make the next one look like a server that refused to start.
        ctx.port = free_port()
        async with stream, _served(stream, ctx.port):
            await asyncio.to_thread(
                run_leg,
                ctx,
                phase=phase,
                leg=leg,
                system_prompt=ctx.instruction.eval_system,
                user_prompt=ctx.instruction.kickoff,
                session_id=str(uuid.uuid4()),
                resume=False,
                timeout_s=ctx.cell.budget.eval_task_timeout_s,
                task_idx=idx,
                consumed_before=0,
            )
        rows.extend(read_phase(prov_dir))
        leg += 1
    return rows


def _single_task_split(split: Split, phase: str, task_id: str) -> Split:
    """A one-task view of the split, so the phase's stream can only dispense that task."""
    from dataclasses import replace

    from shobench.splits import Side

    side = side_for_phase(split, phase)
    one = Side(task_ids=(task_id,), env_kwargs=dict(side.env_kwargs))
    if phase == "rollout":
        return replace(split, pool=one)
    return replace(split, heldout=one)


SUSPENSION_FILE = "suspended.json"
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
    """Write the suspension, stop the containers, and leave without unwinding. Never returns.

    Three things have to be true at once, and only this order gets all three. The record has to
    be on disk before anything else, because it is what a continuation reads. The cell's
    containers have to stop, because a suspended cell may wait hours for a window and a running
    container is a running bill. And the process has to leave without unwinding the stream: an
    orderly close drains whatever task is in flight into a scored row, and shogym replays only
    positions with no row, so the tidy exit is the one that would cost the agent the task it was
    working on. Exiting hard leaves the claim on disk and the position row-less, which is
    precisely the state ``resume=True`` is for.

    Nothing else in the runner exits the process. It is confined here because this is the one
    place where the correct behavior and the tidy behavior are not the same.
    """
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
    # The record first, and outside the guard below. Everything after it is best-effort, but
    # this is not: a suspension nobody can read is not a suspension, and if it cannot be written
    # then failing through the normal path, which at least publishes, beats exiting into silence.
    write_json(run_dir / SUSPENSION_FILE, suspension)
    try:
        write_json(run_dir / "legs.json", ctx.leg_records())
        # The agent's container is already gone, its process being what ended, so this stops the
        # network namespace and the egress observer.
        if ctx.teardown is not None:
            ctx.teardown()
        print(
            f"[shobench] {ctx.cell.name}: suspended on a usage limit after "
            f"{tasks_dispensed} task(s); resume with: {suspension['resume_with']}",
            file=sys.stderr,
        )
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
    stream = build_stream(ctx.cell, ctx.split, "rollout", prov_dir, resume=resuming)
    queued = suspended.pool_queued if resuming else stream.queue_info().remaining
    spent_before = suspended.elapsed_rollout_s if resuming else 0.0
    dispensed_before = suspended.tasks_dispensed if resuming else 0
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
                tasks_dispensed=dispensed_before + info.consumed,
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
        # Dispensed counts the whole rollout, not this process's share of it: a resumed stream
        # numbers its own dispenses from zero while the record it continues already holds the
        # earlier ones.
        stopping["tasks_dispensed"] = dispensed_before + info.consumed
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
) -> Path:
    """Run this cell's phases, then finalize the manifest and publish the results.

    Shared by a fresh cell and a resumed one, and that sharing is the point rather than a
    convenience: a continuation has to end exactly the way an uninterrupted run ends. eval_after
    is the measurement the rollout is supposed to precede, so it belongs on the far side of a
    real rollout terminus and nowhere else. A suspension never reaches this code at all, since
    it leaves the process from inside the rollout, which is what keeps a half-finished rollout
    from publishing an ending and from spending an exhausted window on the measurement.
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
    # A continuation starts from the phases the interrupted run already recorded, not from
    # nothing. eval_before is the half of the paired measurement that ran before the
    # interruption, and a results file without it reports no requested tasks, no deltas, and
    # every after row unpaired: the benchmark's whole question, silently unanswered.
    phase_rows: dict[str, list[TaskResult]] = {
        phase: read_phase(ctx.run_dir / phase) for phase in recorded_phases
    }
    stopping: dict[str, Any] = {}
    for phase in phases:
        if phase == "rollout":
            phase_rows[phase], stopping = await run_rollout_phase(ctx, suspended=suspended)
        else:
            phase_rows[phase] = await run_eval_phase(ctx, phase)
        write_json(ctx.run_dir / "legs.json", ctx.leg_records())

    # How many times a provider limit suspended and resumed this cell, counted off the one
    # place that record lives: a resumption entry is appended per continuation. The report reads
    # this field, so a resumed cell that omitted it was published as one that never paused.
    if "rollout" in phases:
        stopping["usage_limit_resumes"] = len(manifest.get("resumptions", []))

    # Which model answered, read off the traces rather than assumed from the config, and off
    # every leg of the run rather than the ones this process happened to launch.
    manifest["observed_models"] = sorted(
        {model for leg in ctx.leg_records() for model in leg.get("observed_models", [])}
    )
    manifest["home"]["digest_after"] = home_digest(ctx.sandbox.home, exclude=is_noise)
    manifest["home"]["inventory_after"] = home_inventory(ctx.sandbox.home, exclude=is_noise)
    manifest["home"]["changed"] = (
        manifest["home"]["digest_after"] != manifest["home"]["digest_before"]
    )
    manifest["ended_at"] = time.time()
    write_json(ctx.run_dir / "manifest.json", manifest)

    egress_summary = observer.stop()
    results_path = results_dir / f"{ctx.cell.name}.json"
    write_results(
        results_path,
        manifest=manifest,
        phases=phase_rows,
        stopping=stopping,
        egress=egress_summary,
    )
    return results_path


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
    """Run one cell end to end and return the path of its results JSON."""
    instruction = load_instruction(cell.instruction_arm)
    run_id = _run_id(cell)
    run_dir = runs_dir / run_id
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
    seeded = seed_home(spec_for(cell.harness, cell.credential_mode), sandbox.home)
    observer = _Egress(_start_egress(sandbox, run_dir) if capture_egress else None, run_dir)
    try:
        probes = {
            "version": _probe(
                ctx.harness.version_probe(),
                image=agent_image,
                sandbox=sandbox,
                env=ctx.credentials,
            )
        }
        model_probe = ctx.harness.model_probe()
        if model_probe:
            probes["model"] = _probe(
                model_probe, image=agent_image, sandbox=sandbox, env=ctx.credentials
            )
        manifest = build_manifest(ctx, probes=probes)
        manifest["credential_seed"] = seeded
        write_json(run_dir / "manifest.json", manifest)
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

    Everything this needs is on disk, because the process that wrote it is gone: the suspension
    record names the session and the time already spent, and the manifest beside it names the
    cell. The run directory is reused rather than copied, so the agent continues in the home it
    built, against the provenance record it already wrote, in a stream reopened on the position
    it was holding when the window closed.

    What it must not do is start a second measurement of a different thing. The cell is the one
    the manifest recorded, checked against it rather than trusted; the clock is what remained of
    the budget that run was given, read from the record and not from today's cell file; the
    phases already measured are carried into the published result; and the ending is the shared
    one, so a continued cell publishes the same result shape as a cell that was never
    interrupted.
    """
    suspension = Suspension.read(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cell = load_cell_by_name(manifest["cell"]["name"])
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
    sandbox = CellSandbox(
        run_id=suspension.run_id, home=run_dir / "home", workdir=run_dir / "work"
    )
    ctx = RunContext(
        cell=cell,
        split=split,
        instruction=instruction,
        harness=harness_for(cell.harness),
        run_id=suspension.run_id,
        run_dir=run_dir,
        sandbox=sandbox,
        agent_image=agent_image,
        credentials=dict(credentials or {}),
    )
    legs_path = run_dir / "legs.json"
    if legs_path.is_file():
        # The legs the suspended run recorded. This process appends to that record rather than
        # replacing it, so a finished cell shows its whole rollout and not just the last stretch.
        ctx.prior_legs = json.loads(legs_path.read_text(encoding="utf-8"))
    # Written before anything can suspend again, because a manifest that lives only in memory
    # loses every resumption to the next hard exit, and a cell continued three times would
    # publish as though it had been continued once.
    manifest.setdefault("resumptions", []).append(
        {
            "suspended_at": suspension.suspended_at,
            "resumed_at": time.time(),
            "elapsed_rollout_s_before": suspension.elapsed_rollout_s,
            "session_id": suspension.session_id,
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    sandbox.up()
    # The credential is placed again because the sandbox is new even though the home is not;
    # credential files are excluded from every digest, so re-seeding changes no record.
    seed_home(spec_for(cell.harness, cell.credential_mode), sandbox.home)
    observer = _Egress(_start_egress(sandbox, run_dir) if capture_egress else None, run_dir)
    try:
        results_path = await _run_phases(
            ctx,
            manifest=manifest,
            phases=("rollout", "eval_after"),
            results_dir=results_dir,
            observer=observer,
            suspended=suspension,
            # The measurement that ran before the interruption, read back off the run's own
            # provenance so the published result carries both halves of the pair.
            recorded_phases=("eval_before",),
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
    stem = f"shobench-{run_id}"[:50]
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
    "SUSPENDED_EXIT_CODE",
    "SUSPENSION_FILE",
    "Suspension",
    "build_manifest",
    "cleanup",
    "resume_cell",
    "run_cell",
    "run_eval_phase",
    "run_leg",
    "run_rollout_phase",
    "write_home_files",
]
