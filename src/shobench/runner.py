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
the charter asks about, and the runner does not prompt the agent onward. A run stopped by a
provider usage limit is recorded resumable, for a human to relaunch when the window resets.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shobench import egress
from shobench.config import Cell, Instruction, load_instruction
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
from shobench.splits import Split

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

    def to_json(self) -> dict[str, Any]:
        return {
            "leg": self.leg,
            "observed_models": list(self.observed_models),
            "phase": self.phase,
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


async def run_rollout_phase(ctx: RunContext) -> tuple[list[TaskResult], dict[str, Any]]:
    """Serve the improvement pool as one honest run of the harness. No automatic restart.

    Whether a harness sustains autonomous operation is one of the things this measures, so the
    runner does not relaunch a harness that stops. A single invocation is driven against the
    pool; where it ends is the outcome:

    - it exhausts the pool: sustained autonomy, the success case;
    - it ends its run with tasks still available: it stopped on its own, and that is the
      finding, recorded with how far it got;
    - it wedges past the run's time cap: timed out, a different finding;
    - it hits a provider usage limit: the environment stopped it, not the agent, so the run
      ends but is marked resumable. A human relaunches it when the window resets (windows can
      run to hours, and nothing here waits or auto-retries).

    Restarting the harness to push it further would launder "gave up after N tasks" into "ran
    the whole pool", which is the one thing this must not do.
    """
    prov_dir = ctx.run_dir / "rollout"
    prov_dir.mkdir(parents=True, exist_ok=True)
    budget = ctx.cell.budget

    # Claude Code accepts a pinned id, so a run that is later resumed by hand has one to name.
    session_id = str(uuid.uuid4()) if ctx.harness.pins_session_id else None
    stopping: dict[str, Any] = {
        "stop_reason": "unrecorded",
        "legs": [],
        "session_id": session_id,
    }

    stream = build_stream(ctx.cell, ctx.split, "rollout", prov_dir)
    queued = stream.queue_info().remaining
    ctx.port = free_port()
    async with stream, _served(stream, ctx.port):
        consumed_before = stream.queue_info().consumed
        record = await asyncio.to_thread(
            run_leg,
            ctx,
            phase="rollout",
            leg=0,
            system_prompt=ctx.instruction.rollout_system,
            user_prompt=ctx.instruction.kickoff,
            session_id=session_id,
            resume=False,
            timeout_s=budget.rollout_wall_clock_s,
            task_idx=None,
            consumed_before=consumed_before,
        )
        record.tasks_consumed_after = stream.queue_info().consumed
        stopping["legs"].append(record.to_json())
        observed_session = ctx.harness.session_id_from_trace(Path(record.trace_path))
        if observed_session:
            stopping["session_id"] = observed_session

        info = stream.queue_info()
        if record.verdict.kind is StopKind.USAGE_LIMIT:
            # The provider stopped it, not the agent. Resumable by hand when the window clears.
            stopping["stop_reason"] = "usage_limit"
            stopping["resumable"] = True
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
        stopping["tasks_dispensed"] = info.consumed
        stopping["tasks_remaining_in_pool"] = info.remaining
        stopping["pool_queued"] = queued
        # The charter's question is whether it stopped with work still available.
        stopping["stopped_with_tasks_available"] = (
            stopping["stop_reason"] == "agent_stopped_early" and info.remaining > 0
        )

    return read_phase(prov_dir), stopping


# ----- the cell ----------------------------------------------------------------------------


def _run_id(cell: Cell) -> str:
    return f"{cell.name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


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
    capture = None
    try:
        if capture_egress:
            capture = egress.start(
                netns_container=sandbox.netns_container,
                name=f"{sandbox.netns_container}-egress"[:63],
                log_path=run_dir / "egress.tsv",
            )

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

        phase_rows: dict[str, list[TaskResult]] = {}
        stopping: dict[str, Any] = {}
        for phase in phases:
            if phase == "rollout":
                phase_rows[phase], stopping = await run_rollout_phase(ctx)
            else:
                phase_rows[phase] = await run_eval_phase(ctx, phase)
            write_json(run_dir / "legs.json", [leg.to_json() for leg in ctx.legs])

        # Which model answered, read off the traces rather than assumed from the config.
        manifest["observed_models"] = sorted(
            {model for leg in ctx.legs for model in leg.observed_models}
        )
        manifest["home"]["digest_after"] = home_digest(sandbox.home, exclude=is_noise)
        manifest["home"]["inventory_after"] = home_inventory(sandbox.home, exclude=is_noise)
        manifest["home"]["changed"] = (
            manifest["home"]["digest_after"] != manifest["home"]["digest_before"]
        )
        manifest["ended_at"] = time.time()
        write_json(run_dir / "manifest.json", manifest)

        egress_summary: dict[str, Any] = {}
        if capture is not None:
            egress.stop(capture)
            capture = None
            egress_summary = dict(
                egress.write_summary(run_dir / "egress.tsv", run_dir / "egress.json")
            )

        results_path = results_dir / f"{cell.name}.json"
        write_results(
            results_path,
            manifest=manifest,
            phases=phase_rows,
            stopping=stopping,
            egress=egress_summary,
        )
        return results_path
    finally:
        if capture is not None:
            with contextlib.suppress(Exception):
                egress.stop(capture)
        sandbox.down()


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
    "build_manifest",
    "cleanup",
    "run_cell",
    "run_eval_phase",
    "run_leg",
    "run_rollout_phase",
]
