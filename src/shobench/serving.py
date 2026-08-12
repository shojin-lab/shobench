"""Standing up a shogym stream for one phase of one cell.

Serving is shogym's job, so this module builds ``TaskStream`` / ``EvalStream`` from a cell and
a split and runs them behind ``build_stream_server``. It adds exactly three things shogym
leaves to the caller: which task indices belong to which side of the split, which env kwargs
each side needs, and an HTTP endpoint that outlives a single harness process.

The endpoint outliving the harness is the whole design. A rollout is a sequence of harness
invocations inside one wall clock, because a subscription can stop a session mid-run and
because codex is unreliable over one long loop. Every one of those invocations reconnects to
the same live stream, so the queue advances monotonically across them and the record has one
provenance directory per phase, not one per invocation.

Run as a module, this file is the server process:

    python -m shobench.serving --cell smoke-automationbench --phase rollout --run-dir ...
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from shobench.config import Cell, load_cell_by_name
from shobench.containers import run_relative
from shobench.splits import Side, Split, load_split_by_name

# The MCP server name the agent sees. Tool names are built from it (mcp__shogym__get_task under
# Claude Code), so it is part of the initial conditions and is identical in every cell.
SERVER_NAME = "shogym"
DEFAULT_PORT = 8973


def side_for_phase(split: Split, phase: str) -> Side:
    """Which side of the split a phase serves. Both eval phases serve the same one."""
    if phase in ("eval_before", "eval_after"):
        return split.heldout
    if phase == "rollout":
        return split.pool
    raise ValueError(f"unknown phase {phase!r}")


def task_indices(side: Side, *, ceiling: int | None = None) -> list[int]:
    """The side's ids as the integers shogym's ``TaskRef`` requires.

    ``ceiling`` truncates the improvement pool to the cell's serving maximum. Truncating is
    not a quota: the agent may stop anywhere before it, and where it stopped is the reported
    outcome.
    """
    indices = [int(task_id) for task_id in side.task_ids]
    if ceiling is not None:
        indices = indices[:ceiling]
    return indices


def env_factory(env: str, kwargs: dict[str, Any]):
    """A ``env_for`` closure carrying this side's env kwargs.

    shogym's stream calls this once per env at construction and once per dispensed task, and
    has no per-env config channel of its own, so the kwargs ride in the closure.
    """
    import shogym

    def env_for(name: str):
        if name != env:
            raise ValueError(f"stream asked for env {name!r}; this cell serves {env!r}")
        return shogym.make(name, config=dict(kwargs) or None)

    return env_for


# Envs whose upstream this process has already provisioned. shogym caches the fetch on disk and
# guards it with a lock, so a second construction is cheap; this set makes the runner's own warm
# call a no-op after the first, so a concurrent eval phase warms once rather than once per task.
_WARMED_ENVS: set[str] = set()


def warm_env(cell: Cell, *, make: Callable[..., Any] | None = None) -> None:
    """Provision the cell's env once, before an eval phase fans its tasks out concurrently.

    An env's upstream source is fetched into ``~/.cache/shogym`` on first construction and reused
    after (see the env adapters' ``ensure_source``). Constructing one env here, before the fan-out,
    means the fetch and the module import are paid once by the runner rather than raced by the
    first wave of per-task streams. It is isolation-safe by construction: this is read-only
    reference data on the serving side, identical for every task, mounted into no agent container,
    and it consumes no task from any queue. ``make`` is injectable so the warm can be tested
    without standing up a real env.
    """
    if cell.env in _WARMED_ENVS:
        return
    if make is None:
        import shogym

        make = shogym.make
    env = make(cell.env, config=(dict(cell.env_kwargs) or None))
    _WARMED_ENVS.add(cell.env)
    close = getattr(env, "close", None)
    if callable(close):
        # Warming must never fail a phase: a warm env that cannot be torn down is still warm.
        with contextlib.suppress(Exception):
            close()


def build_stream(
    cell: Cell,
    split: Split,
    phase: str,
    prov_dir: Path,
    *,
    resume: bool = False,
    deadline: float | None = None,
):
    """Build the phase's stream.

    Both eval phases use ``EvalStream``, which pins the feedback regime to ``never`` and
    refuses to resume a directory recorded under any other regime: held-out measurement is
    always blind. The rollout's regime is the cell's ``rollout_feedback`` axis. "immediate"
    is the study's premise (improvement grounded in feedback: the sealed task's own outcome
    comes back on the ``done`` that ended it); "never" withholds it and makes the rollout a
    feedback-ablation arm. The regime is recorded per row by the stream itself, so the two
    arms cannot be conflated after the fact.
    """
    from shogym.serve import EvalStream, Immediate, Never, TaskRef, TaskStream

    side = side_for_phase(split, phase)
    kwargs = dict(side.env_kwargs)
    kwargs.update(cell.env_kwargs)
    ceiling = cell.budget.pool_ceiling if phase == "rollout" else None
    refs = [TaskRef(cell.env, idx) for idx in task_indices(side, ceiling=ceiling)]

    if phase == "rollout":
        # The rollout is the only phase that serves the cell's concurrency: above 1, the agent
        # may hold several tasks at once. The eval phase stays one-session-per-task (its
        # concurrency is host-level fan-out, not stream-level), so it never reads this.
        return TaskStream(
            env_factory(cell.env, kwargs),
            refs,
            prov_dir=prov_dir,
            resume=resume,
            deadline=deadline,
            feedback=Immediate() if cell.rollout_feedback == "immediate" else Never(),
            max_in_flight=cell.max_in_flight,
        )
    return EvalStream(
        env_factory(cell.env, kwargs),
        refs,
        prov_dir=prov_dir,
        resume=resume,
        deadline=deadline,
    )


async def serve(
    cell: Cell,
    split: Split,
    phase: str,
    prov_dir: Path,
    *,
    host: str,
    port: int,
    resume: bool = False,
    deadline: float | None = None,
    ready_path: Path | None = None,
) -> None:
    """Run the phase's stream over HTTP until the process is stopped.

    ``ready_path`` is written once the stream exists and before the transport starts, which is
    what the runner waits on rather than sleeping: a harness that connects before the queue is
    built would be told the stream is done.
    """
    from shogym.serve import build_stream_server

    stream = build_stream(cell, split, phase, prov_dir, resume=resume, deadline=deadline)
    print(
        f"[shobench] {cell.name} {phase}: {stream.queue_info().remaining} tasks -> {prov_dir}",
        file=sys.stderr,
        flush=True,
    )
    async with stream:
        if ready_path is not None:
            ready_path.write_text(
                json.dumps(
                    {
                        "cell": cell.name,
                        "phase": phase,
                        # Relative to the run directory (this dir's parent), so the readiness
                        # file stays resolvable through the caller's --run-dir without writing
                        # the operator's absolute path.
                        "prov_dir": run_relative(prov_dir, prov_dir.parent),
                        "url": f"http://{host}:{port}/mcp",
                        "remaining": stream.queue_info().remaining,
                        "pid": os.getpid(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        server = build_stream_server(stream, name=SERVER_NAME)
        await server.run_async(transport="http", host=host, port=port)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve one phase of one cell over HTTP MCP.")
    parser.add_argument("--cell", required=True)
    parser.add_argument("--phase", required=True, choices=["eval_before", "rollout", "eval_after"])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 (container-local network)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deadline", type=float, default=None)
    parser.add_argument("--ready-file", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    cell = load_cell_by_name(args.cell)
    split = load_split_by_name(cell.split)
    prov_dir = args.run_dir / args.phase
    prov_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(
        serve(
            cell,
            split,
            args.phase,
            prov_dir,
            host=args.host,
            port=args.port,
            resume=args.resume,
            deadline=args.deadline,
            ready_path=args.ready_file,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PORT",
    "SERVER_NAME",
    "build_stream",
    "env_factory",
    "serve",
    "side_for_phase",
    "task_indices",
    "warm_env",
]
