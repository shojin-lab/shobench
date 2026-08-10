"""The shobench command line.

    shobench cells                          # the matrix, as configured
    shobench doctor                         # what is installed, what is missing
    shobench creds --cell <name>            # the negative-control protocol for one cell
    shobench build                          # build the three images
    shobench run --cell <name> --go         # run one cell (real spend without --go: a plan)
    shobench resume --run <run-dir> --go    # continue a cell a usage limit suspended
    shobench report [results/]              # the summary table

``--go`` is the whole safety story: every command that spends prints its plan and exits unless
it is present. Nothing here launches the matrix; a cell is run one at a time by name.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from shobench import credentials, report, runner, tau2_data
from shobench.config import load_all_cells, load_cell_by_name, load_instruction, repo_root
from shobench.containers import AGENT_IMAGE, NETNS_IMAGE, CellSandbox, build_image, daemon_available
from shobench.egress import EGRESS_IMAGE
from shobench.pins import SHOGYM_REV
from shobench.runner import SUSPENSION_FILE, resume_cell, run_cell
from shobench.serving import DEFAULT_PORT
from shobench.splits import load_split_by_name

DOCKER_DIR = "docker"


def _cmd_cells(args: argparse.Namespace) -> int:
    rows = []
    for cell in load_all_cells():
        split = load_split_by_name(cell.split)
        rows.append(
            (
                cell.name,
                cell.env,
                cell.harness,
                cell.model,
                cell.credential_mode,
                str(len(split.heldout)),
                str(len(split.pool)),
                f"{cell.budget.rollout_wall_clock_s // 3600}h",
            )
        )
    header = ("cell", "env", "harness", "model", "creds", "heldout", "pool", "budget")
    widths = [max(len(c) for c in col) for col in zip(header, *rows, strict=True)]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    import shutil

    out: dict[str, object] = {
        "shogym_rev_pinned": SHOGYM_REV,
        "docker_available": daemon_available(),
        "docker_cli": bool(shutil.which("docker")),
        "credentials": credentials.inventory(dict(os.environ)),
        "pending_legs": {"/".join(k): v for k, v in credentials.PENDING_LEGS.items()},
    }
    try:
        import shogym

        out["shogym_importable"] = True
        out["shogym_envs"] = sorted(shogym.registered_envs())
    except Exception as exc:  # noqa: BLE001 - the message is the diagnosis
        out["shogym_importable"] = False
        out["shogym_error"] = str(exc)
    for arm in sorted((repo_root() / "instructions").glob("*")):
        if arm.is_dir():
            instruction = load_instruction(arm.name)
            out.setdefault("instruction_arms", {})  # type: ignore[union-attr]
            out["instruction_arms"][arm.name] = {  # type: ignore[index]
                "rollout_system_sha256": instruction.rollout_system_sha256,
                "eval_system_sha256": instruction.eval_system_sha256,
            }
    print(json.dumps(out, indent=2))
    return 0 if out["docker_available"] and out["shogym_importable"] else 1


def _cmd_build(args: argparse.Namespace) -> int:
    root = repo_root()
    for dockerfile, tag in (
        ("agent.Dockerfile", AGENT_IMAGE),
        ("netns.Dockerfile", NETNS_IMAGE),
        ("egress.Dockerfile", EGRESS_IMAGE),
    ):
        build_image(root / DOCKER_DIR / dockerfile, root, tag)
        print(f"built {tag}")
    return 0


def _cmd_creds(args: argparse.Namespace) -> int:
    """Run the negative-control protocol for one cell, without running the cell."""
    cell = load_cell_by_name(args.cell)
    sandbox = CellSandbox(
        run_id=f"creds-{cell.name}",
        home=Path(args.work) / f"creds-{cell.name}" / "home",
        workdir=Path(args.work) / f"creds-{cell.name}" / "work",
    )
    sandbox.up()
    try:
        verdict = credentials.validate_isolation(
            harness=cell.harness,
            mode=cell.credential_mode,
            model=cell.model,
            docker_args=sandbox.docker_args(env={}, mounts={}),
            image=args.image,
            environ=dict(os.environ),
            home=sandbox.home,
        )
    finally:
        sandbox.down()
    print(json.dumps(verdict.to_json(), indent=2))
    return 0 if verdict.trusted else 1


def _cmd_run(args: argparse.Namespace) -> int:
    cell = load_cell_by_name(args.cell)
    split = load_split_by_name(cell.split)
    instruction = load_instruction(cell.instruction_arm)
    phases = (
        tuple(args.phases.split(","))
        if args.phases
        else (
            "eval_before",
            "rollout",
            "eval_after",
        )
    )
    plan = {
        "cell": cell.to_manifest(),
        "split": split.to_manifest(),
        "instruction_sha256": instruction.rollout_system_sha256,
        "phases": list(phases),
        "eval_tasks_per_phase": len(split.heldout),
        "rollout_pool_ceiling": min(len(split.pool), cell.budget.pool_ceiling or len(split.pool)),
        "rollout_wall_clock_hours": cell.budget.rollout_wall_clock_s / 3600,
        "agent_image": args.image,
        "credentials_present": credentials.inventory(dict(os.environ)),
        "required_env_present": {name: bool(os.environ.get(name)) for name in cell.required_env},
    }
    missing_required = [n for n in cell.required_env if not os.environ.get(n)]
    plan["missing_required_env"] = missing_required
    # tau2 data is provisioned, not operator-set: the runner points TAU2_DATA_DIR at the cache
    # itself, so it is reported here rather than listed among required_env.
    if tau2_data.needs_tau2_data(cell.env):
        plan["tau2_data"] = {
            "needed": True,
            "upstream_sha": tau2_data.UPSTREAM_SHA,
            "data_dir": str(tau2_data.resolve_data_dir()),
            "present": tau2_data.is_present(),
            "provision_command": tau2_data.PROVISION_COMMAND,
        }
    if not args.go:
        print(json.dumps(plan, indent=2))
        print(
            "\n[plan only] this cell spends real budget. Re-run with --go to actually run it.",
            file=sys.stderr,
        )
        return 0

    if missing_required:
        # These are serving-side needs: a judge key, a user simulator key. The cell would start,
        # spend, and fail partway, so it does not start.
        print(json.dumps(plan, indent=2), file=sys.stderr)
        print(
            f"\nBLOCKED: the cell needs {missing_required} in the environment and they are "
            "not set. Nothing was spent.",
            file=sys.stderr,
        )
        return 1

    if tau2_data.needs_tau2_data(cell.env):
        # The dataset checkout is a serving-side need too, but a provisioned one. Resolve it, set
        # TAU2_DATA_DIR for the in-process serving stream, and fail loudly with the provisioning
        # command if it is absent rather than starting a cell that would fail partway.
        try:
            os.environ["TAU2_DATA_DIR"] = str(tau2_data.require())
        except tau2_data.Tau2DataError as exc:
            print(json.dumps(plan, indent=2), file=sys.stderr)
            print(f"\nBLOCKED: {exc}\nNothing was spent.", file=sys.stderr)
            return 1

    if not args.skip_credential_check:
        sandbox = CellSandbox(
            run_id=f"precheck-{cell.name}",
            home=Path(args.runs) / f"precheck-{cell.name}" / "home",
            workdir=Path(args.runs) / f"precheck-{cell.name}" / "work",
        )
        sandbox.up()
        try:
            verdict = credentials.validate_isolation(
                harness=cell.harness,
                mode=cell.credential_mode,
                model=cell.model,
                docker_args=sandbox.docker_args(env={}, mounts={}),
                image=args.image,
                environ=dict(os.environ),
                home=sandbox.home,
            )
        finally:
            sandbox.down()
        if not verdict.trusted:
            print(json.dumps(verdict.to_json(), indent=2), file=sys.stderr)
            print(
                f"\nBLOCKED: {verdict.reason}. The cell did not start.",
                file=sys.stderr,
            )
            return 1

    results_path = asyncio.run(
        run_cell(
            cell,
            split,
            runs_dir=Path(args.runs),
            results_dir=Path(args.results),
            port=args.port,
            agent_image=args.image,
            credentials=credentials.agent_env(cell.harness, cell.credential_mode, dict(os.environ)),
            phases=phases,
            capture_egress=not args.no_egress,
        )
    )
    print(f"results: {results_path}")
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    """Continue a cell a provider usage limit suspended, and let it finish.

    The suspension record is the plan: which cell is waiting, how far the rollout got, and how
    much of its wall clock is left. Printing that and stopping is what happens without ``--go``,
    exactly as for a fresh run, because a continuation spends the same way a run does.
    """
    run_dir = Path(args.run)
    record_path = run_dir / SUSPENSION_FILE
    if not record_path.is_file():
        print(
            f"no suspension record at {record_path}. A cell that was not suspended is either "
            "still running or already finished, and there is nothing to continue.",
            file=sys.stderr,
        )
        return 1
    record = json.loads(record_path.read_text(encoding="utf-8"))
    cell = load_cell_by_name(record["cell"])
    manifest_path = run_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    drift = runner.experiment_drift(
        manifest,
        cell=cell,
        split=load_split_by_name(cell.split),
        instruction=load_instruction(cell.instruction_arm),
    )
    missing_required = [n for n in cell.required_env if not os.environ.get(n)]
    # Where the limit fell decides what a continuation needs to know and what it must guard. A
    # rollout suspension is one session on one clock, so its plan is a clock and a dispense count
    # and its one added guard is the spent-clock refusal. An eval suspension is a fan-out of
    # fresh per-task sessions, so it has no clock to be spent and no session to reattach to: its
    # plan is which held-out ids are done and which remain, and it skips the clock guard entirely.
    interrupted_phase = record.get("phase", "rollout")
    plan = {
        "run_dir": str(run_dir),
        "cell": record["cell"],
        "harness": record["harness"],
        "interrupted_phase": interrupted_phase,
        "suspended_at": record["suspended_at"],
        "stop_evidence": record["stop_evidence"],
        "credentials_present": credentials.inventory(dict(os.environ)),
        "required_env_present": {name: bool(os.environ.get(name)) for name in cell.required_env},
        "missing_required_env": missing_required,
        "experiment_drift": drift,
    }
    if interrupted_phase == "rollout":
        plan.update(
            {
                "session_id": record["session_id"],
                "tasks_dispensed_so_far": record["tasks_dispensed"],
                "pool_queued": record["pool_queued"],
                "elapsed_rollout_s": record["elapsed_rollout_s"],
                "remaining_rollout_s": record["remaining_rollout_s"],
                "phases_left": ["rollout", "eval_after"],
            }
        )
    else:
        pending = record.get("pending_task_ids", [])
        plan.update(
            {
                "completed_task_ids": record.get("completed_task_ids", []),
                "pending_task_ids": pending,
                "held_out_tasks_left": len(pending),
                # An eval_before limit fell before the rollout, so all three phases remain; an
                # eval_after limit is the last phase, so only it does.
                "phases_left": (
                    [interrupted_phase, "rollout", "eval_after"]
                    if interrupted_phase == "eval_before"
                    else [interrupted_phase]
                ),
            }
        )
    if not args.go:
        print(json.dumps(plan, indent=2))
        print(
            "\n[plan only] continuing this cell spends real budget. Re-run with --go.",
            file=sys.stderr,
        )
        return 0

    # Everything below refuses before anything spends, and before the suspension record is
    # touched. A continuation is normally started in a new shell hours later, which is exactly
    # where a serving-side variable goes missing, and a failure past this point used to consume
    # the record that made another attempt possible.
    if interrupted_phase == "rollout" and record["remaining_rollout_s"] <= 0:
        # The clock the rollout was given is gone. Continuing would hand it a second budget,
        # which is the one thing a continuation must not do, so this stops and says so rather
        # than running a rollout nobody budgeted. Only a rollout suspension has this clock; an
        # eval phase is bounded per task, not by the rollout wall clock, so it is never refused
        # here.
        print(json.dumps(plan, indent=2), file=sys.stderr)
        print(
            "\nBLOCKED: the rollout wall clock is already spent, so there is nothing to "
            "continue into. Nothing was spent.",
            file=sys.stderr,
        )
        return 1

    if missing_required:
        # The same serving-side needs a fresh run checks. They are absent far more often here,
        # since a continuation is typically a new shell, and the stream would fail to build
        # after the containers were already up.
        print(json.dumps(plan, indent=2), file=sys.stderr)
        print(
            f"\nBLOCKED: the cell needs {missing_required} in the environment and they are "
            "not set. Nothing was spent, and the run is still resumable.",
            file=sys.stderr,
        )
        return 1

    if drift:
        # The cell, split, or instruction moved while this run waited. Continuing under an
        # edited definition would publish one run id describing two experiments.
        print(json.dumps(plan, indent=2), file=sys.stderr)
        print(
            "\nBLOCKED: " + "; ".join(drift) + ". Restore the recorded definition, or start a "
            "fresh cell. Nothing was spent, and the run is still resumable.",
            file=sys.stderr,
        )
        return 1

    results_path = asyncio.run(
        resume_cell(
            run_dir,
            results_dir=Path(args.results),
            agent_image=args.image,
            credentials=credentials.agent_env(cell.harness, cell.credential_mode, dict(os.environ)),
            capture_egress=not args.no_egress,
        )
    )
    print(f"results: {results_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shobench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("cells", help="the matrix, as configured").set_defaults(func=_cmd_cells)
    sub.add_parser("doctor", help="what is installed and what is missing").set_defaults(
        func=_cmd_doctor
    )
    sub.add_parser("build", help="build the agent, holder, and observer images").set_defaults(
        func=_cmd_build
    )

    creds = sub.add_parser("creds", help="run one cell's negative-control protocol")
    creds.add_argument("--cell", required=True)
    creds.add_argument("--image", default=AGENT_IMAGE)
    creds.add_argument("--work", default="runs")
    creds.set_defaults(func=_cmd_creds)

    run = sub.add_parser("run", help="run one cell")
    run.add_argument("--cell", required=True)
    run.add_argument("--go", action="store_true", help="actually run it (real spend)")
    run.add_argument("--runs", default="runs")
    run.add_argument("--results", default="results")
    run.add_argument("--image", default=AGENT_IMAGE)
    run.add_argument("--port", type=int, default=DEFAULT_PORT)
    run.add_argument("--phases", default="", help="comma-separated subset, for debugging")
    run.add_argument("--no-egress", action="store_true")
    run.add_argument(
        "--skip-credential-check",
        action="store_true",
        help="skip the negative control; never appropriate for a reported cell",
    )
    run.set_defaults(func=_cmd_run)

    res = sub.add_parser("resume", help="continue a cell a usage limit suspended")
    res.add_argument("--run", required=True, help="the run directory holding suspended.json")
    res.add_argument("--go", action="store_true", help="actually continue it (real spend)")
    res.add_argument("--results", default="results")
    res.add_argument("--image", default=AGENT_IMAGE)
    res.add_argument("--no-egress", action="store_true")
    res.set_defaults(func=_cmd_resume)

    rep = sub.add_parser("report", help="the summary table")
    rep.add_argument("results", nargs="?", default="results")
    rep.add_argument("--format", choices=["table", "json"], default="table")
    rep.set_defaults(func=lambda a: report.main([str(a.results), "--format", a.format]))

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
