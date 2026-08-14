"""The shobench command line.

    shobench cells                          # the matrix, as configured
    shobench doctor                         # what is installed, what is missing
    shobench creds --cell <name>            # the negative-control protocol for one cell
    shobench build                          # build the three images
    shobench run --cell <name> --go         # run one cell (real spend without --go: a plan)
    shobench stop --run <run-dir>           # end a live run through its normal ending
    shobench resume --run <run-dir> --go    # continue a cell a usage limit suspended
    shobench rerun-eval --run <run-dir> --go # finish an eval_after that lost tasks
    shobench rebookend --run <run-dir> --go # a resumed eval_after for an existing run, as a new run
    shobench report [results/]              # the summary table

``--go`` is the whole safety story: every command that spends prints its plan and exits unless
it is present. Nothing here launches the matrix; a cell is run one at a time by name. ``stop``
carries no ``--go``: it can only reduce what a run spends.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from shobench import credentials, report, runner, tau2_data
from shobench.config import load_all_cells, load_cell_by_name, load_instruction, repo_root
from shobench.containers import AGENT_IMAGE, NETNS_IMAGE, CellSandbox, build_image, daemon_available
from shobench.egress import EGRESS_IMAGE
from shobench.harnesses import harness_for
from shobench.pins import SHOGYM_REV
from shobench.runner import SUSPENSION_FILE, resume_cell, run_cell
from shobench.serving import DEFAULT_PORT
from shobench.splits import load_split_by_name

DOCKER_DIR = "docker"

# Seconds `stop` waits for the run's owner to acknowledge its ask by consuming the request, and
# how often it looks. Expiring leaves the request in place for the owner to read.
STOP_ACK_TIMEOUT_S = 60.0
STOP_ACK_POLL_S = 0.2


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


def _credential_status(spec: credentials.CredentialSpec) -> dict[str, object]:
    """Whether one file-based credential is usable on this host, named but never valued."""
    available, why_not = credentials.credential_available(spec)
    return {
        "path": spec.seed_from,
        "available": available,
        "blocked_by": why_not,
        "hint": "" if available else spec.pending_hint,
    }


def _cmd_doctor(args: argparse.Namespace) -> int:
    import shutil

    out: dict[str, object] = {
        "shogym_rev_pinned": SHOGYM_REV,
        "docker_available": daemon_available(),
        "docker_cli": bool(shutil.which("docker")),
        "credentials": credentials.inventory(dict(os.environ)),
        # Whether a file-based credential is on this host is read from the host every time this
        # runs, rather than from a note. A mode that needed a login last week does not need one
        # after the login, and a doctor that says otherwise is how a cell stays unrunnable.
        "file_credentials": {
            f"{spec.harness}/{spec.mode}": _credential_status(spec)
            for spec in credentials.SPECS.values()
            if spec.seed_from
        },
        "open_questions": {"/".join(k): v for k, v in credentials.OPEN_QUESTIONS.items()},
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


def _tau2_plan(cell) -> dict[str, object] | None:
    """What a plan says about the provisioned tau2 data, or nothing for an env that reads none.

    tau2 data is provisioned, not operator-set: the runner points TAU2_DATA_DIR at the cache
    itself, so it is reported in the plan rather than listed among required_env.
    """
    if not tau2_data.needs_tau2_data(cell.env):
        return None
    return {
        "needed": True,
        "upstream_sha": tau2_data.UPSTREAM_SHA,
        "data_dir": str(tau2_data.resolve_data_dir()),
        "present": tau2_data.is_present(),
        "provision_command": tau2_data.PROVISION_COMMAND,
    }


def _set_tau2_data_dir(cell) -> str | None:
    """Point TAU2_DATA_DIR at the pinned cache for a tau2 cell. Returns why it cannot run, or None.

    Every command that starts a tau2 cell has to do this, and a continuation needs it more than a
    fresh run does rather than less: the assignment is process-local, and a resume is a new process
    started in a new shell hours later, so it inherits nothing from the run it continues. Without
    it the env falls back to shogym's *source* cache, which carries no data at all, and the cell
    fails partway with an upstream FileNotFoundError that names none of this. Shared between the
    two commands so the gate cannot hold on one path and be missing from the other.
    """
    if not tau2_data.needs_tau2_data(cell.env):
        return None
    try:
        os.environ["TAU2_DATA_DIR"] = str(tau2_data.require())
    except tau2_data.Tau2DataError as exc:
        return str(exc)
    return None


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
    tau2_plan = _tau2_plan(cell)
    if tau2_plan is not None:
        plan["tau2_data"] = tau2_plan
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

    # The dataset checkout is a serving-side need too, but a provisioned one. Resolve it, set
    # TAU2_DATA_DIR for the in-process serving stream, and fail loudly with the provisioning
    # command if it is absent rather than starting a cell that would fail partway.
    blocked = _set_tau2_data_dir(cell)
    if blocked:
        print(json.dumps(plan, indent=2), file=sys.stderr)
        print(f"\nBLOCKED: {blocked}\nNothing was spent.", file=sys.stderr)
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


def _cmd_rerun_eval(args: argparse.Namespace) -> int:
    """Finish an eval_after that lost tasks to infrastructure, with no suspension to resume.

    The plan without ``--go`` names the run, the ids still row-less, and the drift check, and
    spends nothing. The eval phase runner re-runs only the pending ids, so the plan's count is
    exactly what a ``--go`` will pay for.

    With ``--refresh-baseline`` the plan also names what catching the bookend's carried before
    rows up to its baseline would add, upgrade, and refuse, computed by the same function the
    spending path acts on, so a plan cannot say something a ``--go`` will not.
    """
    run_dir = Path(args.run)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"no manifest at {manifest_path}; this is not a run directory.", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cell = load_cell_by_name(manifest["cell"]["name"])
    split = load_split_by_name(cell.split)
    drift = runner.experiment_drift(
        manifest,
        cell=cell,
        split=split,
        instruction=load_instruction(cell.instruction_arm),
    )
    heldout_ids = [str(task_id) for task_id in split.heldout.task_ids]
    pending = runner._eval_pending_ids(run_dir / args.phase, heldout_ids)
    missing_required = [name for name in cell.required_env if not os.environ.get(name)]
    plan = {
        "run_dir": str(run_dir),
        "cell": cell.name,
        "phase": args.phase,
        "heldout_ids": len(heldout_ids),
        "already_complete": len(heldout_ids) - len(pending),
        "pending": len(pending),
        "suspension_present": (run_dir / runner.SUSPENSION_FILE).is_file(),
        "rollout_terminus_present": (run_dir / "rollout_stopping.json").is_file(),
        "rollout_ever_ran": (run_dir / "rollout").is_dir(),
        "required_env_present": {name: bool(os.environ.get(name)) for name in cell.required_env},
        "missing_required_env": missing_required,
        "experiment_drift": drift,
    }
    refresh: dict[str, object] = {}
    if args.refresh_baseline:
        try:
            refresh, _ = runner.baseline_refresh_plan(
                run_dir,
                manifest,
                split=split,
                baseline_run_dir=Path(args.baseline) if args.baseline else None,
            )
        except RuntimeError as exc:
            refresh = {"blocked": str(exc)}
        plan["baseline_refresh"] = refresh
    if not args.go:
        print(json.dumps(plan, indent=2))
        print("\nDry plan. Re-run with --go to spend.", file=sys.stderr)
        return 0
    if missing_required:
        print(json.dumps(plan, indent=2), file=sys.stderr)
        print(
            f"\nBLOCKED: the cell needs {missing_required} in the environment and they are "
            "not set. Nothing was spent.",
            file=sys.stderr,
        )
        return 1
    if refresh.get("blocked") or refresh.get("refused"):
        print(json.dumps(plan, indent=2), file=sys.stderr)
        why = refresh.get("blocked") or (
            "the baseline no longer holds the rows this bookend carries for held-out "
            f"{refresh['refused']}."
        )
        print(f"\nBLOCKED: {why} Nothing was spent.", file=sys.stderr)
        return 1
    # A zero-pending invocation still runs the tail: the fan-out may have finished while the
    # prior process died before republishing, and the phase runner is a no-op over complete
    # ids, so the only work left is the accounting and the artifact.
    blocked = _set_tau2_data_dir(cell)
    if blocked:
        print(f"BLOCKED: {blocked}\nNothing was spent.", file=sys.stderr)
        return 1
    results_path = asyncio.run(
        runner.rerun_eval(
            run_dir,
            results_dir=Path(args.results),
            phase=args.phase,
            agent_image=args.image,
            credentials=credentials.agent_env(cell.harness, cell.credential_mode, dict(os.environ)),
            capture_egress=not args.no_egress,
            refresh_baseline=args.refresh_baseline,
            baseline_run_dir=Path(args.baseline) if args.baseline else None,
        )
    )
    print(f"results: {results_path}")
    return 0


def _cmd_rebookend(args: argparse.Namespace) -> int:
    """Give an existing run a resumed eval_after, as a new run; the source stays untouched.

    The plan without ``--go`` names the source, the axes the bookend inherits, the terminal
    session it will fork, how many held-out tasks a ``--go`` will pay for (all of them: a
    rebookend is a fresh bookend, not a repair), the size of the home it will copy, and every
    refusal state, and it spends nothing.
    """
    source_dir = Path(args.run)
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"no manifest at {manifest_path}; this is not a run directory.", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_is_bookend = "rebookend" in manifest
    bookend_original = (
        manifest.get("rebookend", {}).get("rebookend_of") if source_is_bookend else None
    )
    cell = load_cell_by_name(manifest["cell"]["name"])
    recorded_arm = runner.recorded_rollout_feedback(manifest)
    split = load_split_by_name(cell.split)
    # Built by the runner's own recovery rather than by a second copy of the rule, so the plan's
    # verdict is the runner's verdict: the bookend inherits the source's recorded arm and its
    # recorded eval runtime, and pins the eval context to resumed.
    will_run = runner.bookend_cell(cell, manifest)
    drift = runner.experiment_drift(
        manifest,
        cell=will_run,
        split=split,
        instruction=load_instruction(cell.instruction_arm),
        scope=runner.DRIFT_BOOKEND,
    )
    # What the checkout's cell file says where the record says otherwise, so the operator sees
    # what the bookend is declining to read from it. Nothing in here is silently in force: the
    # eval runtime comes from the record, and everything else refuses.
    cell_drift = runner.cell_field_drift(manifest.get("cell", {}), cell.to_manifest())
    terminal_session = runner.terminal_session_in(source_dir)
    # The id alone proves only that the stopping record names something; the plan promises the
    # session it will fork, so the transcript is resolved in the SOURCE home with the same
    # per-harness validation the runner preflight applies to the copy.
    terminal_transcript = (
        harness_for(cell.harness).session_transcript(source_dir / "home", terminal_session)
        if terminal_session is not None
        else None
    )
    source_stopping_path = source_dir / runner.ROLLOUT_STOPPING_FILE
    source_stopping = (
        json.loads(source_stopping_path.read_text(encoding="utf-8"))
        if source_stopping_path.is_file()
        else {}
    )
    home_files = [p for p in (source_dir / "home").rglob("*") if p.is_file()]
    missing_required = [name for name in cell.required_env if not os.environ.get(name)]
    source_resolved = source_dir.resolve()
    outputs_inside_source = [
        str(out)
        for out in (Path(args.runs), Path(args.results))
        if out.resolve() == source_resolved or out.resolve().is_relative_to(source_resolved)
    ]
    # The bookend publishes under its OWN run id, never the cell name: the cell-name artifact
    # is the source's measurement, the one the bookend pairs with, so the two coexist and no
    # deterministic leaf exists for anyone to pre-occupy; the runner still bounds the minted
    # concrete names before its lock. What the plan can state is the scheme and the lock
    # states.
    # The baseline identity the runner will validate: the source's own before-side when it
    # has one, an explicitly named run otherwise, with the same checks the runner applies.
    source_has_before = runner._has_eval_before(source_dir)
    baseline_dir = Path(args.baseline).resolve() if args.baseline else None
    baseline_states: dict[str, object] = {
        "source_has_own_eval_before": source_has_before,
        "baseline_required": not source_has_before and baseline_dir is None,
    }
    baseline_run_id = manifest.get("run_id") if source_has_before else None
    if baseline_dir is not None:
        baseline_manifest_path = baseline_dir / "manifest.json"
        if baseline_manifest_path.is_file():
            baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
            baseline_run_id = baseline_manifest.get("run_id")
            baseline_states.update(
                {
                    "baseline_is_bookend": "rebookend" in baseline_manifest,
                    "baseline_cell_matches": baseline_manifest.get("cell", {}).get("name")
                    == cell.name,
                    "baseline_split_matches": baseline_manifest.get("split", {}).get(
                        "id_digest"
                    )
                    == manifest.get("split", {}).get("id_digest"),
                    "baseline_has_eval_before": runner._has_eval_before(baseline_dir),
                    # The whole pairing verdict, from the same helper the spending path
                    # raises on, so a dry plan cannot say something a --go will not. Empty is
                    # the passing shape.
                    "baseline_pairing_drift": runner.pairing_drift(
                        manifest, baseline_manifest
                    ),
                    # Not a refusal and not nothing: the identities NEITHER archive records,
                    # named before the spend. The bookend's manifest carries the same list.
                    "baseline_pairing_unproven": runner.pairing_unproven(
                        manifest, baseline_manifest
                    ),
                }
            )
        else:
            baseline_states["baseline_is_run_dir"] = False
    elif source_has_before:
        # Self-paired: the before rows are the source's own, so the pairing is a record against
        # itself and every fact it states matches. What it does NOT state is still evidence the
        # plan owes an operator, and the manifest carries it either way.
        baseline_states["baseline_pairing_drift"] = runner.pairing_drift(manifest, manifest)
        baseline_states["baseline_pairing_unproven"] = runner.pairing_unproven(
            manifest, manifest
        )
    # The third comparison, at the only stage a plan can make it. The facts that exist only
    # after a container and a credential are checked by the runner at the moment they become
    # knowable, still before any row.
    execution_lines, execution_unproven = runner.execution_drift(
        manifest,
        runner.current_identity(
            cell=will_run,
            split=split,
            instruction=load_instruction(cell.instruction_arm),
            harness=harness_for(cell.harness),
            image_tag=args.image,
            image_digest_value=runner.image_digest(args.image),
        ),
        stage=runner.IDENTITY_PRE_SPEND,
    )
    source_lock_present = (source_dir / runner.RUN_LOCK_FILE).is_file()
    source_live = False
    if source_lock_present:
        try:
            runner._refuse_live_source(source_resolved)
        except RuntimeError:
            source_live = True
    refusals = {
        "suspension_present": (source_dir / SUSPENSION_FILE).is_file(),
        "source_is_bookend": source_is_bookend,
        "source_lock_present": source_lock_present,
        "source_live": source_live,
        "outputs_inside_source": outputs_inside_source,
        "rollout_terminus_present": source_stopping_path.is_file(),
        "terminal_session_resolvable": terminal_session is not None,
        "terminal_transcript_resolvable": terminal_transcript is not None,
        "experiment_drift": drift,
        "execution_identity_drift": execution_lines,
        "execution_identity_unproven": execution_unproven,
        "missing_required_env": missing_required,
        **baseline_states,
    }
    plan = {
        "source_run_dir": str(source_dir),
        "source_run_id": manifest.get("run_id"),
        "cell": cell.name,
        "harness": cell.harness,
        "axes": {
            "rollout_feedback": recorded_arm,
            "eval_context": "resumed",
            "eval_prompt_used": "rollout_system",
        },
        # The stopping rule the legs will run under, taken from the record rather than from
        # the cell file, and the fields where the file now says something else. Both are
        # written into the bookend's manifest too, so the plan and the artifact agree.
        "eval_runtime_from_record": {
            field: getattr(will_run.budget, field)
            for field in runner.BOOKEND_EVAL_RUNTIME_FIELDS
        },
        "cell_drift": cell_drift,
        "source_stop_reason": source_stopping.get("stop_reason"),
        "terminal_session_id": terminal_session,
        "baseline_run_id": baseline_run_id,
        # Every held-out task, because a rebookend is a fresh bookend rather than a repair:
        # nothing is already complete in a run directory that does not exist yet.
        "heldout_tasks_to_run": len(split.heldout),
        # The artifact is SELF-CONTAINED: it carries the baseline run's eval_before rows
        # as its own before block, labeled eval_before.source_run_id, so the paired delta
        # lives inside the artifact and survives the baseline's cell-stem artifact being
        # evicted by a later same-cell publication. Its name reflects whatever the carried
        # before side can account for; a baseline with holes publishes the incomplete shape.
        "result_artifact": f"<bookend-run-id>.json under {args.results} "
        f"(self-contained: carries eval_before rows from the baseline run "
        f"{baseline_run_id or '<named via --baseline>'}, labeled "
        "eval_before.source_run_id; the source's cell artifact is never touched)",
        "source_home": {
            "files": len(home_files),
            "bytes": sum(p.stat().st_size for p in home_files),
        },
        "credentials_present": credentials.inventory(dict(os.environ)),
        "refusals": refusals,
    }
    tau2_plan = _tau2_plan(cell)
    if tau2_plan is not None:
        plan["tau2_data"] = tau2_plan
    if not args.go:
        print(json.dumps(plan, indent=2))
        print(
            "\n[plan only] a rebookend runs a full held-out eval (real spend). "
            "Re-run with --go.",
            file=sys.stderr,
        )
        return 0
    blockers = []
    if source_is_bookend:
        blockers.append(
            f"the source is itself a rebookend; rebookend the original run "
            f"({bookend_original}) instead"
        )
    if refusals.get("baseline_required"):
        blockers.append(
            "the source has no eval_before of its own; name the baseline run with --baseline"
        )
    for state, why in (
        ("baseline_is_run_dir", "the named baseline is not a run directory"),
        ("baseline_is_bookend", None),
        ("baseline_cell_matches", "the named baseline measured a different cell"),
        ("baseline_split_matches", "the named baseline ran a different held-out split"),
        ("baseline_has_eval_before", "the named baseline has no eval_before provenance"),
    ):
        value = refusals.get(state)
        if state == "baseline_is_bookend":
            if value is True:
                blockers.append("the named baseline is itself a rebookend")
        elif value is False:
            blockers.append(why)
    if refusals["execution_identity_drift"]:
        blockers.append(
            "the execution identity of this checkout does not match the source's record: "
            + "; ".join(refusals["execution_identity_drift"])
        )
    if refusals.get("baseline_pairing_drift"):
        blockers.append(
            "the named baseline was not measured by the same definition as the source: "
            + "; ".join(refusals["baseline_pairing_drift"])
        )
    if refusals["suspension_present"]:
        blockers.append("the source holds a suspension record; finish it with `shobench resume`")
    if not refusals["source_lock_present"]:
        blockers.append(
            "the source has no run.lock, so it cannot be held still for the snapshot"
        )
    if refusals["source_live"]:
        blockers.append("the source is owned by a live process; wait for it to finish")
    if outputs_inside_source:
        blockers.append(
            f"outputs {outputs_inside_source} are inside the source run directory"
        )
    if not refusals["rollout_terminus_present"]:
        blockers.append("the source has no rollout terminus, so there is nothing to resume from")
    elif not refusals["terminal_session_resolvable"]:
        blockers.append("the source's rollout record names no terminal session")
    elif not refusals["terminal_transcript_resolvable"]:
        blockers.append(
            "the recorded terminal session has no resumable transcript in the source home"
        )
    if drift:
        blockers.append("; ".join(drift))
    if missing_required:
        blockers.append(f"the cell needs {missing_required} in the environment")
    if blockers:
        print(json.dumps(plan, indent=2), file=sys.stderr)
        print("\nBLOCKED: " + "; ".join(blockers) + ". Nothing was spent.", file=sys.stderr)
        return 1
    blocked = _set_tau2_data_dir(cell)
    if blocked:
        print(json.dumps(plan, indent=2), file=sys.stderr)
        print(f"\nBLOCKED: {blocked}\nNothing was spent.", file=sys.stderr)
        return 1
    results_path = asyncio.run(
        runner.rebookend_run(
            source_dir,
            runs_dir=Path(args.runs),
            results_dir=Path(args.results),
            baseline_run_dir=baseline_dir,
            agent_image=args.image,
            credentials=credentials.agent_env(cell.harness, cell.credential_mode, dict(os.environ)),
            capture_egress=not args.no_egress,
            allow_partial_baseline=args.allow_partial_baseline,
        )
    )
    print(f"results: {results_path}")
    return 0


def _owner_is_live(run_dir: Path) -> bool:
    """Does a live process hold this run's directory?

    Proven the way `rebookend` proves it: a SHARED flock on the run's own lock file, which every
    mutating owner holds EXCLUSIVE. The run's pid is never consulted, because the lock file is
    never unlinked and a finished run therefore names a pid the system may since have reissued.
    """
    import fcntl

    fd = os.open(run_dir / runner.RUN_LOCK_FILE, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _withdraw_stop_request(run_dir: Path) -> bool:
    """Take back an ask no owner consumed, under a shared lock so no new owner can be reading it.

    A mutating owner takes the run lock EXCLUSIVE, so holding it SHARED here means no owner
    exists for the duration and the request cannot be latched between the check and the removal.
    """
    import fcntl

    fd = os.open(run_dir / runner.RUN_LOCK_FILE, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            # An owner appeared after all, so the ask is theirs to read and must stay.
            return False
        try:
            path = run_dir / runner.STOP_REQUEST_FILE
            existed = path.is_file()
            path.unlink(missing_ok=True)
            return existed
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _cmd_stop(args: argparse.Namespace) -> int:
    """Ask a live run to end through its normal ending, so its records get written.

    Spends nothing, so it takes no ``--go``, and it is safe to call twice and on a finished run.

    A busy lock is not enough to write an ask on: a live owner started before the stop path
    existed never reads a request. So the owner has to ADVERTISE that it is watching
    (``stop_protocol`` in the lock it holds), and one that does not is refused with nothing
    written, because an ask accepted and never consumed is left on disk for the next resume or
    rerun to latch. Delivery is not reported until the runner unlinks the request, which also
    closes the window where the owner finishes between the liveness probe and the write.
    """
    run_dir = Path(args.run)
    if not (run_dir / runner.RUN_LOCK_FILE).is_file():
        print(
            f"{run_dir} has no {runner.RUN_LOCK_FILE}, so no shobench process ever owned it: "
            "this is not a live run directory and there is nothing to stop.",
            file=sys.stderr,
        )
        return 1
    if not _owner_is_live(run_dir):
        # Deliberately not a written request: a run nobody owns will not read one, and a file
        # left behind would end the next process to reopen the directory.
        print(
            f"{run_dir} is not owned by a live process, so it has already finished or was "
            "killed. Nothing was written and nothing was stopped.",
            file=sys.stderr,
        )
        return 0
    holder = runner.read_lock_holder(run_dir)
    if not holder.get("stop_protocol"):
        print(
            f"BLOCKED: {run_dir} is owned by a live process that does not watch for a stop "
            f"(its {runner.RUN_LOCK_FILE} advertises no stop protocol), so an ask written here "
            "would never be read and would be left behind for the next resume or rerun-eval to "
            "act on. Nothing was written and nothing was stopped.\n\n"
            "A process started before this command existed is the usual cause. Let it finish, or "
            "end it by hand and accept that it will leave no rollout terminus and so cannot be "
            "bookended.",
            file=sys.stderr,
        )
        return 1
    request = runner.write_stop_request(run_dir, reason=args.reason)
    print(json.dumps(request, indent=2))
    deadline = time.monotonic() + STOP_ACK_TIMEOUT_S
    while True:
        if not (run_dir / runner.STOP_REQUEST_FILE).is_file():
            print(
                f"\n[shobench] stop acknowledged by the owner of {run_dir}. It ends its current "
                "leg through the normal path, writes the run's records, and publishes what it "
                "has; the run's own output says whether the terminus it reached can be "
                "bookended.",
                file=sys.stderr,
            )
            return 0
        if not _owner_is_live(run_dir):
            withdrawn = _withdraw_stop_request(run_dir)
            print(
                f"\nBLOCKED: the owner of {run_dir} released it without reading the stop, so "
                "nothing was stopped: it had already finished, or it ended some other way. The "
                + ("request was withdrawn" if withdrawn else "request was already gone")
                + ", so nothing is left behind for a later resume or rerun-eval to act on.",
                file=sys.stderr,
            )
            return 1
        if time.monotonic() >= deadline:
            print(
                f"\nBLOCKED: the owner of {run_dir} has not acknowledged the stop within "
                f"{STOP_ACK_TIMEOUT_S:.0f}s. It is still live and still advertises the stop "
                "protocol, so the request is left in place for it to read; re-run this to wait "
                "again, and check that the process is not itself wedged.",
                file=sys.stderr,
            )
            return 1
        time.sleep(STOP_ACK_POLL_S)


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
    # The same provisioned serving-side need a fresh run reports. A continuation is where it goes
    # missing, because the assignment the first process made lives only in that process, so the
    # plan says which tree this one will point the stream at.
    tau2_plan = _tau2_plan(cell)
    if tau2_plan is not None:
        plan["tau2_data"] = tau2_plan
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

    # The provisioned half of the same serving-side need, and the one a continuation cannot
    # inherit: the fresh run set TAU2_DATA_DIR in a process that is gone. Set here, last of the
    # refusals and still before the sandbox comes up, so a tau2 cell a usage limit suspended
    # continues against the pinned data rather than failing at env construction with a path
    # nobody configured, and so a refused continuation leaves this process as it found it.
    blocked = _set_tau2_data_dir(cell)
    if blocked:
        print(json.dumps(plan, indent=2), file=sys.stderr)
        print(
            f"\nBLOCKED: {blocked}\nNothing was spent, and the run is still resumable.",
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

    stop = sub.add_parser(
        "stop", help="end a live run through its normal ending, so its records are written"
    )
    stop.add_argument("--run", required=True, help="the live run directory to end")
    stop.add_argument(
        "--reason",
        default="",
        help="what to record about why, carried into the leg's stop verdict",
    )
    stop.set_defaults(func=_cmd_stop)

    res = sub.add_parser("resume", help="continue a cell a usage limit suspended")
    res.add_argument("--run", required=True, help="the run directory holding suspended.json")
    res.add_argument("--go", action="store_true", help="actually continue it (real spend)")
    res.add_argument("--results", default="results")
    res.add_argument("--image", default=AGENT_IMAGE)
    res.add_argument("--no-egress", action="store_true")
    res.set_defaults(func=_cmd_resume)

    rerun = sub.add_parser(
        "rerun-eval", help="finish an eval_after that lost tasks with no suspension to resume"
    )
    rerun.add_argument("--run", required=True, help="the finished run directory to reopen")
    rerun.add_argument(
        "--phase",
        choices=["eval_after", "eval_before"],
        default="eval_after",
        help="which eval phase to repair (eval_before only for a run that never rolled out)",
    )
    rerun.add_argument(
        "--refresh-baseline",
        action="store_true",
        help=(
            "on a bookend: catch its carried baseline rows up to the baseline they were "
            "snapshotted from, adding ids the carry has none for and upgrading unsettled ones"
        ),
    )
    rerun.add_argument(
        "--baseline",
        default=None,
        help=(
            "the baseline run directory --refresh-baseline re-reads; defaults to the sibling "
            "named by the manifest's baseline_run_id"
        ),
    )
    rerun.add_argument("--go", action="store_true", help="actually re-run the holes (real spend)")
    rerun.add_argument("--results", default="results")
    rerun.add_argument("--image", default=AGENT_IMAGE)
    rerun.add_argument("--no-egress", action="store_true")
    rerun.set_defaults(func=_cmd_rerun_eval)

    rebookend = sub.add_parser(
        "rebookend",
        help="give an existing run a resumed eval_after, as a new run; the source is untouched",
    )
    rebookend.add_argument("--run", required=True, help="the SOURCE run directory to bookend")
    rebookend.add_argument(
        "--baseline",
        default=None,
        help=(
            "the run directory whose eval_before pairs with this bookend; required when the "
            "source has no eval_before of its own, defaults to the source when it does"
        ),
    )
    rebookend.add_argument(
        "--allow-partial-baseline",
        action="store_true",
        help=(
            "carry the baseline's rows even though it cannot account for every held-out id; "
            "the ids frozen this way are recorded in the bookend's manifest"
        ),
    )
    rebookend.add_argument(
        "--go", action="store_true", help="actually run the bookend (real spend)"
    )
    rebookend.add_argument("--runs", default="runs", help="where the NEW run directory is made")
    rebookend.add_argument("--results", default="results")
    rebookend.add_argument("--image", default=AGENT_IMAGE)
    rebookend.add_argument("--no-egress", action="store_true")
    rebookend.set_defaults(func=_cmd_rebookend)

    rep = sub.add_parser("report", help="the summary table")
    rep.add_argument("results", nargs="?", default="results")
    rep.add_argument("--format", choices=["table", "json"], default="table")
    rep.set_defaults(func=lambda a: report.main([str(a.results), "--format", a.format]))

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
