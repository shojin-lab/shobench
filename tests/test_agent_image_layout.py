"""What the image bakes has to survive the mounts a cell runs with.

The agent image and the runtime mounts are written in different files and were never checked
against each other, which is how prime-agent's kernel came to be baked under ``/root`` and then
hidden by the cell HOME the runner mounts over all of it. Unmounted the image was fine; every
real cell got a prime-agent whose IPython tool either failed outright or rebuilt itself over the
network during measured time, and neither shows up as anything but a strange rollout.

So this reads the real Dockerfile and drives the real ``docker_args``, and asserts the one thing
neither file can state on its own: nothing the image installs for the harness lies under a path
the runtime mounts over. The build itself carries the other half of the guard, a verification
layer that fails the build if uv is off PATH or the venv landed back under ``/root``.
"""

from __future__ import annotations

import re
from pathlib import Path

from shobench.config import repo_root
from shobench.containers import CellSandbox

# The image paths that must be reachable at runtime, keyed by the variable that pins each one.
# Every one of these was a default under $HOME before, which is exactly why they are pinned now:
# the kernel venv, the uv-managed interpreter it symlinks its python to, and uv itself, which
# prime-agent looks for on PATH and then in ~/.local/bin.
PINNED_IMAGE_PATHS = ("PRIME_AGENT_KERNEL_VENV", "UV_PYTHON_INSTALL_DIR", "UV_INSTALL_DIR")


def _dockerfile_env() -> dict[str, str]:
    """The ``ENV KEY=VALUE`` lines of the agent Dockerfile, which is where the paths are pinned."""
    body = (repo_root() / "docker" / "agent.Dockerfile").read_text(encoding="utf-8")
    return {
        match.group(1): match.group(2).strip().strip('"')
        for match in re.finditer(r"^ENV\s+([A-Z0-9_]+)=(.*)$", body, re.MULTILINE)
    }


def _mount_targets(args: list[str]) -> list[str]:
    """The container-side paths a leg's ``docker run`` covers with a bind mount."""
    return [
        args[index + 1].split(":")[1]
        for index, token in enumerate(args)
        if token == "-v" and index + 1 < len(args)
    ]


def _mount_sources(args: list[str]) -> list[str]:
    """The host-side paths of those same bind mounts."""
    return [
        args[index + 1].split(":")[0]
        for index, token in enumerate(args)
        if token == "-v" and index + 1 < len(args)
    ]


def test_relative_host_paths_still_reach_docker_absolute(tmp_path: Path, monkeypatch) -> None:
    """Docker reads a non-absolute ``-v`` source as a named volume and refuses it.

    The CLI's default layout hands the sandbox paths relative to the invocation cwd
    (``runs/precheck-<cell>/home``), which is how the first real precheck died with
    "invalid characters for a local volume name" before any spend.
    """
    monkeypatch.chdir(tmp_path)
    sandbox = CellSandbox(run_id="r", home=Path("runs/pre/home"), workdir=Path("runs/pre/work"))

    sources = _mount_sources(sandbox.docker_args(env={}, mounts={Path("runs/cfg"): "/cfg:ro"}))

    assert sources, "no bind mounts in the leg argv at all"
    for source in sources:
        assert Path(source).is_absolute(), f"docker would read {source!r} as a named volume"


def test_the_runtime_mounts_are_the_ones_the_leg_actually_runs_with(tmp_path: Path) -> None:
    """A guard on the guard: the mount list below is read off the real argv, not assumed."""
    sandbox = CellSandbox(run_id="r", home=tmp_path / "home", workdir=tmp_path / "work")

    targets = _mount_targets(sandbox.docker_args(env={}, mounts={tmp_path / "cfg": "/cfg:ro"}))

    assert targets == ["/root", "/work", "/cfg"]


def test_nothing_the_image_bakes_for_the_harness_is_hidden_by_a_runtime_mount(
    tmp_path: Path,
) -> None:
    """Every pinned image path stays visible once the cell's HOME and workdir are mounted."""
    declared = _dockerfile_env()
    sandbox = CellSandbox(run_id="r", home=tmp_path / "home", workdir=tmp_path / "work")
    targets = _mount_targets(sandbox.docker_args(env={}, mounts={tmp_path / "cfg": "/cfg:ro"}))

    for name in PINNED_IMAGE_PATHS:
        assert name in declared, (
            f"{name} is not pinned in the agent Dockerfile, so this path falls back to a "
            "default under $HOME and the runtime mount hides it"
        )
        baked = Path(declared[name])
        for target in targets:
            covered = baked == Path(target) or Path(target) in baked.parents
            assert not covered, (
                f"{name}={baked} lies under the runtime mount at {target}, so a real cell "
                "cannot see it however well the image builds"
            )


def test_the_build_verifies_its_own_bake(tmp_path: Path) -> None:
    """The Dockerfile fails the build rather than shipping a kernel that is not there.

    prime-agent's postinstall swallows its own errors, so a bootstrap that did not happen leaves
    a green build and a cell that discovers it eight hours in. The assertions are part of the
    install layer so a partial success cannot be cached as a whole one.
    """
    body = (repo_root() / "docker" / "agent.Dockerfile").read_text(encoding="utf-8")

    assert "test ! -e /root/.prime/agent/kernel-venv" in body
    assert "test -x /usr/local/bin/uv" in body
    assert "import ipykernel" in body


def test_two_runs_of_one_cell_never_share_container_names(tmp_path: Path) -> None:
    """Truncation must not erase what makes a run unique.

    The prime-opus cell name is long enough that a bare [:50] cut off the whole timestamp, two
    concurrent runs of the cell shared one namespace-holder name, and the second run's up()
    (a docker rm -f before the create) tore down the first run's live network mid-eval.
    """
    cell = "automationbench-prime_agent-claude-opus-5"
    a = CellSandbox(run_id=f"{cell}-20260812T001456Z", home=tmp_path / "a", workdir=tmp_path / "aw")
    b = CellSandbox(run_id=f"{cell}-20260812T110239Z", home=tmp_path / "b", workdir=tmp_path / "bw")
    again = CellSandbox(
        run_id=f"{cell}-20260812T001456Z", home=tmp_path / "c", workdir=tmp_path / "cw"
    )

    assert a.netns_container != b.netns_container
    assert a.network != b.network
    # The same run keeps the same names: resume and rerun reclaim their own holder by name.
    assert again.netns_container == a.netns_container
    # DNS-label budget: the longest derived name (the egress observer suffix) must still fit.
    assert len(f"{a.netns_container}-egress") <= 63


def test_crash_cleanup_and_the_sandbox_agree_on_every_name(tmp_path: Path, monkeypatch) -> None:
    """cleanup() must remove the names up() creates, not a hand-reconstructed formula.

    The first digest fix left cleanup rebuilding the OLD stem by hand, so it silently removed
    nothing: every docker call in it is check=False.
    """
    from shobench import runner

    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(runner, "docker", lambda *a, **k: calls.append(a))

    run_id = "automationbench-prime_agent-claude-opus-5-20260812T001456Z"
    sandbox = CellSandbox(run_id=run_id, home=tmp_path / "h", workdir=tmp_path / "w")
    runner.cleanup(run_id)

    removed = {args[2] for args in calls if args[:2] == ("rm", "-f")}
    assert sandbox.netns_container in removed
    networks = {args[2] for args in calls if args[:2] == ("network", "rm")}
    assert sandbox.network in networks


def test_migrating_a_pre_digest_run_rewrites_names_and_deletes_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    """Two legacy manifests can claim the SAME recorded names, so deleting by record could
    tear down a live neighbor: migration rewrites the record and touches no docker resource."""
    from shobench import runner

    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(runner, "docker", lambda *a, **k: calls.append(a))

    legacy = 'shobench-automationbench-prime_agent-claude-opus-5'[:50]
    shared_record = {"netns_container": f"{legacy}-ns", "network": f"{legacy}-net"}
    run_a = CellSandbox(
        run_id="automationbench-prime_agent-claude-opus-5-20260812T001456Z",
        home=tmp_path / "a",
        workdir=tmp_path / "aw",
    )
    manifest_a = {"container": dict(shared_record)}
    manifest_b = {"container": dict(shared_record)}

    runner._migrate_recorded_containers(manifest_a, run_a)

    assert calls == []
    assert manifest_a["container"]["netns_container"] == run_a.netns_container
    assert manifest_a["container"]["network"] == run_a.network
    # The neighbor's identical record is untouched by A's migration.
    assert manifest_b["container"] == shared_record

    # A post-fix run's recorded names already match: the block is left exactly as recorded.
    same = {"container": {"netns_container": run_a.netns_container, "network": run_a.network}}
    runner._migrate_recorded_containers(same, run_a)
    assert calls == []
