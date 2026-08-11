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
