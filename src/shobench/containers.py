"""Container and credential lifecycle for one cell.

This is the part of the original study's broker pattern that survived: the broker was good at
owning containers and credentials, and it stays responsible for exactly that. Serving moved to
shogym's stream classes. The wandb sink, when a run wants one, stays on this side too, so the
agent container never holds a wandb key and never mounts anything of ours.

The topology per cell:

- a Docker network, created and destroyed with the cell;
- a network-namespace holder on it, which the egress observer attaches to before any agent
  runs and which every agent container then joins;
- an isolated HOME directory on the host, mounted at the agent's ``$HOME``, holding whatever
  the harness writes about itself and nothing from the operator's own logins;
- agent containers, run one leg at a time, with credentials passed as ``-e`` at runtime and
  never baked into an image or written to a committed file.

The agent reaches the task stream at the host's ``host.docker.internal``, which keeps the
serving process outside the container entirely. That is the isolation the measurement needs:
the provenance directory holding scores and held-out answers is not on any path the agent can
open.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# Given a POSIX-relative path inside the agent's HOME, is this file noise rather than part of
# the durable self?
DurableFilter = Callable[[str], bool]

AGENT_IMAGE = "shobench-agent:v0"
NETNS_IMAGE = "shobench-netns:v0"

# Docker Desktop's name for the host from inside a container. Linux hosts need
# --add-host=host.docker.internal:host-gateway, which `Cell.docker_args` adds unconditionally
# because it is harmless where the name already resolves.
HOST_ALIAS = "host.docker.internal"


class DockerError(RuntimeError):
    """A docker command failed. Carries the command and its stderr, because the message
    docker prints is almost always the actual diagnosis."""


def docker(*args: str, check: bool = True, text: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["docker", *args], capture_output=True, text=text)
    if check and result.returncode != 0:
        raise DockerError(f"docker {' '.join(args)} failed ({result.returncode}): {result.stderr}")
    return result


def daemon_available() -> bool:
    return shutil.which("docker") is not None and docker("info", check=False).returncode == 0


def build_image(dockerfile: Path, context: Path, tag: str) -> str:
    docker("build", "-q", "-f", str(dockerfile), "-t", tag, str(context))
    return tag


@dataclass
class CellSandbox:
    """The per-cell network, namespace holder, and isolated HOME.

    Names are derived from the run id so a leftover from a crashed run is identifiable and
    removable, and so two cells never share a network or a HOME.
    """

    run_id: str
    home: Path
    workdir: Path
    network: str = field(init=False)
    netns_container: str = field(init=False)

    def __post_init__(self) -> None:
        # Container names become DNS labels, which are capped at 63 characters.
        stem = f"shobench-{self.run_id}"[:50]
        self.network = f"{stem}-net"
        self.netns_container = f"{stem}-ns"

    def up(self, *, netns_image: str = NETNS_IMAGE) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.workdir.mkdir(parents=True, exist_ok=True)
        if docker("network", "inspect", self.network, check=False).returncode != 0:
            docker("network", "create", self.network)
        docker("rm", "-f", self.netns_container, check=False)
        docker(
            "run",
            "-d",
            "--name",
            self.netns_container,
            "--network",
            self.network,
            "--add-host",
            f"{HOST_ALIAS}:host-gateway",
            netns_image,
            "sleep",
            "infinity",
        )

    def down(self) -> None:
        docker("rm", "-f", self.netns_container, check=False)
        docker("network", "rm", self.network, check=False)

    def docker_args(
        self, *, env: dict[str, str], mounts: dict[Path, str], name: str | None = None
    ) -> list[str]:
        """The ``docker run`` prefix an agent leg uses.

        The leg joins the holder's network namespace rather than the network directly, so the
        observer that attached to that namespace sees the leg's traffic from its first packet.

        ``name`` matters for a leg that has to be killed. Stopping the docker client does not
        stop the container it started, so a leg the runner ends at its budget would otherwise
        keep running, keep spending, and keep talking to the stream after the runner had
        moved on. A named container can be removed.
        """
        args = [
            "run",
            "--rm",
        ]
        if name is not None:
            args += ["--name", name]
        args += [
            "--network",
            f"container:{self.netns_container}",
            "-v",
            f"{self.home}:/root:rw",
            "-v",
            f"{self.workdir}:/work:rw",
            "-w",
            "/work",
        ]
        for host_path, container_path in mounts.items():
            args += ["-v", f"{host_path}:{container_path}"]
        for key, value in env.items():
            args += ["-e", f"{key}={value}"]
        return args


def _digest_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def durable_files(home: Path, *, exclude: DurableFilter) -> list[Path]:
    """The files under ``home`` that count as the agent's durable self."""
    if not home.exists():
        return []
    out = []
    for path in sorted(p for p in home.rglob("*") if p.is_file()):
        if not exclude(path.relative_to(home).as_posix()):
            out.append(path)
    return out


def home_digest(home: Path, *, exclude: DurableFilter) -> str:
    """A content hash over the agent's durable self, for the before and after.

    What is excluded decides what the number means. A digest that includes session
    transcripts, caches and bookkeeping answers "did a session happen", which is always yes
    and therefore says nothing. A digest over what survives a session boundary answers "did
    the rollout write anything the next session can use", which is the benchmark's question.
    """
    import hashlib

    digest = hashlib.sha256()
    for path in durable_files(home, exclude=exclude):
        digest.update(path.relative_to(home).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_digest_file(path)))
    return digest.hexdigest()


def home_inventory(home: Path, *, exclude: DurableFilter) -> list[dict[str, object]]:
    """Every durable file with its size and digest.

    The digest says whether the home changed; the inventory says what changed, which is what a
    reader of the results wants when a cell reports a nonzero delta.
    """
    return [
        {
            "path": path.relative_to(home).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _digest_file(path),
        }
        for path in durable_files(home, exclude=exclude)
    ]


def run_relative(path: Path | str, run_dir: Path) -> str:
    """A run-internal path recorded portably: relative to the run directory, in POSIX form.

    The manifest and the results JSON are written at the run directory's root, so a path stored
    relative to it resolves for any reader that has the run directory, and none of them carry
    the operator's absolute layout. A path that somehow lies outside the run directory falls
    back to its basename rather than leaking the absolute path it came from.
    """
    candidate = Path(path)
    try:
        return candidate.relative_to(run_dir).as_posix()
    except ValueError:
        return candidate.name


def write_json(path: Path, body: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "AGENT_IMAGE",
    "HOST_ALIAS",
    "NETNS_IMAGE",
    "CellSandbox",
    "DockerError",
    "build_image",
    "daemon_available",
    "docker",
    "DurableFilter",
    "durable_files",
    "home_digest",
    "home_inventory",
    "run_relative",
    "write_json",
]
