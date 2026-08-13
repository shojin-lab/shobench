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

import hashlib
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


def image_digest(image: str) -> str | None:
    """The image's content id, which is exactly what a tag is not.

    A tag is a mutable name: rebuilding ``shobench-agent:v0`` on a newer base, a different Node
    or Python, another shell tool or another Prime kernel produces the same tag and the same
    ``--version`` probe while producing a different agent. The image ID is the content, so it is
    what a later comparison can rest on, and every run records it from here on.

    Answers nothing when docker cannot (no daemon, no such image, no docker at all) rather than
    raising or guessing: a manifest that records an honest absence is what lets a reader see the
    identity was never established, and this is not worth failing a run over.

    Deliberately uncached. The runner asks once per run and carries the answer in the run's
    context, which is the scope that means anything: a process-lifetime cache made a second run
    in one process pin the image the FIRST run resolved, so a deliberate rebuild between two
    calls of the exported async API published the old id for the new run's rows.
    """
    if shutil.which("docker") is None:
        return None
    result = docker("image", "inspect", "--format", "{{.Id}}", image, check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def build_image(dockerfile: Path, context: Path, tag: str) -> str:
    docker("build", "-q", "-f", str(dockerfile), "-t", tag, str(context))
    return tag


def run_stem(run_id: str) -> str:
    """The one name stem every docker resource of a run derives from.

    Container names become DNS labels, capped at 63 characters, so the stem is truncated.
    Truncation alone once erased the run timestamp for a long cell name, and two concurrent
    runs of one cell then fought over a single namespace holder: ``up()`` is a docker rm -f
    before the create, so the second run tore down the first run's live network mid-eval. The
    64-bit digest of the FULL run id keeps distinct runs' names distinct whatever the cell
    name's length (a pairwise collision is 1 in 2**64; the 32-bit version this replaced was
    reviewably weak), while the same run keeps the same names across resume and rerun, which
    reclaim their own holder by name. Derivation lives in one function because a cleanup path
    that reconstructed the stem by hand kept the OLD formula after the fix and silently
    removed nothing.
    """
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
    return f"shobench-{run_id}"[:33] + f"-{digest}"


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
        stem = run_stem(self.run_id)
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
        self,
        *,
        env: dict[str, str],
        mounts: dict[Path, str],
        name: str | None = None,
        home: Path | None = None,
        workdir: Path | None = None,
    ) -> list[str]:
        """The ``docker run`` prefix an agent leg uses.

        The leg joins the holder's network namespace rather than the network directly, so the
        observer that attached to that namespace sees the leg's traffic from its first packet.

        ``name`` matters for a leg that has to be killed. Stopping the docker client does not
        stop the container it started, so a leg the runner ends at its budget would otherwise
        keep running, keep spending, and keep talking to the stream after the runner had
        moved on. A named container can be removed.

        ``home`` overrides which HOME is mounted at ``/root``. The rollout mounts the cell's one
        accumulating home (the default); an eval task mounts its own throwaway copy of it, so
        two tasks can run at once without one task's writes reaching the other or the base.

        ``workdir`` overrides which host directory is mounted at ``/work``, the cwd every
        harness runs in. It is the same isolation story as ``home`` and it matters for the same
        reason: ``/work`` is writable, so concurrent eval tasks sharing one would let task A's
        file reach task B, and one phase's ``/work`` would leak into the next. Unlike ``home``,
        ``/work`` is not part of the measured durable self (only the HOME digest is), so an eval
        task gets a fresh empty directory of its own, discarded with its task HOME, rather than a
        copy of anything. The rollout keeps the cell's one accumulating ``/work`` (the default).
        """
        # Host paths must be absolute: docker reads a non-absolute ``-v`` source as a named
        # volume, and the default runs/ layout arrives here relative to the invocation cwd.
        home_path = (self.home if home is None else home).resolve()
        work_path = (self.workdir if workdir is None else workdir).resolve()
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
            f"{home_path}:/root:rw",
            "-v",
            f"{work_path}:/work:rw",
            "-w",
            "/work",
        ]
        for host_path, container_path in mounts.items():
            args += ["-v", f"{host_path.resolve()}:{container_path}"]
        for key, value in env.items():
            args += ["-e", f"{key}={value}"]
        return args


def _digest_file(path: Path) -> str:
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
