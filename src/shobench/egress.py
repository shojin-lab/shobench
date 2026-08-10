"""Per-cell network egress observation.

The scope settled leakage as an observable, not a gate: the sandbox has full internet because
the harnesses need their model APIs, denying web tools does not stop a Bash-capable agent from
fetching an answer, and an agent that chooses to look up a held-out answer is a finding worth
recording. So this records where a cell's containers went and restricts nothing.

The mechanism is a passive capture sidecar sharing the cell's network namespace. A holder
container owns the namespace and joins the cell's Docker network; the observer attaches to
that same namespace before any agent does and runs tshark; every agent container then joins
the holder. Passive capture is the choice because it cannot fail closed: an HTTPS_PROXY the
harness declines to honor would silently observe nothing, and a proxy that breaks is a proxy
that blocks. Attaching the observer to a namespace that exists before the agent does is what
removes the startup race in which the first requests go unseen.

What this sees and what it does not: it sees resolved hostnames (DNS queries) and TLS SNI, so
it answers "which hosts did this cell talk to, and how often". It does not decrypt TLS, so it
never answers "what did it fetch". That is the honest limit and the reason the egress log is
evidence for a human to read alongside the traces rather than an automatic cheating verdict.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# tshark fields, in the order they appear in each TSV line of the egress log.
FIELDS = (
    "frame.time_epoch",
    "ip.dst",
    "tcp.dstport",
    "udp.dstport",
    "dns.qry.name",
    "tls.handshake.extensions_server_name",
)

# Outbound DNS questions and TLS client hellos: the two places a hostname appears in the clear.
DISPLAY_FILTER = "dns.flags.response == 0 || tls.handshake.type == 1"

EGRESS_IMAGE = "shobench-egress:v0"


@dataclass(frozen=True)
class EgressCapture:
    """A running observer, and where its log lands."""

    container: str
    log_path: Path
    netns_container: str


def build_image(dockerfile: Path, context: Path, *, tag: str = EGRESS_IMAGE) -> str:
    subprocess.run(
        ["docker", "build", "-q", "-f", str(dockerfile), "-t", tag, str(context)],
        check=True,
        capture_output=True,
    )
    return tag


def start(
    *,
    netns_container: str,
    name: str,
    log_path: Path,
    image: str = EGRESS_IMAGE,
) -> EgressCapture:
    """Attach a passive observer to ``netns_container``'s network namespace.

    NET_ADMIN and NET_RAW are what a capture needs and all it needs. The observer holds no
    credential and mounts nothing: its output is its stdout, harvested by :func:`stop`, so
    there is no shared path between it and the agent.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch()
    fields: list[str] = []
    for field in FIELDS:
        fields += ["-e", field]
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--network",
            f"container:{netns_container}",
            "--cap-add",
            "NET_ADMIN",
            "--cap-add",
            "NET_RAW",
            image,
            "tshark",
            "-i",
            "any",
            "-l",
            "-n",
            "-Q",
            "-Y",
            DISPLAY_FILTER,
            "-T",
            "fields",
            # tshark spells a tab separator as the two-character escape /t.
            "-E",
            "separator=/t",
            *fields,
        ],
        check=True,
        capture_output=True,
    )
    return EgressCapture(container=name, log_path=log_path, netns_container=netns_container)


def stop(capture: EgressCapture) -> None:
    """Stop the observer and flush its output into the cell's egress log."""
    subprocess.run(["docker", "stop", "-t", "5", capture.container], capture_output=True)
    logs = subprocess.run(["docker", "logs", capture.container], capture_output=True, text=True)
    capture.log_path.write_text(logs.stdout, encoding="utf-8")
    subprocess.run(["docker", "rm", "-f", capture.container], capture_output=True)


def summarize(log_path: Path) -> dict[str, object]:
    """Fold the raw capture into per-host counts, which is what a reviewer actually reads."""
    if not log_path.exists():
        return {"available": False, "reason": "no egress log written"}
    hosts: Counter[str] = Counter()
    lines = 0
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        lines += 1
        parts = line.split("\t")
        padded = parts + [""] * (len(FIELDS) - len(parts))
        for value in (padded[4], padded[5]):
            for host in str(value).split(","):
                host = host.strip().rstrip(".")
                if host:
                    hosts[host] += 1
    return {
        "available": True,
        "path": str(log_path),
        "observations": lines,
        "distinct_hosts": len(hosts),
        "hosts": dict(hosts.most_common()),
        "mechanism": "passive tshark capture in the cell network namespace (DNS + TLS SNI)",
        "limit": "hostnames only; TLS payloads are not decrypted",
    }


def write_summary(log_path: Path, out_path: Path) -> dict[str, object]:
    summary = summarize(log_path)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


__all__ = [
    "DISPLAY_FILTER",
    "EGRESS_IMAGE",
    "FIELDS",
    "EgressCapture",
    "build_image",
    "start",
    "stop",
    "summarize",
    "write_summary",
]
