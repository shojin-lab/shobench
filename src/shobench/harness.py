"""Launching a harness autonomously, and reading how it stopped.

Two jobs live here. The first is building the command that makes a harness fully autonomous
from its first turn, because no cell may depend on a human approving anything mid-run. The
second is classifying how a leg ended, because the scope's stopping metrics count only the
agent's own choice to stop: a session the provider cut off at a usage limit is resumed and
does not count as a stop, and a wedged process is an error rather than either.

The classification is deliberately evidence-first. Each harness's rule names the artifact it
reads (an exit code, a JSON result field, a matched line) and the runner records that evidence
in the results JSON, so a reader can check the call rather than trust it. Anything the rules do
not recognise becomes ``UNKNOWN``, which the runner treats as an error and never as a chosen
stop; a metric that silently absorbs surprises is worse than one that reports them.

The per-harness settings and the citations behind them are in ``docs/harness-autonomy.md``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class StopKind(StrEnum):
    """How one harness leg ended."""

    # The agent finished its turn on its own terms. This is the only kind the stopping metrics
    # count, and only when the stream still had tasks left to give.
    CHOSEN = "chosen_stop"
    # The provider stopped the session at a usage or rate limit. The runner resumes; the
    # stopping metrics ignore it.
    USAGE_LIMIT = "usage_limit"
    # The runner ended the leg because the leg budget elapsed.
    LEG_TIMEOUT = "leg_timeout"
    # The harness failed for a reason that is neither of the above.
    ERROR = "error"
    # No rule matched. Treated as an error, reported as itself.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StopVerdict:
    """A classification and the evidence for it."""

    kind: StopKind
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def resumable(self) -> bool:
        return self.kind is StopKind.USAGE_LIMIT

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "reason": self.reason, "evidence": self.evidence}


@dataclass(frozen=True)
class LaunchSpec:
    """One harness invocation, ready to run inside the cell's container."""

    argv: list[str]
    env: dict[str, str]
    # Files the runner writes into the container's config mount before the leg starts. The
    # mount is read-only and lives outside the agent's working directory, so a harness config
    # never becomes part of what the agent thinks of as itself.
    config_files: dict[str, str] = field(default_factory=dict)
    # Files the runner writes into the cell's isolated HOME, for harnesses that read their
    # configuration only from there. These land inside what the agent can edit, which is the
    # cost of the harness having no config flag; the manifest's home inventory records them so
    # a later edit by the agent is still visible as a change.
    home_files: dict[str, str] = field(default_factory=dict)
    # Harnesses that hang on an open stdin get /dev/null; ones that read the prompt from stdin
    # get it here.
    stdin: str | None = None


@dataclass(frozen=True)
class UsageLimitRule:
    """One evidence-based way a harness announces a usage-limit stop.

    ``where`` names the artifact the rule reads so the recorded evidence says where the match
    came from, and ``citation`` names the source that established it, so a rule nobody can
    trace is visible as such.
    """

    where: str
    pattern: str
    citation: str

    def matches(self, text: str) -> str | None:
        found = re.search(self.pattern, text, re.IGNORECASE)
        return found.group(0) if found else None


class Harness:
    """The interface the runner drives. One subclass per harness."""

    name: str = ""
    usage_limit_rules: Sequence[UsageLimitRule] = ()

    def base_env(self) -> dict[str, str]:
        """Environment every invocation of this harness needs, credentials excluded.

        The credential probe uses this too, which matters more than it looks: a probe missing
        one of these fails for a reason that has nothing to do with the credential, and a
        negative control that fails for the wrong reason proves nothing.
        """
        return {}

    def version_probe(self) -> list[str]:
        """A command that reports the installed harness version, for the cell manifest."""
        raise NotImplementedError

    def model_probe(self) -> list[str] | None:
        """A command that reports which model the harness resolved, when it has one.

        The scope requires the manifest to record which model actually answered rather than
        which one was requested, so a harness that can be asked is asked.
        """
        return None

    def launch(
        self,
        *,
        mcp_url: str,
        system_prompt: str,
        user_prompt: str,
        model: str,
        trace_path: Path,
        session_id: str | None = None,
        resume: bool = False,
        leg_timeout_s: int = 3600,
    ) -> LaunchSpec:
        raise NotImplementedError

    def classify(
        self, *, returncode: int, stdout_path: Path, stderr_path: Path, timed_out: bool
    ) -> StopVerdict:
        raise NotImplementedError

    def observed_models(self, trace_path: Path) -> list[str]:
        """Model identifiers the trace shows actually answered.

        The scope asks the manifest to record which model answered rather than which one was
        requested. A version probe cannot say that; the trace can, when the harness reports it.
        Returning nothing is honest for a harness that does not.
        """
        return []

    # ----- shared helpers -----

    def _match_usage_limit(self, texts: dict[str, str]) -> StopVerdict | None:
        for rule in self.usage_limit_rules:
            text = texts.get(rule.where, "")
            hit = rule.matches(text)
            if hit:
                return StopVerdict(
                    kind=StopKind.USAGE_LIMIT,
                    reason=f"matched {rule.where}: {hit!r}",
                    evidence={
                        "where": rule.where,
                        "pattern": rule.pattern,
                        "match": hit,
                        "citation": rule.citation,
                    },
                )
        return None


def tail(path: Path, *, lines: int = 60, limit: int = 8000) -> str:
    """The end of a log, which is where a harness says why it stopped."""
    if not path.exists():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    return "\n".join(content.splitlines()[-lines:])[-limit:]


def jsonl_events(path: Path, *, limit: int = 400) -> list[dict[str, Any]]:
    """The last events of a stream-json trace, parsed and tolerant of a truncated final line."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            out.append(event)
    return out


__all__ = [
    "Harness",
    "LaunchSpec",
    "StopKind",
    "StopVerdict",
    "UsageLimitRule",
    "jsonl_events",
    "tail",
]
