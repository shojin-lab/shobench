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

Evidence a verdict carries is quoted only where the runner knows what it is quoting. The
harness's raw stderr is the one artifact it never knows that about, so a verdict describes it
instead: see :func:`stderr_evidence` for why, and for what a description has to include to keep
the classification checkable.

The per-harness settings and the citations behind them are in ``docs/harness-autonomy.md``.
"""

from __future__ import annotations

import hashlib
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
    # The runner ended an eval leg that had nothing left to do: its held-out task was sealed and
    # its stream was drained, and the harness kept running anyway. Its own kind because the fact
    # it records is a result about the harness, not an accident of the budget. The value is not
    # the bare word `drained`, which shogym already uses for a row a stream close cut off in
    # flight; these are different events and a reader joining the two records must not merge them.
    DRAINED = "stream_drained"
    # An operator asked for the run to end, and the runner ended the leg through its normal path
    # so the records got written. Its own kind because what it says about the measurement is
    # unlike any of the others: the treatment was cut short by a person, for a reason outside the
    # experiment, and how far the agent had got when that happened is still a real terminus.
    OPERATOR = "operator_stop"
    # The runner ended a rollout leg that had shown no evidence of progress anywhere for the
    # cell's bound. Its own kind because it is a statement about the leg rather than about the
    # operator or the budget: the container was alive and something was still being paid for,
    # and nothing the agent could be observed doing had moved.
    STALLED = "no_progress"
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
        # `resumable` is written down rather than left to be re-derived from `kind` by whoever
        # reads the record. It is the field an operator acts on, since it says whether a run
        # that ended early is waiting for a window to clear or is simply over.
        return {
            "kind": self.kind.value,
            "reason": self.reason,
            "resumable": self.resumable,
            "evidence": self.evidence,
        }


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
    # a later edit by the agent is still visible as a change. Every leg rewrites them, so what
    # belongs here is what only the runner can know and what changes between legs, an endpoint
    # being the example.
    home_files: dict[str, str] = field(default_factory=dict)
    # HOME files the runner creates when they are absent and never rewrites afterwards. These
    # are an initial condition rather than a per-leg setting: what the agent starts with, and
    # then owns. A harness asset the agent is free to edit belongs here, because the rollout
    # measures what the agent made durable, and a later leg that restored the original bytes
    # would erase exactly that between the rollout and the evaluation meant to read it.
    home_seed_files: dict[str, str] = field(default_factory=dict)
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


# Every harness runs with a clean NODE_OPTIONS. An inherited one (a debugger port, a loader
# hook) reaches the harness's own Node runtime and has broken launches before. It is the base
# env every harness extends rather than replaces, so it lives with the base class.
BASE_ENV = {"NODE_OPTIONS": ""}


class Harness:
    """The interface the runner drives. One subclass per harness."""

    name: str = ""
    usage_limit_rules: Sequence[UsageLimitRule] = ()

    # HOME paths the RUNNER writes on every leg, declared here so the durable-self digest can
    # exclude them. They are the same paths ``launch`` returns as ``home_files``, and a
    # characterization test holds the two together. Declaring them separately is what lets the
    # digest be taken before any leg has run: an endpoint the runner rewrites per leg cannot be
    # an agent write however it changes, and counting it made a prime_agent cell that wrote
    # nothing report a changed durable self.
    runner_owned_home_files: tuple[str, ...] = ()

    def home_seed_files(self) -> dict[str, str]:
        """Assets the cell's HOME starts with, which the agent then owns.

        Separate from ``launch`` because the runner places them once, before the baseline digest
        is taken, and they depend on nothing a leg knows. Placing them lazily on the rollout's
        first leg put them on the far side of the baseline, so the vendored bytes themselves
        read as something the rollout had written.
        """
        return {}

    def credential_provider(self, model: str) -> str:
        """Which entry of a multi-provider credential file a cell running ``model`` will present.

        Empty for a harness whose credential file holds one login and names no providers, which
        is every harness here but prime-agent: there is nothing to select, so nothing to say.

        It matters wherever the file can hold several, because those files accumulate. prime's
        auth.json keeps an entry for every provider ever logged in, while a leg looks one of them
        up by id, so a check that judged all of them would refuse a cell over a credential the
        cell never reaches for. Declared on the harness rather than derived by a caller so the
        answer comes from the same place the launch flag does.
        """
        return ""

    def base_env(self) -> dict[str, str]:
        """Environment every invocation of this harness needs, credentials excluded.

        The default is a clean ``NODE_OPTIONS`` and nothing else, which every harness needs and
        which a subclass extends rather than replaces (``{**super().base_env(), ...}``). The
        credential probe uses this too, which matters more than it looks: a probe missing one of
        these fails for a reason that has nothing to do with the credential, and a negative
        control that fails for the wrong reason proves nothing.
        """
        return dict(BASE_ENV)

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
        effort: str = "",
    ) -> LaunchSpec:
        raise NotImplementedError

    def classify(
        self, *, returncode: int, stdout_path: Path, stderr_path: Path, timed_out: bool
    ) -> StopVerdict:
        raise NotImplementedError

    # Can the runner choose the session id before launch? Claude Code accepts one; codex and
    # prime-agent mint their own and announce it in the trace, so resuming those means reading
    # the id back rather than assuming it.
    pins_session_id: bool = False

    # HOME subtrees that hold this harness's recorded conversations, relative to the HOME.
    # Everywhere else they are session byproducts (the durable digest rightly ignores them, and
    # a cold eval copy leaves them behind), but a resumed eval_after fork cannot exist without
    # them: each task's home copy carries these subtrees so the harness can reopen the rollout's
    # terminal session inside the copy, where its own resume lookup goes searching for it.
    session_state_dirs: tuple[str, ...] = ()

    # How long an eval leg of this harness may keep running after its held-out task is sealed and
    # its stream has nothing left to give, or ``None`` for a harness whose leg the first reading
    # that finds it finished ends outright, with no wait at all.
    #
    # A harness that ends its own legs needs a little of this. A leg does not end the instant the
    # row seals: it writes its last message, closes its session, and exits. Claude Code and codex
    # take 8 to 25 seconds over that, so at two minutes the watchdog never fires for them and
    # their legs still end on their own terms, which is what keeps the stopping record comparable
    # across harnesses. A harness that never ends a leg by choosing to has no such wrap-up to
    # protect, and every second it is given is spent on turns nobody will read.
    eval_drain_grace_s: float | None = 120.0

    # Does this harness's trace say which model answered? Declared rather than inferred from an
    # empty list, because the two mean opposite things: a harness that reports models and
    # returned none had none answer, while a harness that reports none has told us nothing. The
    # manifest published the empty list for both and read as the first.
    reports_observed_models: bool = False

    # How this harness is told the cell's reasoning effort, empty when it has no such control.
    # The manifest records the requested effort either way, and records separately whether it
    # was applied: every prime_agent cell asks for xhigh and prime-agent has no effort control,
    # so a manifest that reported the request as the setting was claiming a controlled variable
    # the run did not have.
    effort_flag: str = ""

    def session_id_from_trace(self, trace_path: Path) -> str | None:
        """The session the leg actually ran under, read off its trace.

        This is what a resume has to target. A runner-chosen id would be wrong for any harness
        that mints its own, and resuming the wrong id starts a fresh session that has lost
        everything the rollout had built up in context.
        """
        return None

    def session_transcript(self, home: Path, session_id: str) -> Path | None:
        """The recorded conversation for ``session_id`` under ``home``, proven resumable.

        Each harness resolves this the way its own resume lookup resolves it, and then
        requires the minimum its pinned CLI requires before reopening the file, because
        existence is not resumability: an empty or crashed file whose name carries the id
        passes any filename test, and all three CLIs refuse exactly that file (observed, see
        ``docs/harness-autonomy.md``). Without the validation the refusal arrives per task,
        after every fork's copy, stream, and container are already paid for; with it, the one
        pre-fan-out check refuses the phase instead. Identity is part of the minimum: the
        file must name this session in its own recorded metadata, so a file that merely wears
        the id in its filename cannot stand in for the conversation.

        The base harness records no sessions, so there is nothing to resolve.
        """
        return None

    def observed_models(self, trace_path: Path) -> list[str]:
        """Model identifiers the trace shows actually answered.

        The scope asks the manifest to record which model answered rather than which one was
        requested. A version probe cannot say that; the trace can, when the harness reports it.
        Returning nothing is honest for a harness that does not, and
        ``reports_observed_models`` is what tells a reader which of the two an empty list means.
        """
        return []

    # ----- shared helpers -----

    def _timed_out_verdict(self) -> StopVerdict:
        """The verdict for a leg the runner ended at its budget.

        Identical for every harness because the decision was the runner's, not the agent's: a
        leg the runner cut off is a ``LEG_TIMEOUT`` and never a chosen stop, whatever the trace
        happens to hold. Every ``classify`` returns this first when ``timed_out`` is set.
        """
        return StopVerdict(StopKind.LEG_TIMEOUT, "the runner ended the leg at its budget")

    def drained_verdict(self, *, grace_s: float | None) -> StopVerdict:
        """The verdict for an eval leg the runner ended after its work was already finished.

        Identical for every harness, and never overridden, for the same reason the timeout
        verdict is: the decision was the runner's. What it records is that the held-out task was
        sealed, the stream had nothing remaining and nothing in flight, and the harness was still
        running ``grace_s`` later, or was still running when it was first looked at where the
        grace is ``None``.

        It is a third kind rather than either of the two it sits between, and both exclusions
        matter. It is not a chosen stop, because the agent chose nothing here. It is not a leg
        timeout, because the leg did not exhaust its budget and calling it one would hide the
        finding inside a number the operator picked. The finding is that a harness launched
        autonomously with no quality gate has no terminal condition of its own, and it stays
        legible in the record exactly because this kind is its own.
        """
        return StopVerdict(
            StopKind.DRAINED,
            "the task was sealed and the stream drained; the runner ended a leg that did not end",
            {"grace_s": grace_s},
        )

    def operator_verdict(self, *, request: dict[str, Any]) -> StopVerdict:
        """The verdict for a leg an operator asked the runner to end.

        Identical for every harness, and never overridden, for the same reason the timeout and
        drain verdicts are: the decision was neither the agent's nor the harness's. It is its own
        kind rather than any of the three it sits between, and each exclusion carries a fact a
        reader needs. It is not a chosen stop, because the agent chose nothing and the stopping
        metrics must not count it. It is not a leg timeout, because the budget did not run out and
        reading it as one would say the harness sustained the whole clock. It is not a usage-limit
        suspension, because nothing is waiting for a window to reopen: the run is over, and what a
        continuation would resume is a treatment the operator deliberately ended.

        What it leaves true is the thing the ending exists for. An operator-ended rollout has a
        real terminus, being the agent's state at the moment it was stopped, so the run stays
        bookendable and the shorter treatment is a fact the artifact states rather than a record
        nobody can produce.

        ``request`` is the operator's own record of the ask, carried through so the leg says who
        ended it and why in the same place it says how.
        """
        return StopVerdict(
            StopKind.OPERATOR,
            "an operator asked for this run to end; the runner ended the leg through its "
            "normal path so the run's records were written",
            dict(request),
        )

    def no_progress_verdict(self, *, bound_s: float, silent_s: float) -> StopVerdict:
        """The verdict for a rollout leg the runner ended for showing no progress anywhere.

        Identical for every harness, and its own kind, because what it records is neither the
        agent's decision nor the budget's arrival. The condition is an absence across every source
        a working leg writes to (see :func:`shobench.runner.rollout_progress`), held for the
        cell's own bound while the container was alive, which is why it is not read as a chosen
        stop and not read as a timeout.

        Both numbers are here rather than only the bound, because the bound is the rule and the
        silence is what actually happened: a leg ended a second past its bound and a leg that had
        been silent for twice it are different observations about the same rule.
        """
        return StopVerdict(
            StopKind.STALLED,
            "no trace record, sealed row, session artifact or /work change for the cell's "
            "no-progress bound; the runner ended a leg that was producing nothing",
            {"bound_s": bound_s, "silent_s": silent_s},
        )

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


def stderr_evidence(path: Path, *, matched: Sequence[str] = ()) -> dict[str, Any]:
    """What a published verdict may say about a leg's stderr: everything except its bytes.

    A 2KB tail of the harness's own output used to be lifted in here, and it was the one thing a
    verdict carried that nothing could vouch for. The credential file a harness rewrites mid-leg
    is the case that settles it. A leg is one invocation lasting up to eight hours, so a
    file-backed OAuth client refreshes inside it more than once: a token minted an hour in, read
    by an ordinary config inspection, and overwritten by the next refresh is in the stderr of
    that leg and in no file anything can read afterwards. Redaction performed when the leg ends
    can only name what the file holds then, so a tail lifted from those bytes is a tail nothing
    downstream can promise is clean, and the verdict copies it into legs.json, the suspension
    record, and the published results.

    So the bytes are not lifted at all. The raw stderr stays in the run directory beside the
    trace, which is the operator's own data on the operator's own machine and where a failure is
    actually diagnosed; what crosses into a published artifact is a description of that file.
    The name is the runner's own stem, the counts and the digest are computed from the bytes
    rather than taken from them, and ``matched`` is whichever of the caller's own marker strings
    the classification found. Every field is a number, a runner-chosen name, or a literal from
    this repository's source, so no value here can be a secret whatever the harness wrote.

    The digest is what keeps the record checkable, which is the property the tail used to serve:
    a reader with the run directory can prove the local file is the one this verdict was
    computed from, rather than being asked to trust it.
    """
    described: dict[str, Any] = {
        "file": path.name,
        "bytes": None,
        "lines": None,
        "sha256": None,
        "matched": list(matched),
    }
    size, lines = 0, 0
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                size += len(chunk)
                lines += chunk.count(b"\n")
                digest.update(chunk)
    except OSError:
        # A leg that wrote no stderr file at all, or one this cannot read. Left as the absence it
        # is rather than reported as an empty file, which is a different fact about the run.
        return described
    return {**described, "bytes": size, "lines": lines, "sha256": digest.hexdigest()}


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
    "BASE_ENV",
    "Harness",
    "LaunchSpec",
    "StopKind",
    "StopVerdict",
    "UsageLimitRule",
    "jsonl_events",
    "stderr_evidence",
    "tail",
]
