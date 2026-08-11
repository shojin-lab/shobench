"""A skeleton harness. Copy this file, do not import it.

This module is not registered and nothing runs it. It exists to be copied to
``src/shobench/harnesses/<your_harness>.py`` as the starting point for a new harness, then
filled in. Every method the runner calls is present below with a one-line note on what it must
return and which of the three real harnesses to crib from. Read ``docs/adding-a-harness.md``
for the four steps around this file, and read ``docs/harness-autonomy.md`` for what a new
harness has to establish (and cite) before it can be trusted in a cell.

The base class ``shobench.harness.Harness`` already supplies the machinery every harness
shares: ``base_env`` defaults to a clean NODE_OPTIONS, ``_timed_out_verdict`` is the verdict
for a leg the runner cut off, and ``_match_usage_limit`` matches your ``usage_limit_rules``
against named evidence. Override only what your harness actually does differently.
"""

from __future__ import annotations

from pathlib import Path

from shobench.harness import (
    Harness,
    LaunchSpec,
    StopKind,
    StopVerdict,
    UsageLimitRule,
    stderr_evidence,
)
from shobench.harnesses._trace import _first_event_of_type, _last_event_of_type


class TemplateHarness(Harness):
    """One-paragraph statement of what makes this harness autonomous from its first turn and
    how it announces a stop. Say what has to be bypassed (permissions, sandbox), what the MCP
    wiring is, and where the standing instruction goes (system-prompt channel, or prepended to
    the turn when there is none). Crib the shape from ``ClaudeCode``, ``Codex``, or
    ``PrimeAgent`` depending on which your harness most resembles.
    """

    # The cell's harness name, and the registry key in __init__.py. Lower_snake_case.
    name = "template"

    # True only if the runner can choose the session id before launch (Claude Code accepts
    # --session-id). Leave False for a harness that mints its own id and announces it in the
    # trace; then session_id_from_trace below has to read it back. Crib: ClaudeCode sets True,
    # Codex and PrimeAgent leave it False.
    pins_session_id = False

    # The evidence-based ways this harness announces a usage-limit stop, each naming the artifact
    # it reads (``where``), the regex, and a citation for where the signal was established. The
    # ``where`` keys are matched against the dict you pass to _match_usage_limit in classify.
    # Only ever match these against a turn that actually FAILED, never a clean turn's own text.
    # Crib: Codex reads one terminal event; PrimeAgent reads a structured kind; ClaudeCode reads
    # the result json and text.
    usage_limit_rules = (
        UsageLimitRule(
            where="turn_failed",
            pattern=r"fill in the provider's own limit message",
            citation="where you established this, by source or by observation",
        ),
    )

    def base_env(self) -> dict[str, str]:
        # Environment every invocation needs, credentials excluded. Override ONLY to add to the
        # base (which already sets a clean NODE_OPTIONS): return {**super().base_env(), ...}. If
        # you add nothing, delete this method. Crib: ClaudeCode adds IS_SANDBOX, PrimeAgent adds
        # its MCP token variable, Codex overrides nothing.
        return super().base_env()

    def version_probe(self) -> list[str]:
        # A command that prints the installed harness version, for the manifest. Crib: any of
        # the three (["<binary>", "--version"]).
        raise NotImplementedError

    def model_probe(self) -> list[str] | None:
        # A command that reports which model the harness resolved, or None if it cannot be asked.
        # Crib: PrimeAgent returns one; ClaudeCode and Codex return None (they report the model
        # in the trace instead, via observed_models).
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
        # The full argv that makes this harness autonomous, plus the env and any files it needs.
        # This is genuinely per-harness; there is no shared body. Handle the three launch modes:
        # fresh, resume with an id, and resume without one (--continue). Put the standing
        # instruction in the harness's system-prompt channel if it has one, else prepend it to
        # the turn (Codex, PrimeAgent). Use config_files for a read-only config mount,
        # home_files for config the harness reads only from HOME and the runner refreshes every
        # leg, and home_seed_files for a HOME asset the agent is then free to change. Crib the
        # closest of the three.
        raise NotImplementedError

    def session_id_from_trace(self, trace_path: Path) -> str | None:
        # The session id the leg actually ran under, read off its trace, so a resume targets the
        # real session and not a fresh one. Only needed when pins_session_id is False. Crib:
        # Codex reads thread.started, PrimeAgent reads the session header. Delete if pins_session_id
        # is True.
        header = _first_event_of_type(trace_path, ("session",))
        return None if header is None else str(header.get("id") or "") or None

    def observed_models(self, trace_path: Path) -> list[str]:
        # Model identifiers the trace shows actually answered, or [] for a harness that does not
        # report them. Crib: ClaudeCode reads the result event's modelUsage keys.
        return []

    def classify(
        self, *, returncode: int, stdout_path: Path, stderr_path: Path, timed_out: bool
    ) -> StopVerdict:
        # How this leg ended, as a StopVerdict carrying its evidence. The rules are per-harness;
        # only the two edges are shared. Start with the timeout, then read your terminal signal.
        if timed_out:
            return self._timed_out_verdict()
        terminal = _last_event_of_type(stdout_path, ("your.terminal.event",))
        # A usage limit only counts on a turn that failed; guard the match on that, then:
        #   limit = self._match_usage_limit({"turn_failed": <the failure text>})
        #   if limit is not None:
        #       return limit
        evidence = {
            "returncode": returncode,
            "terminal_event": None if terminal is None else terminal.get("type"),
            # Never a tail of the harness's own output: a leg's stderr can hold a credential the
            # harness minted and overwrote inside the same leg, which nothing downstream can
            # name afterwards. Describe the file instead; the raw bytes stay in the run
            # directory. Read `stderr_evidence` before you put anything else from stderr here.
            "stderr": stderr_evidence(stderr_path),
        }
        # Return CHOSEN only for the agent's own clean stop; ERROR for a failure; and reserve
        # UNKNOWN for a terminal event you cannot read, never as a catch-all. Crib: Codex is the
        # simplest terminal-event classifier of the three.
        return StopVerdict(StopKind.CHOSEN, "the agent ended its turn", evidence)
