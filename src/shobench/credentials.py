"""Per-cell credential isolation, and the negative control that proves it.

Ambient logins were shown to mask a bogus credential entirely in the earlier study: a probe
handed nonsense still worked, because the harness quietly fell back to the operator's own
session. A cell run under that condition is not the cell it claims to be. So the rule here is
that a credential is trusted only after a bogus one has been shown to FAIL in the same
isolated HOME. The negative control runs first; the positive check runs second; a cell whose
negative control passes is refused.

Subscription is the preferred mode for every v0 cell per the scope, and both modes are
supported per harness, so the mode a cell ran under is recorded rather than assumed.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A syntactically plausible value that cannot authenticate anywhere. It is the negative
# control's whole mechanism: if a probe carrying this succeeds, something other than the value
# under test authenticated the call.
BOGUS = "sk-shobench-negative-control-0000000000000000000000000000"


@dataclass(frozen=True)
class CredentialSpec:
    """What one (harness, mode) pair needs, named but never valued here."""

    harness: str
    mode: str
    # Environment variable names the harness reads. Values come from the operator's
    # environment at runtime and are never written to a committed file.
    env_names: tuple[str, ...]
    # Paths under the isolated HOME the harness stores credentials in, for the record and for
    # the isolation check.
    home_paths: tuple[str, ...]
    # A shell-free login step some harnesses need before the credential takes effect.
    login_argv: tuple[str, ...] = ()
    # Whether the login step reads the credential from stdin rather than from the environment.
    login_reads_stdin: bool = False
    # Set when only a human can complete this mode's login. The runner builds the check,
    # marks the cell pending, and does not block other cells on it.
    pending_reason: str = ""
    notes: str = ""


SPECS: dict[tuple[str, str], CredentialSpec] = {
    ("claude_code", "subscription"): CredentialSpec(
        harness="claude_code",
        mode="subscription",
        env_names=("CLAUDE_CODE_OAUTH_TOKEN",),
        home_paths=(".claude/.credentials.json",),
        notes=(
            "The token is passed as -e at runtime and never baked into an image. Claude Code "
            "reads it directly, so no login step runs inside the container."
        ),
    ),
    ("claude_code", "api_key"): CredentialSpec(
        harness="claude_code",
        mode="api_key",
        env_names=("ANTHROPIC_API_KEY",),
        home_paths=(),
        notes="Supported for completeness; subscription is the preferred mode for every cell.",
    ),
    ("codex", "subscription"): CredentialSpec(
        harness="codex",
        mode="subscription",
        env_names=(),
        home_paths=(".codex/auth.json",),
        login_argv=("codex", "login"),
        pending_reason=(
            "The ChatGPT subscription login is a browser flow, so the auth.json has to be "
            "minted once by a human and then copied into the cell's isolated HOME."
        ),
        notes=(
            "Once minted, the runner copies auth.json into the isolated HOME rather than "
            "re-running the browser flow, and the negative control still applies to the copy."
        ),
    ),
    ("codex", "api_key"): CredentialSpec(
        harness="codex",
        mode="api_key",
        env_names=("OPENAI_API_KEY",),
        home_paths=(".codex/auth.json",),
        login_argv=("codex", "login", "--with-api-key"),
        login_reads_stdin=True,
        notes=(
            "The environment variable alone does not authenticate codex; the key must be piped "
            "into `codex login --with-api-key`, which writes ~/.codex/auth.json."
        ),
    ),
    ("prime_agent", "subscription"): CredentialSpec(
        harness="prime_agent",
        mode="subscription",
        env_names=("CLAUDE_CODE_OAUTH_TOKEN",),
        home_paths=(".prime/",),
        login_argv=("prime-agent", "login"),
        notes=(
            "The Anthropic OAuth path through prime_agent is the validated one. The OpenAI "
            "subscription leg is separate and pending; see PENDING_LEGS."
        ),
    ),
    ("prime_agent", "api_key"): CredentialSpec(
        harness="prime_agent",
        mode="api_key",
        env_names=("OPENAI_API_KEY", "ANTHROPIC_API_KEY"),
        home_paths=(".prime/",),
        notes="Validated with an API key; the subscription variant is the preferred mode.",
    ),
}

# Legs that cannot be validated without a human. The runner reports these and skips them; it
# never blocks another cell on them.
PENDING_LEGS = {
    ("prime_agent", "subscription", "openai"): (
        "The OpenAI subscription path through prime_agent needs an interactive login only the "
        "owner can perform. The check is built and runs as soon as the credential exists."
    ),
}


def spec_for(harness: str, mode: str) -> CredentialSpec:
    key = (harness, mode)
    if key not in SPECS:
        raise ValueError(f"no credential spec for {harness!r} in {mode!r} mode")
    return SPECS[key]


def inventory(environ: dict[str, str]) -> dict[str, Any]:
    """Which credential names the environment supplies, by name only.

    Values are never read, printed, or returned. What comes back is the presence map a run
    plan needs in order to say which cells can start.
    """
    agent_names = sorted({name for spec in SPECS.values() for name in spec.env_names})
    server_names = [
        # The hle judge and the tau2 user simulator run on the serving side, never in the
        # agent container.
        "OPENAI_API_KEY",
        # The gated cais/hle dataset.
        "HF_TOKEN",
        # The optional live sink, broker-side by construction.
        "WANDB_API_KEY",
    ]
    return {
        "agent_side": {name: bool(environ.get(name)) for name in agent_names},
        "server_side": {name: bool(environ.get(name)) for name in server_names},
        "missing_agent_side": [n for n in agent_names if not environ.get(n)],
        "missing_server_side": [n for n in server_names if not environ.get(n)],
    }


@dataclass
class ControlResult:
    """What one arm of the negative-control protocol observed."""

    arm: str
    returncode: int
    succeeded: bool
    duration_s: float
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "returncode": self.returncode,
            "succeeded": self.succeeded,
            "duration_s": round(self.duration_s, 3),
            "detail": self.detail,
        }


@dataclass
class IsolationVerdict:
    """The protocol's outcome. ``trusted`` is the only state a cell may start in."""

    harness: str
    mode: str
    trusted: bool
    reason: str
    negative: ControlResult | None = None
    positive: ControlResult | None = None
    pending: str = ""
    env_names_present: dict[str, bool] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "harness": self.harness,
            "mode": self.mode,
            "trusted": self.trusted,
            "reason": self.reason,
            "pending": self.pending,
            "env_names_present": self.env_names_present,
            "negative_control": None if self.negative is None else self.negative.to_json(),
            "positive_check": None if self.positive is None else self.positive.to_json(),
        }


# A probe short enough to cost almost nothing and specific enough that a wrong answer is
# obvious. It exercises the credential and nothing else: no MCP server, no tools.
PROBE_PROMPT = "Reply with exactly: SHOBENCH-OK"
PROBE_EXPECT = "SHOBENCH-OK"


def _probe_argv(harness: str, model: str) -> list[str]:
    if harness == "claude_code":
        return [
            "claude",
            "-p",
            PROBE_PROMPT,
            "--model",
            model,
            "--tools",
            "",
            "--setting-sources",
            "",
            "--permission-mode",
            "bypassPermissions",
            "--output-format",
            "json",
        ]
    if harness == "codex":
        return ["codex", "exec", "--json", "-m", model, "--skip-git-repo-check", PROBE_PROMPT]
    if harness == "prime_agent":
        return ["prime-agent", "-p", "--mode", "json", PROBE_PROMPT, "--model", model]
    raise ValueError(f"no probe for harness {harness!r}")


def run_probe(
    *,
    harness: str,
    model: str,
    docker_args: list[str],
    image: str,
    env: dict[str, str],
    timeout_s: int = 180,
) -> ControlResult:
    """Run the credential probe once in an isolated HOME and say whether it authenticated."""
    argv = ["docker", *docker_args]
    for key, value in env.items():
        argv += ["-e", f"{key}={value}"]
    argv += [image, *_probe_argv(harness, model)]
    started = time.time()
    result = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout_s, stdin=subprocess.DEVNULL
    )
    combined = f"{result.stdout}\n{result.stderr}"
    succeeded = result.returncode == 0 and PROBE_EXPECT in combined
    # The detail is the tail of the output, which is where an auth failure names itself. It is
    # never a place a credential value appears: the probe echoes a fixed string, not its env.
    detail = "\n".join(combined.strip().splitlines()[-6:])[-1200:]
    return ControlResult(
        arm="",
        returncode=result.returncode,
        succeeded=succeeded,
        duration_s=time.time() - started,
        detail=detail,
    )


def validate_isolation(
    *,
    harness: str,
    mode: str,
    model: str,
    docker_args: list[str],
    image: str,
    environ: dict[str, str],
) -> IsolationVerdict:
    """Run the negative control, then the positive check, and say whether the cell may start.

    Order matters. The negative control runs first because it is the one that can reveal a
    broken isolation, and a positive check that runs first would look fine either way.
    """
    spec = spec_for(harness, mode)
    present = {name: bool(environ.get(name)) for name in spec.env_names}

    if spec.pending_reason:
        return IsolationVerdict(
            harness=harness,
            mode=mode,
            trusted=False,
            reason="mode needs a human login before it can be validated",
            pending=spec.pending_reason,
            env_names_present=present,
        )
    missing = [name for name, ok in present.items() if not ok]
    if spec.env_names and len(missing) == len(spec.env_names):
        return IsolationVerdict(
            harness=harness,
            mode=mode,
            trusted=False,
            reason=f"no credential in the environment; needs one of {list(spec.env_names)}",
            env_names_present=present,
        )

    negative = run_probe(
        harness=harness,
        model=model,
        docker_args=docker_args,
        image=image,
        env={name: BOGUS for name in spec.env_names},
    )
    negative.arm = "negative_control"
    if negative.succeeded:
        return IsolationVerdict(
            harness=harness,
            mode=mode,
            trusted=False,
            reason=(
                "the negative control SUCCEEDED with a bogus credential, so something other "
                "than the credential under test authenticated the call and this HOME is not "
                "isolated"
            ),
            negative=negative,
            env_names_present=present,
        )

    real = {name: environ[name] for name in spec.env_names if environ.get(name)}
    positive = run_probe(
        harness=harness, model=model, docker_args=docker_args, image=image, env=real
    )
    positive.arm = "positive_check"
    if not positive.succeeded:
        return IsolationVerdict(
            harness=harness,
            mode=mode,
            trusted=False,
            reason="the real credential did not authenticate in the isolated HOME",
            negative=negative,
            positive=positive,
            env_names_present=present,
        )
    return IsolationVerdict(
        harness=harness,
        mode=mode,
        trusted=True,
        reason="bogus credential failed and the real credential succeeded in the same HOME",
        negative=negative,
        positive=positive,
        env_names_present=present,
    )


def agent_env(harness: str, mode: str, environ: dict[str, str]) -> dict[str, str]:
    """The credential variables the agent container gets, by name from the spec.

    Only what the harness needs crosses into the container. The serving side's keys, the
    dataset token, and the wandb key stay out: the agent holds no broker credential and no
    mount of anything the broker writes.
    """
    spec = spec_for(harness, mode)
    return {name: environ[name] for name in spec.env_names if environ.get(name)}


def write_verdict(path: Path, verdict: IsolationVerdict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict.to_json(), indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "BOGUS",
    "PENDING_LEGS",
    "SPECS",
    "ControlResult",
    "CredentialSpec",
    "IsolationVerdict",
    "agent_env",
    "inventory",
    "run_probe",
    "spec_for",
    "validate_isolation",
    "write_verdict",
]
