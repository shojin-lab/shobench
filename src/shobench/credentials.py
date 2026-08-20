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
import math
import subprocess
import time
from collections.abc import Sequence
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
    # A credential the harness reads from a file rather than the environment. The runner copies
    # it from the operator's own home into the cell's isolated one, because the login that
    # mints it is a browser flow that cannot run in a container. Copying it is not a weakening
    # of the isolation: the cell still gets its own copy in its own HOME, and the negative
    # control still has to fail against a corrupted one before the real one is trusted.
    seed_from: str = ""
    seed_to: str = ""
    # Which credential file schema this harness reads. It decides two things that have to agree:
    # what counts as a usable file on the host, and what a structurally valid file carrying an
    # unusable secret looks like for the negative control. A negative control that fails because
    # the file was the wrong shape tests the parser rather than the credential, which is exactly
    # the thing the negative control exists to rule out.
    seed_schema: str = ""
    # Whether presenting this credential once can destroy the copy the host still holds. An OAuth
    # provider that rotates the refresh token on use mints a new pair and invalidates the pair it
    # was handed, and the positive check presents the credential inside a throwaway HOME: the
    # rotated pair is written into that copy and discarded with it, while the host file is left
    # holding a token the provider has already retired. Observed on the prime_agent subscription
    # arm, where a positive check reported the credential healthy and the real launch 76 seconds
    # later failed to authenticate on a fresh copy of the same host file.
    # What this buys is a GUARD in front of the probe rather than a replacement for it. The
    # rotation is not something presenting a credential does unconditionally: the pinned harness
    # redeems the refresh token only once the clock has passed the entry's expiry, and presents
    # the access token as it stands otherwise. So a static check that refuses anything with less
    # than PREFLIGHT_MIN_LIFETIME_S left keeps the probe out of the window where a refresh can
    # fire, and the live arm still has to authenticate for the cell to be trusted (see
    # :func:`_rotates_as_seeded` and :func:`static_positive_check`). The mark is the spec's half
    # of that question and not the whole of it, because one spec's file can hold either kind:
    # prime's auth.json carries an oauth entry or an api_key entry under a provider id, and an
    # api key is the same key after it is presented.
    # Only the arm this was observed on is marked: codex's auth.json states no expiry a reader
    # can check, so a guard on it would refuse nothing, and the environment-carried modes seed
    # no file to read at all.
    rotates_on_use: bool = False
    # What a human has to do when this mode's credential is not on the host yet. It is a hint
    # rather than a verdict: whether a cell is pending is decided by looking at the credential
    # now (see :func:`credential_available`), never by a field written months ago. A static
    # pending is how every prime_agent cell came to be permanently untrusted, including after
    # the login that was being waited for.
    pending_hint: str = ""
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
        seed_from="~/.codex/auth.json",
        seed_to=".codex/auth.json",
        seed_schema="codex_auth",
        pending_hint=(
            "run `codex login` on the host once and rerun; nothing else blocks on this."
        ),
        notes=(
            "The ChatGPT subscription login is a browser flow that cannot run in a container, "
            "so the auth.json minted once on the host is copied into the cell's isolated HOME. "
            "A file whose auth_mode is not chatgpt means the host is logged in with an API key "
            "and the cell would silently bill it, so the runner checks the mode before copying."
        ),
    ),
    ("codex", "api_key"): CredentialSpec(
        harness="codex",
        mode="api_key",
        env_names=("CODEX_API_KEY",),
        home_paths=(".codex/auth.json",),
        login_argv=("codex", "login", "--with-api-key"),
        login_reads_stdin=True,
        notes=(
            "CODEX_API_KEY supplies a key for a single non-interactive run and is supported "
            "only by `codex exec`, which is the only way this runner invokes codex, so no "
            "auth.json is minted. OPENAI_API_KEY alone does not authenticate codex; the "
            "alternative is piping the key into `codex login --with-api-key`, which writes "
            "~/.codex/auth.json, and the runner keeps that path available for a harness "
            "version that drops the variable."
        ),
    ),
    ("prime_agent", "subscription"): CredentialSpec(
        harness="prime_agent",
        mode="subscription",
        env_names=(),
        home_paths=(".prime/agent/auth.json",),
        login_argv=("prime-agent", "login"),
        seed_from="~/.prime/agent/auth.json",
        seed_to=".prime/agent/auth.json",
        seed_schema="prime_auth",
        rotates_on_use=True,
        pending_hint=(
            "prime-agent's auth.json exists but declares no provider, and a "
            "CLAUDE_CODE_OAUTH_TOKEN does not substitute for one. Observed: with that token "
            "present in the environment, prime-agent answered 'No API key found for "
            "anthropic. Use /login to log into a provider via OAuth or API key.' Run "
            "`prime-agent login` on the host once; the runner copies the resulting auth.json "
            "into each cell's isolated HOME and validates it like any other credential."
        ),
        notes=(
            "prime-agent reads auth.json in preference to the environment, so a cell HOME must "
            "start without a stale one or an injected key is shadowed. A pristine per-cell "
            "HOME is what this runner provides."
        ),
    ),
    ("prime_agent", "api_key"): CredentialSpec(
        harness="prime_agent",
        mode="api_key",
        env_names=("OPENAI_API_KEY", "ANTHROPIC_API_KEY"),
        home_paths=(".prime/agent/auth.json",),
        notes=(
            "Per-provider environment variables are the documented path in a container, "
            "because there is no keyring and auth.json would otherwise shadow them."
        ),
    ),
}

# Open questions about a leg that a credential's arrival does not answer. These are notes for a
# reader, never a gate: whether a cell may start is decided by the negative-control protocol
# against the credential that exists at the time, and nothing here can hold a cell back.
OPEN_QUESTIONS = {
    ("prime_agent", "subscription", "openai"): (
        "The OpenAI subscription path through prime_agent goes through an interactive login "
        "only the owner can perform, so this leg cannot be validated unattended the first time."
    ),
    ("prime_agent", "subscription", "anthropic"): (
        "Recorded in docs/harness-autonomy.md: a CLAUDE_CODE_OAUTH_TOKEN is minted for "
        "Anthropic's own client and may be refused when a third-party harness presents it, "
        "which would make this leg api spend rather than subscription allowance. The negative "
        "control cannot tell those apart, so a trusted verdict here is not a billing claim."
    ),
}


# A year, in milliseconds. The bogus prime credential is minted with an expiry this far out so
# the harness presents the unusable access token rather than trying to refresh it: a refresh
# failure and an authentication failure are both failures, but only the second is the one the
# negative control claims to have tested.
_BOGUS_LIFETIME_MS = 365 * 24 * 60 * 60 * 1000

# The OAuth providers prime-agent ships, used only when the host has no file to mirror.
_PRIME_DEFAULT_PROVIDERS = ("anthropic", "openai-codex")


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _prime_providers(body: Any) -> list[str]:
    """The providers a prime-agent auth.json declares, by name.

    The file is a map of provider id to a typed credential: ``{"type": "oauth", "access": ...,
    "refresh": ..., "expires": ...}`` or ``{"type": "api_key", "key": ...}``. A fresh install
    writes ``{}``, which is a file that exists and authenticates nothing, so counting providers
    rather than checking for the file is what tells those two apart. Names only; no value here
    is read or returned.
    """
    if not isinstance(body, dict):
        return []
    return sorted(
        name
        for name, entry in body.items()
        if isinstance(entry, dict) and entry.get("type") in ("oauth", "api_key")
    )


def _bogus_body(spec: CredentialSpec) -> Any:
    """A file with the right schema and a secret that cannot authenticate anywhere.

    Shape is the whole point. A negative control whose file the harness refuses to parse fails
    for a reason that has nothing to do with the credential, which means it proves nothing about
    isolation and quietly turns the positive check into the only evidence. So each schema is
    reproduced faithfully and only the secret is replaced.

    The prime arm mirrors the providers the host's own file declares, so the bogus arm and the
    real arm differ in exactly one thing: the secret. When there is no host file to mirror there
    is no credential to validate either, and the cell is pending before this is ever called; the
    default is there so a caller that reaches it still gets a well-formed file.
    """
    if spec.seed_schema == "codex_auth":
        return {
            "auth_mode": "chatgpt",
            "OPENAI_API_KEY": None,
            "tokens": {
                "id_token": BOGUS,
                "access_token": BOGUS,
                "refresh_token": BOGUS,
                "account_id": "shobench-negative-control",
            },
        }
    if spec.seed_schema == "prime_auth":
        providers = _prime_providers(_read_json(Path(spec.seed_from).expanduser()))
        expires = int(time.time() * 1000) + _BOGUS_LIFETIME_MS
        return {
            provider: {
                "type": "oauth",
                "access": BOGUS,
                "refresh": BOGUS,
                "expires": expires,
            }
            for provider in (providers or _PRIME_DEFAULT_PROVIDERS)
        }
    raise ValueError(f"no bogus credential shape for schema {spec.seed_schema!r}")


def describe_seed(spec: CredentialSpec, body: Any) -> str:
    """What a seeded file is, in terms a record can carry: names and modes, never values."""
    if spec.seed_schema == "codex_auth":
        mode = body.get("auth_mode") if isinstance(body, dict) else None
        return f"auth_mode={mode}"
    if spec.seed_schema == "prime_auth":
        return f"providers={_prime_providers(body)}"
    return "unknown schema"


# How much life a seeded OAuth credential must have left before an eval phase fans out.
#
# The fan-out is what makes this different from an ordinary expiry check. Every held-out task
# copies the one seeded file into a HOME of its own and starts within seconds of its siblings, so
# a credential that is about to expire is not one leg's problem: each copy reaches for a refresh
# of its own, and a provider that rotates the refresh token on use invalidates every copy that
# was a moment behind the first. One prime eval phase lost 119 of 120 legs to "No API key for
# provider" in bursts of near-identical ~19s failures, with the file's own expiry hours away, so
# the margin is a floor rather than the whole story: it rules out the credential that was already
# dead before the phase began, and the launch stagger is what addresses the rest.
#
# The same margin is what makes the positive probe safe to run against a rotating credential. The
# harness reaches for a refresh only once the clock has passed the entry's expiry, so an entry with
# a quarter of an hour left cannot be refreshed by a probe that finishes in seconds, and the rotated
# pair that broke a launch cannot be minted inside a HOME that is about to be deleted.
PREFLIGHT_MIN_LIFETIME_S = 900


def _oauth_expiry_ms(entry: Any) -> float | None:
    """The expiry an oauth entry declares, in epoch milliseconds, when it declares a usable one.

    ``None`` covers every way there is no number to compare: another credential type, a missing
    field, a string, a bool (which is an ``int`` in Python and is not an expiry anywhere), and a
    NaN or an infinity, none of which order against anything.
    """
    if not isinstance(entry, dict) or entry.get("type") != "oauth":
        return None
    expires = entry.get("expires")
    if isinstance(expires, bool) or not isinstance(expires, int | float):
        return None
    value = float(expires)
    return value if math.isfinite(value) else None


def _prime_entry_usable(provider: str, entry: Any, *, now: float | None = None) -> str:
    """What is wrong with this one provider's credential, or an empty string when nothing is.

    Judged against what the pinned prime-agent 0.7.1 actually does with the entry it looks up by
    provider id, read out of the bundle rather than assumed. An oauth entry's ``access`` IS the
    api key it presents (``getApiKey(credentials) { return credentials.access }``, for both the
    anthropic and openai-codex providers), its ``refresh`` is the only thing
    ``refreshToken(credentials.refresh)`` has to work with, and its ``expires`` is compared
    against the clock to choose between the two. An api_key entry is presented as it stands.

    So an empty field the CLI reads is a credential that authenticates nothing whatever the
    file's shape suggests, and an oauth entry carrying no comparable expiry is worse than it
    looks: ``Date.now() >= undefined`` is false, so the CLI never refreshes it and presents a
    token it cannot know is dead. Neither passes.
    """
    if not isinstance(entry, dict):
        return f"declares no credential for {provider}"
    kind = entry.get("type")
    if kind == "api_key":
        if not str(entry.get("key") or ""):
            return f"carries no key on its api_key entry for {provider}"
        return ""
    if kind != "oauth":
        return f"carries a credential for {provider} of type {kind!r}, which authenticates nothing"
    empty = [field for field in ("access", "refresh") if not str(entry.get(field) or "")]
    if empty:
        return f"carries an empty {' and '.join(empty)} on its oauth entry for {provider}"
    expiry = _oauth_expiry_ms(entry)
    if expiry is None:
        return f"carries no usable expiry on its oauth entry for {provider}"
    if expiry < ((time.time() if now is None else now) + PREFLIGHT_MIN_LIFETIME_S) * 1000:
        return (
            f"has under {PREFLIGHT_MIN_LIFETIME_S}s of life left on its oauth credential for "
            f"{provider}"
        )
    return ""


def preflight_seeded_credential(
    spec: CredentialSpec, home: Path, *, provider: str = "", now: float | None = None
) -> tuple[bool, str]:
    """Is the credential already seeded into this cell's HOME still usable, judged for free.

    Structure and expiry, never a provider call. A probe that authenticated for real would spend
    on every phase of every cell, and it would not settle the question anyway: the failure this
    exists in front of is a whole phase of legs refused at once, which one successful call a
    minute earlier cannot rule out.

    Generic by schema rather than by harness, which is the honest shape of it. A mode that
    carries its credential in the environment seeds no file, so there is nothing here to read
    without reading a secret, and it passes. A mode that seeds a FILE is both the one this can
    check and the one where the fan-out hands N homes copies of a single refreshable token.

    ``provider`` names the ONE entry of a multi-provider file this cell will present, and it is
    the difference between a useful check and a harmful one. prime's auth.json holds every
    provider ever logged in, while a leg looks up exactly one by id: a real gpt-terra home
    carries a live openai-codex login beside a long-expired anthropic one, and judging every
    entry refuses that cell over a credential it never reaches for. A caller that names no
    provider gets the weakest honest check instead, which is that the file declares something.

    Returns ``(ok, why_not)``. The reason names the file and what is wrong with it, never a value.
    """
    if not spec.seed_to:
        return True, ""
    path = home / spec.seed_to
    if not path.is_file():
        return False, f"the cell home has no {spec.seed_to} for its eval tasks to copy"
    body = _read_json(path)
    if body is None:
        return False, f"the cell home's {spec.seed_to} is not readable JSON"
    if spec.seed_schema == "prime_auth":
        if not provider:
            if not _prime_providers(body):
                return False, (
                    f"the cell home's {spec.seed_to} declares no provider, so it authenticates "
                    "nothing"
                )
            return True, ""
        wrong = _prime_entry_usable(provider, (body or {}).get(provider), now=now)
        if wrong:
            return False, (
                f"the cell home's {spec.seed_to} {wrong}, and {provider} is the provider this "
                "cell's model resolves to"
            )
        return True, ""
    if spec.seed_schema == "codex_auth":
        if body.get("auth_mode") != "chatgpt":
            return False, f"the cell home's {spec.seed_to} is not a chatgpt subscription login"
        tokens = body.get("tokens")
        if not isinstance(tokens, dict) or not tokens.get("access_token"):
            return False, f"the cell home's {spec.seed_to} carries no access token"
        return True, ""
    return True, ""


def refresh_seeded_credential(spec: CredentialSpec, home: Path, *, provider: str = "") -> str:
    """Replace ONE provider's entry in the cell HOME's credential when the host's outlives it.

    prime-agent specific, and it has to be: prime's auth.json is the only seeded schema that
    states its own expiry in a field this can read, so it is the only one where "fresher" is a
    computable fact rather than a guess. codex's auth.json carries no readable expiry (the
    lifetime lives inside the token), so nothing here touches it.

    Per provider rather than per file, which is a correctness property and not a refinement.
    These files hold several logins, and the file-level comparison this replaced (the soonest
    expiry on either side) is not an ordering: a host file whose anthropic entry outlives the
    cell's, but whose openai-codex entry is older, compared as fresher and overwrote a live
    openai-codex credential with a staler one. Replacing a single key cannot do that to any
    provider but the one asked for, and cannot do it to that one either, since the host's entry
    has to outlive what is there.

    The direction of the copy is the point. The cell HOME's entry is whatever the last leg's
    harness refreshed it to and is usually the newer of the two; the host's is newer only when
    something else on the host logged in or refreshed while this cell was running. Taking the
    later of the two is the only refresh available without a browser flow or a provider call,
    and a credential file is excluded from the durable digest either way, so this cannot touch
    what the cell measures.

    Returns a short description for the operator's log, never a value.
    """
    if spec.seed_schema != "prime_auth" or not spec.seed_from or not spec.seed_to or not provider:
        return ""
    target = home / spec.seed_to
    host = _read_json(Path(spec.seed_from).expanduser())
    seeded = _read_json(target)
    if not isinstance(host, dict) or not isinstance(seeded, dict):
        return ""
    host_expiry = _oauth_expiry_ms(host.get(provider))
    if host_expiry is None:
        return ""
    seeded_expiry = _oauth_expiry_ms(seeded.get(provider))
    if seeded_expiry is None and provider in seeded:
        # An entry this cannot compare (an api_key, or an oauth entry with no readable expiry)
        # is left where it is rather than replaced on a guess about which one is better.
        return ""
    if seeded_expiry is not None and seeded_expiry >= host_expiry:
        return ""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({**seeded, provider: host.get(provider)}), encoding="utf-8")
    target.chmod(0o600)
    return (
        f"replaced the {provider} entry of {spec.seed_to} from {spec.seed_from}, which carried "
        "the later expiry; every other provider's entry is untouched"
    )


def credential_available(spec: CredentialSpec) -> tuple[bool, str]:
    """Is this mode's credential on the host right now, and if not, what is missing.

    Asked fresh every time, because a mode that needed a human login yesterday does not need one
    today. The file half checks that the file both exists and declares something usable, since
    both harnesses write an empty well-formed file at first run and an empty file is the case a
    mere existence check gets wrong. The environment half is deliberately not checked here: a
    spec that also names environment variables reports on those separately, and a mode with no
    file credential at all has nothing for this to say.
    """
    if not spec.seed_from:
        return True, ""
    source = Path(spec.seed_from).expanduser()
    if not source.is_file():
        return False, f"no {spec.seed_from} on the host to seed the cell's HOME from"
    body = _read_json(source)
    if body is None:
        return False, f"{spec.seed_from} is not readable JSON"
    if spec.seed_schema == "prime_auth" and not _prime_providers(body):
        return False, f"{spec.seed_from} declares no provider, so it authenticates nothing"
    if spec.seed_schema == "codex_auth" and body.get("auth_mode") != "chatgpt":
        # An auth.json minted by an API-key login would bill the key rather than the
        # subscription, and the cell would report a mode it did not run under.
        return False, f"{spec.seed_from} is not a chatgpt subscription login"
    return True, ""


def seed_home(spec: CredentialSpec, home: Path, *, bogus: bool = False) -> str:
    """Place a file-based credential into the cell's isolated HOME.

    Returns a short description of what was placed, for the record. The bogus arm writes a
    structurally valid file carrying an unusable token, so the negative control tests the
    credential rather than the file's existence or its shape: a probe that failed merely because
    the file was missing, or because the harness could not parse it, would prove nothing about
    isolation.
    """
    if not spec.seed_from:
        return "no file-based credential for this mode"
    target = home / spec.seed_to
    target.parent.mkdir(parents=True, exist_ok=True)
    if bogus:
        body = _bogus_body(spec)
        target.write_text(json.dumps(body), encoding="utf-8")
        target.chmod(0o600)
        return f"wrote a bogus {spec.seed_to} ({describe_seed(spec, body)})"
    source = Path(spec.seed_from).expanduser()
    if not source.is_file():
        return f"MISSING: no {spec.seed_from} on the host to copy"
    raw = source.read_text(encoding="utf-8")
    target.write_text(raw, encoding="utf-8")
    target.chmod(0o600)
    return f"copied {spec.seed_from} ({describe_seed(spec, _read_json(target))})"


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
    # How the arm reached its answer: "probe" for a real call in an isolated HOME, "static" for a
    # credential read rather than presented, "static+probe" for a read that cleared the way for a
    # call. A reader who cannot tell those apart would take a structural pass for evidence that
    # the credential authenticates, which alone it is not, so the method travels in the record
    # beside the outcome and the detail says what each half found. The composite is spelled out
    # rather than folded into "probe" because the static half is what the arm did FIRST and is
    # the whole of the arm when it fails, so a record that named only the probe would describe a
    # call that never happened.
    method: str = "probe"

    def to_json(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "method": self.method,
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
#
# Two defences keep a failed run from reading as an authenticated one, and both are needed.
# The expectation never appears in the prompt, because a harness can echo its input into its
# own output stream and still exit 0 on an authentication failure (prime-agent does both).
# And the expectation is matched only against the model's own answer text, extracted from the
# harness's structured stream by ``_probe_answer``, because the surrounding stream carries
# incidental numerics (prime-agent stamps millisecond timestamps on every message, and every
# epoch millisecond in a three-hour window of late 2026 begins with this sum's digits).
PROBE_PROMPT = "Add 179312 and 41, then reply with only the digits of the sum."
PROBE_EXPECT = "179353"


def _probe_answer(harness: str, stdout: str) -> str:
    """The model's own words in the probe output, and nothing else.

    Each harness's stream wraps the answer differently, and everything outside the answer
    (event metadata, tool results, echoed input, timestamps) is exactly what the probe must
    not match against. An output this cannot parse yields no answer, so an unrecognized
    stream fails the probe rather than passing it.
    """
    texts: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if harness == "claude_code":
            # `claude -p --output-format json`: one terminal object whose `result` is the text.
            if event.get("type") == "result":
                texts.append(str(event.get("result") or ""))
        elif harness == "codex":
            # `codex exec --json`: item.completed events, the answer is an agent_message item.
            item = event.get("item")
            if (
                event.get("type") == "item.completed"
                and isinstance(item, dict)
                and item.get("type") == "agent_message"
            ):
                texts.append(str(item.get("text") or ""))
        elif harness == "prime_agent":
            # `prime-agent -p --mode json`: assistant text parts inside message_end, with
            # agent_end's message list as the shape that survives a cut stream. The same dual
            # read the harness uses for observed_models.
            messages = (
                event.get("messages")
                if event.get("type") == "agent_end"
                else [event.get("message")]
                if event.get("type") == "message_end"
                else []
            )
            for message in messages or ():
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    texts.append(content)
                for part in content if isinstance(content, list) else ():
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(str(part.get("text") or ""))
    return "\n".join(texts)


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
        return [
            "codex",
            "exec",
            "--json",
            "-m",
            model,
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            PROBE_PROMPT,
        ]
    if harness == "prime_agent":
        from shobench.harnesses.prime_agent import PrimeAgent

        # The same explicit provider the real launch passes, so the probe exercises the exact
        # resolution path a leg will use rather than a luckier or unluckier one.
        return [
            "prime-agent",
            "-p",
            "--mode",
            "json",
            "--provider",
            PrimeAgent.provider_for(model),
            PROBE_PROMPT,
            "--model",
            model,
        ]
    raise ValueError(f"no probe for harness {harness!r}")


def run_probe(
    *,
    harness: str,
    model: str,
    docker_args: list[str],
    image: str,
    env: dict[str, str],
    timeout_s: int = 300,
    credential_file: Path | None = None,
) -> ControlResult:
    """Run the credential probe once in an isolated HOME and say whether it authenticated.

    The harness's own base environment is merged in first, so the probe fails only for
    credential reasons. A probe that dies on a missing IS_SANDBOX would fail identically with
    a real credential and a bogus one, which would make the negative control prove nothing.

    ``credential_file`` is the file this probe was seeded with, named so the secrets inside it
    can be redacted out of the recorded detail alongside the environment values.
    """
    from shobench.harnesses import harness_for
    from shobench.redact import redactor_for

    argv = ["docker", *docker_args]
    for key, value in {**harness_for(harness).base_env(), **env}.items():
        argv += ["-e", f"{key}={value}"]
    argv += [image, *_probe_argv(harness, model)]
    started = time.time()
    result = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout_s, stdin=subprocess.DEVNULL
    )
    combined = f"{result.stdout}\n{result.stderr}"
    succeeded = result.returncode == 0 and PROBE_EXPECT in _probe_answer(harness, result.stdout)
    # The detail is the tail of the output, which is where an auth failure names itself. The
    # probe echoes a fixed string rather than its environment, so a credential should never
    # reach it, but this detail is written to a durable verdict file and a harness that dumps
    # its config on an auth failure would put one there. Redacted against exactly what this
    # probe was handed, which is the same rule the runner applies to a leg's trace.
    detail = "\n".join(combined.strip().splitlines()[-6:])[-1200:]
    detail = redactor_for(
        environment=env,
        credential_files=() if credential_file is None else (credential_file,),
    ).text(detail)
    return ControlResult(
        arm="",
        returncode=result.returncode,
        succeeded=succeeded,
        duration_s=time.time() - started,
        detail=detail,
    )


def _rotates_as_seeded(spec: CredentialSpec, home: Path, provider: str) -> bool:
    """Could presenting the ONE entry this cell will present destroy the host's copy of it.

    ``rotates_on_use`` says the spec CAN rotate; the seeded entry says whether this cell's could.
    prime's auth.json holds either kind under a provider id, and only oauth rotates: redeeming its
    refresh token mints a new pair and retires the pair it was handed, while an api_key entry is
    presented as it stands and is the same key afterwards. So an api_key entry has nothing to
    spend and needs no guard in front of its probe.

    A marked spec whose file states no entry type is guarded on the spec's word alone. The guard
    costs a cell nothing it was going to keep, since a credential too close to its expiry to guard
    is one the eval fan-out would refuse minutes later anyway.
    """
    if not spec.rotates_on_use:
        return False
    if spec.seed_schema != "prime_auth":
        return True
    body = _read_json(home / spec.seed_to)
    entry = body.get(provider) if isinstance(body, dict) else None
    return isinstance(entry, dict) and entry.get("type") == "oauth"


def static_positive_check(
    spec: CredentialSpec, home: Path, *, provider: str = "", now: float | None = None
) -> ControlResult:
    """The static half of the positive arm: read the file before anything presents it.

    Structure and expiry, against the same seeded file the cell's legs will copy, using the same
    reader the eval fan-out already trusts to refuse a phase before it spends
    (:func:`preflight_seeded_credential`). ``provider`` names the one entry the cell will present,
    resolved by the harness at the call site so this and the launch cannot drift.

    This is a guard and not a verdict. It answers one question, which is whether presenting the
    credential can rotate it, and the answer is no whenever the entry has more than
    ``PREFLIGHT_MIN_LIFETIME_S`` of life left: the harness redeems a refresh token only once the
    clock has passed the entry's expiry and presents the access token as it stands otherwise. So a
    passing check is not evidence that the credential authenticates, it is permission to go and
    find out, and the live probe that follows is what the cell is trusted on. A failing check is
    the whole of the arm, because the credential it was asked about is too close to spent to
    present safely and too close to spent to run a cell on either.

    ``returncode`` mirrors the outcome because no process ran to have one, and ``method`` says so.
    """
    started = time.time()
    ok, why_not = preflight_seeded_credential(spec, home, provider=provider, now=now)
    detail = (
        (
            f"checked statically first: {spec.seed_to} is well-formed and holds more than "
            f"{PREFLIGHT_MIN_LIFETIME_S}s of life, so presenting it cannot reach the refresh the "
            "harness runs on an expired credential, which would mint a new pair inside a "
            "throwaway HOME and leave the host holding a token the provider had retired"
        )
        if ok
        else f"checked statically first: {why_not}, so nothing was presented"
    )
    return ControlResult(
        arm="",
        returncode=0 if ok else 1,
        succeeded=ok,
        duration_s=time.time() - started,
        detail=detail,
        method="static",
    )


def validate_isolation(
    *,
    harness: str,
    mode: str,
    model: str,
    docker_args: list[str],
    image: str,
    environ: dict[str, str],
    home: Path,
) -> IsolationVerdict:
    """Run the negative control, then the positive check, and say whether the cell may start.

    Order matters. The negative control runs first because it is the one that can reveal a
    broken isolation, and a positive check that runs first would look fine either way.

    Every credential is proved by a live probe. What the credential decides is whether anything
    has to be checked before the probe may run, and that is settled by the entry the cell will
    present rather than by the spec or the harness's name. An entry that could rotate is read
    first: the harness refreshes it only once its expiry has passed, so a static check that finds
    real life left on it is what keeps the probe out of the window where authenticating would mint
    a new pair inside a HOME about to be deleted and retire the host's copy. The credential that
    fails that check never reaches the probe, and loses little by it: the margin is the one the
    eval fan-out refuses a whole phase on, so an entry too close to spent to present here is one
    this run could not have finished on anyway. The negative control is untouched either way,
    since it presents a bogus secret with a year of stated life and there is nothing there to
    spend or refresh.

    Whether the cell is pending is decided by looking at the credential now, not by a field on
    the spec. A static pending is how every prime_agent cell stayed untrusted forever: the check
    returned before it ever looked at the auth file, so the login it was waiting for could never
    clear it, and prime_agent is the harness this study most wants a trusted cell of.
    """
    from shobench.harnesses import harness_for

    spec = spec_for(harness, mode)
    present = {name: bool(environ.get(name)) for name in spec.env_names}

    available, why_not = credential_available(spec)
    if not available:
        return IsolationVerdict(
            harness=harness,
            mode=mode,
            trusted=False,
            reason=why_not,
            pending=spec.pending_hint or "log in on the host once, then rerun",
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

    seeded_bogus = seed_home(spec, home, bogus=True)
    negative = run_probe(
        harness=harness,
        model=model,
        docker_args=docker_args,
        image=image,
        env={name: BOGUS for name in spec.env_names},
        credential_file=(home / spec.seed_to) if spec.seed_to else None,
    )
    negative.arm = "negative_control"
    negative.detail = f"[{seeded_bogus}] {negative.detail}"
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

    seeded_real = seed_home(spec, home)
    # The one entry the cell will present, by the same resolution the launch passes as
    # --provider. Empty for every harness whose credential file holds a single login.
    provider = harness_for(harness).credential_provider(model)
    guard = (
        static_positive_check(spec, home, provider=provider)
        if _rotates_as_seeded(spec, home, provider)
        else None
    )
    passed = "bogus credential failed and the real credential succeeded in the same HOME"
    if guard is not None and not guard.succeeded:
        positive = guard
        failed = f"the real credential did not pass the check in front of its probe: {guard.detail}"
    else:
        real = {name: environ[name] for name in spec.env_names if environ.get(name)}
        positive = run_probe(
            harness=harness,
            model=model,
            docker_args=docker_args,
            image=image,
            env=real,
            credential_file=(home / spec.seed_to) if spec.seed_to else None,
        )
        failed = "the real credential did not authenticate in the isolated HOME"
        if guard is not None:
            positive.method = "static+probe"
            positive.detail = f"{guard.detail}. Then presented it: {positive.detail}"
            passed = (
                "bogus credential failed and the real credential authenticated in the same HOME, "
                "presented only after a static check found enough life left on it that presenting "
                "it could not rotate the host's copy"
            )
    positive.arm = "positive_check"
    positive.detail = f"[{seeded_real}] {positive.detail}"
    if not positive.succeeded:
        return IsolationVerdict(
            harness=harness,
            mode=mode,
            trusted=False,
            reason=failed,
            negative=negative,
            positive=positive,
            env_names_present=present,
        )
    return IsolationVerdict(
        harness=harness,
        mode=mode,
        trusted=True,
        reason=passed,
        negative=negative,
        positive=positive,
        env_names_present=present,
    )


def effective_mode(spec: CredentialSpec, home: Path, *, env_names: Sequence[str] = ()) -> dict:
    """What the cell is actually about to run under, read from what was seeded.

    The cell file says ``credential_mode = "subscription"`` and the manifest used to copy that
    string across as though saying it made it so. It does not: the mode is a property of the
    credential, and the credential is a file this runner placed a moment ago and can read. A
    codex auth.json minted by an API-key login bills the key; a prime auth.json whose provider
    entries are ``api_key`` rather than ``oauth`` does the same. Both would have been published
    as subscription runs, and subscription billing is the reason the scope allows no token
    ceiling, so a cell that quietly ran on API spend is a different experiment.

    Names and modes only. No value in the seeded file is read into the returned record.
    """
    body = _read_json(home / spec.seed_to) if spec.seed_to else None
    if spec.seed_schema == "codex_auth":
        effective = "subscription" if (body or {}).get("auth_mode") == "chatgpt" else "api_key"
    elif spec.seed_schema == "prime_auth":
        types = {
            (entry or {}).get("type")
            for entry in (body or {}).values()
            if isinstance(entry, dict)
        }
        effective = "subscription" if types == {"oauth"} else ("api_key" if types else "unknown")
    else:
        # Nothing was seeded, so the credential is whichever environment variable the spec
        # names, and the name is the mode. Reported as unknown when none of them arrived.
        effective = spec.mode if set(env_names) & set(spec.env_names) else "unknown"
    return {
        "requested": spec.mode,
        "effective": effective,
        "matches_requested": effective == spec.mode,
        "source": "seeded file" if spec.seed_to else "environment",
        "evidence": describe_seed(spec, body) if spec.seed_to else f"names={list(spec.env_names)}",
    }


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
    "PREFLIGHT_MIN_LIFETIME_S",
    "seed_home",
    "OPEN_QUESTIONS",
    "SPECS",
    "ControlResult",
    "CredentialSpec",
    "IsolationVerdict",
    "agent_env",
    "credential_available",
    "describe_seed",
    "effective_mode",
    "inventory",
    "preflight_seeded_credential",
    "refresh_seeded_credential",
    "run_probe",
    "spec_for",
    "static_positive_check",
    "validate_isolation",
    "write_verdict",
]
