"""A prime_agent cell whose credential has arrived can be trusted, and its control is real.

Two failures met here, and the second only became reachable once the first was fixed.

The gate returned a static "needs a human login" for prime_agent before it ever looked at the
auth file, so the login it was waiting for could not clear it and every prime_agent cell was
permanently untrusted. Underneath that, the bogus arm wrote codex's ``auth_mode``/``tokens``
schema into prime-agent's ``auth.json`` whatever the harness, so the prime negative control
would have failed on the file's SHAPE. A negative control that fails because the harness cannot
parse the file proves nothing about isolation, which is the one thing it exists to prove.

Both credential schemas here are the real ones, read off the installed packages: codex's
``auth_mode`` with a ``tokens`` object, and prime-agent's map of provider id to a typed
credential (``{"type": "oauth", "access", "refresh", "expires"}`` or
``{"type": "api_key", "key"}``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from shobench import credentials
from shobench.credentials import (
    BOGUS,
    credential_available,
    seed_home,
    spec_for,
    validate_isolation,
)


def _host_prime_auth(tmp_path: Path, body: object) -> object:
    """Point the prime spec's ``seed_from`` at a host file this test controls."""
    path = tmp_path / "prime-auth.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


@pytest.fixture
def prime_spec(tmp_path, monkeypatch):
    """The real prime_agent subscription spec, sourced from a host file under the test's control."""

    def _with(body: object):
        path = _host_prime_auth(tmp_path, body)
        spec = spec_for("prime_agent", "subscription")
        patched = credentials.CredentialSpec(**{**spec.__dict__, "seed_from": str(path)})
        monkeypatch.setitem(credentials.SPECS, ("prime_agent", "subscription"), patched)
        return patched

    return _with


_LOGGED_IN = {
    "anthropic": {
        "type": "oauth",
        "access": "host-access-token-value",
        "refresh": "host-refresh-token-value",
        "expires": 4102444800000,
    }
}


def test_an_empty_prime_auth_file_is_not_a_credential(prime_spec) -> None:
    """A fresh prime-agent install writes ``{}``, which exists and authenticates nothing."""
    available, why_not = credential_available(prime_spec({}))

    assert not available
    assert "declares no provider" in why_not


def test_a_prime_cell_whose_login_has_happened_is_no_longer_pending(prime_spec, tmp_path) -> None:
    """The fix: once the auth file declares a provider, the cell reaches its negative control.

    Driven through the real ``validate_isolation``, with the probe replaced by something that
    records which arm ran. Before the fix it returned pending without running either arm, so a
    prime_agent cell could never be trusted however many times the owner logged in.
    """
    spec = prime_spec(_LOGGED_IN)
    arms: list[str] = []

    def fake_probe(*, harness, model, docker_args, image, env, timeout_s=300, credential_file=None):
        # The bogus arm fails, which is the outcome a correctly isolated HOME produces. prime's
        # credential rotates on use, so the positive arm is static and no probe runs for it.
        seeded = json.loads((tmp_path / "home" / spec.seed_to).read_text(encoding="utf-8"))
        is_bogus = seeded["anthropic"]["access"] == BOGUS
        arms.append("negative" if is_bogus else "positive")
        return credentials.ControlResult(
            arm="", returncode=0 if not is_bogus else 1, succeeded=not is_bogus, duration_s=0.0
        )

    original = credentials.run_probe
    credentials.run_probe = fake_probe
    try:
        verdict = validate_isolation(
            harness="prime_agent",
            mode="subscription",
            model="claude-opus-5",
            docker_args=[],
            image="img",
            environ={},
            home=tmp_path / "home",
        )
    finally:
        credentials.run_probe = original

    assert arms == ["negative"]
    assert verdict.trusted
    assert verdict.pending == ""


def test_a_rotating_credential_is_read_by_the_positive_check_and_never_presented(
    prime_spec, tmp_path, monkeypatch
) -> None:
    """The positive check may not spend the credential it is checking.

    Anthropic's OAuth mints a new refresh pair whenever one is redeemed and retires the pair it
    was handed. A positive check that authenticates for real does that redemption inside a HOME it
    is about to delete: the rotated pair dies with the sandbox and the host file is left holding a
    token the provider has already retired, so the check reports the credential healthy and the
    real launch a minute later cannot authenticate at all. The credential is read instead, and the
    record says so rather than letting a structural pass read as a live one.
    """
    spec = prime_spec(_LOGGED_IN)
    home = tmp_path / "home"
    arms: list[str] = []

    def fake_probe(*, harness, model, docker_args, image, env, timeout_s=300, credential_file=None):
        seeded = json.loads((home / spec.seed_to).read_text(encoding="utf-8"))
        assert seeded["anthropic"]["access"] == BOGUS, "a probe was handed the real credential"
        arms.append("negative")
        return credentials.ControlResult(arm="", returncode=1, succeeded=False, duration_s=0.0)

    monkeypatch.setattr(credentials, "run_probe", fake_probe)

    verdict = validate_isolation(
        harness="prime_agent",
        mode="subscription",
        model="claude-opus-5",
        docker_args=[],
        image="img",
        environ={},
        home=home,
    )

    assert arms == ["negative"]
    assert verdict.trusted
    recorded = verdict.to_json()["positive_check"]
    assert recorded["method"] == "static"
    assert "rotates it" in recorded["detail"]
    assert "reading the seeded file" in verdict.reason


def test_a_rotating_credential_with_no_life_left_still_fails_its_positive_check(
    prime_spec, tmp_path, monkeypatch
) -> None:
    """Static is not a pass: the same reader the eval fan-out gates on refuses a spent file."""
    spec = prime_spec(
        {
            "anthropic": {
                "type": "oauth",
                "access": "host-access-token-value",
                "refresh": "host-refresh-token-value",
                "expires": int(time.time() * 1000),
            }
        }
    )
    home = tmp_path / "home"

    monkeypatch.setattr(
        credentials,
        "run_probe",
        lambda **kwargs: credentials.ControlResult(
            arm="", returncode=1, succeeded=False, duration_s=0.0
        ),
    )

    verdict = validate_isolation(
        harness="prime_agent",
        mode="subscription",
        model="claude-opus-5",
        docker_args=[],
        image="img",
        environ={},
        home=home,
    )

    assert not verdict.trusted
    assert "life left" in verdict.reason
    assert spec.seed_to in verdict.reason


def test_a_credential_that_does_not_rotate_keeps_its_live_positive_check(
    tmp_path, monkeypatch
) -> None:
    """The static arm is scoped to the specs that need it; codex still proves it authenticates."""
    path = tmp_path / "codex-auth.json"
    path.write_text(
        json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "host-token-value"}}),
        encoding="utf-8",
    )
    spec = spec_for("codex", "subscription")
    patched = credentials.CredentialSpec(**{**spec.__dict__, "seed_from": str(path)})
    monkeypatch.setitem(credentials.SPECS, ("codex", "subscription"), patched)
    home = tmp_path / "home"
    arms: list[str] = []

    def fake_probe(*, harness, model, docker_args, image, env, timeout_s=300, credential_file=None):
        seeded = json.loads((home / patched.seed_to).read_text(encoding="utf-8"))
        is_bogus = seeded["tokens"]["access_token"] == BOGUS
        arms.append("negative" if is_bogus else "positive")
        return credentials.ControlResult(
            arm="", returncode=0 if not is_bogus else 1, succeeded=not is_bogus, duration_s=0.0
        )

    monkeypatch.setattr(credentials, "run_probe", fake_probe)

    verdict = validate_isolation(
        harness="codex",
        mode="subscription",
        model="gpt-5.6-terra",
        docker_args=[],
        image="img",
        environ={},
        home=home,
    )

    assert arms == ["negative", "positive"]
    assert verdict.trusted
    assert verdict.to_json()["positive_check"]["method"] == "probe"


def test_a_prime_cell_with_no_login_is_pending_with_something_to_do(prime_spec, tmp_path) -> None:
    """The other side of the same gate: no provider, no cell, and a hint that says why."""
    prime_spec({})

    verdict = validate_isolation(
        harness="prime_agent",
        mode="subscription",
        model="claude-opus-5",
        docker_args=[],
        image="img",
        environ={},
        home=tmp_path / "home",
    )

    assert not verdict.trusted
    assert "declares no provider" in verdict.reason
    assert "prime-agent login" in verdict.pending


def test_the_bogus_prime_credential_is_the_shape_prime_agent_reads(prime_spec, tmp_path) -> None:
    """Right schema, wrong secret: the control tests the credential and not the parser."""
    spec = prime_spec(_LOGGED_IN)
    home = tmp_path / "home"

    seed_home(spec, home, bogus=True)

    body = json.loads((home / spec.seed_to).read_text(encoding="utf-8"))
    # The same provider the host is logged in to, so the two arms differ in the secret alone.
    assert list(body) == ["anthropic"]
    entry = body["anthropic"]
    assert entry["type"] == "oauth"
    assert entry["access"] == BOGUS and entry["refresh"] == BOGUS
    # Unexpired, so the harness presents the unusable token rather than failing on a refresh.
    assert entry["expires"] > time.time() * 1000
    # Not codex's schema, which is what used to be written here whatever the harness.
    assert "auth_mode" not in body and "tokens" not in body


def test_the_bogus_codex_credential_is_still_codex_shaped(tmp_path) -> None:
    """The codex arm keeps the schema codex reads; the fix is per-harness, not a replacement."""
    spec = spec_for("codex", "subscription")
    home = tmp_path / "home"

    seed_home(spec, home, bogus=True)

    body = json.loads((home / spec.seed_to).read_text(encoding="utf-8"))
    assert body["auth_mode"] == "chatgpt"
    assert body["tokens"]["access_token"] == BOGUS


def test_a_codex_auth_file_from_an_api_key_login_is_refused(tmp_path, monkeypatch) -> None:
    """An api-key auth.json would bill the key while the cell reported a subscription run."""
    path = tmp_path / "codex-auth.json"
    path.write_text(json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": "k"}), encoding="utf-8")
    spec = spec_for("codex", "subscription")
    patched = credentials.CredentialSpec(**{**spec.__dict__, "seed_from": str(path)})

    available, why_not = credential_available(patched)

    assert not available
    assert "not a chatgpt subscription login" in why_not


def test_a_seeded_file_is_described_by_shape_and_never_by_value(prime_spec, tmp_path) -> None:
    """What the manifest records about a credential: which providers, never which secrets."""
    spec = prime_spec(_LOGGED_IN)

    described = seed_home(spec, tmp_path / "home")

    assert "providers=['anthropic']" in described
    assert "host-access-token-value" not in described


def test_the_probe_expectation_is_not_derivable_from_the_prompt() -> None:
    """The invariant the negative control rests on: an echo of the prompt can never pass.

    prime-agent echoes its input into its json event stream and exits 0 on an authentication
    failure, so an expectation that appears in the prompt would make a 401 read as a working
    credential, and the negative control would report the HOME isolation broken.
    """
    assert credentials.PROBE_EXPECT not in credentials.PROBE_PROMPT


def _run_probe_with(monkeypatch, harness: str, stdout: str, returncode: int = 0):
    class _Result:
        pass

    _Result.returncode = returncode
    _Result.stdout = stdout
    _Result.stderr = ""
    monkeypatch.setattr(credentials.subprocess, "run", lambda *a, **k: _Result())
    return credentials.run_probe(
        harness=harness,
        model="claude-opus-5",
        docker_args=["run", "--rm"],
        image="img",
        env={},
    )


def test_a_prime_401_stream_with_echo_and_timestamps_is_not_a_success(monkeypatch) -> None:
    """The two ways a failed exit-0 prime-agent run can carry the expectation anyway.

    The user message echoes the prompt, and every message is stamped with an epoch-millisecond
    timestamp. In a three-hour window of late 2026 those timestamps begin with the probe sum's
    digits, so a match against anything but the assistant's own words would deterministically
    pass the negative control and block a good cell.
    """
    stream = "\n".join(
        [
            json.dumps({"type": "session", "id": "s", "timestamp": 1793530000000}),
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": credentials.PROBE_PROMPT}],
                        "timestamp": 1793530000000,
                    },
                }
            ),
            json.dumps({"type": "auto_retry_end", "success": False, "attempt": 1}),
        ]
    )

    probe = _run_probe_with(monkeypatch, "prime_agent", stream)

    assert probe.returncode == 0
    assert not probe.succeeded


def test_a_prime_assistant_answer_is_a_success(monkeypatch) -> None:
    """The real shape a live prime-agent run emits, so the fix cannot overshoot."""
    stream = json.dumps(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": credentials.PROBE_EXPECT}],
                "timestamp": 1754955000000,
            },
        }
    )

    assert _run_probe_with(monkeypatch, "prime_agent", stream).succeeded


def test_a_prime_answer_surviving_only_in_agent_end_is_a_success(monkeypatch) -> None:
    """A cut stream keeps its messages only in the terminal agent_end event."""
    stream = json.dumps(
        {
            "type": "agent_end",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": credentials.PROBE_PROMPT}]},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": credentials.PROBE_EXPECT}],
                },
            ],
        }
    )

    assert _run_probe_with(monkeypatch, "prime_agent", stream).succeeded


def test_a_codex_answer_counts_only_from_its_agent_message(monkeypatch) -> None:
    """The expectation in a tool result or metadata is not the model answering."""
    answer = json.dumps(
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": credentials.PROBE_EXPECT},
        }
    )
    noise = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": f"t-{credentials.PROBE_EXPECT}"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "mcp_tool_call", "result": credentials.PROBE_EXPECT},
                }
            ),
        ]
    )

    assert _run_probe_with(monkeypatch, "codex", answer).succeeded
    assert not _run_probe_with(monkeypatch, "codex", noise).succeeded


def test_a_claude_answer_counts_only_from_its_result_text(monkeypatch) -> None:
    """Same scope rule for claude: the result field is the answer, other fields are not."""
    answer = json.dumps({"type": "result", "result": credentials.PROBE_EXPECT})
    noise = json.dumps({"type": "system", "session_id": credentials.PROBE_EXPECT})

    assert _run_probe_with(monkeypatch, "claude_code", answer).succeeded
    assert not _run_probe_with(monkeypatch, "claude_code", noise).succeeded


def test_an_unparseable_probe_output_fails_closed(monkeypatch) -> None:
    """A stream shape this code does not recognize blocks the cell rather than passing it."""
    assert not _run_probe_with(monkeypatch, "prime_agent", "plain text, no json events").succeeded
