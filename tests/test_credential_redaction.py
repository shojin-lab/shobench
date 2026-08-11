"""No artifact this runner publishes carries a credential value.

Every v0 harness has a shell and full internet, and the rollout's whole instruction is to
improve itself, so a session that inspects its own environment or its own auth file is an
ordinary thing for it to do. Whatever it prints goes into the leg's trace verbatim, and the
runner then lifts tails of that trace into the stop evidence, ``legs.json``, the manifest, and
the results JSON. These tests drive those real paths with a real token in the output and assert
that what lands on disk carries the marker instead.

Two things are deliberate about how they are written. The traces are the actual stream-json each
harness emits, so a redaction that only worked on a shape the harnesses never produce would fail
here. And each test that asserts something is redacted has a sibling assertion on an unwatched
runner, so a redactor that silently stopped matching would be visible rather than passing.
"""

from __future__ import annotations

import asyncio
import base64
import json
import urllib.parse
from pathlib import Path

import pytest

from shobench import runner
from shobench.config import load_cell_by_name, load_instruction
from shobench.containers import CellSandbox
from shobench.credentials import spec_for
from shobench.redact import MARKER, Redactor, redactor_for, secrets_in_file
from shobench.runner import RunContext, build_manifest, run_leg
from shobench.splits import load_split_by_name

# A value with the shape of the real thing: long, unbroken, and not a word. Never a real token;
# the point of exact-value redaction is that any string works as long as the runner provisioned
# it, so the test provisions this one.
TOKEN = "sk-ant-oat01-TESTONLYbutlongenoughtolookreal-0123456789"

_SMOKE_CELL = "smoke-automationbench-claude-code"


def _context(tmp_path: Path, *, redactor: Redactor) -> RunContext:
    cell = load_cell_by_name(_SMOKE_CELL)
    run_dir = tmp_path / "runs" / "run-1"
    sandbox = CellSandbox(run_id="run-1", home=run_dir / "home", workdir=run_dir / "work")
    sandbox.home.mkdir(parents=True, exist_ok=True)
    sandbox.workdir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        cell=cell,
        split=load_split_by_name(cell.split),
        instruction=load_instruction(cell.instruction_arm),
        harness=runner.harness_for(cell.harness),
        run_id="run-1",
        run_dir=run_dir,
        sandbox=sandbox,
        credentials={"CLAUDE_CODE_OAUTH_TOKEN": TOKEN},
        redactor=redactor,
    )


def _agent_that_dumps_its_environment(monkeypatch) -> None:
    """Stand in for the container with something that writes what a real leg's would.

    Only the daemon is replaced. The stand-in writes to the same file handles ``run_leg`` opened
    and emits real Claude Code stream-json, so everything downstream of the process (the
    redaction, the classification, the trace reads, the leg record) runs for real against bytes a
    harness actually produces.
    """
    import subprocess as real_subprocess

    def fake_run(argv, **kwargs):
        out = kwargs["stdout"]
        err = kwargs["stderr"]
        # An assistant turn in which the agent ran `env` and the tool result came back with the
        # token in it, then the result event the classifier reads.
        for event in (
            {"type": "system", "subtype": "init", "session_id": "sess-1"},
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "content": f"NODE_OPTIONS=\nCLAUDE_CODE_OAUTH_TOKEN={TOKEN}\n",
                        }
                    ],
                },
            },
            {
                "type": "result",
                "is_error": False,
                "subtype": "success",
                "terminal_reason": "completed",
                "api_error_status": None,
                "result": "read my environment",
                "modelUsage": {"claude-opus-5": {"inputTokens": 10}},
            },
        ):
            out.write(json.dumps(event) + "\n")
        err.write(f"warning: could not reach telemetry with token {TOKEN}\n")
        out.flush()
        err.flush()
        return real_subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)


def _run_one_leg(ctx: RunContext) -> runner.LegRecord:
    return run_leg(
        ctx,
        phase="rollout",
        leg=0,
        system_prompt="SYS",
        user_prompt="USR",
        session_id="sess-1",
        resume=False,
        timeout_s=60,
        task_idx=None,
        consumed_before=0,
    )


def test_a_leg_that_printed_its_token_leaves_no_copy_of_it_on_disk(tmp_path, monkeypatch) -> None:
    """The trace, the stderr file, the stop evidence, and legs.json all carry the marker."""
    _agent_that_dumps_its_environment(monkeypatch)
    ctx = _context(tmp_path, redactor=Redactor([TOKEN]))

    record = _run_one_leg(ctx)

    traces = ctx.run_dir / "rollout" / "traces"
    stream = (traces / "leg-0000.stream.jsonl").read_text(encoding="utf-8")
    stderr = (traces / "leg-0000.err.txt").read_text(encoding="utf-8")
    assert TOKEN not in stream and MARKER in stream
    assert TOKEN not in stderr and MARKER in stderr
    # The 2KB stderr tail the verdict carries is read after the redaction, which is the whole
    # reason the redaction happens where it does rather than at the end of the phase.
    assert TOKEN not in json.dumps(record.verdict.to_json())
    assert MARKER in record.verdict.evidence["stderr_tail"]

    legs = ctx.publish_json(ctx.run_dir / "legs.json", ctx.leg_records())
    assert TOKEN not in legs.read_text(encoding="utf-8")
    # The trace still parses as what it was: the redaction replaced a value, not the JSONL.
    assert record.observed_models == ["claude-opus-5"]
    assert record.session_id == "sess-1"


def test_without_the_redactor_the_same_leg_publishes_the_token(tmp_path, monkeypatch) -> None:
    """The mutation check for the test above: an unwatched cell leaks in all four places."""
    _agent_that_dumps_its_environment(monkeypatch)
    ctx = _context(tmp_path, redactor=Redactor())

    record = _run_one_leg(ctx)

    traces = ctx.run_dir / "rollout" / "traces"
    assert TOKEN in (traces / "leg-0000.stream.jsonl").read_text(encoding="utf-8")
    assert TOKEN in (traces / "leg-0000.err.txt").read_text(encoding="utf-8")
    assert TOKEN in record.verdict.evidence["stderr_tail"]
    legs = ctx.publish_json(ctx.run_dir / "legs.json", ctx.leg_records())
    assert TOKEN in legs.read_text(encoding="utf-8")


def test_the_results_json_is_redacted_through_its_own_write_path(tmp_path) -> None:
    """The one artifact assembled from every other is redacted once, at the end, as a whole."""
    from shobench.results import write_results

    ctx = _context(tmp_path, redactor=Redactor([TOKEN]))
    manifest = build_manifest(ctx, probes={"version": f"2.1.226 (token {TOKEN})"})
    stopping = {"stop_reason": "harness_error", "stop_evidence": {"stderr_tail": TOKEN}}

    path = write_results(
        tmp_path / "results.json",
        manifest=manifest,
        phases={"eval_before": [], "rollout": [], "eval_after": []},
        stopping=stopping,
        redact=ctx.redactor.json,
    )

    body = path.read_text(encoding="utf-8")
    doc = json.loads(body)
    assert TOKEN not in body
    assert doc["manifest"]["harness_probes"]["version"] == f"2.1.226 (token {MARKER})"
    assert doc["rollout"]["stopping"]["stop_evidence"]["stderr_tail"] == MARKER
    # Unredacted, the same call publishes both copies. Without this the assertion above would
    # pass on a results file that simply never contained the token.
    raw = write_results(
        tmp_path / "raw.json",
        manifest=manifest,
        phases={"eval_before": [], "rollout": [], "eval_after": []},
        stopping=stopping,
    )
    assert raw.read_text(encoding="utf-8").count(TOKEN) == 2


@pytest.mark.parametrize(
    "encode",
    [
        lambda v: v,
        lambda v: json.dumps(v)[1:-1],
        lambda v: urllib.parse.quote(v, safe=""),
        lambda v: base64.b64encode(v.encode()).decode(),
        lambda v: base64.urlsafe_b64encode(v.encode()).decode().rstrip("="),
    ],
    ids=["raw", "json-escaped", "url-encoded", "base64", "urlsafe-base64-unpadded"],
)
def test_the_cheap_ways_of_printing_a_secret_are_covered(encode) -> None:
    """An agent that pipes its environment through base64 is not doing anything exotic."""
    redactor = Redactor([TOKEN])
    assert redactor.text(f"leaked: {encode(TOKEN)}") == f"leaked: {MARKER}"


def test_short_and_spaced_values_are_never_treated_as_secrets() -> None:
    """A schema word or a sentence must not become a needle, or redaction corrupts artifacts."""
    redactor = Redactor(["oauth", "chatgpt", "a value with spaces in it that is long enough"])
    assert not redactor
    body = "auth_mode=chatgpt type=oauth"
    assert redactor.text(body) == body


def test_a_seeded_codex_auth_file_names_its_tokens_and_not_its_schema(tmp_path) -> None:
    """The real ``~/.codex/auth.json`` shape: the tokens are watched, ``chatgpt`` is not."""
    path = tmp_path / "auth.json"
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "OPENAI_API_KEY": None,
                "tokens": {
                    "id_token": f"id-{TOKEN}",
                    "access_token": f"access-{TOKEN}",
                    "refresh_token": f"refresh-{TOKEN}",
                    "account_id": "acct-0000",
                },
            }
        ),
        encoding="utf-8",
    )

    found = secrets_in_file(path)

    assert found == {f"id-{TOKEN}", f"access-{TOKEN}", f"refresh-{TOKEN}"}
    assert "chatgpt" not in found


def test_a_seeded_prime_auth_file_names_its_tokens_and_not_its_provider(tmp_path) -> None:
    """The real ``~/.prime/agent/auth.json`` shape: a provider map of typed credentials."""
    path = tmp_path / "auth.json"
    path.write_text(
        json.dumps(
            {
                "anthropic": {
                    "type": "oauth",
                    "access": f"access-{TOKEN}",
                    "refresh": f"refresh-{TOKEN}",
                    "expires": 1800000000000,
                },
                "openai-codex": {"type": "api_key", "key": f"key-{TOKEN}"},
            }
        ),
        encoding="utf-8",
    )

    found = secrets_in_file(path)

    assert found == {f"access-{TOKEN}", f"refresh-{TOKEN}", f"key-{TOKEN}"}
    assert "oauth" not in found and "anthropic" not in found


def test_a_credential_file_that_will_not_parse_is_not_fatal(tmp_path) -> None:
    """A shape this cannot read must not take the cell down; it protects what it can name."""
    path = tmp_path / "auth.json"
    path.write_text("not json at all", encoding="utf-8")

    redactor = redactor_for(
        environment={"CLAUDE_CODE_OAUTH_TOKEN": TOKEN}, credential_files=(path,)
    )

    assert redactor.text(TOKEN) == MARKER


def test_the_cell_redactor_watches_both_the_environment_and_the_seeded_file(tmp_path) -> None:
    """Built the way a cell builds it, from the two channels a credential actually arrives by."""
    ctx = _context(tmp_path, redactor=Redactor())
    spec = spec_for("codex", "subscription")
    seeded = ctx.sandbox.home / spec.seed_to
    seeded.parent.mkdir(parents=True, exist_ok=True)
    seeded.write_text(
        json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": f"file-{TOKEN}"}}),
        encoding="utf-8",
    )

    redactor = runner._cell_redactor(ctx, spec)

    assert redactor.text(f"env={TOKEN} file=file-{TOKEN}") == f"env={MARKER} file={MARKER}"


def test_the_version_probe_is_handed_no_credential(tmp_path, monkeypatch) -> None:
    """A version probe reports what the image installed, so it never needs a token.

    Driven through the real ``run_cell`` with no phases, because the property under test is
    which environment the call site passes, not what ``_probe`` does with one.
    """
    seen: list[dict[str, str]] = []

    def fake_probe(argv, *, image, sandbox, env, redactor=None):
        seen.append(dict(env))
        return "2.1.226 (Claude Code)"

    monkeypatch.setattr(runner, "_probe", fake_probe)
    monkeypatch.setattr(CellSandbox, "up", lambda self, **kw: None)
    monkeypatch.setattr(CellSandbox, "down", lambda self: None)
    cell = load_cell_by_name(_SMOKE_CELL)

    asyncio.run(
        runner.run_cell(
            cell,
            load_split_by_name(cell.split),
            runs_dir=tmp_path / "runs",
            results_dir=tmp_path / "results",
            credentials={"CLAUDE_CODE_OAUTH_TOKEN": TOKEN},
            phases=(),
            capture_egress=False,
        )
    )

    # claude_code has no model probe, so the version probe is the only one, and it gets nothing.
    assert seen == [{}]
