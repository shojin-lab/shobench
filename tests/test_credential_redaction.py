"""No artifact this runner publishes carries a credential value.

Every v0 harness has a shell and full internet, and the rollout's whole instruction is to
improve itself, so a session that inspects its own environment or its own auth file is an
ordinary thing for it to do. Whatever it prints goes into the leg's trace verbatim, and the
runner writes the manifest, ``legs.json``, the suspension record and the results JSON out of
what it read there. These tests drive those real paths with a real token in the output and
assert what lands on disk.

Two mechanisms are under test and they are not equally strong, which is why the assertions
differ. Redaction replaces values the runner can name, and it is what keeps the operator's own
local trace clean; it cannot name a value a harness minted and overwrote inside one leg. The
publication boundary is the other: a verdict quotes no byte a leg wrote, so a value nothing can
name still cannot reach a published file. The tests below say which of the two each one is
about, and the ones about redaction have a sibling assertion on an unwatched runner so a
redactor that silently stopped matching would be visible rather than passing.

The traces are the actual stream-json each harness emits, so a redaction that only worked on a
shape the harnesses never produce would fail here.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import shutil
import time
import urllib.parse
from collections.abc import Callable
from pathlib import Path

import pytest

from shobench import runner
from shobench.config import load_cell_by_name, load_instruction
from shobench.containers import CellSandbox
from shobench.credentials import spec_for
from shobench.harness import stderr_evidence
from shobench.redact import MARKER, Redactor, redactor_for, secrets_in_file
from shobench.results import write_results
from shobench.runner import RunContext, build_manifest, run_leg
from shobench.splits import load_split_by_name

# A value with the shape of the real thing: long, unbroken, and not a word. Never a real token;
# the point of exact-value redaction is that any string works as long as the runner provisioned
# it, so the test provisions this one.
TOKEN = "sk-ant-oat01-TESTONLYbutlongenoughtolookreal-0123456789"

_SMOKE_CELL = "smoke-automationbench-claude-code"


class _FinishedProcess:
    """What the leg supervisor needs of a process that has already done everything it will do.

    The stand-ins below write their whole trace while ``Popen`` is being called, so by the time
    the supervisor waits there is nothing left to run and ``wait`` answers at once with the exit
    status the real docker client would have carried. ``kill`` is a failed assertion rather than
    a no-op: a supervisor reaching for it here would be ending a leg that had already ended.
    """

    def __init__(self, argv: list[str], returncode: int) -> None:
        self.args = argv
        self.returncode = returncode
        self.stdin = None

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def kill(self) -> None:
        raise AssertionError("the supervisor killed a leg that had already exited")


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
    def fake_popen(argv, **kwargs):
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
        return _FinishedProcess(argv, 0)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)


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
    # The verdict is read after the redaction, which is the whole reason the redaction happens
    # where it does rather than at the end of the phase. It describes the stderr it classified
    # rather than quoting it, so what it carries is the file's size and not its bytes.
    assert TOKEN not in json.dumps(record.verdict.to_json())
    assert record.verdict.evidence["stderr"]["bytes"] == len(stderr.encode("utf-8"))

    legs = ctx.publish_json(ctx.run_dir / "legs.json", ctx.leg_records())
    assert TOKEN not in legs.read_text(encoding="utf-8")
    # The trace still parses as what it was: the redaction replaced a value, not the JSONL.
    assert record.observed_models == ["claude-opus-5"]
    assert record.session_id == "sess-1"


def test_without_the_redactor_the_same_leg_keeps_the_token_in_its_own_trace(
    tmp_path, monkeypatch
) -> None:
    """The mutation check for the test above: an unwatched cell's trace keeps the value.

    Which is what makes the assertions above about the trace mean something. The published file
    is a separate question and does not rest on this: the same unwatched leg still publishes no
    trace bytes, because the verdict never quotes any.
    """
    _agent_that_dumps_its_environment(monkeypatch)
    ctx = _context(tmp_path, redactor=Redactor())

    _run_one_leg(ctx)

    traces = ctx.run_dir / "rollout" / "traces"
    assert TOKEN in (traces / "leg-0000.stream.jsonl").read_text(encoding="utf-8")
    assert TOKEN in (traces / "leg-0000.err.txt").read_text(encoding="utf-8")
    legs = ctx.publish_json(ctx.run_dir / "legs.json", ctx.leg_records())
    assert TOKEN not in legs.read_text(encoding="utf-8")


def test_the_results_json_is_redacted_through_its_own_write_path(tmp_path) -> None:
    """The one artifact assembled from every other is redacted once, at the end, as a whole."""
    from shobench.results import write_results

    ctx = _context(tmp_path, redactor=Redactor([TOKEN]))
    manifest = build_manifest(ctx, probes={"version": f"2.1.226 (token {TOKEN})"})
    stopping = {"stop_reason": "harness_error", "stop_evidence": {"detail": TOKEN}}

    path = write_results(
        tmp_path / "results.json",
        manifest=manifest,
        phases={"eval_before": [], "rollout": [], "eval_after": []},
        stopping=stopping,
        heldout_ids=(),
        redact=ctx.redactor.json,
    )

    body = path.read_text(encoding="utf-8")
    doc = json.loads(body)
    assert TOKEN not in body
    assert doc["manifest"]["harness_probes"]["version"] == f"2.1.226 (token {MARKER})"
    assert doc["rollout"]["stopping"]["stop_evidence"]["detail"] == MARKER
    # Unredacted, the same call publishes both copies. Without this the assertion above would
    # pass on a results file that simply never contained the token.
    raw = write_results(
        tmp_path / "raw.json",
        manifest=manifest,
        phases={"eval_before": [], "rollout": [], "eval_after": []},
        stopping=stopping,
        heldout_ids=(),
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

    runner._watch_cell_credential(ctx, spec)

    assert ctx.redactor.text(f"env={TOKEN} file=file-{TOKEN}") == f"env={MARKER} file={MARKER}"
    # And it keeps the path, because the file is the harness's to rewrite while the cell runs.
    assert ctx.credential_home_paths == (spec.seed_to,)


# ----- a credential the harness mints after the cell started -----------------------------------
#
# The seeded value is not the only value a cell ever holds. Every file-backed OAuth client in v0
# refreshes an expired token and persists the new one back over the same file: prime-agent does
# it under a lock in AuthStorage, codex rewrites ~/.codex/auth.json with a new access token and
# last_refresh, and Claude Code mints its own .credentials.json from the OAuth token it is given.
# A redactor built once at seeding time cannot match any of those, so what a later shell or
# config inspection printed used to go through every redaction untouched.

_REFRESHED = "sk-ant-oat01-REFRESHEDbutlongenoughtolookreal-987654321"


def _claude_credentials(token: str) -> str:
    """The shape Claude Code writes into ``~/.claude/.credentials.json``."""
    return json.dumps(
        {
            "claudeAiOauth": {
                "accessToken": token,
                "refreshToken": f"refresh-{token}",
                "expiresAt": 1800000000000,
                "scopes": ["user:inference"],
                "subscriptionType": "max",
            }
        }
    )


def _agent_that_refreshes_its_token(monkeypatch, home: Path) -> None:
    """A leg that does what a refresh does: rewrite the auth file, then print what it now holds.

    The two halves are the whole failure. The value is minted after the cell built its redactor,
    and it reaches the trace the ordinary way, through an agent reading its own configuration.
    """
    def fake_popen(argv, **kwargs):
        credentials = home / ".claude" / ".credentials.json"
        credentials.parent.mkdir(parents=True, exist_ok=True)
        credentials.write_text(_claude_credentials(_REFRESHED), encoding="utf-8")
        out = kwargs["stdout"]
        err = kwargs["stderr"]
        for event in (
            {"type": "system", "subtype": "init", "session_id": "sess-1"},
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "content": credentials.read_text(encoding="utf-8")}
                    ],
                },
            },
            {
                "type": "result",
                "is_error": False,
                "subtype": "success",
                "terminal_reason": "completed",
                "api_error_status": None,
                "result": "read my own credentials file",
                "modelUsage": {"claude-opus-5": {"inputTokens": 10}},
            },
        ):
            out.write(json.dumps(event) + "\n")
        err.write(f"warning: refreshed the session token to {_REFRESHED}\n")
        out.flush()
        err.flush()
        return _FinishedProcess(argv, 0)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)


def test_a_token_minted_during_the_leg_is_redacted_like_the_one_the_cell_provisioned(
    tmp_path, monkeypatch
) -> None:
    """The refreshed value is covered, and so is the one the cell started with.

    Both halves matter. The new value is what everything after the refresh carries, and the old
    one is still in whatever was written before it, so a redactor that swapped rather than grew
    would trade one leak for another.
    """
    ctx = _context(tmp_path, redactor=Redactor())
    runner._watch_cell_credential(ctx, spec_for("claude_code", "subscription"))
    watched_before = ctx.redactor.count
    _agent_that_refreshes_its_token(monkeypatch, ctx.sandbox.home)

    record = _run_one_leg(ctx)

    traces = ctx.run_dir / "rollout" / "traces"
    stream = (traces / "leg-0000.stream.jsonl").read_text(encoding="utf-8")
    stderr = (traces / "leg-0000.err.txt").read_text(encoding="utf-8")
    for body in (stream, stderr, json.dumps(record.verdict.to_json())):
        assert _REFRESHED not in body
        assert TOKEN not in body
    assert MARKER in stream and MARKER in stderr
    # The provisioned value is still watched for alongside the minted one.
    assert ctx.redactor.count > watched_before
    assert ctx.redactor.text(TOKEN) == MARKER


def test_a_leg_whose_harness_keeps_no_credential_file_is_left_alone(tmp_path, monkeypatch) -> None:
    """The sibling assertion: with nothing to re-read, the refreshed value is on disk in full.

    Without it, the test above would pass against a fixture that never produced the leak, and a
    redaction that quietly stopped extending would look identical to one that worked.
    """
    ctx = _context(tmp_path, redactor=Redactor())
    _agent_that_refreshes_its_token(monkeypatch, ctx.sandbox.home)

    _run_one_leg(ctx)

    stream = (ctx.run_dir / "rollout" / "traces" / "leg-0000.stream.jsonl").read_text()
    assert _REFRESHED in stream


def test_an_eval_task_credential_is_read_before_its_home_is_discarded(
    tmp_path, monkeypatch
) -> None:
    """An eval task runs against a private copy of the HOME that is deleted the moment it ends.

    So a token refreshed inside that copy can be learned then or never, and a redactor that only
    looked at the cell's own HOME would never see it. The leg is handed the task's home exactly as
    the eval phase hands it one.
    """
    ctx = _context(tmp_path, redactor=Redactor())
    runner._watch_cell_credential(ctx, spec_for("claude_code", "subscription"))
    task_home = tmp_path / "eval_before" / "homes" / "task-00007"
    task_home.mkdir(parents=True)
    _agent_that_refreshes_its_token(monkeypatch, task_home)

    run_leg(
        ctx,
        phase="eval_before",
        leg=7,
        system_prompt="SYS",
        user_prompt="USR",
        session_id="sess-7",
        resume=False,
        timeout_s=60,
        task_idx=7,
        consumed_before=0,
        home=task_home,
        workdir=tmp_path / "work",
    )
    shutil.rmtree(task_home)  # what the phase does the moment the task is done

    trace = ctx.run_dir / "eval_before" / "traces" / "task-00007-leg-0007.stream.jsonl"
    assert _REFRESHED not in trace.read_text(encoding="utf-8")
    # And the value is watched from here on, for every artifact the cell has left to publish.
    assert ctx.redactor.text(_REFRESHED) == MARKER


def test_a_token_minted_outside_a_leg_is_covered_before_anything_is_published(
    tmp_path, monkeypatch
) -> None:
    """A refresh does not only happen inside a leg: the model probe authenticates, and so does
    anything else a harness leaves running. So the credential file is read once more immediately
    before the manifest and the results are written, and the count the manifest publishes is the
    one the whole run watched rather than the one it started with."""

    class _Observer:
        summary: dict = {}

        def stop(self) -> dict:
            return {}

    ctx = _context(tmp_path, redactor=Redactor())
    runner._watch_cell_credential(ctx, spec_for("claude_code", "subscription"))
    manifest = build_manifest(ctx, probes={"version": f"2.1.226 (token {_REFRESHED})"})
    watched_before = manifest["redaction"]["forms_watched"]
    # No leg runs here: the file simply holds a value now that it did not hold before.
    credentials = ctx.sandbox.home / ".claude" / ".credentials.json"
    credentials.parent.mkdir(parents=True, exist_ok=True)
    credentials.write_text(_claude_credentials(_REFRESHED), encoding="utf-8")

    results = asyncio.run(
        runner._run_phases(
            ctx,
            manifest=manifest,
            phases=(),
            results_dir=tmp_path / "results",
            observer=_Observer(),
        )
    )

    published = (ctx.run_dir / "manifest.json").read_text(encoding="utf-8")
    assert _REFRESHED not in published and MARKER in published
    assert _REFRESHED not in results.read_text(encoding="utf-8")
    assert json.loads(published)["redaction"]["forms_watched"] > watched_before


# ----- a credential the harness replaced inside the same leg -----------------------------------
#
# One leg is one invocation lasting up to eight hours, so a file-backed OAuth client refreshes
# inside it more than once. The value in the middle is the hard one: minted after the cell built
# its redactor, printed by an ordinary config inspection, and gone from the file before the leg
# ends, so a read taken when the process exits cannot name it and no redaction can replace it.
# Two things answer it, and the tests below separate them: the runner watches the file while the
# leg runs, which usually catches the generation; and a verdict quotes no byte a leg wrote, which
# holds whether or not the watcher caught it.

_INTERMEDIATE = "sk-ant-oat01-INTERMEDIATEbutlongenoughtolookreal-13579"


def _agent_that_overwrites_the_token_it_printed(
    monkeypatch, home: Path, *, settle: Callable[[], None] | None = None
) -> None:
    """A leg that refreshes twice: mint a token, print it, replace it, exit.

    ``settle`` is called between minting the first value and printing it, and is how a test says
    whether the runner gets a chance to observe that generation at all. Without it the two writes
    are microseconds apart, which is the case no poller can be relied on to catch.
    """
    def fake_popen(argv, **kwargs):
        credentials = home / ".claude" / ".credentials.json"
        credentials.parent.mkdir(parents=True, exist_ok=True)
        credentials.write_text(_claude_credentials(_INTERMEDIATE), encoding="utf-8")
        if settle is not None:
            settle()
        out, err = kwargs["stdout"], kwargs["stderr"]
        for event in (
            {"type": "system", "subtype": "init", "session_id": "sess-1"},
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "content": credentials.read_text(encoding="utf-8")}
                    ],
                },
            },
            {
                "type": "result",
                "is_error": True,
                "subtype": "error_during_execution",
                "terminal_reason": "error",
                "api_error_status": 500,
                "result": "read my own credentials file",
                "modelUsage": {"claude-opus-5": {"inputTokens": 10}},
            },
        ):
            out.write(json.dumps(event) + "\n")
        err.write(f"[debug] loaded session credentials {_INTERMEDIATE}\n")
        # The refresh that makes the value above unnameable: the file no longer holds it, and
        # nothing else on the host ever did.
        credentials.write_text(_claude_credentials(_REFRESHED), encoding="utf-8")
        err.write(f"[debug] refreshed session credentials to {_REFRESHED}\n")
        out.flush()
        err.flush()
        return _FinishedProcess(argv, 1)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)


def test_a_token_the_harness_replaced_mid_leg_reaches_no_published_artifact(
    tmp_path, monkeypatch
) -> None:
    """The value nothing can name, driven through the leg and the results file that follow it.

    The watcher is pinned out of the way so this is the case where redaction has already lost:
    the intermediate value is in the leg's stderr and in no file the runner can still read. What
    it must not be in is anything published, and that does not depend on naming it, because the
    verdict describes the stderr it classified rather than quoting it.
    """
    monkeypatch.setattr(runner, "CREDENTIAL_POLL_S", 3600.0)
    ctx = _context(tmp_path, redactor=Redactor())
    runner._watch_cell_credential(ctx, spec_for("claude_code", "subscription"))
    _agent_that_overwrites_the_token_it_printed(monkeypatch, ctx.sandbox.home)

    record = _run_one_leg(ctx)
    legs = ctx.publish_json(ctx.run_dir / "legs.json", ctx.leg_records())
    results = write_results(
        tmp_path / "results" / f"{ctx.cell.name}.json",
        manifest=build_manifest(ctx, probes={}),
        phases={"eval_before": [], "rollout": [], "eval_after": []},
        stopping={
            "stop_reason": "harness_error",
            "legs": [record.to_json()],
            "stop_evidence": record.verdict.to_json(),
        },
        heldout_ids=(),
        redact=ctx.redactor.json,
    )

    # The finished name, which is the artifact a reader of this cell reaches for.
    assert results.name == f"{ctx.cell.name}.json"
    for published in (
        json.dumps(record.verdict.to_json()),
        legs.read_text(encoding="utf-8"),
        results.read_text(encoding="utf-8"),
    ):
        assert _INTERMEDIATE not in published
        assert _REFRESHED not in published
        assert TOKEN not in published
    # The residual, asserted where it can be seen rather than described in a comment. The value
    # the harness replaced is still in the operator's own local trace, because by the time the
    # leg ended nothing could name it; the value it ended on was named and replaced there, which
    # is what shows the redaction ran at all.
    local = (ctx.run_dir / "rollout" / "traces" / "leg-0000.err.txt").read_text(encoding="utf-8")
    assert _INTERMEDIATE in local
    assert _REFRESHED not in local and MARKER in local


def test_the_runner_learns_a_generation_the_harness_later_overwrote(tmp_path, monkeypatch) -> None:
    """The other half: the file is read while the leg runs, not only when it ends.

    Same leg, same two writes, with the runner given the chance to observe the first generation.
    The local trace comes out clean as well, which is what the watching buys: the published side
    was already covered by the test above.
    """
    monkeypatch.setattr(runner, "CREDENTIAL_POLL_S", 0.02)
    ctx = _context(tmp_path, redactor=Redactor())
    runner._watch_cell_credential(ctx, spec_for("claude_code", "subscription"))

    def until_the_runner_has_read_it() -> None:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and ctx.redactor.text(_INTERMEDIATE) != MARKER:
            time.sleep(0.01)

    _agent_that_overwrites_the_token_it_printed(
        monkeypatch, ctx.sandbox.home, settle=until_the_runner_has_read_it
    )

    record = _run_one_leg(ctx)

    traces = ctx.run_dir / "rollout" / "traces"
    stream = (traces / "leg-0000.stream.jsonl").read_text(encoding="utf-8")
    local = (traces / "leg-0000.err.txt").read_text(encoding="utf-8")
    for body in (stream, local, json.dumps(record.verdict.to_json())):
        assert _INTERMEDIATE not in body
        assert _REFRESHED not in body
    assert MARKER in stream and MARKER in local


def test_no_harness_lifts_stderr_bytes_into_the_evidence_it_publishes(tmp_path) -> None:
    """The boundary itself, at every harness the runner can be asked to classify a leg of.

    A leg's stderr is the one artifact whose contents nothing can vouch for, so no verdict may
    quote it. The nonce stands in for whatever the harness printed; the sibling assertion on
    ``tail`` is what proves the bytes were sitting there to be lifted, so a verdict that dropped
    them did so rather than never having had them.
    """
    from shobench.harness import tail
    from shobench.harnesses import harness_for

    nonce = "stderr-nonce-that-no-verdict-may-quote-0123456789"
    traces = {
        # Each harness's own terminal shape, so classification takes its ordinary path rather
        # than the one an unreadable trace forces.
        "claude_code": [
            {
                "type": "result",
                "is_error": True,
                "subtype": "error_during_execution",
                "terminal_reason": "error",
                "api_error_status": 500,
                "result": "it went wrong",
            }
        ],
        "codex": [{"type": "thread.started", "thread_id": "t"}, {"type": "turn.failed"}],
        "prime_agent": [
            {"type": "session", "id": "s"},
            {"type": "agent_end", "messages": [{"role": "assistant", "stopReason": "error"}]},
        ],
    }
    for name, events in traces.items():
        stdout_path = tmp_path / f"{name}.stream.jsonl"
        stdout_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
        stderr_path = tmp_path / f"{name}.err.txt"
        stderr_path.write_text(f"harness noise\n{nonce}\nmore noise\n", encoding="utf-8")

        verdict = harness_for(name).classify(
            returncode=1, stdout_path=stdout_path, stderr_path=stderr_path, timed_out=False
        )

        assert nonce in tail(stderr_path)
        assert nonce not in json.dumps(verdict.to_json())
        # And what it says instead is enough to find those bytes and prove they are the ones.
        described = verdict.evidence["stderr"]
        assert described["file"] == stderr_path.name
        assert described["bytes"] == stderr_path.stat().st_size
        assert described["sha256"] == hashlib.sha256(stderr_path.read_bytes()).hexdigest()


def test_a_stderr_file_that_never_existed_is_described_as_absent(tmp_path) -> None:
    """An absent file and an empty one are different facts about a leg, and read as different."""
    written = tmp_path / "written-but-empty.err.txt"
    written.write_bytes(b"")

    absent = stderr_evidence(tmp_path / "never-written.err.txt")
    empty = stderr_evidence(written)

    assert absent["bytes"] is None and absent["sha256"] is None
    assert empty["bytes"] == 0 and empty["lines"] == 0


def test_extending_a_redactor_never_forgets_what_it_already_watched() -> None:
    """The property the cell's growing redactor rests on, at the unit it lives in."""
    first = Redactor([TOKEN])
    grown = first.extended([_REFRESHED])

    assert grown.text(f"{TOKEN} {_REFRESHED}") == f"{MARKER} {MARKER}"
    assert grown.count > first.count
    # A value it already watches changes nothing, and a value too short to be a secret is refused.
    assert grown.extended([TOKEN]) is grown
    assert grown.extended(["oauth"]) is grown
    # The original is untouched, which is what makes the swap safe while other threads redact.
    assert first.text(_REFRESHED) == _REFRESHED


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
