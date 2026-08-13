"""What the held-out fan-out costs when nothing goes wrong, and what it costs when one thing does.

Three mechanisms are under test here, and none of them changes a published number.

- **The credential preflight.** One check, before the phase, of the file every task is about to
  copy. A fan-out multiplies a single dead credential by the size of the held-out set, so this
  refuses loudly rather than discovering it once per task.
- **The launch stagger.** The first wave is the one moment N containers start at the same
  instant holding copies of one refreshable token. Spacing it changes no ordering and no
  accounting, which is what the tests below hold it to.
- **The drain watchdog.** An eval leg whose task is sealed and whose stream is drained has
  nothing left to do, and a harness that keeps running anyway spends wall clock and tokens on
  nothing. The ending is recorded as its own verdict kind, because "this harness does not stop
  by itself" is a finding rather than an accident.

Everything here is offline and keyless. The streams are real single-task ``EvalStream``s driven
in process, the same way the eval suspension tests drive them; the container is stood in for at
the one seam the runner owns, which is the leg supervisor.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from shobench import runner
from shobench.config import load_all_cells, load_cell_by_name, load_instruction
from shobench.containers import CellSandbox
from shobench.credentials import (
    PREFLIGHT_MIN_LIFETIME_S,
    preflight_seeded_credential,
    refresh_seeded_credential,
    spec_for,
)
from shobench.harness import Harness, StopKind, StopVerdict
from shobench.harnesses import harness_for
from shobench.results import TaskResult
from shobench.runner import DrainWatchdog, LegRecord, RunContext, run_leg
from shobench.splits import Side, Split

_SMOKE_CELL = "smoke-automationbench-claude-code"
_PRIME_CELL = "hle-prime_agent-claude-opus-5"
_TERRA_CELL = "hle-prime_agent-gpt-56-terra"


def _ctx(
    tmp_path: Path, *, cell_name: str = _SMOKE_CELL, heldout: tuple[str, ...] = ("1",)
) -> RunContext:
    cell = load_cell_by_name(cell_name)
    run_dir = tmp_path / "run"
    sandbox = CellSandbox(run_id="test", home=run_dir / "home", workdir=run_dir / "work")
    sandbox.home.mkdir(parents=True)
    sandbox.workdir.mkdir(parents=True)
    split = Split(
        env=cell.env,
        heldout=Side(task_ids=heldout),
        pool=Side(task_ids=("900", "901")),
        provenance={"kind": "adopted"},
        source=tmp_path / "split.json",
    )
    return RunContext(
        cell=cell,
        split=split,
        instruction=load_instruction(cell.instruction_arm),
        harness=harness_for(cell.harness),
        run_id="test",
        run_dir=run_dir,
        sandbox=sandbox,
    )


class _FakeStream:
    """A stream that never drains: an async context manager whose queue stays busy.

    Never draining is what keeps the drain watchdog out of the tests that are not about it. A
    phase whose fake stream reported a finished task would have its legs ended by the watchdog,
    which is a different test entirely.
    """

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def queue_info(self):
        from shogym.serve import QueueInfo

        return QueueInfo(remaining=1, consumed=0, in_flight=0)


@contextlib.asynccontextmanager
async def _fake_served(stream: object, port: int):
    yield


def _capture_launches(monkeypatch, launched: list[int], at: dict[int, float] | None = None) -> None:
    """Route the fan-out through fakes that record which task launched, and when."""

    def fake_run_leg(ctx_arg: RunContext, **kw: object) -> LegRecord:
        idx = int(kw["task_idx"])  # type: ignore[arg-type]
        launched.append(idx)
        if at is not None:
            at[idx] = time.monotonic()
        return LegRecord(
            leg=idx,
            phase=str(kw["phase"]),
            task_idx=idx,
            started_at=0.0,
            ended_at=1.0,
            returncode=0,
            verdict=StopVerdict(StopKind.CHOSEN, "it stopped on its own"),
            tasks_consumed_before=0,
            tasks_consumed_after=0,
            trace_path="t",
            run_dir=ctx_arg.run_dir,
        )

    def fake_read_phase(prov_dir: Path) -> list[TaskResult]:
        idx = int(prov_dir.name.split("-")[1])
        if idx not in launched:
            return []
        return [
            TaskResult(
                seq=idx, position=0, task_idx=idx, closure="sealed", reward=1.0, success=True
            )
        ]

    monkeypatch.setattr(runner, "warm_env", lambda cell: None)
    monkeypatch.setattr(runner, "build_stream", lambda *a, **k: _FakeStream())
    monkeypatch.setattr(runner, "_served", _fake_served)
    monkeypatch.setattr(runner, "run_leg", fake_run_leg)
    monkeypatch.setattr(runner, "read_phase", fake_read_phase)


# ----- the credential preflight ----------------------------------------------------------------


def _oauth(lifetime_s: float, **overrides: object) -> dict[str, object]:
    """One provider's oauth entry in the schema prime 0.7.1 reads, with no secret in it.

    The fields are the ones the pinned CLI actually uses: ``access`` is what ``getApiKey``
    returns and presents, ``refresh`` is what ``refreshToken`` is called with, and ``expires`` is
    what it compares against the clock to choose between them.
    """
    return {
        "type": "oauth",
        "access": "not-a-token",
        "refresh": "not-a-token",
        "expires": int((time.time() + lifetime_s) * 1000),
        **overrides,
    }


def _prime_auth(**providers: object) -> str:
    """A prime auth.json: a flat map of provider id to that provider's credential."""
    return json.dumps(dict(providers))


def _seed_prime_auth(
    ctx: RunContext, *, lifetime_s: float, monkeypatch=None, host: Path | None = None
) -> Path:
    """Put a prime credential in the cell home, and point the spec's host source somewhere safe.

    The host source is redirected because the refresh half of the preflight reads it for real. A
    test that left it alone would read the operator's own login and re-seed the cell home from
    it, which is neither hermetic nor anything a test has any business touching.
    """
    if monkeypatch is not None:
        from shobench import credentials

        spec = spec_for("prime_agent", "subscription")
        monkeypatch.setitem(
            credentials.SPECS,
            ("prime_agent", "subscription"),
            replace(spec, seed_from=str(host or ctx.run_dir / "no-host-login.json")),
        )
    path = ctx.sandbox.home / ".prime" / "agent" / "auth.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # The cell is claude-backed, so anthropic is the entry its legs will present.
    path.write_text(_prime_auth(anthropic=_oauth(lifetime_s)), encoding="utf-8")
    return path


def test_the_selected_provider_is_the_one_the_launch_names() -> None:
    """The preflight and the launch must not be able to disagree about which credential a cell
    uses, so both come from the harness. A harness whose credential file holds one login names
    no provider at all, which is what makes the check generic rather than prime-shaped."""
    prime = harness_for("prime_agent")
    for model in ("claude-opus-5", "gpt-5.6-terra"):
        spec = prime.launch(
            mcp_url="http://h:1/mcp",
            system_prompt="s",
            user_prompt="u",
            model=model,
            trace_path=Path("/dev/null"),
        )
        named = spec.argv[spec.argv.index("--provider") + 1]
        assert prime.credential_provider(model) == named
    assert prime.credential_provider("claude-opus-5") == "anthropic"
    assert prime.credential_provider("gpt-5.6-terra") == "openai-codex"
    for other in ("claude_code", "codex"):
        assert harness_for(other).credential_provider("claude-opus-5") == ""


def test_the_preflight_judges_only_the_credential_the_cell_will_present(tmp_path: Path) -> None:
    """A prime auth.json accumulates an entry for every provider ever logged in, and a leg looks
    up exactly one of them by id. The real gpt-terra home is the case: a live openai-codex login
    beside a long-expired anthropic one. Judging every entry refuses that cell over a credential
    no leg of it ever presents, which is the thing this must not do."""
    spec = spec_for("prime_agent", "subscription")
    prime = harness_for("prime_agent")
    terra = prime.credential_provider("gpt-5.6-terra")
    opus = prime.credential_provider("claude-opus-5")
    home = tmp_path / "home"
    auth = home / ".prime" / "agent" / "auth.json"
    auth.parent.mkdir(parents=True)

    auth.write_text(
        _prime_auth(anthropic=_oauth(-99999), **{"openai-codex": _oauth(86400)}), encoding="utf-8"
    )
    assert preflight_seeded_credential(spec, home, provider=terra) == (True, "")
    ok, why = preflight_seeded_credential(spec, home, provider=opus)
    assert not ok and "anthropic" in why

    # The mirror, which is the same rule seen from the other side: a claude cell is not refused
    # over a stale openai entry it will never reach for.
    auth.write_text(
        _prime_auth(anthropic=_oauth(86400), **{"openai-codex": _oauth(-99999)}), encoding="utf-8"
    )
    assert preflight_seeded_credential(spec, home, provider=opus) == (True, "")
    ok, why = preflight_seeded_credential(spec, home, provider=terra)
    assert not ok and "openai-codex" in why

    # A provider with no entry of its own is a refusal, not a pass: the file authenticates
    # something, just not this cell.
    auth.write_text(_prime_auth(anthropic=_oauth(86400)), encoding="utf-8")
    ok, why = preflight_seeded_credential(spec, home, provider=terra)
    assert not ok and "declares no credential for openai-codex" in why


def test_the_preflight_refuses_an_entry_prime_could_not_present(tmp_path: Path) -> None:
    """Shape is not usability. Each of these parses, declares a provider and a credential type,
    and would authenticate nothing: the pinned CLI presents ``access``, refreshes with
    ``refresh``, and compares ``expires`` against the clock, so an empty one of those is a dead
    credential. An oauth entry with no comparable expiry is the subtle one, because
    ``Date.now() >= undefined`` is false: the CLI never refreshes it and presents a token it
    cannot know is dead."""
    spec = spec_for("prime_agent", "subscription")
    home = tmp_path / "home"
    auth = home / ".prime" / "agent" / "auth.json"
    auth.parent.mkdir(parents=True)

    unusable = {
        "an oauth entry with no fields at all": {"type": "oauth"},
        "an api_key entry with no key": {"type": "api_key"},
        "an empty access": _oauth(86400, access=""),
        "an empty refresh": _oauth(86400, refresh=""),
        "a non-finite expiry": _oauth(86400, expires=float("inf")),
        "a boolean expiry": _oauth(86400, expires=True),
        "a credential type nothing presents": {"type": "keychain", "key": "k"},
    }
    for label, entry in unusable.items():
        auth.write_text(_prime_auth(anthropic=entry), encoding="utf-8")
        ok, why = preflight_seeded_credential(spec, home, provider="anthropic")
        assert not ok, label
        assert "anthropic" in why, label

    for label, entry in {
        "a live oauth entry": _oauth(PREFLIGHT_MIN_LIFETIME_S + 3600),
        "an api_key entry with a key": {"type": "api_key", "key": "not-a-key"},
    }.items():
        auth.write_text(_prime_auth(anthropic=entry), encoding="utf-8")
        assert preflight_seeded_credential(spec, home, provider="anthropic") == (True, ""), label


def test_the_preflight_reads_structure_and_expiry_and_asks_no_provider(tmp_path: Path) -> None:
    """The states that do not depend on which provider was selected, across both seeded schemas.

    A mode that seeds nothing passes: its credential is an environment value, and the only way to
    inspect one is to read it, which is the thing this module never does. A caller that names no
    provider gets the weakest honest check, because guessing which entry a cell will present is
    how a good credential gets refused.
    """
    prime = spec_for("prime_agent", "subscription")
    codex = spec_for("codex", "subscription")
    home = tmp_path / "home"
    home.mkdir()

    assert preflight_seeded_credential(spec_for("claude_code", "subscription"), home) == (True, "")

    ok, why = preflight_seeded_credential(prime, home, provider="anthropic")
    assert not ok and "no .prime/agent/auth.json" in why

    auth = home / ".prime" / "agent" / "auth.json"
    auth.parent.mkdir(parents=True)
    auth.write_text("{not json", encoding="utf-8")
    ok, why = preflight_seeded_credential(prime, home, provider="anthropic")
    assert not ok and "not readable JSON" in why

    auth.write_text("{}", encoding="utf-8")
    ok, why = preflight_seeded_credential(prime, home, provider="anthropic")
    assert not ok and "declares no credential for anthropic" in why

    # Alive, but not for long enough to survive a phase that fans out under it.
    auth.write_text(_prime_auth(anthropic=_oauth(PREFLIGHT_MIN_LIFETIME_S - 60)), encoding="utf-8")
    ok, why = preflight_seeded_credential(prime, home, provider="anthropic")
    assert not ok and "life left" in why

    # Unnamed provider: an empty file is still a refusal, and a file with any usable entry passes
    # without any one of them being judged.
    auth.write_text("{}", encoding="utf-8")
    ok, why = preflight_seeded_credential(prime, home)
    assert not ok and "declares no provider" in why
    auth.write_text(_prime_auth(anthropic=_oauth(-99999)), encoding="utf-8")
    assert preflight_seeded_credential(prime, home) == (True, "")

    codex_auth = home / ".codex" / "auth.json"
    codex_auth.parent.mkdir(parents=True)
    codex_auth.write_text(json.dumps({"auth_mode": "apikey"}), encoding="utf-8")
    ok, why = preflight_seeded_credential(codex, home)
    assert not ok and "chatgpt" in why
    codex_auth.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {}}), encoding="utf-8")
    ok, why = preflight_seeded_credential(codex, home)
    assert not ok and "no access token" in why
    codex_auth.write_text(
        json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "x"}}), encoding="utf-8"
    )
    assert preflight_seeded_credential(codex, home) == (True, "")


def test_the_refresh_replaces_one_provider_and_can_regress_none(tmp_path: Path) -> None:
    """A multi-provider file has no single freshness to compare. Ordering the files by their
    soonest expiry says this host file is fresher (300s beats 100s) while its openai-codex entry
    is older than the one it would overwrite (300s against 1000s), so a whole-file copy trades a
    live credential for a staler one. Replacing the selected provider's entry cannot do that to
    any other provider, and cannot do it to that one either."""
    host = tmp_path / "host-auth.json"
    home = tmp_path / "home"
    (home / ".prime" / "agent").mkdir(parents=True)
    seeded = home / ".prime" / "agent" / "auth.json"
    spec = replace(spec_for("prime_agent", "subscription"), seed_from=str(host))

    host.write_text(
        _prime_auth(anthropic=_oauth(5000), **{"openai-codex": _oauth(300)}), encoding="utf-8"
    )
    seeded.write_text(
        _prime_auth(anthropic=_oauth(100), **{"openai-codex": _oauth(1000)}), encoding="utf-8"
    )
    before = json.loads(seeded.read_text())

    # The gpt cell's provider: the host's is older, so nothing moves.
    assert refresh_seeded_credential(spec, home, provider="openai-codex") == ""
    assert json.loads(seeded.read_text()) == before

    # The claude cell's provider: the host's outlives it, so that ONE entry is replaced and the
    # live openai-codex credential beside it is left byte-identical.
    note = refresh_seeded_credential(spec, home, provider="anthropic")
    after = json.loads(seeded.read_text())
    assert "anthropic" in note
    assert after["anthropic"] == json.loads(host.read_text())["anthropic"]
    assert after["openai-codex"] == before["openai-codex"]

    # And the other direction for the same provider: a cell home that is already ahead keeps what
    # it has, so a refresh can never walk a credential backwards.
    assert refresh_seeded_credential(spec, home, provider="anthropic") == ""
    assert json.loads(seeded.read_text()) == after


def test_the_refresh_declines_what_it_cannot_order(tmp_path: Path) -> None:
    """Every case where there is no fact to act on: no provider named, a host with nothing for
    that provider, a seeded entry with no comparable expiry, and a schema that states none."""
    host = tmp_path / "host-auth.json"
    home = tmp_path / "home"
    (home / ".prime" / "agent").mkdir(parents=True)
    seeded = home / ".prime" / "agent" / "auth.json"
    spec = replace(spec_for("prime_agent", "subscription"), seed_from=str(host))

    host.write_text(_prime_auth(anthropic=_oauth(99999)), encoding="utf-8")
    seeded.write_text(_prime_auth(anthropic=_oauth(100)), encoding="utf-8")
    kept = json.loads(seeded.read_text())
    assert refresh_seeded_credential(spec, home) == ""
    assert refresh_seeded_credential(spec, home, provider="openai-codex") == ""
    assert json.loads(seeded.read_text()) == kept

    # An api_key entry cannot be ordered against an oauth one, so it is left where it is rather
    # than replaced on a guess about which is better.
    seeded.write_text(
        _prime_auth(anthropic={"type": "api_key", "key": "not-a-key"}), encoding="utf-8"
    )
    kept = json.loads(seeded.read_text())
    assert refresh_seeded_credential(spec, home, provider="anthropic") == ""
    assert json.loads(seeded.read_text()) == kept

    # codex states no readable expiry at all, so there is no fresher of the two to compute.
    assert refresh_seeded_credential(spec_for("codex", "subscription"), home, provider="x") == ""


def test_a_dead_credential_refuses_the_phase_before_a_single_task_launches(
    tmp_path: Path, monkeypatch
) -> None:
    """The refusal is the point: nothing is copied, nothing is served, nothing is launched."""
    ctx = _ctx(tmp_path, cell_name=_PRIME_CELL, heldout=("1", "2"))
    _seed_prime_auth(ctx, lifetime_s=60, monkeypatch=monkeypatch)
    launched: list[int] = []
    _capture_launches(monkeypatch, launched)

    with pytest.raises(RuntimeError, match="life left"):
        asyncio.run(runner.run_eval_phase(ctx, "eval_before"))

    assert launched == []
    assert not (ctx.run_dir / "eval_before" / "homes").exists()


def test_a_credential_with_life_left_lets_the_phase_run(tmp_path: Path, monkeypatch) -> None:
    """The other half of the refusal, so a preflight that refused everything would be visible."""
    ctx = _ctx(tmp_path, cell_name=_PRIME_CELL, heldout=("1", "2"))
    _seed_prime_auth(ctx, lifetime_s=PREFLIGHT_MIN_LIFETIME_S + 7200, monkeypatch=monkeypatch)
    monkeypatch.setattr(runner, "EVAL_LAUNCH_STAGGER_S", 0.0)
    launched: list[int] = []
    _capture_launches(monkeypatch, launched)

    rows = asyncio.run(runner.run_eval_phase(ctx, "eval_before"))

    assert sorted(launched) == [1, 2]
    assert [r.task_idx for r in rows] == [1, 2]


def test_a_gpt_cell_runs_against_the_home_a_real_gpt_cell_has(
    tmp_path: Path, monkeypatch
) -> None:
    """The whole phase against the shape the real gpt-terra homes carry: a live openai-codex
    login beside a long-expired anthropic one. This is the case a preflight that judged every
    entry would refuse, and the refusal would land on a paused measurement's resume rather than
    on anything actually wrong."""
    ctx = _ctx(tmp_path, cell_name=_TERRA_CELL, heldout=("1", "2"))
    assert ctx.cell.model == "gpt-5.6-terra"
    _seed_prime_auth(ctx, lifetime_s=0, monkeypatch=monkeypatch)
    (ctx.sandbox.home / ".prime" / "agent" / "auth.json").write_text(
        _prime_auth(
            anthropic=_oauth(-99999), **{"openai-codex": _oauth(PREFLIGHT_MIN_LIFETIME_S + 7200)}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "EVAL_LAUNCH_STAGGER_S", 0.0)
    launched: list[int] = []
    _capture_launches(monkeypatch, launched)

    rows = asyncio.run(runner.run_eval_phase(ctx, "eval_before"))

    assert sorted(launched) == [1, 2]
    assert [r.task_idx for r in rows] == [1, 2]


# ----- the launch stagger ------------------------------------------------------------------------


def test_the_first_wave_is_spaced_and_no_task_launches_around_it(
    tmp_path: Path, monkeypatch
) -> None:
    """MORE tasks than slots, which is the only arrangement that can catch the bypass.

    With exactly as many tasks as the concurrency limit there is no coroutine left over to
    overtake the wave, so a stagger that spaced only the sleeping coroutines passed. With eight
    tasks at a concurrency of four, spacing ahead of the gate let 5, 6, 7 and 8 take the free
    slots the sleepers had not claimed yet and launch first, unspaced, so the real first wave was
    both unstaggered and reversed. What this asserts is what the mechanism is for: the first four
    LAUNCHES are the first four pending ids, each a gap after the last, and no fifth task starts
    before the wave is out.
    """
    gap = 0.05
    limit = 4
    monkeypatch.setattr(runner, "EVAL_LAUNCH_STAGGER_S", gap)
    ids = ("1", "2", "3", "4", "5", "6", "7", "8")
    ctx = _ctx(tmp_path, cell_name=_PRIME_CELL, heldout=ids)
    ctx = replace(
        ctx, cell=replace(ctx.cell, budget=replace(ctx.cell.budget, eval_concurrency=limit))
    )
    _seed_prime_auth(ctx, lifetime_s=PREFLIGHT_MIN_LIFETIME_S + 7200, monkeypatch=monkeypatch)
    launched: list[int] = []
    at: dict[int, float] = {}
    _capture_launches(monkeypatch, launched, at)

    staggered = asyncio.run(runner.run_eval_phase(ctx, "eval_before"))

    # The wave is the first four ids, in order, and nothing overtook it.
    assert launched[:limit] == [1, 2, 3, 4]
    wave = [at[idx] for idx in launched[:limit]]
    # Pairwise, so a wave that arrived in one burst with a long tail cannot pass on its span.
    assert min(later - earlier for earlier, later in zip(wave, wave[1:], strict=False)) >= gap * 0.8
    assert min(at[idx] for idx in launched[limit:]) >= wave[-1]
    assert sorted(launched) == [1, 2, 3, 4, 5, 6, 7, 8]

    # The same phase with no spacing at all: same ids, same rows, same order.
    monkeypatch.setattr(runner, "EVAL_LAUNCH_STAGGER_S", 0.0)
    plain_ctx = _ctx(tmp_path / "plain", cell_name=_PRIME_CELL, heldout=ids)
    plain_ctx = replace(
        plain_ctx,
        cell=replace(plain_ctx.cell, budget=replace(plain_ctx.cell.budget, eval_concurrency=limit)),
    )
    _seed_prime_auth(plain_ctx, lifetime_s=PREFLIGHT_MIN_LIFETIME_S + 7200, monkeypatch=monkeypatch)
    plain_launched: list[int] = []
    _capture_launches(monkeypatch, plain_launched)
    plain = asyncio.run(runner.run_eval_phase(plain_ctx, "eval_before"))

    assert sorted(launched) == sorted(plain_launched)
    assert [r.task_idx for r in staggered] == [r.task_idx for r in plain] == list(range(1, 9))


def test_the_stagger_stops_after_the_first_wave(tmp_path: Path, monkeypatch) -> None:
    """Only the first wave is spaced. Past it a slot opens only when a task finishes, so the
    launches are already spread and paying the gap again would be wall clock for nothing: at a
    two-second gap over a 120-task phase it would be four minutes of sleeping."""
    gap = 0.05
    limit = 2
    monkeypatch.setattr(runner, "EVAL_LAUNCH_STAGGER_S", gap)
    ctx = _ctx(tmp_path, cell_name=_PRIME_CELL, heldout=tuple(str(i) for i in range(1, 9)))
    ctx = replace(
        ctx, cell=replace(ctx.cell, budget=replace(ctx.cell.budget, eval_concurrency=limit))
    )
    _seed_prime_auth(ctx, lifetime_s=PREFLIGHT_MIN_LIFETIME_S + 7200, monkeypatch=monkeypatch)
    launched: list[int] = []
    at: dict[int, float] = {}
    _capture_launches(monkeypatch, launched, at)

    started = time.monotonic()
    asyncio.run(runner.run_eval_phase(ctx, "eval_before"))

    # Eight tasks, two staggered admissions: one gap of sleeping, not seven.
    assert len(launched) == 8
    assert time.monotonic() - started < gap * 4
    assert at[launched[1]] - at[launched[0]] >= gap * 0.8


def test_a_credential_the_harness_never_writes_is_not_staggered(
    tmp_path: Path, monkeypatch
) -> None:
    """A token handed to the container as an environment variable is not refreshed by the
    harness, so there is nothing for N homes to race and nothing to space out."""
    monkeypatch.setattr(runner, "EVAL_LAUNCH_STAGGER_S", 5.0)
    ctx = _ctx(tmp_path, heldout=("1", "2", "3"))  # claude_code: no seeded credential file
    launched: list[int] = []
    _capture_launches(monkeypatch, launched)

    started = time.monotonic()
    rows = asyncio.run(runner.run_eval_phase(ctx, "eval_before"))

    assert time.monotonic() - started < 5.0
    assert [r.task_idx for r in rows] == [1, 2, 3]


# ----- the drain watchdog: when it fires -----------------------------------------------------


def _against_a_real_stream(prov_dir: Path, task_id: int, body) -> None:
    """Drive one real single-task ``EvalStream`` and hand ``body`` the live stream and a client.

    Real rather than faked because the condition is a joint reading of shogym's own queue and
    shogym's own provenance, and a fake of either would be a fake of the thing under test.
    """
    import shogym
    from fastmcp import Client
    from shogym.serve import EvalStream, TaskRef, build_stream_server

    async def play() -> None:
        stream = EvalStream(shogym.make, [TaskRef("wordle_v1", task_id)], prov_dir=prov_dir)
        async with stream:
            server = build_stream_server(stream, name="shogym")
            async with Client(server) as client:
                await body(stream, client)

    asyncio.run(play())


def test_the_finished_condition_needs_the_queue_and_the_row_to_agree(tmp_path: Path) -> None:
    """Neither reading is enough on its own, so the predicate is checked at all three states an
    eval task passes through: nothing pulled, one in flight, one sealed."""
    prov = tmp_path / "task-00000"
    prov.mkdir(parents=True)
    seen: list[bool] = []

    async def body(stream, client):
        seen.append(runner._eval_task_is_finished(stream, prov, 0))
        await client.call_tool("get_task", {})
        seen.append(runner._eval_task_is_finished(stream, prov, 0))
        await client.call_tool("terminate", {})
        seen.append(runner._eval_task_is_finished(stream, prov, 0))

    _against_a_real_stream(prov, 0, body)
    # An untouched queue is a leg that has not started, not a leg with nothing left to do.
    assert seen == [False, False, True]


def test_the_watchdog_fires_only_after_the_grace_and_only_when_finished(tmp_path: Path) -> None:
    """The grace is a wait, not a trigger: an unsealed task is watched however long it takes, and
    a sealed one is given the whole period before its leg is ended."""
    prov = tmp_path / "task-00000"
    prov.mkdir(parents=True)
    grace = 0.2

    async def body(stream, client):
        watchdog = DrainWatchdog(threading.Event(), grace)
        await client.call_tool("get_task", {})
        # In flight: the watchdog waits, and goes on waiting.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                runner._watch_for_drain(stream, prov, 0, watchdog, poll_s=0.01), timeout=grace * 3
            )
        assert not watchdog.fired.is_set()

        await client.call_tool("terminate", {})
        started = time.monotonic()
        assert await runner._watch_for_drain(stream, prov, 0, watchdog, poll_s=0.01) is True
        assert watchdog.fired.is_set()
        # It waited the grace out rather than firing on the first finished reading.
        assert time.monotonic() - started >= grace

    _against_a_real_stream(prov, 0, body)


class _PollClock:
    """A monotonic clock that hands out one poll interval per reading.

    The watchdog reads a clock once per poll, so this runs the real loop at test speed over the
    arithmetic a real phase does. Only the clock is stood in for: the stream, the sealed row and
    the loop reading them are the run's own, and waiting the real graces out would spend two and a
    quarter minutes of suite time on a subtraction.
    """

    def __init__(self) -> None:
        self.readings: list[float] = []
        self._next = 0.0

    def monotonic(self) -> float:
        self.readings.append(self._next)
        self._next += runner.EVAL_DRAIN_POLL_S
        return self.readings[-1]


def test_prime_ends_a_finished_leg_on_sight_and_the_others_wait_their_grace(
    tmp_path: Path, monkeypatch
) -> None:
    """Two endings, and which one a leg gets is the harness's own declaration.

    Claude Code and codex end their legs 8 to 25 seconds after the seal, so two minutes is a wait
    they never reach the end of and their voluntary stops stay reachable. prime is not waiting for
    an ending it makes, and a wait for it is a clock racing the next billable dispatch, so the
    first reading that finds its leg finished is the ending. A zero would not do this: it would
    still be compared on the following poll.
    """
    prov = tmp_path / "task-00000"
    prov.mkdir(parents=True)
    real_finished = runner._eval_task_is_finished
    seen: dict[str, tuple[int, float]] = {}

    async def watch_out(stream, name: str) -> tuple[int, float]:
        """One harness's watchdog against an already finished leg: how many readings found it
        finished before the watchdog fired, and how many simulated seconds passed over them."""
        clock = _PollClock()
        polls = 0

        def counted(*args: object, **kw: object) -> bool:
            nonlocal polls
            finished = real_finished(*args, **kw)
            polls += 1 if finished else 0
            return finished

        monkeypatch.setattr(runner, "_eval_task_is_finished", counted)
        monkeypatch.setattr(runner, "time", clock)
        watchdog = DrainWatchdog(threading.Event(), harness_for(name).eval_drain_grace_s)
        assert await runner._watch_for_drain(stream, prov, 0, watchdog, poll_s=0.001) is True
        assert watchdog.fired.is_set()
        waited = clock.readings[-1] - clock.readings[0] if clock.readings else 0.0
        return polls, waited

    async def body(stream, client):
        await client.call_tool("get_task", {})
        await client.call_tool("terminate", {})
        for name in ("prime_agent", "claude_code", "codex"):
            seen[name] = await watch_out(stream, name)

    _against_a_real_stream(prov, 0, body)

    # prime: the reading that finds the leg finished is the one that ends it, nothing waited and
    # no clock consulted, so the exposure is the poll interval and not a number racing a dispatch.
    assert seen["prime_agent"] == (1, 0.0)
    # claude_code and codex: unchanged, two minutes of polls before a leg is taken from them.
    assert seen["claude_code"] == (25, 120.0)
    assert seen["codex"] == (25, 120.0)
    # A harness that declares nothing waits the long one, so the change is prime's alone.
    assert Harness.eval_drain_grace_s == 120.0


def test_the_phase_bounds_every_leg_by_its_own_harness_grace(tmp_path: Path, monkeypatch) -> None:
    """What the harness declares reaches the leg: the watchdog a phase builds carries its own
    harness's grace, which is the one place the runner reads it."""
    monkeypatch.setattr(runner, "EVAL_LAUNCH_STAGGER_S", 0.0)

    def graces_of(cell_name: str, *, at: Path) -> list[float | None]:
        launched: list[int] = []
        _capture_launches(monkeypatch, launched)
        captured = runner.run_leg
        seen: list[float | None] = []

        def record(ctx_arg: RunContext, **kw: object) -> LegRecord:
            watchdog = kw["watchdog"]
            assert isinstance(watchdog, DrainWatchdog)
            seen.append(watchdog.grace_s)
            return captured(ctx_arg, **kw)  # type: ignore[arg-type]

        monkeypatch.setattr(runner, "run_leg", record)
        ctx = _ctx(at, cell_name=cell_name, heldout=("1", "2"))
        if ctx.harness.name == "prime_agent":
            _seed_prime_auth(
                ctx, lifetime_s=PREFLIGHT_MIN_LIFETIME_S + 7200, monkeypatch=monkeypatch
            )
        asyncio.run(runner.run_eval_phase(ctx, "eval_before"))
        return seen

    assert graces_of(_PRIME_CELL, at=tmp_path / "prime") == [None, None]
    assert graces_of(_SMOKE_CELL, at=tmp_path / "claude") == [120.0, 120.0]


# ----- the drain watchdog: what it does to the leg --------------------------------------------


class _NeverExits:
    """A docker client that never exits on its own, so only the supervisor can end it."""

    stdin = None

    def __init__(self) -> None:
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        if self.killed:
            return -9
        raise subprocess.TimeoutExpired("docker", timeout or 0)

    def kill(self) -> None:
        self.killed = True


class _ExitsAtOnce:
    """The claude and codex shape: the harness ends its own leg while the watchdog watches."""

    stdin = None

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        raise AssertionError("the supervisor killed a leg that had already exited")


def _supervising(monkeypatch, proc: object, removed: list[str]) -> None:
    def fake_docker(*args: str, **kw: object) -> None:
        removed.append(args[-1])

    monkeypatch.setattr(runner, "LEG_POLL_S", 0.01)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda argv, **kw: proc)
    monkeypatch.setattr(runner, "docker", fake_docker)


def _supervise(
    tmp_path: Path, *, timeout_s: int, watchdog: DrainWatchdog
) -> tuple[int, bool, bool]:
    with (tmp_path / "o").open("w") as out, (tmp_path / "e").open("w") as err:
        return runner._supervise(
            ["docker", "run"],
            out=out,
            err=err,
            stdin_data=None,
            timeout_s=timeout_s,
            container="cell-eval-t1",
            watchdog=watchdog,
        )


def test_the_supervisor_ends_a_fired_leg_the_way_the_timeout_path_does(
    tmp_path: Path, monkeypatch
) -> None:
    """Client killed first, then the container removed by name: a container that outlived its leg
    would keep spending and keep holding the task it was handed."""
    proc = _NeverExits()
    removed: list[str] = []
    _supervising(monkeypatch, proc, removed)
    watchdog = DrainWatchdog(threading.Event(), 1.0)
    watchdog.fired.set()

    assert _supervise(tmp_path, timeout_s=600, watchdog=watchdog) == (-1, False, True)
    assert proc.killed
    assert removed == ["cell-eval-t1"]


def test_the_supervisor_still_reports_an_elapsed_budget_as_a_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    """The watchdog is an additional ending, never a replacement for the one the budget imposes."""
    removed: list[str] = []
    _supervising(monkeypatch, _NeverExits(), removed)

    result = _supervise(tmp_path, timeout_s=0, watchdog=DrainWatchdog(threading.Event(), 1.0))

    assert result == (-1, True, False)
    assert removed == ["cell-eval-t1"]


def test_a_leg_that_ends_first_is_neither_drained_nor_killed(tmp_path: Path, monkeypatch) -> None:
    """A harness that exits on its own keeps its own exit status, and nothing is removed."""
    removed: list[str] = []
    _supervising(monkeypatch, _ExitsAtOnce(), removed)

    result = _supervise(tmp_path, timeout_s=600, watchdog=DrainWatchdog(threading.Event(), 1.0))

    assert result == (0, False, False)
    assert removed == []


def _leg(ctx: RunContext, **kw: object) -> LegRecord:
    return run_leg(
        ctx,
        phase="eval_before",
        leg=1,
        system_prompt="SYS",
        user_prompt="USR",
        session_id="sess-1",
        resume=False,
        timeout_s=60,
        task_idx=1,
        consumed_before=0,
        **kw,  # type: ignore[arg-type]
    )


def test_a_drained_leg_carries_its_own_verdict_and_never_the_harness_classification(
    tmp_path: Path, monkeypatch
) -> None:
    """What ended the leg was a decision here, so the trace is not consulted for it, and the kind
    is its own: not the chosen stop the agent did not make, not the budget that did not run out.
    """
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(runner, "_supervise", lambda *a, **kw: (-1, False, True))

    def refuse(**kw: object) -> StopVerdict:
        raise AssertionError("a drained leg's trace was classified")

    monkeypatch.setattr(ctx.harness, "classify", refuse)

    record = _leg(ctx, watchdog=DrainWatchdog(threading.Event(), 120.0))

    assert record.verdict.kind is StopKind.DRAINED
    assert record.verdict.kind is not StopKind.CHOSEN
    assert record.verdict.kind is not StopKind.LEG_TIMEOUT
    assert record.verdict.resumable is False
    assert record.verdict.evidence["grace_s"] == 120.0
    # It reaches the durable record as itself, which is where the finding stays legible.
    assert ctx.leg_records()[0]["verdict"]["kind"] == "stream_drained"


def test_a_leg_the_watchdog_did_not_end_is_classified_by_its_harness(
    tmp_path: Path, monkeypatch
) -> None:
    """The sibling assertion: a runner that called everything drained would pass the test above."""
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(runner, "_supervise", lambda *a, **kw: (0, False, False))
    seen: dict[str, object] = {}

    def fake_classify(**kw: object) -> StopVerdict:
        seen.update(kw)
        return StopVerdict(StopKind.CHOSEN, "the last message ended the turn")

    monkeypatch.setattr(ctx.harness, "classify", fake_classify)

    record = _leg(ctx, watchdog=DrainWatchdog(threading.Event(), 120.0))

    assert record.verdict.kind is StopKind.CHOSEN
    assert seen["timed_out"] is False


def test_every_harness_reports_a_drained_leg_the_same_way() -> None:
    """The decision was the runner's, so the verdict is identical across harnesses, and its kind
    is not the word shogym already uses for a row a stream close cut off in flight."""
    for name in ("claude_code", "codex", "prime_agent"):
        verdict = harness_for(name).drained_verdict(grace_s=120.0)
        assert verdict.kind is StopKind.DRAINED
        assert verdict.kind.value == "stream_drained"
        assert verdict.kind.value != "drained"
        assert not verdict.resumable
        assert "sealed" in verdict.reason


# ----- the rollout is untouched ----------------------------------------------------------------


def test_the_rollout_leg_is_never_handed_a_watchdog(tmp_path: Path, monkeypatch) -> None:
    """A rollout leg facing an empty queue is the charter's own question. Ending it here would
    answer that question for the agent, so the rollout passes no watchdog at all."""
    ctx = _ctx(tmp_path)
    ctx = replace(
        ctx,
        cell=replace(ctx.cell, env="wordle_v1"),
        split=replace(ctx.split, env="wordle_v1", pool=Side(task_ids=("3", "4"))),
    )
    seen: dict[str, object] = {}

    def fake_run_leg(ctx_arg: RunContext, **kw: object) -> LegRecord:
        seen.update(kw)
        return LegRecord(
            leg=0,
            phase="rollout",
            task_idx=None,
            started_at=0.0,
            ended_at=1.0,
            returncode=0,
            verdict=StopVerdict(StopKind.CHOSEN, "it stopped on its own"),
            tasks_consumed_before=0,
            tasks_consumed_after=0,
            trace_path="t",
            run_dir=ctx_arg.run_dir,
        )

    monkeypatch.setattr(runner, "run_leg", fake_run_leg)
    asyncio.run(runner.run_rollout_phase(ctx))

    assert seen["phase"] == "rollout"
    assert "watchdog" not in seen


# ----- the prompt cache ------------------------------------------------------------------------


def test_prime_asks_for_the_long_prompt_cache_and_reaches_the_leg_with_it(tmp_path: Path) -> None:
    """Claude Code already runs with 1h retention and prime-agent defaults to short, so a prime
    cell was rewriting a cache every five minutes for the same conversation. The variable is
    prime-agent's own, it is read only by its providers, and it changes nothing the agent sees."""
    spec = harness_for("prime_agent").launch(
        mcp_url="http://h:1/mcp",
        system_prompt="s",
        user_prompt="u",
        model="claude-opus-5",
        trace_path=tmp_path / "t",
    )
    assert spec.env["PI_CACHE_RETENTION"] == "long"
    for other in ("claude_code", "codex"):
        assert "PI_CACHE_RETENTION" not in harness_for(other).base_env()


# ----- the backstop behind the watchdog --------------------------------------------------------


def test_every_prime_cell_bounds_an_eval_task_behind_the_watchdog() -> None:
    """Belt and braces: the watchdog ends these legs minutes in, and the timeout is what catches a
    leg the watchdog could not reach. Pooled time-to-terminal p99 was 665s, so 900 leaves the
    measurement itself untouched."""
    for cell in load_all_cells():
        if cell.harness != "prime_agent":
            continue
        assert cell.budget.eval_task_timeout_s == 900, cell.name
