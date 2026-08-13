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
from shobench.harness import StopKind, StopVerdict
from shobench.harnesses import harness_for
from shobench.results import TaskResult
from shobench.runner import DrainWatchdog, LegRecord, RunContext, run_leg
from shobench.splits import Side, Split

_SMOKE_CELL = "smoke-automationbench-claude-code"
_PRIME_CELL = "hle-prime_agent-claude-opus-5"


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


def _prime_auth(*, lifetime_s: float, kind: str = "oauth") -> str:
    """A prime auth.json with the schema the real one has and secrets that are not secrets."""
    entry: dict[str, object] = {"type": kind, "key": "not-a-key"}
    if kind == "oauth":
        entry = {
            "type": "oauth",
            "access": "not-a-token",
            "refresh": "not-a-token",
            "expires": int((time.time() + lifetime_s) * 1000),
        }
    return json.dumps({"anthropic": entry})


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
    path.write_text(_prime_auth(lifetime_s=lifetime_s), encoding="utf-8")
    return path


def test_the_preflight_reads_structure_and_expiry_and_asks_no_provider(tmp_path: Path) -> None:
    """Every state the check distinguishes, against the two schemas that seed a file.

    A mode that seeds nothing passes: its credential is an environment value, and the only way to
    inspect one is to read it, which is the thing this module never does.
    """
    prime = spec_for("prime_agent", "subscription")
    codex = spec_for("codex", "subscription")
    home = tmp_path / "home"
    home.mkdir()

    assert preflight_seeded_credential(spec_for("claude_code", "subscription"), home) == (True, "")

    ok, why = preflight_seeded_credential(prime, home)
    assert not ok and "no .prime/agent/auth.json" in why

    auth = home / ".prime" / "agent" / "auth.json"
    auth.parent.mkdir(parents=True)
    auth.write_text("{not json", encoding="utf-8")
    ok, why = preflight_seeded_credential(prime, home)
    assert not ok and "not readable JSON" in why

    auth.write_text("{}", encoding="utf-8")
    ok, why = preflight_seeded_credential(prime, home)
    assert not ok and "declares no provider" in why

    # Alive, but not for long enough to survive a phase that fans out under it.
    auth.write_text(_prime_auth(lifetime_s=PREFLIGHT_MIN_LIFETIME_S - 60), encoding="utf-8")
    ok, why = preflight_seeded_credential(prime, home)
    assert not ok and "anthropic" in why

    auth.write_text(_prime_auth(lifetime_s=PREFLIGHT_MIN_LIFETIME_S + 3600), encoding="utf-8")
    assert preflight_seeded_credential(prime, home) == (True, "")

    # An entry that declares no expiry at all is left alone: nothing here can prove it is stale.
    auth.write_text(_prime_auth(lifetime_s=0, kind="api_key"), encoding="utf-8")
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


def test_the_refresh_takes_the_later_expiry_of_the_two_files(tmp_path: Path) -> None:
    """The cell home's credential is usually the fresher one and stays; the host's replaces it
    only when the host has since logged in or refreshed. Expiry decides, not who wrote last."""
    host = tmp_path / "host-auth.json"
    home = tmp_path / "home"
    (home / ".prime" / "agent").mkdir(parents=True)
    seeded = home / ".prime" / "agent" / "auth.json"
    spec = replace(spec_for("prime_agent", "subscription"), seed_from=str(host))

    host.write_text(_prime_auth(lifetime_s=3600), encoding="utf-8")
    seeded.write_text(_prime_auth(lifetime_s=7200), encoding="utf-8")
    kept = json.loads(seeded.read_text())
    assert refresh_seeded_credential(spec, home) == ""
    assert json.loads(seeded.read_text()) == kept

    host.write_text(_prime_auth(lifetime_s=99999), encoding="utf-8")
    assert "re-seeded" in refresh_seeded_credential(spec, home)
    assert (
        json.loads(seeded.read_text())["anthropic"]["expires"]
        == json.loads(host.read_text())["anthropic"]["expires"]
    )

    # codex declares no readable expiry, so there is no fresher of the two to compute.
    assert refresh_seeded_credential(spec_for("codex", "subscription"), home) == ""


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


# ----- the launch stagger ------------------------------------------------------------------------


def test_the_first_wave_is_spaced_and_the_phase_is_otherwise_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    """The spacing is real, and it is the only thing that differs: the same ids run, they come
    back in the same order, and every one of them is accounted for either way."""
    gap = 0.05
    monkeypatch.setattr(runner, "EVAL_LAUNCH_STAGGER_S", gap)
    ctx = _ctx(tmp_path, cell_name=_PRIME_CELL, heldout=("4", "1", "3", "2"))
    ctx = replace(ctx, cell=replace(ctx.cell, budget=replace(ctx.cell.budget, eval_concurrency=4)))
    _seed_prime_auth(ctx, lifetime_s=PREFLIGHT_MIN_LIFETIME_S + 7200, monkeypatch=monkeypatch)
    launched: list[int] = []
    at: dict[int, float] = {}
    _capture_launches(monkeypatch, launched, at)

    staggered = asyncio.run(runner.run_eval_phase(ctx, "eval_before"))

    # The wave really was spaced: its first and last launches are three gaps apart, not zero.
    assert at[2] - at[4] >= 3 * gap * 0.8
    assert sorted(launched) == [1, 2, 3, 4]

    # The same phase with no spacing at all: same ids, same rows, same order.
    monkeypatch.setattr(runner, "EVAL_LAUNCH_STAGGER_S", 0.0)
    plain_ctx = _ctx(tmp_path / "plain", cell_name=_PRIME_CELL, heldout=("4", "1", "3", "2"))
    _seed_prime_auth(plain_ctx, lifetime_s=PREFLIGHT_MIN_LIFETIME_S + 7200, monkeypatch=monkeypatch)
    plain_launched: list[int] = []
    _capture_launches(monkeypatch, plain_launched)
    plain = asyncio.run(runner.run_eval_phase(plain_ctx, "eval_before"))

    assert sorted(launched) == sorted(plain_launched)
    assert [r.task_idx for r in staggered] == [r.task_idx for r in plain] == [1, 2, 3, 4]


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
