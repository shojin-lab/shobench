"""The held-out eval runs its tasks concurrently, each against its own isolated home.

Three properties decide whether the parallel eval is still the same measurement, and each has a
test here that needs neither Docker nor a credential:

- **Isolation.** A task copies the phase's base home, reads the accumulated durable self out of
  it, and writes only into its copy. Its writes never reach the base home or a sibling's copy,
  and only the durable channel plus the credential cross into the fresh session.
- **Concurrency correctness.** N tasks run at once, bounded by the knob; every result is present
  and keyed to its task; a task that fails to run lands unscored rather than sinking the batch;
  and the reported rows are ordered by task id whatever order the tasks finished in.
- **Cache reuse.** The env's upstream is provisioned once and reused, and the eval prompt prefix
  is byte-identical across tasks, which is the precondition for the provider's prompt cache to
  carry it warm from one task to the next.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from shobench import runner, serving
from shobench.config import load_cell_by_name, load_instruction
from shobench.containers import CellSandbox, home_digest
from shobench.harness import StopKind, StopVerdict
from shobench.harnesses import harness_for
from shobench.results import TaskResult
from shobench.runner import (
    LegRecord,
    RunContext,
    _copy_task_home,
    _eval_container_name,
    is_noise,
)
from shobench.splits import Side, Split

# ----- per-task home isolation ---------------------------------------------------------------


def _seed_base_home(base: Path) -> None:
    """A base home shaped like a post-rollout one: durable self, a credential, and noise."""
    memory = base / ".claude" / "projects" / "-work" / "memory"
    memory.mkdir(parents=True)
    (memory / "note.md").write_text("accumulated lesson\n", encoding="utf-8")
    skill = base / ".claude" / "skills" / "s"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("a learned skill\n", encoding="utf-8")
    # A file-based credential: noise for the digest, but the harness cannot authenticate without
    # it, so it must cross into the copy.
    (base / ".codex").mkdir(parents=True)
    (base / ".codex" / "auth.json").write_text('{"auth_mode": "chatgpt"}', encoding="utf-8")
    # Pure noise that must never bloat the copy or reach the fresh session.
    cache = base / ".cache" / "big"
    cache.mkdir(parents=True)
    (cache / "blob").write_text("x" * 100_000, encoding="utf-8")
    (base / ".claude" / "projects" / "-work" / "abc-123.jsonl").write_text("transcript\n")


def test_a_task_home_copies_the_durable_self_and_the_credential_but_not_the_noise(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    _seed_base_home(base)
    task_home = tmp_path / "task"

    _copy_task_home(base, task_home)

    # The durable self the eval measures crossed, and so did the credential.
    note = task_home / ".claude/projects/-work/memory/note.md"
    assert note.read_text() == "accumulated lesson\n"
    assert (task_home / ".claude/skills/s/SKILL.md").exists()
    assert (task_home / ".codex/auth.json").exists()
    # The noise did not: the copy is the durable channel plus the credential, nothing else.
    assert not (task_home / ".cache/big/blob").exists()
    assert not (task_home / ".claude/projects/-work/abc-123.jsonl").exists()


def test_a_task_write_reaches_neither_the_base_home_nor_a_sibling(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _seed_base_home(base)
    base_digest = home_digest(base, exclude=is_noise)

    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    _copy_task_home(base, home_a)
    _copy_task_home(base, home_b)

    # Task A does what a running agent does: writes a new file and edits an existing one.
    (home_a / "agent_wrote.txt").write_text("side effect\n", encoding="utf-8")
    (home_a / ".claude/projects/-work/memory/note.md").write_text("MUTATED\n", encoding="utf-8")

    # The base home is untouched, both by content and by digest.
    assert home_digest(base, exclude=is_noise) == base_digest
    assert (base / ".claude/projects/-work/memory/note.md").read_text() == "accumulated lesson\n"
    # The sibling never sees task A's write and still reads the pristine accumulated self.
    assert not (home_b / "agent_wrote.txt").exists()
    assert (home_b / ".claude/projects/-work/memory/note.md").read_text() == "accumulated lesson\n"


def test_copying_an_empty_base_home_yields_an_empty_isolated_home(tmp_path: Path) -> None:
    """eval_before under a keyless harness starts from a home with nothing durable in it. The
    copy of that is an empty directory, which is exactly the pristine fresh session eval wants."""
    base = tmp_path / "base"
    base.mkdir()
    task_home = tmp_path / "task"
    _copy_task_home(base, task_home)
    assert task_home.is_dir()
    assert list(task_home.rglob("*")) == []


def test_the_copy_carries_the_rollout_conversation_only_when_asked(tmp_path: Path) -> None:
    """A resumed fork needs the transcript in its copy; a cold copy still leaves it behind.

    ``keep`` names the harness's session-state subtrees and nothing else, so the rest of the
    noise stays out of the copy either way.
    """
    base = tmp_path / "base"
    _seed_base_home(base)

    cold = tmp_path / "cold"
    _copy_task_home(base, cold)
    assert not (cold / ".claude/projects/-work/abc-123.jsonl").exists()

    forked = tmp_path / "forked"
    _copy_task_home(base, forked, keep=(".claude/projects",))
    assert (forked / ".claude/projects/-work/abc-123.jsonl").read_text() == "transcript\n"
    assert not (forked / ".cache/big/blob").exists()


# ----- container naming under the length cap -------------------------------------------------


def test_concurrent_container_names_are_unique_and_keep_the_task_id() -> None:
    """Docker caps names at 63 chars. With a long run id the naive `<netns>-eval-t00005`[:63]
    truncates the id off the end, so two live containers collide and a timeout could `rm -f` the
    wrong one. The names must stay <=63, distinct per id, and carry the full id."""
    # A run id long enough that the netns holder name is already near the cap.
    netns = ("shobench-stress-automationbench-claude_code-clau"[:50]) + "-ns"
    names = {i: _eval_container_name(netns, "eval_before", i) for i in range(120)}
    assert all(len(name) <= 63 for name in names.values())
    assert len(set(names.values())) == 120  # no collisions among the whole held-out set
    assert names[5].endswith("-eval-t00005")
    assert names[119].endswith("-eval-t00119")


# ----- the per-task /work mount --------------------------------------------------------------


def _work_mount(args: list[str]) -> str:
    """The ``-v`` value the generated docker args map to ``/work:rw``."""
    return next(
        args[i + 1]
        for i, a in enumerate(args)
        if a == "-v" and args[i + 1].endswith(":/work:rw")
    )


def test_each_eval_task_mounts_its_own_work_directory(tmp_path: Path) -> None:
    """Finding 1: every eval task's ``/work`` is a directory of its own, never the cell-wide one,
    so two concurrent tasks cannot share the writable cwd. Inspected on the generated docker args,
    the same construction the reviewer drove against the real image."""
    sandbox = CellSandbox(run_id="r", home=tmp_path / "home", workdir=tmp_path / "cellwork")
    work_a = tmp_path / "task-a"
    work_b = tmp_path / "task-b"
    args_a = sandbox.docker_args(env={}, mounts={}, home=tmp_path / "ha", workdir=work_a)
    args_b = sandbox.docker_args(env={}, mounts={}, home=tmp_path / "hb", workdir=work_b)

    # Distinct per-task mounts, each its own directory.
    assert _work_mount(args_a) == f"{work_a}:/work:rw"
    assert _work_mount(args_b) == f"{work_b}:/work:rw"
    assert _work_mount(args_a) != _work_mount(args_b)
    # The cwd is still /work, so the harness runs where it always has, just in isolation.
    assert args_a[args_a.index("-w") + 1] == "/work"

    # The rollout keeps the cell's one accumulating /work (the default), unchanged by the fix.
    rollout_args = sandbox.docker_args(env={}, mounts={})
    assert _work_mount(rollout_args) == f"{tmp_path / 'cellwork'}:/work:rw"


# ----- concurrency correctness ---------------------------------------------------------------


class _FakeStream:
    """Stands in for a shogym stream: an async context manager and nothing more."""

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


@contextlib.asynccontextmanager
async def _fake_served(stream: object, port: int):
    yield


def _ctx(tmp_path: Path, heldout_ids: tuple[str, ...]) -> RunContext:
    cell = load_cell_by_name("smoke-automationbench-claude-code")
    split = Split(
        env=cell.env,
        heldout=Side(task_ids=heldout_ids),
        pool=Side(task_ids=("900", "901")),
        provenance={"kind": "adopted"},
        source=tmp_path / "split.json",
    )
    run_dir = tmp_path / "run"
    sandbox = CellSandbox(run_id="test", home=run_dir / "home", workdir=run_dir / "work")
    sandbox.home.mkdir(parents=True)
    # One durable file in the base home, so every task must be able to read the accumulated self.
    memory = sandbox.home / ".claude" / "projects" / "-work" / "memory"
    memory.mkdir(parents=True)
    (memory / "note.md").write_text("accumulated lesson\n", encoding="utf-8")
    return RunContext(
        cell=cell,
        split=split,
        instruction=load_instruction(cell.instruction_arm),
        harness=harness_for(cell.harness),
        run_id="test",
        run_dir=run_dir,
        sandbox=sandbox,
    )


def test_the_eval_phase_runs_in_parallel_keys_every_result_and_survives_a_failure(
    tmp_path: Path, monkeypatch
) -> None:
    # Ids deliberately out of ascending order and larger than the concurrency limit.
    heldout = ("7", "3", "5", "1", "6", "2", "8", "4")
    failing = 5
    ctx = _ctx(tmp_path, heldout)
    ctx = replace(ctx, cell=replace(ctx.cell, budget=replace(ctx.cell.budget, eval_concurrency=3)))
    limit = ctx.cell.budget.eval_concurrency

    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0
    prompts: list[str] = []
    homes_seen: dict[int, Path] = {}
    works_seen: dict[int, Path] = {}

    def fake_run_leg(ctx_arg: RunContext, **kw: object) -> LegRecord:
        nonlocal in_flight, max_in_flight
        idx = int(kw["task_idx"])  # type: ignore[arg-type]
        home = Path(kw["home"])  # type: ignore[arg-type]
        work = Path(kw["workdir"])  # type: ignore[arg-type]
        # Every task reads the accumulated durable self out of its own isolated home.
        note = home / ".claude/projects/-work/memory/note.md"
        assert note.read_text() == "accumulated lesson\n"
        # And writes into a /work of its own that starts empty: a fresh scratch cwd, never a
        # shared one, so a file this task drops there cannot be read by a sibling.
        assert work.is_dir() and list(work.iterdir()) == []
        (work / "from-task.txt").write_text(str(idx), encoding="utf-8")
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            homes_seen[idx] = home
            works_seen[idx] = work
            prompts.append(str(kw["system_prompt"]))
        # Hold the slot long enough that the bounded set of siblings genuinely overlap.
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        if idx == failing:
            raise RuntimeError("this task could not be run")
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
        if idx not in homes_seen or idx == failing:
            return []  # never ran, or ran and failed: unscored, like reconcile would record
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

    rows = asyncio.run(runner.run_eval_phase(ctx, "eval_before"))

    # Every requested task is present, keyed to its id, and reported in ascending id order
    # regardless of the shuffled input order or the order the tasks finished in. The failing one
    # included: it produced no row of its own, and an id that is simply absent from this list is
    # an id that disappears from the published measurement while the manifest goes on saying it
    # was requested. It is here, unscored, saying why.
    assert [r.task_idx for r in rows] == [1, 2, 3, 4, 5, 6, 7, 8]
    lost = next(r for r in rows if r.task_idx == failing)
    assert lost.closure == "missing" and not lost.scored
    assert lost.reward is None and lost.success is None
    assert "this task could not be run" in lost.diagnostic
    # The batch really ran in parallel, and never past the knob.
    assert 2 <= max_in_flight <= limit
    assert max_in_flight == limit
    # Each task ran against a distinct isolated home, and all were discarded afterward.
    assert len(set(homes_seen.values())) == len(homes_seen)
    for home in homes_seen.values():
        assert not home.exists()
    # And a distinct /work of its own, discarded with the home: no eval-writable directory is
    # shared, which is what keeps one task's file from reaching another (finding 1).
    assert len(set(works_seen.values())) == len(works_seen)
    for work in works_seen.values():
        assert not work.exists()
    # The failing task left a provenance breadcrumb rather than vanishing.
    assert (ctx.run_dir / "eval_before" / f"task-{failing:05d}" / "runner-error.txt").exists()

    # The model-visible prefix (the eval system prompt) is byte-identical across every task, so a
    # provider prompt cache warmed by one task is reused by the next.
    assert set(prompts) == {ctx.instruction.eval_system}


# ----- cache reuse -----------------------------------------------------------------------------


def test_warm_env_provisions_the_upstream_once_and_reuses_it(monkeypatch) -> None:
    cell = load_cell_by_name("smoke-automationbench-claude-code")
    serving._WARMED_ENVS.discard(cell.env)
    calls: list[str] = []

    def fake_make(name: str, config: object = None) -> object:
        calls.append(name)
        return object()

    try:
        serving.warm_env(cell, make=fake_make)
        serving.warm_env(cell, make=fake_make)
        # Two warms, one provision: the second reuses the on-disk cache the first fetched.
        assert calls == [cell.env]
    finally:
        serving._WARMED_ENVS.discard(cell.env)


def test_warm_env_survives_an_env_that_cannot_be_torn_down(monkeypatch) -> None:
    cell = load_cell_by_name("smoke-automationbench-claude-code")
    serving._WARMED_ENVS.discard(cell.env)

    class _Env:
        def close(self) -> None:
            raise RuntimeError("teardown blew up")

    try:
        serving.warm_env(cell, make=lambda name, config=None: _Env())  # must not raise
        assert cell.env in serving._WARMED_ENVS
    finally:
        serving._WARMED_ENVS.discard(cell.env)


# ----- eval_after forks the rollout's terminal session ---------------------------------------


_ROLLOUT_SID = "cccccccc-4444-4444-4444-cccccccccccc"


def _rollout_terminus(ctx: RunContext, session_id: str = _ROLLOUT_SID) -> None:
    """The record a finished rollout leaves behind: the stopping file naming its terminal
    session, and that session's transcript in the cell's base home."""
    (ctx.run_dir / "rollout_stopping.json").write_text(
        json.dumps({"stop_reason": "pool_exhausted", "session_id": session_id}),
        encoding="utf-8",
    )
    transcript = ctx.sandbox.home / ".claude" / "projects" / "-work" / f"{session_id}.jsonl"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text('{"type":"user"}\n', encoding="utf-8")


def _capture_launches(monkeypatch, launches: dict[int, dict]) -> None:
    """Route the eval fan-out through fakes that record what each task's leg was launched with."""

    def fake_run_leg(ctx_arg: RunContext, **kw: object) -> LegRecord:
        idx = int(kw["task_idx"])  # type: ignore[arg-type]
        home = Path(kw["home"])  # type: ignore[arg-type]
        launches[idx] = {
            "session_id": kw["session_id"],
            "resume": kw["resume"],
            "transcript_in_copy": (
                home / ".claude/projects/-work" / f"{_ROLLOUT_SID}.jsonl"
            ).exists(),
        }
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
        if idx not in launches:
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


def test_a_resumed_eval_after_forks_the_rollout_session_into_every_task(
    tmp_path: Path, monkeypatch
) -> None:
    """Every task resumes the rollout's terminal session, independently: same id in every
    launch, and each fork's own home copy carries the transcript the resume reopens."""
    ctx = _ctx(tmp_path, ("1", "2", "3"))
    assert ctx.cell.eval_context == "resumed"
    _rollout_terminus(ctx)
    launches: dict[int, dict] = {}
    _capture_launches(monkeypatch, launches)

    rows = asyncio.run(runner.run_eval_phase(ctx, "eval_after"))

    assert [r.task_idx for r in rows] == [1, 2, 3]
    assert set(launches) == {1, 2, 3}
    for record in launches.values():
        assert record["session_id"] == _ROLLOUT_SID
        assert record["resume"] is True
        assert record["transcript_in_copy"] is True


def test_eval_before_never_resumes_whatever_the_axis_says(tmp_path: Path, monkeypatch) -> None:
    """The before-bookend has no conversation to carry, so even a resumed cell with a rollout
    terminus already on disk (a resume re-running an interrupted eval_before) starts cold."""
    ctx = _ctx(tmp_path, ("1", "2"))
    _rollout_terminus(ctx)
    launches: dict[int, dict] = {}
    _capture_launches(monkeypatch, launches)

    asyncio.run(runner.run_eval_phase(ctx, "eval_before"))

    assert set(launches) == {1, 2}
    for record in launches.values():
        assert record["resume"] is False
        assert record["session_id"] != _ROLLOUT_SID  # a fresh pinned id, never the rollout's
        assert record["transcript_in_copy"] is False


def test_a_cold_cell_runs_eval_after_cold(tmp_path: Path, monkeypatch) -> None:
    """The recorded ablation: eval_context = "cold" starts fresh even when a resumable rollout
    session is sitting right there, and the transcript stays out of the copies."""
    ctx = _ctx(tmp_path, ("1", "2"))
    ctx = replace(ctx, cell=replace(ctx.cell, eval_context="cold"))
    _rollout_terminus(ctx)
    launches: dict[int, dict] = {}
    _capture_launches(monkeypatch, launches)

    asyncio.run(runner.run_eval_phase(ctx, "eval_after"))

    assert set(launches) == {1, 2}
    for record in launches.values():
        assert record["resume"] is False
        assert record["session_id"] != _ROLLOUT_SID
        assert record["transcript_in_copy"] is False


def test_a_resumed_eval_after_with_no_rollout_session_fails_before_spending(
    tmp_path: Path, monkeypatch
) -> None:
    """No session to fork is a refusal, never a silent fall-back to cold: a phase that ran cold
    under the resumed label would publish a mislabeled measurement. Nothing launches, nothing
    is copied, no stream is built."""
    ctx = _ctx(tmp_path, ("1", "2"))
    launches: dict[int, dict] = {}
    _capture_launches(monkeypatch, launches)

    # No rollout_stopping.json at all: the record names no session.
    with pytest.raises(RuntimeError, match="eval_context"):
        asyncio.run(runner.run_eval_phase(ctx, "eval_after"))
    assert launches == {}
    assert not (ctx.run_dir / "eval_after" / "homes").exists()

    # A session id whose transcript is not in the base home is the same refusal: the id alone
    # resumes nothing, and each fork would discover that only after it had spent.
    (ctx.run_dir / "rollout_stopping.json").write_text(
        json.dumps({"stop_reason": "pool_exhausted", "session_id": _ROLLOUT_SID}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="transcript"):
        asyncio.run(runner.run_eval_phase(ctx, "eval_after"))
    assert launches == {}


def test_the_terminal_session_falls_back_to_the_last_rollout_leg(tmp_path: Path) -> None:
    """A stopping record without a top-level id still names its legs, and the LAST leg's
    session is the terminal one: an earlier leg's session was ended by whatever forced the
    next leg, so forking it would resume a conversation the rollout itself abandoned."""
    ctx = _ctx(tmp_path, ("1",))
    (ctx.run_dir / "rollout_stopping.json").write_text(
        json.dumps(
            {
                "stop_reason": "pool_exhausted",
                "session_id": None,
                "legs": [{"session_id": "early-sid"}, {"session_id": _ROLLOUT_SID}],
            }
        ),
        encoding="utf-8",
    )
    transcript = ctx.sandbox.home / ".claude" / "projects" / "-work" / f"{_ROLLOUT_SID}.jsonl"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text('{"type":"user"}\n', encoding="utf-8")

    assert runner._rollout_terminal_session(ctx) == _ROLLOUT_SID


def test_the_eval_launch_prefix_is_identical_across_tasks(tmp_path: Path) -> None:
    """What a provider caches is the system prompt plus the tool manifest, both byte-identical
    across held-out tasks. Only the per-task connection (the stream url and the fresh session id)
    differs, and neither is part of the cached prefix."""
    harness = harness_for("claude_code")
    instruction = load_instruction("get-better")
    specs = [
        harness.launch(
            mcp_url=f"http://host.docker.internal:{port}/mcp",
            system_prompt=instruction.eval_system,
            user_prompt=instruction.kickoff,
            model="claude-opus-5",
            trace_path=tmp_path / f"t-{port}.jsonl",
            session_id=f"session-{port}",
        )
        for port in (5001, 5002, 5003)
    ]

    def _appended_system_prompt(argv: list[str]) -> str:
        return argv[argv.index("--append-system-prompt") + 1]

    prompts = {_appended_system_prompt(spec.argv) for spec in specs}
    assert prompts == {instruction.eval_system}
