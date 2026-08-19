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
import os
import stat
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from shobench import runner, serving
from shobench.config import load_cell_by_name, load_instruction
from shobench.containers import CellSandbox, home_digest, work_digest
from shobench.harness import StopKind, StopVerdict
from shobench.harnesses import harness_for
from shobench.results import TaskResult
from shobench.runner import (
    LegRecord,
    RunContext,
    _copy_task_home,
    _copy_work_tree,
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


def test_a_prime_fork_copy_carries_the_kernel_and_child_state_independently(
    tmp_path: Path,
) -> None:
    """Prime's persisted session is the transcript PLUS its artifact tree.

    At the pinned 0.7.1 the kernel snapshot is revived from session-artifacts/<id>/ right
    after kernel start and completed RLM children are read back from its sub-* dirs, so a fork
    that copied the transcript alone would reopen the conversation with the learned kernel
    state and every child conversation silently gone. Both must cross, each fork must get its
    own copy, and the operational subtrees beside them (leases, daemon workers) must not.
    """
    sid = "eeeeeeee-6666-6666-6666-eeeeeeeeeeee"
    base = tmp_path / "base"
    (base / ".prime/agent/sessions").mkdir(parents=True)
    (base / ".prime/agent/sessions" / f"{sid}.jsonl").write_text(
        json.dumps({"type": "session", "version": 3, "id": sid, "cwd": "/work"}) + "\n",
        encoding="utf-8",
    )
    artifacts = base / ".prime/agent/session-artifacts" / sid
    (artifacts / "sub-child1").mkdir(parents=True)
    (artifacts / "kernel-state.dill").write_bytes(b"pickled namespace")
    (artifacts / "kernel-state.json").write_text("{}", encoding="utf-8")
    (artifacts / "sub-child1" / "child.jsonl").write_text('{"type":"session"}\n')
    # The operational neighbors that must stay behind: a lease and a daemon worker.
    lease = base / ".prime/agent/session-leases/deadbeef.lock"
    lease.mkdir(parents=True)
    (lease / "owner.json").write_text("{}", encoding="utf-8")
    (base / ".prime/agent/daemon-workers").mkdir(parents=True)
    (base / ".prime/agent/daemon-workers" / "worker.json").write_text("{}", encoding="utf-8")

    keep = harness_for("prime_agent").session_state_dirs
    fork_a = tmp_path / "fork-a"
    fork_b = tmp_path / "fork-b"
    _copy_task_home(base, fork_a, keep=keep)
    _copy_task_home(base, fork_b, keep=keep)

    for fork in (fork_a, fork_b):
        assert (fork / ".prime/agent/sessions" / f"{sid}.jsonl").is_file()
        forked = fork / ".prime/agent/session-artifacts" / sid
        assert (forked / "kernel-state.dill").read_bytes() == b"pickled namespace"
        assert (forked / "kernel-state.json").is_file()
        assert (forked / "sub-child1/child.jsonl").is_file()
        assert not (fork / ".prime/agent/session-leases").exists()
        assert not (fork / ".prime/agent/daemon-workers").exists()

    # Independent copies: one fork's kernel mutating reaches neither the base nor its sibling.
    (fork_a / ".prime/agent/session-artifacts" / sid / "kernel-state.dill").write_bytes(b"MUT")
    assert (artifacts / "kernel-state.dill").read_bytes() == b"pickled namespace"
    assert (
        fork_b / ".prime/agent/session-artifacts" / sid / "kernel-state.dill"
    ).read_bytes() == b"pickled namespace"


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


def _seed_base_work(base: Path) -> None:
    """A base /work shaped like a post-rollout one: what an agent builds in its own cwd."""
    base.mkdir(parents=True, exist_ok=True)
    (base / "helper.py").write_text("def batch_get(ids):\n    ...\n", encoding="utf-8")
    notes = base / "notes"
    notes.mkdir()
    (notes / "what-works.md").write_text("batch, never one at a time\n", encoding="utf-8")
    # A name the durable filter calls noise in a HOME. /work has no such filter: whatever is
    # here, the rollout put here.
    (base / "scratch.log").write_text("a log the agent keeps for itself\n", encoding="utf-8")
    (base / "empty-dir").mkdir()
    (base / "shortcut.md").symlink_to(Path("notes/what-works.md"))
    # Two CONTAINER paths, both ordinary in a rollout and neither meaning on the host what it
    # means at /work: one absolute (the shape a venv's interpreter link takes), one leaving
    # /work altogether.
    (base / "tool").symlink_to(Path("/home/oai/tool"))
    (base / "up-and-out").symlink_to(Path("../elsewhere/thing"))
    # Modes an agent sets on purpose: a helper it made runnable, and a directory it closed.
    (base / "run-me.sh").write_text("#!/bin/sh\necho ready\n", encoding="utf-8")
    os.chmod(base / "run-me.sh", 0o755)
    locked = base / "locked"
    locked.mkdir()
    (locked / "answer.md").write_text("do not edit this\n", encoding="utf-8")
    os.chmod(locked, 0o500)


def test_a_task_work_copy_carries_the_whole_rollout_cwd(tmp_path: Path) -> None:
    """Everything the rollout left in its cwd crosses, unfiltered, links included."""
    base = tmp_path / "work"
    _seed_base_work(base)
    task_work = tmp_path / "task"

    _copy_work_tree(base, task_work)

    assert (task_work / "helper.py").read_text().startswith("def batch_get")
    assert (task_work / "notes/what-works.md").read_text() == "batch, never one at a time\n"
    assert (task_work / "scratch.log").is_file()
    assert (task_work / "empty-dir").is_dir()
    # A link stays a link, so it names in the task what it named in the rollout: the copy is
    # mounted back at /work under the same image, and resolving one on the host would drop the
    # container paths and de-alias the in-tree one.
    assert (task_work / "shortcut.md").is_symlink()
    assert (task_work / "shortcut.md").readlink() == Path("notes/what-works.md")
    assert (task_work / "shortcut.md").read_text() == "batch, never one at a time\n"
    assert (task_work / "tool").readlink() == Path("/home/oai/tool")
    assert (task_work / "up-and-out").readlink() == Path("../elsewhere/thing")
    # And the alias is still an alias rather than two independent files.
    (task_work / "notes/what-works.md").write_text("edited\n", encoding="utf-8")
    assert (task_work / "shortcut.md").read_text() == "edited\n"


def test_a_hard_link_alias_survives_the_copy(tmp_path: Path) -> None:
    """Two names on one inode stay one inode, which is not two files that merely match.

    An agent that hard-linked ``active.py`` to ``helper.py`` edited both by editing either. Copied
    path by path, the exam gets two unrelated files, and the same edit now reaches one of them:
    a cwd that behaves differently from the one the rollout worked in.
    """
    base = tmp_path / "work"
    _seed_base_work(base)
    os.link(base / "helper.py", base / "active.py")
    task_work = tmp_path / "task"

    _copy_work_tree(base, task_work)

    assert (task_work / "helper.py").stat().st_ino == (task_work / "active.py").stat().st_ino
    (task_work / "helper.py").write_text("edited through one name\n", encoding="utf-8")
    assert (task_work / "active.py").read_text() == "edited through one name\n"
    # The alias lives inside the copy, so the exam's edit still reaches neither source name.
    assert (base / "helper.py").read_text().startswith("def batch_get")
    assert (base / "active.py").read_text().startswith("def batch_get")


def test_the_copy_carries_the_modes_the_rollout_set(tmp_path: Path) -> None:
    """A cwd is read through its permissions, so the exam has to inherit them.

    A helper the rollout made executable is a program the session can run, and a directory the
    rollout closed is one the session has to open the same way. Directory modes were recreated
    under the host umask, so a private directory arrived at the exam world-readable, which is
    both a cwd the rollout never had and a mode-only improvement the record could not see.
    """
    base = tmp_path / "work"
    _seed_base_work(base)
    task_work = tmp_path / "task"

    _copy_work_tree(base, task_work)

    def mode(path: Path) -> int:
        return stat.S_IMODE(path.stat().st_mode)

    assert mode(task_work / "run-me.sh") == mode(base / "run-me.sh") == 0o755
    assert mode(task_work / "locked") == mode(base / "locked") == 0o500
    # Applied after the directory was filled, so what it closes is still inside it.
    assert (task_work / "locked/answer.md").read_text() == "do not edit this\n"
    assert mode(task_work / "notes") == mode(base / "notes")


def test_a_task_cwd_is_discarded_even_when_the_rollout_locked_a_directory(tmp_path: Path) -> None:
    """The copy carries the agent's modes, and a read-only directory is one the host cannot
    unlink out of. Left to the plain removal, that copy survives its task silently: it sits on
    the disk of every phase after it and merges into the next attempt at the same held-out id,
    which is the stale-leftover case the copy starts by clearing."""
    base = tmp_path / "work"
    _seed_base_work(base)
    task_work = tmp_path / "task"
    _copy_work_tree(base, task_work)
    (task_work / "the-dead-leg-wrote-this.py").write_text("half a thought\n", encoding="utf-8")

    runner._discard_work_tree(task_work)

    assert not task_work.exists()
    # And the base keeps the modes the discard had to work around.
    assert stat.S_IMODE((base / "locked").stat().st_mode) == 0o500


def test_the_copy_is_the_tree_the_record_describes(tmp_path: Path) -> None:
    """The two halves meet: what a manifest records about a rollout's cwd is what the exam gets.

    Checked by digesting the copy with the record's own definition, which reads kinds, link
    targets and alias groups. A copy that dropped a log, resolved a link into bytes, or split an
    alias into two files would digest differently from the tree it came from.
    """
    base = tmp_path / "work"
    _seed_base_work(base)
    os.link(base / "helper.py", base / "active.py")

    task_work = tmp_path / "task"
    _copy_work_tree(base, task_work)

    assert work_digest(task_work) == work_digest(base)


def test_a_task_work_write_reaches_neither_the_base_nor_a_sibling(tmp_path: Path) -> None:
    base = tmp_path / "work"
    _seed_base_work(base)
    base_digest = home_digest(base, exclude=is_noise)

    work_a = tmp_path / "a"
    work_b = tmp_path / "b"
    _copy_work_tree(base, work_a)
    _copy_work_tree(base, work_b)

    # Task A does what a running agent does in its cwd: writes a file and edits an inherited one.
    (work_a / "the-exam-wrote-this.py").write_text("scratch\n", encoding="utf-8")
    (work_a / "notes/what-works.md").write_text("MUTATED\n", encoding="utf-8")

    assert home_digest(base, exclude=is_noise) == base_digest
    assert (base / "notes/what-works.md").read_text() == "batch, never one at a time\n"
    assert not (work_b / "the-exam-wrote-this.py").exists()
    assert (work_b / "notes/what-works.md").read_text() == "batch, never one at a time\n"


def test_a_copy_clears_what_a_killed_attempt_left_in_the_task_cwd(tmp_path: Path) -> None:
    """A re-run of an id whose first attempt died before its cleanup starts from the rollout's
    cwd, never from the dead leg's own writes sitting on top of it."""
    base = tmp_path / "work"
    _seed_base_work(base)
    task_work = tmp_path / "task"
    _copy_work_tree(base, task_work)
    (task_work / "the-killed-leg-wrote-this.py").write_text("half a thought\n", encoding="utf-8")

    _copy_work_tree(base, task_work)

    assert not (task_work / "the-killed-leg-wrote-this.py").exists()
    assert (task_work / "helper.py").is_file()


def test_copying_a_base_work_that_holds_nothing_yields_an_empty_one(tmp_path: Path) -> None:
    """A run that has not rolled out yet has written nothing into its cwd, and on a fresh run
    the directory does not exist at all. Both copy to the empty cwd a cold baseline needs."""
    absent = tmp_path / "task-from-absent"
    _copy_work_tree(tmp_path / "never-created", absent)
    assert absent.is_dir() and list(absent.rglob("*")) == []

    empty_base = tmp_path / "created-and-empty"
    empty_base.mkdir()
    from_empty = tmp_path / "task-from-empty"
    _copy_work_tree(empty_base, from_empty)
    assert from_empty.is_dir() and list(from_empty.rglob("*")) == []


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
    session, and that session's transcript in the cell's base home. The transcript names its
    session the way a real one does, because the preflight validates rather than globs."""
    (ctx.run_dir / "rollout_stopping.json").write_text(
        json.dumps({"stop_reason": "pool_exhausted", "session_id": session_id}),
        encoding="utf-8",
    )
    transcript = ctx.sandbox.home / ".claude" / "projects" / "-work" / f"{session_id}.jsonl"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(
        json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": "kickoff"},
                "timestamp": "2026-08-12T00:00:00.000Z",
                "sessionId": session_id,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _capture_launches(monkeypatch, launches: dict[int, dict]) -> None:
    """Route the eval fan-out through fakes that record what each task's leg was launched with."""

    def fake_run_leg(ctx_arg: RunContext, **kw: object) -> LegRecord:
        idx = int(kw["task_idx"])  # type: ignore[arg-type]
        home = Path(kw["home"])  # type: ignore[arg-type]
        launches[idx] = {
            "session_id": kw["session_id"],
            "resume": kw["resume"],
            "system_prompt": kw["system_prompt"],
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
    launch, each fork's own home copy carries the transcript the resume reopens, and the
    standing instruction is the ROLLOUT's: the conversation already holds the objective, and
    swapping the instruction mid-conversation would measure an agent that never existed."""
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
        assert record["system_prompt"] == ctx.instruction.rollout_system


def _work_reading_legs(
    monkeypatch, seen: dict[int, dict], barrier: threading.Barrier | None = None
) -> None:
    """Route the fan-out through a leg that reads its own /work and writes into it."""
    lock = threading.Lock()

    def fake_run_leg(ctx_arg: RunContext, **kw: object) -> LegRecord:
        idx = int(kw["task_idx"])  # type: ignore[arg-type]
        work = Path(kw["workdir"])  # type: ignore[arg-type]
        inherited = sorted(p.relative_to(work).as_posix() for p in work.rglob("*"))
        (work / f"task-{idx}.py").write_text("what this session tried\n", encoding="utf-8")
        if barrier is not None:
            # Every task has written before any of them looks, so a shared /work would show.
            barrier.wait()
        with lock:
            seen[idx] = {
                "inherited": inherited,
                "own": sorted(p.name for p in work.glob("task-*.py")),
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
        if idx not in seen:
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


def test_every_eval_after_task_starts_from_the_rollout_cwd_and_none_sees_a_sibling(
    tmp_path: Path, monkeypatch
) -> None:
    """The transfer, through the real phase: three tasks run at once, each reads what the rollout
    left in /work, each writes into a copy of its own, and the rollout's cwd comes out unmoved."""
    ctx = _ctx(tmp_path, ("1", "2", "3"))
    ctx = replace(ctx, cell=replace(ctx.cell, budget=replace(ctx.cell.budget, eval_concurrency=3)))
    _rollout_terminus(ctx)
    _seed_base_work(ctx.sandbox.workdir)
    base_digest = home_digest(ctx.sandbox.workdir, exclude=is_noise)
    seen: dict[int, dict] = {}
    _work_reading_legs(monkeypatch, seen, barrier=threading.Barrier(3, timeout=30))

    rows = asyncio.run(runner.run_eval_phase(ctx, "eval_after"))

    assert [r.task_idx for r in rows] == [1, 2, 3]
    assert set(seen) == {1, 2, 3}
    for idx, record in seen.items():
        # The rollout's own helper and notes, in every task, before that task wrote anything.
        assert "helper.py" in record["inherited"]
        assert "notes/what-works.md" in record["inherited"]
        # And of the three files the three concurrent tasks wrote, a task sees only its own.
        assert record["own"] == [f"task-{idx}.py"]
    # The rollout's cwd is the record of what the rollout did, and no task moved it.
    assert home_digest(ctx.sandbox.workdir, exclude=is_noise) == base_digest
    assert not list(ctx.sandbox.workdir.glob("task-*.py"))
    # And every copy is gone, the rollout's read-only directory inside each of them included: a
    # copy that outlived its task would sit here and merge into the next attempt at that id.
    assert not list((ctx.run_dir / "eval_after" / "work").iterdir())


def test_the_task_copies_run_off_the_event_loop(tmp_path: Path, monkeypatch) -> None:
    """A task's copies share their loop with the stream server and the drain watcher of every
    task already launched, so a tree copied inline pauses live held-out sessions for as long as
    the copy runs, and serializes the setup the concurrency knob was meant to overlap. Both
    trees are copied in a worker thread, and the loop keeps turning while they do.

    Slowed deliberately, because size is what makes this visible: a rollout that cloned a
    repository or built a venv leaves a tree whose copy takes long enough to time a live task
    out, and the archive's own cwds are far too small to show it.
    """
    ctx = _ctx(tmp_path, ("1", "2"))
    _seed_base_work(ctx.sandbox.workdir)
    seen: dict[int, dict] = {}
    _work_reading_legs(monkeypatch, seen)
    copy_threads: set[int] = set()

    def slowed(real):
        def copy(*args, **kwargs):
            copy_threads.add(threading.get_ident())
            time.sleep(0.25)
            return real(*args, **kwargs)

        return copy

    monkeypatch.setattr(runner, "_copy_task_home", slowed(runner._copy_task_home))
    monkeypatch.setattr(runner, "_copy_work_tree", slowed(runner._copy_work_tree))

    async def drive() -> tuple[int, int]:
        beats = 0

        async def pulse() -> None:
            nonlocal beats
            while True:
                await asyncio.sleep(0.01)
                beats += 1

        beating = asyncio.create_task(pulse())
        try:
            await runner.run_eval_phase(ctx, "eval_before")
        finally:
            beating.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await beating
        return threading.get_ident(), beats

    loop_thread, beats = asyncio.run(drive())

    assert set(seen) == {1, 2}
    # A second of filesystem work, none of it on the thread the loop runs on.
    assert copy_threads and loop_thread not in copy_threads
    # And the loop kept turning throughout, which is what an already-launched task's server and
    # drain watcher need. Run inline, the same phase lets it turn once.
    assert beats > 10


def test_an_eval_before_task_starts_from_an_empty_cwd(tmp_path: Path, monkeypatch) -> None:
    """The baseline stays cold, and not because the phase is named eval_before: the rollout is
    the only thing that ever writes the cell's /work and eval_before runs before it, so what
    every task copies is empty. Re-measuring a before after a rollout is refused outright, so no
    path carries an accumulated cwd into a baseline."""
    ctx = _ctx(tmp_path, ("1", "2"))
    seen: dict[int, dict] = {}
    _work_reading_legs(monkeypatch, seen)

    asyncio.run(runner.run_eval_phase(ctx, "eval_before"))

    assert set(seen) == {1, 2}
    assert all(record["inherited"] == [] for record in seen.values())


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
        assert record["system_prompt"] == ctx.instruction.eval_system  # blind, as always


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
        assert record["system_prompt"] == ctx.instruction.eval_system  # cold stays blind


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
    transcript.write_text(
        json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": "kickoff"},
                "timestamp": "2026-08-12T00:00:00.000Z",
                "sessionId": _ROLLOUT_SID,
            }
        )
        + "\n",
        encoding="utf-8",
    )

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
