"""Rebookend: a resumed eval_after for an existing run, published as a new run.

The already-run cells measured their after-bookend cold, and the resumed default cannot reach
them retroactively: their runs are archived artifacts. Rebookend is the entry that gives such
a run the resumed measurement without touching it. Everything here holds the two properties
the entry exists for: the SOURCE run is read and never written (its bytes are the experiment's
record), and the NEW run is an honest artifact of its own (the source's axes inherited, the
eval context resumed, a provenance block naming what it bookends, and the incomplete name a
run with no before-side deserves).

None of this needs Docker or a credential. The fan-out is driven through the same fakes the
resumed eval_after tests use, so the real preflight, home copy, manifest build, and publish
path all run.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from shobench import runner
from shobench.config import load_cell_by_name, load_instruction
from shobench.containers import CellSandbox
from shobench.harness import StopKind, StopVerdict
from shobench.results import TaskResult
from shobench.runner import (
    ROLLOUT_STOPPING_FILE,
    SUSPENSION_FILE,
    LegRecord,
    RunContext,
    build_manifest,
)
from shobench.splits import Side, Split, load_split_by_name

_SID = "cccccccc-4444-4444-4444-cccccccccccc"


class _FakeStream:
    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


@contextlib.asynccontextmanager
async def _fake_served(stream: object, port: int):
    yield


def _synthetic_definitions(tmp_path: Path):
    """A cell and split shaped like the archived never-arm sources: wordle over the smoke
    cell, one-at-a-time eval, the feedback-ablation arm."""
    cell = replace(
        load_cell_by_name("smoke-automationbench-claude-code"),
        env="wordle_v1",
        rollout_feedback="never",
        budget=replace(
            load_cell_by_name("smoke-automationbench-claude-code").budget,
            eval_concurrency=1,
            eval_task_timeout_s=120,
        ),
    )
    split = Split(
        env="wordle_v1",
        heldout=Side(task_ids=("0", "1", "2")),
        pool=Side(task_ids=("3", "4")),
        provenance={"kind": "adopted"},
        source=tmp_path / "split.json",
    )
    return cell, split


def _source_run(
    tmp_path: Path,
    cell,
    split,
    *,
    session_id: str | None = _SID,
    with_terminus: bool = True,
    with_before: bool = True,
) -> Path:
    """An archived source run: manifest, terminus, and the accumulated post-rollout home.

    The manifest is built by the real builder from a COLD-era cell, because the runs this
    entry exists for were measured before the resumed default existed; the terminus and the
    terminal transcript are the ones the resumed fork machinery resolves.
    """
    source_dir = tmp_path / "source-run"
    home = source_dir / "home"
    memory = home / ".claude" / "projects" / "-work" / "memory"
    memory.mkdir(parents=True)
    (memory / "note.md").write_text("accumulated lesson\n", encoding="utf-8")
    transcript = home / ".claude" / "projects" / "-work" / f"{_SID}.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": "kickoff"},
                "timestamp": "2026-08-12T00:00:00.000Z",
                "sessionId": _SID,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # Noise the eval copies leave behind but the rebookend home copy must still preserve in
    # the source: the untouched guarantee is over every byte, not the durable subset.
    cache = home / ".cache"
    cache.mkdir()
    (cache / "blob").write_text("x" * 512, encoding="utf-8")
    source_cell = replace(cell, eval_context="cold")
    ctx = RunContext(
        cell=source_cell,
        split=split,
        instruction=load_instruction(source_cell.instruction_arm),
        harness=runner.harness_for(source_cell.harness),
        run_id="source-run-20260101T000000Z",
        run_dir=source_dir,
        sandbox=CellSandbox(run_id="src", home=home, workdir=source_dir / "work"),
    )
    runner.write_json(source_dir / "manifest.json", build_manifest(ctx, probes={"version": "t"}))
    if with_terminus:
        runner.write_json(
            source_dir / ROLLOUT_STOPPING_FILE,
            {"stop_reason": "agent_stopped_early", "session_id": session_id},
        )
    # Every real run writes its lock on the way in, and a lock-less source is refused as
    # unholdable, so the fixture is a lockable archive like the runs this entry exists for.
    (source_dir / runner.RUN_LOCK_FILE).write_text("{}", encoding="utf-8")
    if with_before:
        # A source that measured its own eval_before is the self-paired default; the v0
        # shapes (rollout-only sources with a separate baseline run) set with_before=False.
        for task_id in split.heldout.task_ids:
            (source_dir / "eval_before" / f"task-{int(task_id):05d}").mkdir(
                parents=True, exist_ok=True
            )
    return source_dir


def _fingerprint(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _wire_fakes(monkeypatch, cell, split, launches: dict[int, dict]) -> None:
    """The same provider-free fan-out the resumed eval_after tests drive, plus the loaders:
    the cell under test is synthetic, so the checkout loaders hand back the recorded
    definitions the drift check verifies, exactly as they would for a committed cell."""

    def fake_run_leg(ctx_arg: RunContext, **kw: object) -> LegRecord:
        idx = int(kw["task_idx"])  # type: ignore[arg-type]
        home = Path(kw["home"])  # type: ignore[arg-type]
        launches[idx] = {
            "session_id": kw["session_id"],
            "resume": kw["resume"],
            "system_prompt": kw["system_prompt"],
            # The bound the leg ran under, which is the stopping rule whatever row it
            # produces was scored by.
            "timeout_s": kw["timeout_s"],
            "transcript_in_copy": (
                home / ".claude/projects/-work" / f"{_SID}.jsonl"
            ).is_file(),
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
        if not prov_dir.name.startswith("task-"):
            return []
        idx = int(prov_dir.name.split("-")[1])
        if "eval_before" in prov_dir.parts:
            # Baseline provenance: the archived before row the creation-time carry reads.
            return [
                TaskResult(
                    seq=idx, position=0, task_idx=idx, closure="sealed", reward=0.25,
                    success=False,
                )
            ]
        if idx not in launches:
            return []
        return [
            TaskResult(
                seq=idx, position=0, task_idx=idx, closure="sealed", reward=1.0, success=True
            )
        ]

    monkeypatch.setattr(runner, "load_cell_by_name", lambda name, **kw: cell)
    monkeypatch.setattr(runner, "load_split_by_name", lambda name, **kw: split)
    monkeypatch.setattr(
        CellSandbox,
        "up",
        lambda self, **kw: (
            self.home.mkdir(parents=True, exist_ok=True),
            self.workdir.mkdir(parents=True, exist_ok=True),
        ),
    )
    monkeypatch.setattr(CellSandbox, "down", lambda self: None)
    monkeypatch.setattr(runner, "seed_home", lambda spec, home: {})
    monkeypatch.setattr(runner, "_probe", lambda *a, **kw: "probe")
    monkeypatch.setattr(runner, "warm_env", lambda cell_arg: None)
    monkeypatch.setattr(runner, "build_stream", lambda *a, **kw: _FakeStream())
    monkeypatch.setattr(runner, "_served", _fake_served)
    monkeypatch.setattr(runner, "run_leg", fake_run_leg)
    monkeypatch.setattr(runner, "read_phase", fake_read_phase)


def test_rebookend_leaves_the_source_untouched_and_publishes_an_honest_bookend(
    tmp_path: Path, monkeypatch
) -> None:
    """The whole entry, end to end: the source's bytes are the experiment's record and none of
    them move, while the new run inherits the source's axes, forces the resumed context, forks
    the source's terminal session out of its own copied home, and publishes under the
    incomplete name a before-less run deserves."""
    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    before = _fingerprint(source_dir)
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    results_path = asyncio.run(
        runner.rebookend_run(
            source_dir,
            runs_dir=tmp_path / "runs",
            results_dir=tmp_path / "results",
            capture_egress=False,
        )
    )

    # The source is byte-identical: same files, same contents, nothing added or removed.
    assert _fingerprint(source_dir) == before

    # Every held-out task forked the SOURCE's terminal session under the rollout instruction,
    # with the transcript present in the fork's own per-task copy: the resumed machinery
    # composed with the copied home rather than being reimplemented.
    assert set(launches) == {0, 1, 2}
    instruction = load_instruction(cell.instruction_arm)
    for record in launches.values():
        assert record["session_id"] == _SID
        assert record["resume"] is True
        assert record["transcript_in_copy"] is True
        assert record["system_prompt"] == instruction.rollout_system

    # The new run is its own directory with its own lock and its own copied terminus, so a
    # usage limit mid-bookend suspends and resumes through the ordinary machinery.
    run_dirs = [p for p in (tmp_path / "runs").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    new_run = run_dirs[0]
    assert new_run.name.startswith(cell.name)
    assert (new_run / ROLLOUT_STOPPING_FILE).is_file()

    # The manifest says what this run is: the source's arm, the resumed context, the rollout
    # instruction on the resumed side, and the provenance block naming the source.
    manifest = json.loads((new_run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cell"]["rollout_feedback"] == "never"
    assert manifest["cell"]["eval_context"] == "resumed"
    assert manifest["instruction"]["eval_prompt_used"] == "rollout_system"
    marker = manifest["rebookend"]
    assert {key: marker[key] for key in marker if key not in ("source_cell", "cell_drift")} == {
        "rebookend_of": "source-run-20260101T000000Z",
        # The source measured its own eval_before, so the baseline defaults to it: the
        # self-paired case, stated in the marker rather than assumed by the reader.
        "baseline_run_id": "source-run-20260101T000000Z",
        "source_rollout_feedback": "never",
        "source_stop_reason": "agent_stopped_early",
        # The stopping rule the legs ran under, taken from the record so the pair's two sides
        # share one. A settled checkout states the same values, and the marker says them
        # anyway, because a reader must not have to infer which side the rule came from.
        "eval_runtime_from_record": {
            "eval_task_timeout_s": cell.budget.eval_task_timeout_s,
            "eval_concurrency": cell.budget.eval_concurrency,
        },
    }
    # The source's recorded cell block is kept whole beside the block this run ran under, and
    # every field the checkout's file states differently is named with both values. Nothing but
    # the axis a rebookend exists to change moved here, and that is what the record says.
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    assert marker["source_cell"] == source_manifest["cell"]
    assert marker["cell_drift"] == {
        "eval_context": {"recorded": "cold", "checkout": "resumed"}
    }

    # Published honestly, under its OWN name, and SELF-CONTAINED: the before block is the
    # baseline's carried rows, labeled with the run they came from, so the paired delta
    # lives inside the artifact and no other file is needed to read it. The carried before
    # side is whole here, so the artifact accounts for every id and takes the finished name.
    assert results_path.name == f"{new_run.name}.json"
    published = json.loads(results_path.read_text(encoding="utf-8"))
    assert published["eval_after"]["summary"]["n_scored"] == 3
    assert published["eval_before"]["source_run_id"] == "source-run-20260101T000000Z"
    assert published["eval_before"]["summary"]["n_scored"] == 3
    assert len(published["paired"]) == 3
    assert all(p["reward_delta"] == pytest.approx(0.75) for p in published["paired"])
    assert published["manifest"]["rebookend"]["rebookend_of"] == "source-run-20260101T000000Z"


def test_rebookend_refuses_a_source_without_a_terminus(tmp_path: Path, monkeypatch) -> None:
    """No rollout ending means no conversation end to resume from: the never-terminus sources
    (a rollout that never reached its stopping record) are correctly unrebookendable."""
    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split, with_terminus=False)
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    with pytest.raises(RuntimeError, match="terminus"):
        asyncio.run(
            runner.rebookend_run(
                source_dir,
                runs_dir=tmp_path / "runs",
                results_dir=tmp_path / "results",
                capture_egress=False,
            )
        )
    assert launches == {}
    assert not (tmp_path / "runs").exists()


def test_rebookend_refuses_a_terminus_that_names_no_session(tmp_path: Path, monkeypatch) -> None:
    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split, session_id=None)
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    with pytest.raises(RuntimeError, match="terminal session"):
        asyncio.run(
            runner.rebookend_run(
                source_dir,
                runs_dir=tmp_path / "runs",
                results_dir=tmp_path / "results",
                capture_egress=False,
            )
        )
    assert launches == {}
    assert not (tmp_path / "runs").exists()


def test_rebookend_refuses_a_suspended_source(tmp_path: Path, monkeypatch) -> None:
    """A suspended run is not finished: its ending belongs to resume, and bookending a run
    mid-interruption would measure the far side of a terminus that does not exist yet."""
    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    (source_dir / SUSPENSION_FILE).write_text("{}", encoding="utf-8")
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    with pytest.raises(RuntimeError, match="suspension"):
        asyncio.run(
            runner.rebookend_run(
                source_dir,
                runs_dir=tmp_path / "runs",
                results_dir=tmp_path / "results",
                capture_egress=False,
            )
        )
    assert launches == {}
    assert not (tmp_path / "runs").exists()


def test_rebookend_preflight_validates_the_transcript_in_the_copied_home(
    tmp_path: Path, monkeypatch
) -> None:
    """The id can be recorded while the conversation is gone; the refusal then comes from the
    resumed preflight, against the COPIED home, before any task is launched, and the source
    stays untouched even though the copy was already made."""
    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    (source_dir / "home" / ".claude" / "projects" / "-work" / f"{_SID}.jsonl").unlink()
    before = _fingerprint(source_dir)
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    with pytest.raises(RuntimeError, match="resumable transcript"):
        asyncio.run(
            runner.rebookend_run(
                source_dir,
                runs_dir=tmp_path / "runs",
                results_dir=tmp_path / "results",
                capture_egress=False,
            )
        )
    assert launches == {}
    assert _fingerprint(source_dir) == before


def test_rebookend_refuses_outputs_inside_the_source(tmp_path: Path, monkeypatch) -> None:
    """The untouched guarantee is over the tree, so no output may land at or under the source,
    whatever the operator typed: the new lock alone would already be a write into the archive."""
    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    before = _fingerprint(source_dir)
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    for runs_dir, results_dir in (
        (source_dir, tmp_path / "results"),
        (source_dir / "nested" / "runs", tmp_path / "results"),
        (tmp_path / "runs", source_dir),
        (tmp_path / "runs", source_dir / "results"),
    ):
        with pytest.raises(RuntimeError, match="inside the source"):
            asyncio.run(
                runner.rebookend_run(
                    source_dir,
                    runs_dir=runs_dir,
                    results_dir=results_dir,
                    capture_egress=False,
                )
            )
    assert launches == {}
    assert _fingerprint(source_dir) == before


def test_the_snapshot_materializes_symlinks_so_no_writer_reaches_the_source(
    tmp_path: Path, monkeypatch
) -> None:
    """The reviewed write-through, closed: a source home whose ``.codex`` is a symlink used to
    be copied AS a link, and the first writer into the new home (the credential reseed, in the
    reproduction) wrote through it into the archive. The snapshot is now materialized: every
    link becomes the bytes it pointed at, nothing in the new tree references the source, and a
    writer can do its worst."""
    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    secrets = source_dir / "home" / "codex-real"
    secrets.mkdir()
    (secrets / "auth.json").write_text('{"auth_mode": "chatgpt"}', encoding="utf-8")
    (source_dir / "home" / ".codex").symlink_to(secrets)
    before = _fingerprint(source_dir)
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    asyncio.run(
        runner.rebookend_run(
            source_dir,
            runs_dir=tmp_path / "runs",
            results_dir=tmp_path / "results",
            capture_egress=False,
        )
    )

    new_run = next(p for p in (tmp_path / "runs").iterdir() if p.is_dir())
    new_home = new_run / "home"
    # The snapshot is self-contained: the link became a real directory with the real bytes,
    # and no link anywhere in the tree can reach outside it.
    assert not (new_home / ".codex").is_symlink()
    assert (new_home / ".codex" / "auth.json").read_text() == '{"auth_mode": "chatgpt"}'
    assert not any(p.is_symlink() for p in new_home.rglob("*"))
    # The reproduction's write, thrown at the copy: it stays in the copy.
    (new_home / ".codex" / "auth.json").write_text('{"auth_mode": "OVERWRITTEN"}')
    assert (secrets / "auth.json").read_text() == '{"auth_mode": "chatgpt"}'
    assert _fingerprint(source_dir) == before


def test_the_snapshot_materializes_valid_relative_links_from_their_own_parent(
    tmp_path: Path, monkeypatch
) -> None:
    """The round-2 silent loss, closed. copytree tested a link's textual target against the
    process CWD, so valid RELATIVE links (the shape of every link in the real prime homes)
    read as dangling and vanished. The materializer resolves from the link's parent, so the
    CWD is irrelevant: valid file and directory links become their bytes, two links to one
    target both materialize, an absolute in-source link materializes, and only the genuinely
    dangling link drops."""
    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    home = source_dir / "home"
    real = home / "real"
    real.mkdir()
    (real / "payload").write_text("payload bytes", encoding="utf-8")
    (home / "file-link").symlink_to("real/payload")
    (home / "dir-link").symlink_to("real")
    (home / "dir-link-two").symlink_to("real")
    (home / "abs-link").symlink_to(real / "payload")
    (home / "dangling").symlink_to("no-such-target")
    before = _fingerprint(source_dir)
    # The repro condition: run somewhere the RAW target strings resolve to nothing.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    asyncio.run(
        runner.rebookend_run(
            source_dir,
            runs_dir=tmp_path / "runs",
            results_dir=tmp_path / "results",
            capture_egress=False,
        )
    )

    new_home = next(p for p in (tmp_path / "runs").iterdir() if p.is_dir()) / "home"
    assert (new_home / "file-link").read_text() == "payload bytes"
    assert (new_home / "dir-link" / "payload").read_text() == "payload bytes"
    assert (new_home / "dir-link-two" / "payload").read_text() == "payload bytes"
    assert (new_home / "abs-link").read_text() == "payload bytes"
    assert not (new_home / "dangling").exists()
    assert not any(p.is_symlink() for p in new_home.rglob("*"))
    assert _fingerprint(source_dir) == before


def test_the_snapshot_refuses_a_link_escaping_the_source_home(
    tmp_path: Path, monkeypatch
) -> None:
    """A link resolving outside the source home refuses loudly rather than importing bytes
    the run never saw: the snapshot must be OF the source, and with the write-through history
    a silent import is the same class of surprise. (Both real homes carry only in-home
    relative links, so this policy costs the report set nothing.)"""
    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    outside = source_dir / "secrets"
    outside.mkdir()
    (outside / "auth.json").write_text('{"auth_mode": "chatgpt"}', encoding="utf-8")
    (source_dir / "home" / ".codex").symlink_to(outside)
    before = _fingerprint(source_dir)
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    with pytest.raises(RuntimeError, match="outside the source home"):
        asyncio.run(
            runner.rebookend_run(
                source_dir,
                runs_dir=tmp_path / "runs",
                results_dir=tmp_path / "results",
                capture_egress=False,
            )
        )
    assert launches == {}
    assert _fingerprint(source_dir) == before


def test_the_materializer_fails_loudly_on_a_link_cycle(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / "sub").mkdir(parents=True)
    (home / "sub" / "back").symlink_to("..")

    with pytest.raises(RuntimeError, match="cycles"):
        runner._materialize_home(home, tmp_path / "copy")


def test_rebookend_refuses_its_minted_names_resolving_into_the_source(
    tmp_path: Path, monkeypatch
) -> None:
    """The bookend's own names are unpredictable before minting, so nothing can pre-occupy
    them in the wild; the check still bounds the minted concrete names before the lock, and
    with the mint pinned deterministic here, a link planted at the exact leaf refuses before
    anything launches."""

    class _FixedUuid:
        hex = "deadbeefcafe0123"

    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    archived = source_dir / "archive-byte"
    archived.write_text("ARCHIVED BYTES", encoding="utf-8")
    monkeypatch.setattr(runner, "_run_id", lambda c: f"{c.name}-20260101T000000Z")
    monkeypatch.setattr(runner.uuid, "uuid4", lambda: _FixedUuid())
    minted = f"{cell.name}-20260101T000000Z-rbdeadbeef"
    results = tmp_path / "results"
    results.mkdir()
    (results / f"{minted}.incomplete.json").symlink_to(archived)
    before = _fingerprint(source_dir)
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    with pytest.raises(RuntimeError, match="resolves into the source"):
        asyncio.run(
            runner.rebookend_run(
                source_dir,
                runs_dir=tmp_path / "runs",
                results_dir=results,
                capture_egress=False,
            )
        )
    assert launches == {}
    assert archived.read_text() == "ARCHIVED BYTES"
    assert _fingerprint(source_dir) == before


def test_rebookend_refuses_a_source_a_live_process_owns(tmp_path: Path, monkeypatch) -> None:
    """A settled-looking terminus under a live owner is not an archived state: the source lock
    is probed non-mutatingly, and a held lock refuses before anything is copied."""
    import fcntl

    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    lock_path = source_dir / runner.RUN_LOCK_FILE
    lock_path.write_text("{}", encoding="utf-8")
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    holder = open(lock_path)
    try:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="live process"):
            asyncio.run(
                runner.rebookend_run(
                    source_dir,
                    runs_dir=tmp_path / "runs",
                    results_dir=tmp_path / "results",
                    capture_egress=False,
                )
            )
        assert launches == {}
        assert not (tmp_path / "runs").exists()
    finally:
        holder.close()

    # Every hard ending releases the lock in the kernel, so a lock file with no holder is a
    # finished run and the probe passes without writing anything.
    stat_before = lock_path.stat()
    runner._refuse_live_source(source_dir)
    assert lock_path.stat().st_mtime == stat_before.st_mtime
    assert lock_path.read_text(encoding="utf-8") == "{}"


def test_two_rebookends_of_one_source_in_one_second_get_distinct_runs(
    tmp_path: Path, monkeypatch
) -> None:
    """The timestamped stem has one-second resolution, so uniqueness cannot ride on the clock:
    both calls in the same second must each get a fresh run rather than one stealing the
    other's lock refusal."""
    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    monkeypatch.setattr(runner, "_run_id", lambda c: f"{c.name}-20260101T000000Z")
    first: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, first)
    asyncio.run(
        runner.rebookend_run(
            source_dir,
            runs_dir=tmp_path / "runs",
            results_dir=tmp_path / "results",
            capture_egress=False,
        )
    )
    second: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, second)
    asyncio.run(
        runner.rebookend_run(
            source_dir,
            runs_dir=tmp_path / "runs",
            results_dir=tmp_path / "results",
            capture_egress=False,
        )
    )

    run_dirs = [p for p in (tmp_path / "runs").iterdir() if p.is_dir()]
    assert len(run_dirs) == 2
    assert len({p.name for p in run_dirs}) == 2
    assert set(first) == set(second) == {0, 1, 2}
    # And two coexisting artifacts, one per bookend, neither under the cell name.
    artifacts = sorted(p.name for p in (tmp_path / "results").iterdir())
    assert len(artifacts) == 2
    assert artifacts == sorted(f"{p.name}.json" for p in run_dirs)


def test_a_suspended_bookend_resumes_with_nothing_recorded(tmp_path: Path, monkeypatch) -> None:
    """A bookend that hits a usage limit must publish the same shape an uninterrupted bookend
    publishes. Its stopping file is the SOURCE's terminus, kept for the resumed preflight, and
    an ordinary eval_after resume would republish it as this run's rollout; the rebookend
    marker narrows the recorded set to nothing instead."""
    cell, split = _synthetic_definitions(tmp_path)
    run_dir = tmp_path / "bookend-run"
    home = run_dir / "home"
    home.mkdir(parents=True)
    ctx = RunContext(
        cell=replace(cell, eval_context="resumed"),
        split=split,
        instruction=load_instruction(cell.instruction_arm),
        harness=runner.harness_for(cell.harness),
        run_id="bookend-run-1",
        run_dir=run_dir,
        sandbox=CellSandbox(run_id="b", home=home, workdir=run_dir / "work"),
    )
    manifest = build_manifest(ctx, probes={"version": "t"})
    manifest["rebookend"] = {
        "rebookend_of": "source-run-20260101T000000Z",
        "source_rollout_feedback": "never",
        "source_stop_reason": "agent_stopped_early",
    }
    runner.write_json(run_dir / "manifest.json", manifest)
    runner.write_json(
        run_dir / ROLLOUT_STOPPING_FILE,
        {"stop_reason": "agent_stopped_early", "session_id": _SID},
    )
    runner.write_json(
        run_dir / SUSPENSION_FILE,
        {
            "schema": "shobench.suspension/1",
            "run_id": "bookend-run-1",
            "cell": cell.name,
            "harness": cell.harness,
            "phase": "eval_after",
            "legs_before": 0,
            "completed_task_ids": [0],
            "pending_task_ids": [1, 2],
            "stop_evidence": {"kind": "usage_limit", "reason": "t", "resumable": True},
            "suspended_at": 1.0,
            "resume_with": "uv run shobench resume",
        },
    )

    captured: dict[str, object] = {}

    async def fake_run_phases(ctx_arg, *, manifest, phases, results_dir, observer, **kwargs):
        captured["phases"] = phases
        captured["recorded_phases"] = kwargs.get("recorded_phases")
        captured["artifact"] = kwargs.get("artifact")
        return results_dir / "x.json"

    monkeypatch.setattr(runner, "_run_phases", fake_run_phases)
    monkeypatch.setattr(runner, "load_cell_by_name", lambda name, **kw: cell)
    monkeypatch.setattr(runner, "load_split_by_name", lambda name, **kw: split)
    monkeypatch.setattr(
        CellSandbox, "up", lambda self, **kw: self.home.mkdir(parents=True, exist_ok=True)
    )
    monkeypatch.setattr(CellSandbox, "down", lambda self: None)
    monkeypatch.setattr(runner, "seed_home", lambda spec, home_arg: {})
    monkeypatch.setattr(runner, "_start_egress", lambda sandbox, rd: None)

    asyncio.run(
        runner.resume_cell(run_dir, results_dir=tmp_path / "results", capture_egress=False)
    )

    assert captured["phases"] == ("eval_after",)
    assert captured["recorded_phases"] == ()
    # The reopened bookend keeps its own artifact stem: the cell-name fallback is the
    # source's artifact, and republishing under it is the round-5 destruction resurrected.
    assert captured["artifact"] == "bookend-run-1"


def _published_pair(tmp_path: Path) -> tuple[Path, str, str]:
    """A results directory holding a real source artifact and a real bookend artifact, both
    written by the real publisher, joined only by the bookend's provenance block."""
    from shobench.results import TaskResult, write_results

    results = tmp_path / "results"
    results.mkdir(exist_ok=True)

    def row(idx: int, reward: float) -> TaskResult:
        return TaskResult(
            seq=idx, position=idx, task_idx=idx, closure="sealed", reward=reward,
            success=reward >= 1.0,
        )

    source_manifest = {
        "run_id": "cell-a-20260101T000000Z",
        "cell": {
            "name": "cell-a", "env": "wordle_v1", "harness": "claude_code", "model": "m",
            "rollout_feedback": "never", "eval_context": "cold",
        },
    }
    write_results(
        results / "cell-a.json",
        manifest=source_manifest,
        phases={
            "eval_before": [row(1, 0.2), row(2, 0.4)],
            "eval_after": [row(1, 0.3), row(2, 0.4)],
            "rollout": [],
        },
        stopping={"stop_reason": "agent_stopped_early"},
        heldout_ids=[1, 2],
    )
    bookend_manifest = {
        "run_id": "cell-a-20260102T000000Z-rb01234567",
        "cell": {
            "name": "cell-a", "env": "wordle_v1", "harness": "claude_code", "model": "m",
            "rollout_feedback": "never", "eval_context": "resumed",
        },
        "rebookend": {
            "rebookend_of": "cell-a-20260101T000000Z",
            "source_rollout_feedback": "never",
            "source_stop_reason": "agent_stopped_early",
        },
    }
    write_results(
        results / f"{bookend_manifest['run_id']}.json",
        manifest=bookend_manifest,
        phases={
            "eval_before": [],
            "eval_after": [row(1, 0.9), row(2, 0.8)],
            "rollout": [],
        },
        stopping={},
        heldout_ids=[1, 2],
    )
    return results, str(source_manifest["run_id"]), str(bookend_manifest["run_id"])


def test_the_assembler_pairs_a_bookend_with_its_source(tmp_path: Path) -> None:
    """The report's own machinery: a bookend artifact alone is n_paired 0 by construction,
    because its before side lives in the SOURCE artifact its provenance names. Assembled, the
    source's before rows pair with the bookend's after rows through the same pair_evals every
    publisher uses, and both rows carry the identity that keeps duplicate cell names apart."""
    from shobench.report import assemble, load_results, render_table, report_cell

    results, source_id, bookend_id = _published_pair(tmp_path)
    docs = assemble(load_results(results))
    reports = {r.run_id: r for r in (report_cell(d, resamples=200, seed=1) for d in docs)}

    source = reports[source_id]
    assert source.pairing == "self"
    assert source.n_paired == 2
    assert source.mean_delta == pytest.approx(0.05)
    assert source.eval_context == "cold"

    bookend = reports[bookend_id]
    assert bookend.pairing == "assembled"
    assert bookend.rebookend_of == source_id
    assert bookend.n_paired == 2
    # The assembled delta is bookend-after minus SOURCE-before: (0.9-0.2, 0.8-0.4).
    assert bookend.mean_before == pytest.approx(0.3)
    assert bookend.mean_after == pytest.approx(0.85)
    assert bookend.mean_delta == pytest.approx(0.55)
    assert bookend.rollout_feedback == "never"
    assert bookend.eval_context == "resumed"
    assert bookend.complete is True

    table = render_table(sorted(reports.values(), key=lambda r: r.run_id))
    assert "20260102T000000Z-rb01234567" in table  # the run column disambiguates
    assert "never+resumed" in table and "never+cold" in table
    assert "of 20260101T000000Z" in table
    as_json = [r.to_json() for r in reports.values()]
    assert {j["run_id"] for j in as_json} == {source_id, bookend_id}
    assert any(j["rebookend_of"] == source_id for j in as_json)


def test_a_bookend_without_its_baseline_surfaces_explicitly(tmp_path: Path) -> None:
    """A measurement that cannot find its other half is a fact the reader needs, never a
    silent unpaired zero: the row says BASELINE MISSING in the table and in the JSON."""
    from shobench.report import assemble, load_results, render_table, report_cell

    results, source_id, bookend_id = _published_pair(tmp_path)
    (results / "cell-a.json").unlink()
    docs = assemble(load_results(results))
    (report,) = [report_cell(d, resamples=100, seed=1) for d in docs]

    assert report.run_id == bookend_id
    assert report.pairing == "baseline_missing"
    assert report.n_paired == 0
    table = render_table([report])
    assert "BASELINE MISSING" in table
    assert report.to_json()["pairing"] == "baseline_missing"


def test_two_bookends_of_one_source_both_assemble(tmp_path: Path) -> None:
    from shobench.report import assemble, load_results, report_cell
    from shobench.results import TaskResult, write_results

    results, source_id, first_id = _published_pair(tmp_path)
    second_manifest = {
        "run_id": "cell-a-20260103T000000Z-rb89abcdef",
        "cell": {
            "name": "cell-a", "env": "wordle_v1", "harness": "claude_code", "model": "m",
            "rollout_feedback": "never", "eval_context": "resumed",
        },
        "rebookend": {
            "rebookend_of": source_id,
            "source_rollout_feedback": "never",
            "source_stop_reason": "agent_stopped_early",
        },
    }
    write_results(
        results / f"{second_manifest['run_id']}.json",
        manifest=second_manifest,
        phases={
            "eval_before": [],
            "eval_after": [
                TaskResult(seq=1, position=1, task_idx=1, closure="sealed", reward=0.5,
                           success=False),
                TaskResult(seq=2, position=2, task_idx=2, closure="sealed", reward=0.6,
                           success=False),
            ],
            "rollout": [],
        },
        stopping={},
        heldout_ids=[1, 2],
    )

    docs = assemble(load_results(results))
    reports = {r.run_id: r for r in (report_cell(d, resamples=100, seed=1) for d in docs)}
    assert reports[first_id].pairing == "assembled"
    assert reports[second_manifest["run_id"]].pairing == "assembled"
    assert reports[first_id].n_paired == 2
    assert reports[second_manifest["run_id"]].n_paired == 2
    assert reports[second_manifest["run_id"]].mean_delta == pytest.approx(0.25)


def test_rebookend_refuses_a_bookend_as_its_source(tmp_path: Path, monkeypatch, capsys) -> None:
    """A bookend of a bookend re-measures the same terminal state as rebookending the
    original directly: the bookend's home IS the source's terminal home, copied, and its own
    eval_after advanced no rollout. Chaining adds provenance to unwind and no information, so
    the runner refuses at acceptance, naming the original, and the plan surfaces the state."""
    from shobench.cli import main as cli_main

    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["rebookend"] = {
        "rebookend_of": "the-original-run",
        "source_rollout_feedback": "never",
        "source_stop_reason": "agent_stopped_early",
    }
    runner.write_json(source_dir / "manifest.json", manifest)
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    with pytest.raises(RuntimeError, match="the-original-run"):
        asyncio.run(
            runner.rebookend_run(
                source_dir,
                runs_dir=tmp_path / "runs",
                results_dir=tmp_path / "results",
                capture_egress=False,
            )
        )
    assert launches == {}
    assert not (tmp_path / "runs").exists()

    cli_source = _real_cell_source(tmp_path / "cli")
    cli_manifest = json.loads((cli_source / "manifest.json").read_text(encoding="utf-8"))
    cli_manifest["rebookend"] = {"rebookend_of": "the-original-run"}
    runner.write_json(cli_source / "manifest.json", cli_manifest)
    assert cli_main(["rebookend", "--run", str(cli_source)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["refusals"]["source_is_bookend"] is True
    assert cli_main(["rebookend", "--run", str(cli_source), "--go"]) == 1
    err = capsys.readouterr().err
    assert "BLOCKED" in err and "the-original-run" in err


def test_the_assembler_labels_chains_and_cycles_invalid_never_assembled(
    tmp_path: Path,
) -> None:
    """The reporter's independent defense: an artifact whose named source is itself a bookend
    can never assemble, because a bookend's before-side does not exist. A chain's second hop
    and both halves of a cycle are labeled invalid provenance, explicitly, in the table and
    the JSON; the first hop still assembles against the real source."""
    from shobench.report import assemble, load_results, render_table, report_cell
    from shobench.results import TaskResult, write_results

    results, source_id, first_id = _published_pair(tmp_path)
    chained_manifest = {
        "run_id": "cell-a-20260104T000000Z-rbfeedfeed",
        "cell": {
            "name": "cell-a", "env": "wordle_v1", "harness": "claude_code", "model": "m",
            "rollout_feedback": "never", "eval_context": "resumed",
        },
        "rebookend": {"rebookend_of": first_id},
    }
    write_results(
        results / f"{chained_manifest['run_id']}.json",
        manifest=chained_manifest,
        phases={
            "eval_before": [],
            "eval_after": [
                TaskResult(seq=1, position=1, task_idx=1, closure="sealed", reward=0.6,
                           success=False),
            ],
            "rollout": [],
        },
        stopping={},
        heldout_ids=[1, 2],
    )
    docs = assemble(load_results(results))
    reports = {r.run_id: r for r in (report_cell(d, resamples=100, seed=1) for d in docs)}
    assert reports[first_id].pairing == "assembled"
    chained = reports[chained_manifest["run_id"]]
    assert chained.pairing == "invalid_provenance"
    assert chained.n_paired == 0
    table = render_table(sorted(reports.values(), key=lambda r: r.run_id))
    assert "INVALID PROVENANCE" in table
    assert chained.to_json()["pairing"] == "invalid_provenance"

    # A cycle among bookends is the same refusal on both halves.
    def bookend_doc(run_id: str, of: str) -> dict:
        return {
            "schema": "shobench.results/1",
            "manifest": {
                "run_id": run_id,
                "cell": {"name": "c"},
                "rebookend": {"rebookend_of": of},
            },
            "heldout": {"task_ids": [1]},
            "eval_before": {"tasks": [], "summary": {}},
            "eval_after": {"tasks": [], "summary": {}},
            "rollout": {"summary": {}, "stopping": {}, "tasks": []},
            "paired": [],
            "unpaired": [],
        }

    cycle = assemble([bookend_doc("x", "y"), bookend_doc("y", "x")])
    assert [report_cell(d, resamples=50, seed=1).pairing for d in cycle] == [
        "invalid_provenance",
        "invalid_provenance",
    ]


def test_legacy_artifacts_render_their_recorded_arms(tmp_path: Path) -> None:
    """The pre-axis backfill semantics reach the report: absence means never and cold, the
    way recorded_rollout_feedback, recorded_eval_context, and write_results already define
    it, so the real legacy artifacts render never+cold and immediate+cold rather than
    question marks. Pinned against the checked-in artifact shape and a synthetic pre-axis
    manifest."""
    from shobench.config import repo_root
    from shobench.report import load_results, report_cell

    legacy = {
        "schema": "shobench.results/1",
        "manifest": {"run_id": "r", "cell": {"name": "c"}},
        "heldout": {"task_ids": []},
        "eval_before": {"tasks": [], "summary": {}},
        "eval_after": {"tasks": [], "summary": {}},
        "rollout": {"summary": {}, "stopping": {}, "tasks": []},
        "paired": [],
        "unpaired": [],
    }
    report = report_cell(legacy, resamples=50, seed=1)
    assert report.rollout_feedback == "never"
    assert report.eval_context == "cold"

    for doc in load_results(repo_root() / "results"):
        report = report_cell(doc, resamples=50, seed=1)
        assert report.rollout_feedback in ("never", "immediate"), report.cell
        assert report.eval_context in ("cold", "resumed"), report.cell
        assert "?" not in f"{report.rollout_feedback}{report.eval_context}"


def _baseline_run(tmp_path: Path, cell, split, *, name: str = "baseline-run") -> Path:
    """A deferred-baseline run: its own manifest over the same cell and split, its own
    eval_before provenance, and nothing else. The v0 baselines are exactly this shape."""
    baseline_dir = tmp_path / name
    home = baseline_dir / "home"
    home.mkdir(parents=True)
    ctx = RunContext(
        cell=replace(cell, eval_context="cold"),
        split=split,
        instruction=load_instruction(cell.instruction_arm),
        harness=runner.harness_for(cell.harness),
        run_id=f"{name}-20260101T000000Z",
        run_dir=baseline_dir,
        sandbox=CellSandbox(run_id="b", home=home, workdir=baseline_dir / "work"),
    )
    runner.write_json(
        baseline_dir / "manifest.json", build_manifest(ctx, probes={"version": "t"})
    )
    for task_id in split.heldout.task_ids:
        (baseline_dir / "eval_before" / f"task-{int(task_id):05d}").mkdir(parents=True)
    return baseline_dir


def test_a_rollout_only_source_pairs_through_its_named_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    """The v0 shape: the rollout source measured no eval_before of its own, the baseline is a
    separate run, and the marker records both identities: the source as terminal-state
    lineage, the baseline as the pairing partner the assembler selects."""
    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split, with_before=False)
    baseline_dir = _baseline_run(tmp_path, cell, split)
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    asyncio.run(
        runner.rebookend_run(
            source_dir,
            runs_dir=tmp_path / "runs",
            results_dir=tmp_path / "results",
            baseline_run_dir=baseline_dir,
            capture_egress=False,
        )
    )

    new_run = next(p for p in (tmp_path / "runs").iterdir() if p.is_dir())
    manifest = json.loads((new_run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rebookend"]["rebookend_of"] == "source-run-20260101T000000Z"
    assert manifest["rebookend"]["baseline_run_id"] == "baseline-run-20260101T000000Z"
    assert set(launches) == {0, 1, 2}
    # The artifact self-contains the BASELINE's before rows, labeled, and pairs inside.
    artifact = json.loads(
        next((tmp_path / "results").glob(f"{new_run.name}*.json")).read_text(encoding="utf-8")
    )
    assert artifact["eval_before"]["source_run_id"] == "baseline-run-20260101T000000Z"
    assert artifact["eval_before"]["summary"]["n_scored"] == 3
    assert len(artifact["paired"]) == 3
    # And the carry survives in the run dir for any reopening to republish from.
    assert (new_run / runner.BASELINE_BEFORE_FILE).is_file()


def test_a_before_less_source_requires_a_named_baseline(tmp_path: Path, monkeypatch) -> None:
    """Without a baseline the bookend has nothing to pair with, and pairing against the
    source's emptiness rendered every v0 row 0/120: the absence refuses in the runner and
    blocks in the plan, naming why."""

    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split, with_before=False)
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    with pytest.raises(RuntimeError, match="no eval_before of its own"):
        asyncio.run(
            runner.rebookend_run(
                source_dir,
                runs_dir=tmp_path / "runs",
                results_dir=tmp_path / "results",
                capture_egress=False,
            )
        )
    assert launches == {}
    assert not (tmp_path / "runs").exists()


def test_cli_plan_surfaces_the_baseline_states(tmp_path: Path, capsys) -> None:
    from shobench.cli import main as cli_main

    source_dir = _real_cell_source(tmp_path)
    import shutil as _shutil

    _shutil.rmtree(source_dir / "eval_before")

    assert cli_main(["rebookend", "--run", str(source_dir)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["refusals"]["source_has_own_eval_before"] is False
    assert plan["refusals"]["baseline_required"] is True
    assert plan["baseline_run_id"] is None
    assert cli_main(["rebookend", "--run", str(source_dir), "--go"]) == 1
    err = capsys.readouterr().err
    assert "BLOCKED" in err and "--baseline" in err

    baseline_dir = _real_cell_source(tmp_path / "b")
    assert (
        cli_main(
            ["rebookend", "--run", str(source_dir), "--baseline", str(baseline_dir)]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["refusals"]["baseline_required"] is False
    assert plan["refusals"]["baseline_cell_matches"] is True
    assert plan["refusals"]["baseline_split_matches"] is True
    assert plan["refusals"]["baseline_has_eval_before"] is True
    assert plan["baseline_run_id"] == "source-run-20260101T000000Z"


def test_a_baseline_over_a_different_split_or_cell_refuses(
    tmp_path: Path, monkeypatch
) -> None:
    """The load-bearing check: before rows over different held-out ids would pair task
    numbers that are not the same tasks, and a different cell is not this measurement."""
    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split, with_before=False)
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    other_split = Split(
        env="wordle_v1",
        heldout=Side(task_ids=("7", "8")),
        pool=Side(task_ids=("9",)),
        provenance={"kind": "adopted"},
        source=tmp_path / "other-split.json",
    )
    mismatched = _baseline_run(tmp_path, cell, other_split, name="baseline-other-split")
    with pytest.raises(RuntimeError, match="same held-out ids"):
        asyncio.run(
            runner.rebookend_run(
                source_dir,
                runs_dir=tmp_path / "runs",
                results_dir=tmp_path / "results",
                baseline_run_dir=mismatched,
                capture_egress=False,
            )
        )

    foreign = _baseline_run(tmp_path, cell, split, name="baseline-other-cell")
    manifest = json.loads((foreign / "manifest.json").read_text(encoding="utf-8"))
    manifest["cell"]["name"] = "some-other-cell"
    runner.write_json(foreign / "manifest.json", manifest)
    with pytest.raises(RuntimeError, match="pairing across cells"):
        asyncio.run(
            runner.rebookend_run(
                source_dir,
                runs_dir=tmp_path / "runs",
                results_dir=tmp_path / "results",
                baseline_run_dir=foreign,
                capture_egress=False,
            )
        )

    bookend_baseline = _baseline_run(tmp_path, cell, split, name="baseline-bookend")
    manifest = json.loads((bookend_baseline / "manifest.json").read_text(encoding="utf-8"))
    manifest["rebookend"] = {"rebookend_of": "elsewhere"}
    runner.write_json(bookend_baseline / "manifest.json", manifest)
    with pytest.raises(RuntimeError, match="no before-side to"):
        asyncio.run(
            runner.rebookend_run(
                source_dir,
                runs_dir=tmp_path / "runs",
                results_dir=tmp_path / "results",
                baseline_run_dir=bookend_baseline,
                capture_egress=False,
            )
        )
    assert launches == {}
    assert not (tmp_path / "runs").exists()


def test_a_legacy_bookend_still_joins_against_a_loaded_baseline_artifact(
    tmp_path: Path,
) -> None:
    """The LEGACY path: a bookend published before the carry existed has no rows of its own
    and no source_run_id label, so the assembler joins it against the loaded baseline
    artifact, exactly as before. New bookends are self-contained and never reach this join."""
    from shobench.report import assemble, load_results, report_cell
    from shobench.results import TaskResult, write_results

    results = tmp_path / "results"
    results.mkdir()
    ids = list(range(120))

    def row(idx: int, reward: float) -> TaskResult:
        return TaskResult(
            seq=idx, position=idx, task_idx=idx, closure="sealed", reward=reward,
            success=False,
        )

    write_results(
        results / "cell-a.json",
        manifest={
            "run_id": "rollout-source",
            "cell": {"name": "cell-a", "rollout_feedback": "never", "eval_context": "cold"},
        },
        phases={"eval_before": [], "eval_after": [], "rollout": []},
        stopping={"stop_reason": "agent_stopped_early"},
        heldout_ids=ids,
    )
    write_results(
        results / "baseline-run.json",
        manifest={
            "run_id": "baseline-run",
            "cell": {"name": "cell-a", "rollout_feedback": "never", "eval_context": "cold"},
        },
        phases={
            "eval_before": [row(i, 0.25) for i in ids if i not in (7, 11)],
            "eval_after": [],
            "rollout": [],
        },
        stopping={},
        heldout_ids=ids,
    )
    write_results(
        results / "bk.json",
        manifest={
            "run_id": "bk",
            "cell": {"name": "cell-a", "rollout_feedback": "never", "eval_context": "resumed"},
            "rebookend": {
                "rebookend_of": "rollout-source",
                "baseline_run_id": "baseline-run",
                "source_rollout_feedback": "never",
                "source_stop_reason": "agent_stopped_early",
            },
        },
        phases={
            "eval_before": [],
            "eval_after": [row(i, 0.75) for i in ids],
            "rollout": [],
        },
        stopping={},
        heldout_ids=ids,
    )

    docs = assemble(load_results(results))
    reports = {r.run_id: r for r in (report_cell(d, resamples=100, seed=1) for d in docs)}
    bookend = reports["bk"]
    assert bookend.pairing == "assembled"
    assert bookend.baseline_run_id == "baseline-run"
    assert bookend.rebookend_of == "rollout-source"
    assert bookend.n_paired == 118
    assert bookend.n_unpaired == 2
    assert bookend.mean_delta == pytest.approx(0.5)


def _self_contained_bookend(results: Path, *, ids: list[int], before_holes: tuple[int, ...]) -> str:
    """A bookend artifact published the way the runner now publishes one: the baseline's
    before rows carried in and labeled, the pairing computed inside write_results."""
    from shobench.results import TaskResult, write_results

    def row(idx: int, reward: float) -> TaskResult:
        return TaskResult(
            seq=idx, position=idx, task_idx=idx, closure="sealed", reward=reward,
            success=False,
        )

    run_id = "cell-a-20260105T000000Z-rbc0ffee00"
    write_results(
        results / f"{run_id}.json",
        manifest={
            "run_id": run_id,
            "cell": {"name": "cell-a", "rollout_feedback": "never", "eval_context": "resumed"},
            "rebookend": {
                "rebookend_of": "rollout-source",
                "baseline_run_id": "baseline-run",
                "source_rollout_feedback": "never",
                "source_stop_reason": "agent_stopped_early",
            },
        },
        phases={
            "eval_before": [row(i, 0.25) for i in ids if i not in before_holes],
            "eval_after": [row(i, 0.75) for i in ids],
            "rollout": [],
        },
        stopping={},
        heldout_ids=ids,
        before_source_run_id="baseline-run",
    )
    return run_id


def test_a_self_contained_bookend_reports_with_zero_cross_file_dependence(
    tmp_path: Path,
) -> None:
    """The round-9 property, at the real scale: the artifact carries the baseline's 118 of
    120 scored before rows, labeled, and the 118/120 pairing lives INSIDE it, so the report
    renders it correctly with no other file loaded at all: the baseline artifact being
    evicted by a later same-cell publication (the real report directory's state for three of
    the four baselines) changes nothing."""
    from shobench.report import assemble, load_results, render_table, report_cell

    results = tmp_path / "results"
    results.mkdir()
    ids = list(range(120))
    run_id = _self_contained_bookend(results, ids=ids, before_holes=(7, 11))

    # The artifact itself carries the pairing.
    artifact = json.loads((results / f"{run_id}.incomplete.json").read_text(encoding="utf-8"))
    assert artifact["eval_before"]["source_run_id"] == "baseline-run"
    assert len(artifact["paired"]) == 118

    # And the report needs nothing else: this is the ONLY file in the directory.
    docs = assemble(load_results(results))
    (report,) = [report_cell(d, resamples=100, seed=1) for d in docs]
    assert report.pairing == "assembled"
    assert report.n_paired == 118
    assert report.n_unpaired == 2
    assert report.mean_delta == pytest.approx(0.5)
    assert report.baseline_run_id == "baseline-run"
    assert report.rebookend_of == "rollout-source"
    table = render_table([report])
    assert "of baseline-run" in table


def test_a_marker_bearing_run_refuses_to_publish_without_its_carried_rows(
    tmp_path: Path,
) -> None:
    """Publication is where self-containment is enforced: a rebookend whose carried baseline
    rows are gone must not publish an empty before under the resumed label, whichever
    process is publishing (the first run, a resume, a repair)."""
    cell, split = _synthetic_definitions(tmp_path)
    run_dir = tmp_path / "bookend-run"
    home = run_dir / "home"
    home.mkdir(parents=True)
    ctx = RunContext(
        cell=replace(cell, eval_context="resumed"),
        split=split,
        instruction=load_instruction(cell.instruction_arm),
        harness=runner.harness_for(cell.harness),
        run_id="bookend-run-1",
        run_dir=run_dir,
        sandbox=CellSandbox(run_id="b", home=home, workdir=run_dir / "work"),
    )
    manifest = build_manifest(ctx, probes={"version": "t"})
    manifest["rebookend"] = {
        "rebookend_of": "src",
        "baseline_run_id": "baseline",
        "source_rollout_feedback": "never",
        "source_stop_reason": "agent_stopped_early",
    }

    with pytest.raises(RuntimeError, match="carried baseline rows"):
        asyncio.run(
            runner._run_phases(
                ctx,
                manifest=manifest,
                phases=(),
                results_dir=tmp_path / "results",
                observer=runner._Egress(None, run_dir),
                artifact="bookend-run-1",
            )
        )
    assert not (tmp_path / "results").exists()


def test_the_source_is_held_still_for_the_whole_snapshot(tmp_path: Path, monkeypatch) -> None:
    """A probe released before the copy left the copy racing any mutator that acquired in
    between (a concurrent rerun's write landed in the published snapshot, in review). The
    shared hold spans the materialization: a would-be exclusive owner is refused for as long
    as the copy runs, which is asserted from inside the copy itself."""
    import fcntl

    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    (source_dir / runner.RUN_LOCK_FILE).write_text("{}", encoding="utf-8")
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    real_materialize = runner._materialize_home
    held: dict[str, bool] = {}

    def probing_materialize(source, destination, **kw):
        fd = os.open(source_dir / runner.RUN_LOCK_FILE, os.O_RDONLY)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held["exclusive_acquirable_mid_copy"] = True
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                held["exclusive_acquirable_mid_copy"] = False
        finally:
            os.close(fd)
        return real_materialize(source, destination, **kw)

    monkeypatch.setattr(runner, "_materialize_home", probing_materialize)

    asyncio.run(
        runner.rebookend_run(
            source_dir,
            runs_dir=tmp_path / "runs",
            results_dir=tmp_path / "results",
            capture_egress=False,
        )
    )

    # A mutator's exclusive acquisition failed WHILE the snapshot was being taken, and the
    # bookend itself completed: the hold is shared, so the reader held it and the writer
    # could not.
    assert held == {"exclusive_acquirable_mid_copy": False}
    assert set(launches) == {0, 1, 2}


def test_a_source_mutated_between_read_and_snapshot_refuses(
    tmp_path: Path, monkeypatch
) -> None:
    """The window between the manifest read and the hold: a mutator that ran whole inside it
    leaves no live lock to refuse, so the manifest is re-read under the hold and any drift
    refuses rather than snapshotting new bytes under the old definition."""
    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    real_acquire = runner._acquire_run_lock

    def mutating_acquire(run_dir: Path) -> int:
        manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["mutated_after_the_read"] = True
        runner.write_json(source_dir / "manifest.json", manifest)
        return real_acquire(run_dir)

    monkeypatch.setattr(runner, "_acquire_run_lock", mutating_acquire)

    with pytest.raises(RuntimeError, match="manifest changed"):
        asyncio.run(
            runner.rebookend_run(
                source_dir,
                runs_dir=tmp_path / "runs",
                results_dir=tmp_path / "results",
                capture_egress=False,
            )
        )
    assert launches == {}


def test_the_snapshot_keeps_directory_modes(tmp_path: Path, monkeypatch) -> None:
    """mkdir alone stamped every directory with the process defaults, which widened the real
    homes' 0700 directories (session leases, daemon caches) to 0755: a loosened mode is not
    the snapshot. Directory metadata is copied after contents, for ordinary directories and
    materialized link targets alike."""
    import stat as stat_module

    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    home = source_dir / "home"
    locked = home / "locked"
    locked.mkdir()
    (locked / "secret.txt").write_text("s", encoding="utf-8")
    os.chmod(locked, 0o700)
    real = home / "real-target"
    real.mkdir()
    (real / "payload").write_text("p", encoding="utf-8")
    os.chmod(real, 0o700)
    (home / "dir-link").symlink_to("real-target")
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    asyncio.run(
        runner.rebookend_run(
            source_dir,
            runs_dir=tmp_path / "runs",
            results_dir=tmp_path / "results",
            capture_egress=False,
        )
    )

    new_home = next(p for p in (tmp_path / "runs").iterdir() if p.is_dir()) / "home"
    for rel in ("locked", "real-target", "dir-link"):
        mode = stat_module.S_IMODE((new_home / rel).stat().st_mode)
        assert mode == 0o700, (rel, oct(mode))


def test_a_case_variant_in_home_link_materializes(tmp_path: Path, monkeypatch) -> None:
    """On a case-insensitive volume one directory answers to many spellings, and a lexical
    prefix test refused a valid in-home link whose target spelled the home differently. The
    boundary is filesystem identity now, so the spelling is irrelevant."""
    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    home = source_dir / "home"
    (home / "real").mkdir()
    (home / "real" / "payload").write_text("payload bytes", encoding="utf-8")
    variant = Path(str(home).replace("source-run", "SOURCE-RUN")) / "real" / "payload"
    if not variant.exists():
        pytest.skip("filesystem is case-sensitive; the casing trap cannot exist here")
    (home / "case-link").symlink_to(variant)
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    asyncio.run(
        runner.rebookend_run(
            source_dir,
            runs_dir=tmp_path / "runs",
            results_dir=tmp_path / "results",
            capture_egress=False,
        )
    )

    new_home = next(p for p in (tmp_path / "runs").iterdir() if p.is_dir()) / "home"
    assert (new_home / "case-link").read_text() == "payload bytes"
    assert not (new_home / "case-link").is_symlink()


def _real_cell_source(tmp_path: Path) -> Path:
    """A source built from the committed smoke cell, so the CLI's checkout loaders and drift
    check run for real."""
    cell = load_cell_by_name("smoke-automationbench-claude-code")
    split = load_split_by_name(cell.split)
    source_dir = tmp_path / "source-run"
    home = source_dir / "home"
    home.mkdir(parents=True)
    (home / "notes.md").write_text("post-rollout self\n", encoding="utf-8")
    transcript = home / ".claude" / "projects" / "-work" / f"{_SID}.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": "kickoff"},
                "timestamp": "2026-08-12T00:00:00.000Z",
                "sessionId": _SID,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ctx = RunContext(
        cell=cell,
        split=split,
        instruction=load_instruction(cell.instruction_arm),
        harness=runner.harness_for(cell.harness),
        run_id="source-run-20260101T000000Z",
        run_dir=source_dir,
        sandbox=CellSandbox(run_id="src", home=home, workdir=source_dir / "work"),
    )
    runner.write_json(source_dir / "manifest.json", build_manifest(ctx, probes={"version": "t"}))
    runner.write_json(
        source_dir / ROLLOUT_STOPPING_FILE,
        {"stop_reason": "pool_exhausted", "session_id": _SID},
    )
    (source_dir / runner.RUN_LOCK_FILE).write_text("{}", encoding="utf-8")
    (source_dir / "eval_before" / "task-00000").mkdir(parents=True, exist_ok=True)
    return source_dir


def test_cli_rebookend_plans_without_spending(tmp_path: Path, capsys) -> None:
    """The dry plan names everything a --go would commit to, and exits clean."""
    from shobench.cli import main as cli_main

    source_dir = _real_cell_source(tmp_path)
    cell = load_cell_by_name("smoke-automationbench-claude-code")
    split = load_split_by_name(cell.split)

    assert cli_main(["rebookend", "--run", str(source_dir)]) == 0
    plan = json.loads(capsys.readouterr().out)

    assert plan["source_run_id"] == "source-run-20260101T000000Z"
    assert plan["cell"] == cell.name
    assert plan["axes"] == {
        "rollout_feedback": cell.rollout_feedback,
        "eval_context": "resumed",
        "eval_prompt_used": "rollout_system",
    }
    assert plan["source_stop_reason"] == "pool_exhausted"
    assert plan["terminal_session_id"] == _SID
    # A rebookend is a fresh bookend, never a repair: every held-out task is pending.
    assert plan["heldout_tasks_to_run"] == len(split.heldout)
    assert plan["source_home"]["files"] >= 1
    assert plan["source_home"]["bytes"] > 0
    assert plan["refusals"]["suspension_present"] is False
    assert plan["refusals"]["source_lock_present"] is True
    assert plan["refusals"]["source_live"] is False
    assert plan["refusals"]["outputs_inside_source"] == []
    assert "result_leaves_inside_source" not in plan["refusals"]
    assert plan["result_artifact"].startswith("<bookend-run-id>.json")
    assert plan["refusals"]["rollout_terminus_present"] is True
    assert plan["refusals"]["terminal_session_resolvable"] is True
    assert plan["refusals"]["terminal_transcript_resolvable"] is True
    assert plan["refusals"]["experiment_drift"] == []
    # The plan names the real partner: the artifact is self-contained, carrying the
    # baseline run's before rows under an explicit label.
    assert plan["result_artifact"].startswith("<bookend-run-id>.json")
    assert "source-run-20260101T000000Z" in plan["result_artifact"]
    assert "eval_before.source_run_id" in plan["result_artifact"]


def test_cli_plan_refuses_an_unresolvable_transcript(tmp_path: Path, capsys) -> None:
    """The reviewed fixture shape: a stopping record that names an id whose transcript is not
    in the source home. The plan used to say resolvable and only the runner refused, after
    minting; the plan now runs the same per-harness validation and blocks the --go."""
    from shobench.cli import main as cli_main

    source_dir = _real_cell_source(tmp_path)
    transcript = source_dir / "home" / ".claude" / "projects" / "-work" / f"{_SID}.jsonl"
    transcript.unlink()

    assert cli_main(["rebookend", "--run", str(source_dir)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["refusals"]["terminal_session_resolvable"] is True
    assert plan["refusals"]["terminal_transcript_resolvable"] is False

    assert cli_main(["rebookend", "--run", str(source_dir), "--go"]) == 1
    err = capsys.readouterr().err
    assert "BLOCKED" in err and "no resumable transcript" in err
    assert not (tmp_path / "runs").exists()


def test_cli_rebookend_reports_and_blocks_on_a_refusal_state(tmp_path: Path, capsys) -> None:
    """A missing terminus is visible in the dry plan and blocks a --go before anything spends:
    the runner entry is never reached."""
    from shobench.cli import main as cli_main

    source_dir = _real_cell_source(tmp_path)
    (source_dir / ROLLOUT_STOPPING_FILE).unlink()

    assert cli_main(["rebookend", "--run", str(source_dir)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["refusals"]["rollout_terminus_present"] is False
    assert plan["terminal_session_id"] is None

    assert cli_main(["rebookend", "--run", str(source_dir), "--go"]) == 1
    err = capsys.readouterr().err
    assert "BLOCKED" in err and "Nothing was spent" in err
    assert not (tmp_path / "runs").exists()


def test_the_bookend_and_the_source_results_coexist(tmp_path: Path, monkeypatch) -> None:
    """The whole point of the namespace: the cell-name artifact is the SOURCE's measurement,
    the one the bookend pairs with, and sharing that stem destroyed it (write_results keeps
    one artifact per stem by design; reproduced). The bookend publishes under its own run id,
    so the source result, in either of its shapes, and every bookend all coexist."""
    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    results = tmp_path / "results"
    results.mkdir()
    # The source's published artifact, in the report set's real shape (incomplete), plus the
    # finished-name shape for good measure: a same-stem publish would have removed one and
    # replaced the other.
    (results / f"{cell.name}.incomplete.json").write_text('{"marker": "SOURCE"}')
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    results_path = asyncio.run(
        runner.rebookend_run(
            source_dir,
            runs_dir=tmp_path / "runs",
            results_dir=results,
            capture_egress=False,
        )
    )

    new_run = next(p for p in (tmp_path / "runs").iterdir() if p.is_dir())
    assert results_path.name == f"{new_run.name}.json"
    assert (results / f"{cell.name}.incomplete.json").read_text() == '{"marker": "SOURCE"}'
    assert json.loads(results_path.read_text())["manifest"]["rebookend"][
        "rebookend_of"
    ] == "source-run-20260101T000000Z"


def test_rebookend_refuses_an_unlockable_source(tmp_path: Path, monkeypatch, capsys) -> None:
    """A source without a lock file cannot be held still: a mutator would CREATE the lock and
    write mid-copy (reproduced), and creating it from here would write into the archive. The
    refusal names the operator's own workaround, and the plan surfaces the state; the real
    report-set sources all carry their locks, so this costs them nothing."""
    from shobench.cli import main as cli_main

    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    (source_dir / runner.RUN_LOCK_FILE).unlink()
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    with pytest.raises(RuntimeError, match="cannot be held still"):
        asyncio.run(
            runner.rebookend_run(
                source_dir,
                runs_dir=tmp_path / "runs",
                results_dir=tmp_path / "results",
                capture_egress=False,
            )
        )
    assert launches == {}

    cli_source = _real_cell_source(tmp_path / "cli")
    (cli_source / runner.RUN_LOCK_FILE).unlink()
    assert cli_main(["rebookend", "--run", str(cli_source)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["refusals"]["source_lock_present"] is False
    assert cli_main(["rebookend", "--run", str(cli_source), "--go"]) == 1
    err = capsys.readouterr().err
    assert "BLOCKED" in err and "run.lock" in err


def test_a_suspension_written_between_probe_and_hold_refuses(
    tmp_path: Path, monkeypatch
) -> None:
    """A usage-limit ending writes suspended.json without touching the manifest, so the
    manifest recheck alone missed it (reproduced interleaving: an owner took the lock, wrote
    the suspension, released, and the copy proceeded). Suspension eligibility is re-proven
    under the hold, before anything is copied."""
    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    real_acquire = runner._acquire_run_lock

    def suspending_acquire(run_dir: Path) -> int:
        # The mutator that ran whole between the early probe and the hold: it suspended the
        # source and released the lock, leaving the manifest untouched.
        runner.write_json(source_dir / SUSPENSION_FILE, {"phase": "eval_after"})
        return real_acquire(run_dir)

    monkeypatch.setattr(runner, "_acquire_run_lock", suspending_acquire)

    with pytest.raises(RuntimeError, match="suspended between the plan and the snapshot"):
        asyncio.run(
            runner.rebookend_run(
                source_dir,
                runs_dir=tmp_path / "runs",
                results_dir=tmp_path / "results",
                capture_egress=False,
            )
        )
    assert launches == {}


# ----- the drift comparison a bookend applies -------------------------------------------------
#
# A rebookend is a NEW run over a rollout that already ended, so the cell file's digest is the
# wrong question: it moves for a comment and for a swapped model alike, and it refused the real
# planned bookends after their cells' eval timeout was retuned. What the bookend measures has to
# be the source's arm, and the rule its rows are scored by has to be the rule the before side was
# scored by, so the arm and the eval runtime are taken from the RECORD and everything refuses on
# drift.

# The two planned bookends recorded this bound, and their cells now carry a shorter one. Every
# other field of those cells is untouched, which is exactly the shape modeled here.
_RECORDED_EVAL_TIMEOUT_S = 1800


def _retuned_timeout_source(cell_name: str):
    """The recorded definitions of a run whose cell has since had only its eval timeout retuned.

    Built from the COMMITTED cell rather than a synthetic one, so the comparison runs against
    the real field set: same cell name, the bound the run recorded, and the digest of the file
    as it read then.
    """
    cell = load_cell_by_name(cell_name)
    current_text = cell.source.read_text(encoding="utf-8")
    recorded_text = current_text.replace(
        f"eval_task_timeout_s = {cell.budget.eval_task_timeout_s}",
        f"eval_task_timeout_s = {_RECORDED_EVAL_TIMEOUT_S}",
    )
    assert recorded_text != current_text, "the fixture models a cell whose timeout moved"
    split = load_split_by_name(cell.split)
    instruction = load_instruction(cell.instruction_arm)
    recorded_cell = cell.to_manifest()
    recorded_cell["budget"] = {
        **recorded_cell["budget"],
        "eval_task_timeout_s": _RECORDED_EVAL_TIMEOUT_S,
    }
    recorded_cell["config_sha256"] = hashlib.sha256(recorded_text.encode("utf-8")).hexdigest()
    manifest = {
        "run_id": f"{cell_name}-20260813T003200Z",
        "cell": recorded_cell,
        "split": split.to_manifest(),
        "instruction": instruction.to_manifest(),
    }
    return manifest, cell, split, instruction


def _bookend_drift(manifest, cell, split, instruction, *, recover: bool = True):
    """The refusal lines a rebookend would raise, over the cell it would actually run."""
    return runner.experiment_drift(
        manifest,
        cell=runner.bookend_cell(cell, manifest) if recover else cell,
        split=split,
        instruction=instruction,
        scope=runner.DRIFT_BOOKEND,
    )


@pytest.mark.parametrize(
    "cell_name",
    [
        "automationbench-prime_agent-claude-opus-5",
        "automationbench-prime_agent-gpt-56-terra",
    ],
)
def test_a_bookend_of_a_retuned_cell_runs_under_the_recorded_bound(cell_name: str) -> None:
    """The two real refusals, and the reason the fix is inheritance rather than permission.

    Both sources finished their rollouts before the timeout moved, so the edit cannot reach
    them and the bookend is measurable. It runs under the RECORDED bound, not the file's: the
    before side it will be paired against was scored by that bound, and a task that seals
    between the two would otherwise be scoreable before and force-stopped after.
    """
    manifest, cell, split, instruction = _retuned_timeout_source(cell_name)

    assert _bookend_drift(manifest, cell, split, instruction) == []
    will_run = runner.bookend_cell(cell, manifest)
    assert will_run.budget.eval_task_timeout_s == _RECORDED_EVAL_TIMEOUT_S
    assert will_run.budget.eval_task_timeout_s != cell.budget.eval_task_timeout_s
    assert will_run.budget.eval_concurrency == manifest["cell"]["budget"]["eval_concurrency"]

    # The same edit still stops a continuation dead: resume and rerun-eval write more of a
    # measurement that already exists, and nothing about that one may move.
    continuation = runner.experiment_drift(
        manifest, cell=cell, split=split, instruction=instruction
    )
    assert continuation and "cell config changed" in continuation[0]

    # Inherited, never hidden: the record names what the checkout would have run instead.
    drift = runner.cell_field_drift(manifest["cell"], cell.to_manifest())
    assert drift["budget.eval_task_timeout_s"] == {
        "recorded": _RECORDED_EVAL_TIMEOUT_S,
        "checkout": cell.budget.eval_task_timeout_s,
    }
    assert "config_sha256" in drift, "the reader is told the file itself moved"


def test_the_bookend_inherits_the_recorded_eval_concurrency() -> None:
    """Concurrency is not score-neutral: concurrent legs share the host, the network and the
    provider account while their per-task clocks run, so contention and throttling become
    timeouts and unscored rows. It comes from the record for the same reason the bound does."""
    manifest, cell, split, instruction = _retuned_timeout_source(
        "automationbench-prime_agent-claude-opus-5"
    )
    manifest["cell"]["budget"]["eval_concurrency"] = cell.budget.eval_concurrency + 3

    will_run = runner.bookend_cell(cell, manifest)
    assert will_run.budget.eval_concurrency == cell.budget.eval_concurrency + 3
    assert _bookend_drift(manifest, cell, split, instruction) == []


def test_an_unrecoverable_eval_runtime_refuses() -> None:
    """A record with no bound to inherit is an absence, not a value. Nothing here can say what
    a run measured before the field existed, so it refuses rather than lending the file's."""
    manifest, cell, split, instruction = _retuned_timeout_source(
        "automationbench-prime_agent-claude-opus-5"
    )
    del manifest["cell"]["budget"]["eval_task_timeout_s"]

    will_run = runner.bookend_cell(cell, manifest)
    assert will_run.budget.eval_task_timeout_s == cell.budget.eval_task_timeout_s
    drift = _bookend_drift(manifest, cell, split, instruction)
    assert drift and any("budget.eval_task_timeout_s" in line for line in drift), drift
    assert any(runner.CELL_FIELD_ABSENT in line for line in drift), drift


@pytest.mark.parametrize(
    ("field", "recorded_value", "names"),
    [
        ("model", "claude-opus-4", "cell model"),
        ("env", "wordle_v1", "cell env"),
        ("harness", "claude_code", "cell harness"),
        ("effort", "low", "cell effort"),
        ("credential_mode", "api_key", "cell credential_mode"),
        ("env_kwargs", {"judge_model": "a-different-judge"}, "cell env_kwargs"),
        ("required_env", ["OPENAI_API_KEY"], "cell required_env"),
        ("max_in_flight", 1, "cell max_in_flight"),
    ],
)
def test_a_bookend_refuses_a_measurement_change(field, recorded_value, names) -> None:
    """Everything the bookend's own eval sessions are made of: a bookend under any of these
    measures something other than the run it claims to follow."""
    manifest, cell, split, instruction = _retuned_timeout_source(
        "automationbench-prime_agent-claude-opus-5"
    )
    manifest["cell"][field] = recorded_value

    drift = _bookend_drift(manifest, cell, split, instruction)
    assert drift and any(names in line for line in drift), drift


def test_a_bookend_holds_its_inheritances_to_their_word() -> None:
    """The arm and the eval runtime are inherited rather than read, and the comparison is what
    proves the inheritance happened: a cell handed in with the checkout's values instead is
    refused, not published under a rule its before side never wore."""
    manifest, cell, split, instruction = _retuned_timeout_source(
        "automationbench-prime_agent-claude-opus-5"
    )
    manifest["cell"]["rollout_feedback"] = "never"

    assert _bookend_drift(manifest, cell, split, instruction) == []
    drift = _bookend_drift(
        manifest, replace(cell, eval_context="resumed"), split, instruction, recover=False
    )
    assert any("cell rollout_feedback" in line for line in drift), drift
    assert any("cell budget.eval_task_timeout_s" in line for line in drift), drift


@pytest.mark.parametrize("field", ["rollout_wall_clock_s", "pool_ceiling"])
def test_a_bookend_refuses_a_changed_rollout_budget(field: str) -> None:
    """The bookend runs no rollout, so these cannot change its eval; it inherits the home that
    rollout built and labels the arm, so publishing today's numbers would name a rollout
    nobody ran."""
    manifest, cell, split, instruction = _retuned_timeout_source(
        "automationbench-prime_agent-claude-opus-5"
    )
    manifest["cell"]["budget"][field] = 7

    drift = _bookend_drift(manifest, cell, split, instruction)
    assert drift and any(f"cell budget.{field}" in line for line in drift), drift


@pytest.mark.parametrize(
    ("block", "key", "names"),
    [
        ("split", "id_digest", "split ids"),
        ("instruction", "rollout_system_sha256", "rollout instruction"),
        ("instruction", "eval_system_sha256", "eval instruction"),
    ],
)
def test_a_bookend_refuses_a_changed_split_or_instruction(block, key, names) -> None:
    """The held-out ids and the prompts are compared under both scopes: a bookend over other
    ids pairs task numbers that are not the same tasks, and one under a reworded prompt
    measures a different question."""
    manifest, cell, split, instruction = _retuned_timeout_source(
        "automationbench-prime_agent-claude-opus-5"
    )
    manifest[block][key] = "0" * 64

    drift = _bookend_drift(manifest, cell, split, instruction)
    assert drift and any(names in line for line in drift), drift


def test_an_axis_the_record_predates_refuses() -> None:
    """The comparison walks the UNION of the two field sets, so a cell axis added after a run
    was recorded surfaces instead of passing.

    Comparing only the recorded fields was the hole: every historical manifest lacks a newly
    added field, so classifying the new axis as measurement-defining changed nothing and the
    fail-closed rule was an enumeration nobody enforced.
    """
    manifest, cell, split, instruction = _retuned_timeout_source(
        "automationbench-prime_agent-claude-opus-5"
    )
    del manifest["cell"]["effort"]

    drift = _bookend_drift(manifest, cell, split, instruction)
    assert drift and any("cell effort" in line for line in drift), drift
    assert runner.cell_field_drift(manifest["cell"], cell.to_manifest())["effort"] == {
        "recorded": runner.CELL_FIELD_ABSENT,
        "checkout": cell.effort,
    }


def test_a_field_the_cell_no_longer_carries_refuses() -> None:
    """The reverse direction, which the intersection also passed: a field the record carries
    and the cell has since dropped is a definition the checkout can no longer state."""
    manifest, cell, split, instruction = _retuned_timeout_source(
        "automationbench-prime_agent-claude-opus-5"
    )
    manifest["cell"]["budget"]["leg_wall_clock_s"] = 900

    drift = _bookend_drift(manifest, cell, split, instruction)
    assert drift and any("cell budget.leg_wall_clock_s" in line for line in drift), drift


def test_only_the_versioned_legacy_axes_read_absence_as_a_value() -> None:
    """A pre-axis manifest carries no rollout_feedback and no eval_context, and those two
    absences have known meanings: never was the only rollout posture then, cold the only eval
    posture. Nothing else is normalized, which is what keeps the union honest."""
    manifest, cell, split, instruction = _retuned_timeout_source(
        "automationbench-prime_agent-claude-opus-5"
    )
    # A wave-1 record: written before either axis existed, so both keys are simply absent.
    del manifest["cell"]["rollout_feedback"]
    del manifest["cell"]["eval_context"]

    # The arm recovers to never and the comparison agrees; the eval context is the axis the
    # bookend changes, so it is reported rather than refused.
    assert runner.bookend_cell(cell, manifest).rollout_feedback == "never"
    assert _bookend_drift(manifest, cell, split, instruction) == []
    drift = runner.cell_field_drift(manifest["cell"], cell.to_manifest())
    assert drift["rollout_feedback"] == {"recorded": "never", "checkout": cell.rollout_feedback}
    assert drift["eval_context"] == {"recorded": "cold", "checkout": cell.eval_context}


def test_every_cell_field_is_judged() -> None:
    """A cell axis added later must not fall into the uncompared set by default.

    The uncompared fields are listed in the runner and the refusing ones are everything else,
    so this is where the second list is written down: a new field breaks it, and whoever adds
    the field decides which side it belongs on rather than inheriting an answer.
    """
    fields = set(
        runner._flat_cell(
            load_cell_by_name("automationbench-prime_agent-claude-opus-5").to_manifest()
        )
    )
    uncompared = set(runner.BOOKEND_UNCOMPARED_CELL_FIELDS)
    assert uncompared <= fields, "an uncompared field no cell carries is a stale judgement"
    assert fields - uncompared == {
        "name",
        "env",
        "harness",
        "model",
        "effort",
        "max_in_flight",
        "rollout_feedback",
        "credential_mode",
        "env_kwargs",
        "required_env",
        "budget.rollout_wall_clock_s",
        "budget.pool_ceiling",
        "budget.eval_task_timeout_s",
        "budget.eval_concurrency",
    }


def test_an_unknown_drift_scope_is_refused() -> None:
    """The scope decides what may change, so a misspelled one must not quietly pick a default."""
    manifest, cell, split, instruction = _retuned_timeout_source(
        "automationbench-prime_agent-claude-opus-5"
    )

    with pytest.raises(ValueError, match="unknown drift scope"):
        runner.experiment_drift(
            manifest, cell=cell, split=split, instruction=instruction, scope="whatever"
        )


def test_a_baseline_measured_under_another_bound_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """The pairing check the stopping rule deserves. The bookend runs under the SOURCE's
    recorded bound, so a baseline scored under a different one would put the two sides of the
    pair under two rules, and the delta would measure the bounds as much as the agent."""
    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split, with_before=False)
    baseline_dir = _baseline_run(tmp_path, cell, split)
    baseline_manifest = json.loads(
        (baseline_dir / "manifest.json").read_text(encoding="utf-8")
    )
    baseline_manifest["cell"]["budget"]["eval_task_timeout_s"] = 999
    runner.write_json(baseline_dir / "manifest.json", baseline_manifest)
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    with pytest.raises(RuntimeError, match="different stopping rules"):
        asyncio.run(
            runner.rebookend_run(
                source_dir,
                runs_dir=tmp_path / "runs",
                results_dir=tmp_path / "results",
                baseline_run_dir=baseline_dir,
                capture_egress=False,
            )
        )
    assert launches == {}
    assert not (tmp_path / "runs").exists()


def test_the_bookend_runs_and_records_the_recorded_eval_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    """End to end: the legs run under the source's recorded bound, the manifest says so, and
    the checkout's differing values are named beside it, so a reader of the numbers sees which
    rule scored them without finding two checkouts to diff."""
    cell, split = _synthetic_definitions(tmp_path)
    source_dir = _source_run(tmp_path, cell, split)
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    # The source's before side was scored under a longer bound, and the file it hashed no
    # longer exists.
    source_manifest["cell"]["budget"]["eval_task_timeout_s"] = 300
    source_manifest["cell"]["config_sha256"] = "0" * 64
    runner.write_json(source_dir / "manifest.json", source_manifest)
    launches: dict[int, dict] = {}
    _wire_fakes(monkeypatch, cell, split, launches)

    results_path = asyncio.run(
        runner.rebookend_run(
            source_dir,
            runs_dir=tmp_path / "runs",
            results_dir=tmp_path / "results",
            capture_egress=False,
        )
    )

    # Every leg ran under the RECORDED bound, not the cell file's.
    assert set(launches) == {0, 1, 2}
    assert {record["timeout_s"] for record in launches.values()} == {300}
    assert cell.budget.eval_task_timeout_s != 300

    new_run = next(p for p in (tmp_path / "runs").iterdir() if p.is_dir())
    manifest = json.loads((new_run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cell"]["budget"]["eval_task_timeout_s"] == 300
    assert manifest["rebookend"]["eval_runtime_from_record"] == {
        "eval_task_timeout_s": 300,
        "eval_concurrency": cell.budget.eval_concurrency,
    }
    assert manifest["rebookend"]["source_cell"] == source_manifest["cell"]
    assert manifest["rebookend"]["cell_drift"]["budget.eval_task_timeout_s"] == {
        "recorded": 300,
        "checkout": cell.budget.eval_task_timeout_s,
    }
    assert "config_sha256" in manifest["rebookend"]["cell_drift"]
    # And it reaches the published artifact, which is what anyone reading the numbers holds.
    published = json.loads(results_path.read_text(encoding="utf-8"))
    assert published["manifest"]["rebookend"]["eval_runtime_from_record"] == {
        "eval_task_timeout_s": 300,
        "eval_concurrency": cell.budget.eval_concurrency,
    }


def test_cli_plans_a_bookend_whose_cell_timeout_was_retuned(tmp_path: Path, capsys) -> None:
    """The refusal an operator actually met, at the entry they met it in: the plan runs the
    checkout's real loaders, reports no drift, names the bound the legs will run under, and
    names what the file says instead."""
    from shobench.cli import main as cli_main

    source_dir = _real_cell_source(tmp_path)
    cell = load_cell_by_name("smoke-automationbench-claude-code")
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["cell"]["budget"]["eval_task_timeout_s"] = _RECORDED_EVAL_TIMEOUT_S
    manifest["cell"]["config_sha256"] = "0" * 64
    runner.write_json(source_dir / "manifest.json", manifest)

    assert cli_main(["rebookend", "--run", str(source_dir)]) == 0
    plan = json.loads(capsys.readouterr().out)

    assert plan["refusals"]["experiment_drift"] == []
    assert plan["eval_runtime_from_record"] == {
        "eval_task_timeout_s": _RECORDED_EVAL_TIMEOUT_S,
        "eval_concurrency": cell.budget.eval_concurrency,
    }
    assert plan["cell_drift"]["budget.eval_task_timeout_s"] == {
        "recorded": _RECORDED_EVAL_TIMEOUT_S,
        "checkout": cell.budget.eval_task_timeout_s,
    }
    assert "config_sha256" in plan["cell_drift"]


def test_cli_blocks_a_bookend_whose_cell_measures_something_else(
    tmp_path: Path, capsys
) -> None:
    """A model swap is not a runtime detail, and the block lands before anything is minted."""
    from shobench.cli import main as cli_main

    source_dir = _real_cell_source(tmp_path)
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["cell"]["model"] = "a-model-this-run-never-used"
    runner.write_json(source_dir / "manifest.json", manifest)

    assert cli_main(["rebookend", "--run", str(source_dir)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert any("cell model changed" in line for line in plan["refusals"]["experiment_drift"])

    assert cli_main(["rebookend", "--run", str(source_dir), "--go"]) == 1
    err = capsys.readouterr().err
    assert "BLOCKED" in err and "cell model changed" in err
    assert not (tmp_path / "runs").exists()


def test_cli_blocks_a_baseline_measured_under_another_bound(tmp_path: Path, capsys) -> None:
    """The plan states the pairing's stopping rule as a refusal state of its own, so the
    operator sees it before the runner raises."""
    from shobench.cli import main as cli_main

    source_dir = _real_cell_source(tmp_path)
    baseline_dir = tmp_path / "baseline"
    shutil.copytree(source_dir, baseline_dir)
    baseline_manifest = json.loads(
        (baseline_dir / "manifest.json").read_text(encoding="utf-8")
    )
    baseline_manifest["run_id"] = "baseline-run-20260101T000000Z"
    baseline_manifest["cell"]["budget"]["eval_task_timeout_s"] = 999
    runner.write_json(baseline_dir / "manifest.json", baseline_manifest)

    assert (
        cli_main(
            ["rebookend", "--run", str(source_dir), "--baseline", str(baseline_dir)]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["refusals"]["baseline_eval_runtime_matches"] is False

    assert (
        cli_main(
            ["rebookend", "--run", str(source_dir), "--baseline", str(baseline_dir), "--go"]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "BLOCKED" in err and "different stopping rules" in err
    assert not (tmp_path / "runs").exists()
