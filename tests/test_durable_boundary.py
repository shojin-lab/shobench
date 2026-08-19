"""What the cell publishes as "the rollout wrote this" has to be the rollout's, and only the
rollout's.

This is the measurement the whole study rests on, so the boundary is stated once and then
checked here rather than left to be inferred from where the calls happen to sit:

- the baseline is the state after the runner has placed everything it owns, and before any phase
  runs, so the seeds a cell starts with sit on the same side of the line as the rollout that may
  improve them;
- each eval task reads that state through a throwaway copy and writes only into the copy;
- the rollout is the only phase that runs against the cell's own HOME and ``/work``;
- the after-state is read at the rollout's terminus, not at the end of the cell;
- both persistent agent-visible channels are recorded, HOME because it is the durable self and
  ``/work`` because it is writable, persists for the whole rollout, and used to appear in no
  record at all.

The false positive these are written against was deterministic: the runner seeded prime-agent's
settings file and the whole vendored skill package into the base HOME on the rollout's first
leg, which is after the baseline, so a prime_agent cell that wrote nothing published four
changed files as its own durable output.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from shobench import runner
from shobench.config import load_cell_by_name, load_instruction
from shobench.containers import CellSandbox, home_digest, work_digest
from shobench.harnesses import harness_for
from shobench.runner import RunContext, build_manifest, durable_filter, is_noise, write_home_files
from shobench.splits import load_split_by_name

_SMOKE_CELL = "smoke-automationbench-claude-code"


def _context(tmp_path: Path, harness: str) -> RunContext:
    cell = load_cell_by_name(_SMOKE_CELL)
    run_dir = tmp_path / "run"
    sandbox = CellSandbox(run_id="r", home=run_dir / "home", workdir=run_dir / "work")
    sandbox.home.mkdir(parents=True, exist_ok=True)
    sandbox.workdir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        cell=cell,
        split=load_split_by_name(cell.split),
        instruction=load_instruction(cell.instruction_arm),
        harness=harness_for(harness),
        run_id="r",
        run_dir=run_dir,
        sandbox=sandbox,
    )


def test_a_prime_cell_that_wrote_nothing_reports_an_unchanged_durable_self(tmp_path) -> None:
    """The deterministic false positive, driven through the real seeding and the real launch.

    Everything the runner owns is placed before the baseline, and the one file it rewrites per
    leg is out of the digest entirely, so a leg that ran and did nothing else moves neither.
    """
    ctx = _context(tmp_path, "prime_agent")

    seeded = runner._place_runner_files(ctx)
    manifest = build_manifest(ctx, probes={})
    # A leg's launch, which writes the per-leg settings file and re-offers the seeds.
    spec = ctx.harness.launch(
        mcp_url="http://host.docker.internal:8973/mcp",
        system_prompt="S",
        user_prompt="U",
        model="claude-opus-5",
        trace_path=tmp_path / "t.jsonl",
        leg_timeout_s=60,
    )
    write_home_files(ctx.sandbox.home, spec)
    runner._snapshot_durable_state(ctx, manifest)

    assert seeded, "the prime harness seeds a skill package; this test is about it"
    assert manifest["home"]["changed"] is False
    assert manifest["home"]["digest_after"] == manifest["home"]["digest_before"]
    # The skill is in the record on both sides, because the agent may improve it and that
    # improvement is exactly what the rollout is supposed to be able to show.
    inventory = {row["path"] for row in manifest["home"]["inventory_after"]}
    assert any(path.startswith(".prime/agent/skills/shogym-stream/") for path in inventory)
    # The endpoint file the runner rewrites every leg is not in it, because it never can be
    # anything but the runner's.
    assert ".prime/agent/settings.json" not in inventory


def test_the_old_lazy_seeding_would_have_reported_a_change(tmp_path) -> None:
    """The mutation check for the test above, written as the behaviour that used to ship.

    A baseline taken before the runner places its own files, with the placement happening on the
    leg, is the arrangement that made an idle prime_agent cell look productive.
    """
    ctx = _context(tmp_path, "prime_agent")

    baseline = home_digest(ctx.sandbox.home, exclude=is_noise)
    spec = ctx.harness.launch(
        mcp_url="http://host.docker.internal:8973/mcp",
        system_prompt="S",
        user_prompt="U",
        model="claude-opus-5",
        trace_path=tmp_path / "t.jsonl",
        leg_timeout_s=60,
    )
    write_home_files(ctx.sandbox.home, spec)

    assert home_digest(ctx.sandbox.home, exclude=is_noise) != baseline


def test_an_agent_write_still_moves_the_digest(tmp_path) -> None:
    """The filter excludes what the runner owns and nothing else; a real write must still show."""
    ctx = _context(tmp_path, "prime_agent")
    runner._place_runner_files(ctx)
    manifest = build_manifest(ctx, probes={})

    memory = ctx.sandbox.home / ".prime" / "agent" / "memory" / "what-i-learned.md"
    memory.parent.mkdir(parents=True, exist_ok=True)
    memory.write_text("tau2 telecom tasks reward reading the policy first\n", encoding="utf-8")
    # And an edit to the seed, which is the agent improving an asset it was given.
    skill = ctx.sandbox.home / ".prime/agent/skills/shogym-stream/SKILL.md"
    skill.write_text(skill.read_text(encoding="utf-8") + "\nmy notes\n", encoding="utf-8")
    runner._snapshot_durable_state(ctx, manifest)

    assert manifest["home"]["changed"] is True
    inventory = {row["path"] for row in manifest["home"]["inventory_after"]}
    assert ".prime/agent/memory/what-i-learned.md" in inventory


@pytest.mark.parametrize("harness", ["claude_code", "codex", "prime_agent"])
def test_every_harness_declares_the_home_files_its_launch_actually_writes(harness) -> None:
    """The digest's exclusion list and the launch spec cannot drift apart.

    The exclusion is declared on the class so the digest can be taken before any leg has run.
    That only stays correct while it matches what ``launch`` really writes, which is what this
    holds: a harness that grows a per-leg home file and forgets to declare it would publish that
    file as an agent write on every cell.
    """
    instance = harness_for(harness)
    spec = instance.launch(
        mcp_url="http://host.docker.internal:8973/mcp",
        system_prompt="S",
        user_prompt="U",
        model="claude-opus-5",
        trace_path=Path("/t"),
        leg_timeout_s=60,
    )

    assert set(spec.home_files) == set(instance.runner_owned_home_files)
    assert set(spec.home_seed_files) == set(instance.home_seed_files())


def test_work_is_inventoried_whole_because_the_exam_inherits_it_whole(tmp_path) -> None:
    """A rollout that wrote its notes into its cwd used to publish a manifest mentioning none.

    Recorded whole, unlike the HOME beside it: every entry of ``/work`` is copied into every
    held-out session whatever the durability filter would call it, so a ``.log`` the filter drops
    is still a file the agent reads.
    """
    ctx = _context(tmp_path, "claude_code")
    manifest = build_manifest(ctx, probes={})

    (ctx.sandbox.workdir / "AGENTS.md").write_text("read the policy first\n", encoding="utf-8")
    (ctx.sandbox.workdir / "scratch.log").write_text("noise\n", encoding="utf-8")
    (ctx.sandbox.home / "notes.log").write_text("noise\n", encoding="utf-8")
    runner._snapshot_durable_state(ctx, manifest)

    assert manifest["work"]["changed"] is True
    inventory = {row["path"] for row in manifest["work"]["inventory_after"]}
    assert inventory == {"AGENTS.md", "scratch.log"}
    # The same name is still noise on the HOME side, whose filter this does not touch.
    assert "notes.log" not in {row["path"] for row in manifest["home"]["inventory_after"]}
    assert manifest["work"]["measured_at"] == "rollout_end"


def test_the_work_record_is_the_tree_the_exam_gets(tmp_path) -> None:
    """Entry for entry, with what decides how each behaves in the session that reads it.

    A container path like ``/home/oai/tool`` is a link the exam gets and a dangling nothing on
    this host, so hashing what it resolves to would drop it, and an alias is not two files that
    happen to match, because an edit through one name is an edit through both.
    """
    ctx = _context(tmp_path, "claude_code")
    work = ctx.sandbox.workdir
    (work / "helper.py").write_text("real\n", encoding="utf-8")
    (work / "scratch.log").write_text("a log the agent keeps\n", encoding="utf-8")
    (work / "notes").mkdir()
    (work / "tool").symlink_to("/home/oai/tool")
    (work / "shortcut.md").symlink_to("helper.py")
    os.link(work / "helper.py", work / "active.py")

    manifest = build_manifest(ctx, probes={})
    runner._snapshot_durable_state(ctx, manifest)
    rows = {row["path"]: row for row in manifest["work"]["inventory_after"]}

    assert set(rows) == {p.relative_to(work).as_posix() for p in work.rglob("*")}
    assert rows["notes"]["kind"] == "dir"
    assert rows["scratch.log"]["kind"] == "file"
    # A link is recorded by what it names, not by what it currently resolves to here.
    assert rows["tool"] == {"path": "tool", "kind": "link", "target": "/home/oai/tool"}
    assert rows["shortcut.md"]["target"] == "helper.py"
    assert "sha256" not in rows["shortcut.md"]
    # Both names of one inode say they are one inode, and say it identically.
    assert rows["helper.py"]["alias"] == rows["active.py"]["alias"] == "active.py"


def test_two_work_trees_a_projection_calls_equal_digest_differently(tmp_path) -> None:
    """The digest has to separate trees that hand a session different behavior.

    All three hold two paths and the same bytes, which the durable-self projection hashed
    identically. Editing ``alias.md`` changes one file in the first two and both in the third.
    """

    def tree(name: str, build) -> Path:
        root = tmp_path / name
        root.mkdir()
        (root / "helper.py").write_text("real\n", encoding="utf-8")
        build(root)
        return root

    linked = tree("linked", lambda r: (r / "alias.md").symlink_to("helper.py"))
    copied = tree("copied", lambda r: (r / "alias.md").write_text("real\n", encoding="utf-8"))
    hard = tree("hard", lambda r: os.link(r / "helper.py", r / "alias.md"))

    digests = {work_digest(linked), work_digest(copied), work_digest(hard)}
    assert len(digests) == 3

    # A link that starts naming something else is a different tree too.
    before = work_digest(linked)
    (linked / "alias.md").unlink()
    (linked / "alias.md").symlink_to("elsewhere.py")
    assert work_digest(linked) != before


def test_an_eval_task_copy_cannot_move_either_channel(tmp_path) -> None:
    """The invariant the publish-time check exists to hold: eval tasks work on copies.

    Driven through the real ``_copy_task_home``, and then through the real check, so a change
    that quietly pointed an eval task at the base home would fail here rather than silently
    crediting the rollout with the eval's writes.
    """
    ctx = _context(tmp_path, "claude_code")
    runner._place_runner_files(ctx)
    manifest = build_manifest(ctx, probes={})
    runner._snapshot_durable_state(ctx, manifest)

    task_home = tmp_path / "task-home"
    runner._copy_task_home(ctx.sandbox.home, task_home)
    (task_home / "the-eval-session-wrote-this.md").write_text("x", encoding="utf-8")
    (tmp_path / "task-work").mkdir()
    runner._check_evals_left_the_snapshot_alone(ctx, manifest)

    assert manifest["home"]["unchanged_by_evals"] is True
    assert manifest["work"]["unchanged_by_evals"] is True
    assert manifest["home"]["changed"] is False


def test_a_write_into_the_base_home_after_the_rollout_is_reported_not_absorbed(tmp_path) -> None:
    """If something ever does move the base home during an eval, the record says so.

    The snapshot is the rollout's and stays the rollout's; the mismatch is published beside it
    rather than folded into a digest the reader would then read as a rollout write.
    """
    ctx = _context(tmp_path, "claude_code")
    manifest = build_manifest(ctx, probes={})
    runner._snapshot_durable_state(ctx, manifest)
    snapshot = manifest["home"]["digest_after"]

    (ctx.sandbox.home / "written-during-eval-after.md").write_text("x", encoding="utf-8")
    runner._check_evals_left_the_snapshot_alone(ctx, manifest)

    assert manifest["home"]["unchanged_by_evals"] is False
    assert manifest["home"]["digest_after"] == snapshot


def test_the_durable_filter_is_harness_specific(tmp_path) -> None:
    """claude_code writes no per-leg home file, so nothing extra is excluded for it."""
    prime = durable_filter(harness_for("prime_agent"))
    claude = durable_filter(harness_for("claude_code"))

    assert prime(".prime/agent/settings.json") is True
    assert claude(".prime/agent/settings.json") is False
    for exclude in (prime, claude):
        assert exclude(".cache/anything") is True
        assert exclude(".claude/memory/notes.md") is False


def test_the_snapshot_is_taken_when_the_rollout_ends_not_when_the_cell_does(
    tmp_path, monkeypatch
) -> None:
    """Order, driven through the real phase loop: the digest lands before eval_after runs.

    An after-digest taken at the end of the cell is the arrangement that lets an eval phase's
    writes be published as the rollout's, and nothing in the record distinguishes the two.
    """
    ctx = _context(tmp_path, "claude_code")
    manifest = build_manifest(ctx, probes={})
    order: list[str] = []

    async def fake_rollout(ctx, *, suspended=None):
        order.append("rollout")
        return [], {"stop_reason": "pool_exhausted"}

    async def fake_eval(ctx, phase):
        order.append(phase)
        # Whatever the snapshot says at this point is what eval_after was handed.
        order.append(f"digest_after={manifest['home']['digest_after'] is not None}")
        return []

    monkeypatch.setattr(runner, "run_rollout_phase", fake_rollout)
    monkeypatch.setattr(runner, "run_eval_phase", fake_eval)

    class _Observer:
        summary: dict = {}

        def stop(self):
            return {}

    asyncio.run(
        runner._run_phases(
            ctx,
            manifest=manifest,
            phases=("eval_before", "rollout", "eval_after"),
            results_dir=tmp_path / "results",
            observer=_Observer(),
        )
    )

    assert order == [
        "eval_before",
        "digest_after=False",
        "rollout",
        "eval_after",
        "digest_after=True",
    ]
    assert manifest["home"]["measured_at"] == "rollout_end"
