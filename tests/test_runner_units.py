"""Unit coverage for the parts that decide what a number means.

Everything here is offline and keyless. The pieces worth testing are the ones where a quiet
mistake becomes a wrong published number: which ids a phase serves, which rows pair, what the
interval is computed over, and how a leg's ending is classified.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shobench import egress
from shobench.config import load_all_cells, load_cell_by_name, load_instruction, repo_root
from shobench.containers import CellSandbox, home_digest, write_json
from shobench.harness import StopKind, StopVerdict
from shobench.harnesses import harness_for
from shobench.report import paired_bootstrap, report_cell
from shobench.results import TaskResult, eval_summary, pair_evals, write_results
from shobench.runner import LegRecord, RunContext, build_manifest, is_noise
from shobench.serving import side_for_phase, task_indices
from shobench.splits import load_split_by_name, splits_dir

# ----- splits ------------------------------------------------------------------------------


def test_every_committed_split_loads_and_is_disjoint() -> None:
    for path in sorted(splits_dir().glob("*.json")):
        split = load_split_by_name(path.stem)
        assert len(split.heldout) > 0
        assert len(split.pool) > 0


def test_automationbench_adopts_the_published_heldout_120() -> None:
    split = load_split_by_name("automationbench")
    assert len(split.heldout) == 120
    assert len(split.pool) == 480
    assert split.total_tasks == 600
    assert split.provenance["kind"] == "adopted"
    # Every id addresses one of the env's 600 tasks, and the two sides cover all of them.
    ids = {int(i) for i in split.heldout.task_ids} | {int(i) for i in split.pool.task_ids}
    assert ids == set(range(600))


def test_tau2_honors_upstreams_declared_split() -> None:
    split = load_split_by_name("tau2_telecom")
    assert (len(split.heldout), len(split.pool)) == (40, 74)
    assert split.heldout.env_kwargs == {"task_split": "test"}
    assert split.pool.env_kwargs == {"task_split": "train"}
    # The sides index into different env constructions, so equal integers are different tasks.
    # The labels are what proves they are disjoint.
    assert not set(split.heldout.labels) & set(split.pool.labels)


def test_hle_split_is_seeded_disjoint_and_published() -> None:
    split = load_split_by_name("hle")
    assert (len(split.heldout), len(split.pool)) == (120, 300)
    assert split.provenance["kind"] == "seeded"
    assert split.provenance["seed"]
    assert not set(split.heldout.task_ids) & set(split.pool.task_ids)
    assert split.total_tasks == 1726


def test_split_manifest_refuses_an_overlapping_split(tmp_path: Path) -> None:
    from shobench.splits import SCHEMA, load_split

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "env": "automationbench",
                "provenance": {"kind": "adopted"},
                "heldout": {"task_ids": ["1", "2"]},
                "pool": {"task_ids": ["2", "3"]},
            }
        )
    )
    with pytest.raises(ValueError, match="not disjoint"):
        load_split(bad)


# ----- what a phase serves -----------------------------------------------------------------


def test_both_eval_phases_serve_the_same_heldout_ids_in_the_same_order() -> None:
    split = load_split_by_name("automationbench")
    before = task_indices(side_for_phase(split, "eval_before"))
    after = task_indices(side_for_phase(split, "eval_after"))
    assert before == after
    assert len(before) == 120


def test_the_pool_ceiling_truncates_and_never_extends() -> None:
    split = load_split_by_name("automationbench")
    pool = side_for_phase(split, "rollout")
    assert len(task_indices(pool, ceiling=10)) == 10
    assert len(task_indices(pool, ceiling=10_000)) == 480


# ----- cells and instructions ---------------------------------------------------------------


def test_every_cell_config_loads_and_names_a_committed_split() -> None:
    cells = load_all_cells()
    assert len(cells) >= 13
    for cell in cells:
        load_split_by_name(cell.split)
        load_instruction(cell.instruction_arm)


def test_the_v0_matrix_is_three_envs_by_four_harness_model_pairs() -> None:
    cells = [c for c in load_all_cells() if not c.name.startswith("smoke")]
    assert len(cells) == 12
    assert {c.env for c in cells} == {"automationbench", "tau2_telecom", "hle"}
    pairs = {(c.harness, c.model) for c in cells}
    assert pairs == {
        ("claude_code", "claude-opus-5"),
        ("codex", "gpt-5.6-terra"),
        ("prime_agent", "claude-opus-5"),
        ("prime_agent", "gpt-5.6-terra"),
    }


def test_the_instruction_is_byte_identical_across_every_cell() -> None:
    digests = {load_instruction(c.instruction_arm).rollout_system_sha256 for c in load_all_cells()}
    assert len(digests) == 1


def test_the_improvement_objective_is_absent_from_the_eval_instruction() -> None:
    instruction = load_instruction("get-better")
    assert instruction.rollout_system.startswith("Get Better.")
    assert "Get Better" not in instruction.eval_system


def test_the_instruction_names_no_env_and_no_env_specific_tool() -> None:
    """The prompt is env-agnostic by design, so it may not name an env or an env's scoring
    tool. `get_task` is the stream's own tool and is the one allowed name; `{done: true}` is
    that tool's own reply field, not a tool the agent is told to call."""
    text = load_instruction("get-better").rollout_system.lower()
    for env_name in ("automationbench", "tau2", "telecom", "hle", "wordle", "orca"):
        assert env_name not in text
    for tool_name in ("submit_answer", "`done`", "call `done", "send_message"):
        assert tool_name not in text


# ----- pairing and the interval --------------------------------------------------------------


def _row(idx: int, reward: float | None, *, closure: str = "sealed") -> TaskResult:
    return TaskResult(
        seq=idx,
        position=idx,
        task_idx=idx,
        closure=closure,
        reward=reward,
        success=None if reward is None else reward >= 1.0,
    )


def test_pairing_keys_on_the_task_index() -> None:
    paired, unpaired = pair_evals([_row(1, 0.2), _row(2, 0.4)], [_row(2, 0.9), _row(1, 0.5)])
    assert [p["task_idx"] for p in paired] == [1, 2]
    assert [p["reward_delta"] for p in paired] == pytest.approx([0.3, 0.5])
    assert unpaired == []


def test_a_task_scored_in_only_one_phase_is_reported_not_dropped() -> None:
    paired, unpaired = pair_evals([_row(1, 0.2), _row(2, 0.4)], [_row(1, 0.5)])
    assert len(paired) == 1
    assert len(unpaired) == 1
    assert unpaired[0]["task_idx"] == 2


def test_an_unscored_closure_never_counts_as_a_zero() -> None:
    rows = [_row(1, 0.5), _row(2, None, closure="timeout")]
    summary = eval_summary(rows)
    assert summary["n_requested"] == 2
    assert summary["n_scored"] == 1
    assert summary["mean_reward"] == pytest.approx(0.5)
    assert summary["closures"] == {"sealed": 1, "timeout": 1}


def test_the_bootstrap_interval_brackets_the_mean_and_is_reproducible() -> None:
    deltas = [0.1, 0.2, -0.05, 0.3, 0.0, 0.15]
    mean, low, high = paired_bootstrap(deltas, resamples=2000, seed=1)
    assert mean == pytest.approx(sum(deltas) / len(deltas))
    assert low < mean < high
    assert paired_bootstrap(deltas, resamples=2000, seed=1) == (mean, low, high)


def test_a_constant_delta_gives_a_zero_width_interval() -> None:
    mean, low, high = paired_bootstrap([0.25] * 20, resamples=500, seed=3)
    assert (mean, low, high) == pytest.approx((0.25, 0.25, 0.25))


def test_report_cell_reads_a_results_document_end_to_end() -> None:
    doc = {
        "schema": "shobench.results/1",
        "manifest": {"cell": {"name": "c", "env": "e", "harness": "h", "model": "m"}},
        "paired": [
            {"task_idx": 1, "reward_before": 0.2, "reward_after": 0.5, "reward_delta": 0.3},
            {"task_idx": 2, "reward_before": 0.4, "reward_after": 0.4, "reward_delta": 0.0},
        ],
        "unpaired": [],
        "eval_before": {"summary": {"full_solve_rate": 0.0}},
        "eval_after": {"summary": {"full_solve_rate": 0.5}},
        "rollout": {
            "summary": {"tasks_attempted": 7, "tasks_scored": 6},
            "stopping": {"stop_reason": "agent_chose_to_stop", "usage_limit_resumes": 2},
        },
    }
    out = report_cell(doc, resamples=500, seed=7)
    assert out.n_paired == 2
    assert out.mean_delta == pytest.approx(0.15)
    assert out.stop_reason == "agent_chose_to_stop"
    assert out.resumes == 2


# ----- stop classification -------------------------------------------------------------------


def _write(path: Path, events: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


def test_claude_clean_finish_is_a_chosen_stop(tmp_path: Path) -> None:
    out = _write(
        tmp_path / "s.jsonl",
        [
            {"type": "system", "subtype": "init"},
            {
                "type": "result",
                "is_error": False,
                "subtype": "success",
                "terminal_reason": "completed",
                "api_error_status": None,
                "result": "done",
            },
        ],
    )
    verdict = harness_for("claude_code").classify(
        returncode=0, stdout_path=out, stderr_path=tmp_path / "e.txt", timed_out=False
    )
    assert verdict.kind is StopKind.CHOSEN


def test_claude_subtype_success_does_not_hide_an_error(tmp_path: Path) -> None:
    """Observed on a bad token: subtype stays "success" while is_error is true. A classifier
    that branched on subtype would call an auth failure a chosen stop."""
    out = _write(
        tmp_path / "s.jsonl",
        [
            {
                "type": "result",
                "is_error": True,
                "subtype": "success",
                "terminal_reason": "api_error",
                "api_error_status": 401,
                "result": "Failed to authenticate. API Error: 401 Invalid bearer token",
            }
        ],
    )
    verdict = harness_for("claude_code").classify(
        returncode=1, stdout_path=out, stderr_path=tmp_path / "e.txt", timed_out=False
    )
    assert verdict.kind is StopKind.ERROR


def test_claude_429_is_a_usage_limit_and_is_resumable(tmp_path: Path) -> None:
    out = _write(
        tmp_path / "s.jsonl",
        [
            {
                "type": "result",
                "is_error": True,
                "subtype": "success",
                "terminal_reason": "api_error",
                "api_error_status": 429,
                "result": "You've hit your session limit · resets 3:45pm",
            }
        ],
    )
    verdict = harness_for("claude_code").classify(
        returncode=1, stdout_path=out, stderr_path=tmp_path / "e.txt", timed_out=False
    )
    assert verdict.kind is StopKind.USAGE_LIMIT
    assert verdict.resumable


def test_an_agent_writing_about_limits_is_not_a_usage_limit(tmp_path: Path) -> None:
    """The result text of a clean turn is the agent's own words. Pattern-matching it for a
    limit message would record an agent that merely discussed one as having hit one."""
    out = _write(
        tmp_path / "s.jsonl",
        [
            {
                "type": "result",
                "is_error": False,
                "subtype": "success",
                "terminal_reason": "completed",
                "api_error_status": None,
                "result": "I noted that you've hit your session limit before, so I paced myself.",
            }
        ],
    )
    verdict = harness_for("claude_code").classify(
        returncode=0, stdout_path=out, stderr_path=tmp_path / "e.txt", timed_out=False
    )
    assert verdict.kind is StopKind.CHOSEN


def test_claude_context_limit_is_an_error_not_a_usage_limit(tmp_path: Path) -> None:
    """blocking_limit is the prompt token limit. Resuming on it would loop forever."""
    out = _write(
        tmp_path / "s.jsonl",
        [
            {
                "type": "result",
                "is_error": True,
                "subtype": "success",
                "terminal_reason": "blocking_limit",
                "api_error_status": None,
                "result": "Prompt is too long",
            }
        ],
    )
    verdict = harness_for("claude_code").classify(
        returncode=1, stdout_path=out, stderr_path=tmp_path / "e.txt", timed_out=False
    )
    assert verdict.kind is StopKind.ERROR
    assert not verdict.resumable


def test_codex_retry_chatter_is_not_a_stop(tmp_path: Path) -> None:
    """codex retries transient failures and emits error events while doing so. Only the
    terminal turn event decides."""
    out = _write(
        tmp_path / "s.jsonl",
        [
            {"type": "thread.started", "thread_id": "t"},
            {"type": "error", "message": "Reconnecting... 2/5 (unexpected status 429)"},
            {"type": "turn.completed", "usage": {}},
        ],
    )
    verdict = harness_for("codex").classify(
        returncode=0, stdout_path=out, stderr_path=tmp_path / "e.txt", timed_out=False
    )
    assert verdict.kind is StopKind.CHOSEN


def test_codex_usage_limit_reads_the_terminal_turn(tmp_path: Path) -> None:
    out = _write(
        tmp_path / "s.jsonl",
        [
            {"type": "thread.started", "thread_id": "t"},
            {
                "type": "turn.failed",
                "error": {"message": "You've hit your usage limit. Try again at 5pm."},
            },
        ],
    )
    verdict = harness_for("codex").classify(
        returncode=1, stdout_path=out, stderr_path=tmp_path / "e.txt", timed_out=False
    )
    assert verdict.kind is StopKind.USAGE_LIMIT


def test_codex_missing_terminal_event_is_an_interruption(tmp_path: Path) -> None:
    out = _write(tmp_path / "s.jsonl", [{"type": "thread.started", "thread_id": "t"}])
    verdict = harness_for("codex").classify(
        returncode=1, stdout_path=out, stderr_path=tmp_path / "e.txt", timed_out=False
    )
    assert verdict.kind is StopKind.ERROR


def test_prime_agent_ignores_the_exit_code(tmp_path: Path) -> None:
    """A json-mode run that errored still exits 0, so exit 0 alone must not read as a stop."""
    out = _write(
        tmp_path / "s.jsonl",
        [
            {"type": "session", "id": "s"},
            {
                "type": "agent_end",
                "messages": [
                    {
                        "role": "assistant",
                        "stopReason": "error",
                        "errorMessage": "Provider server error",
                    }
                ],
            },
        ],
    )
    verdict = harness_for("prime_agent").classify(
        returncode=0, stdout_path=out, stderr_path=tmp_path / "e.txt", timed_out=False
    )
    assert verdict.kind is StopKind.ERROR


def test_prime_agent_rate_limit_diagnostic_is_a_usage_limit(tmp_path: Path) -> None:
    out = _write(
        tmp_path / "s.jsonl",
        [
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "stopReason": "error",
                    "diagnostics": [
                        {
                            "type": "provider_stream_failure",
                            "details": {"kind": "rate_limit", "status": 429},
                        }
                    ],
                },
            }
        ],
    )
    verdict = harness_for("prime_agent").classify(
        returncode=0, stdout_path=out, stderr_path=tmp_path / "e.txt", timed_out=False
    )
    assert verdict.kind is StopKind.USAGE_LIMIT


def test_prime_agent_host_limit_is_a_cutoff_not_a_stop(tmp_path: Path) -> None:
    """Reaching an autonomous budget does not imply task success, so it must never be
    recorded as the agent's own choice to stop."""
    out = _write(tmp_path / "s.jsonl", [{"type": "session", "id": "s"}])
    err = tmp_path / "e.txt"
    err.write_text("Autonomous run stopped before terminal evidence; maxTurns reached\n")
    verdict = harness_for("prime_agent").classify(
        returncode=1, stdout_path=out, stderr_path=err, timed_out=False
    )
    assert verdict.kind is StopKind.LEG_TIMEOUT
    assert verdict.kind is not StopKind.CHOSEN


def test_a_runner_timeout_is_never_a_chosen_stop(tmp_path: Path) -> None:
    for name in ("claude_code", "codex", "prime_agent"):
        verdict = harness_for(name).classify(
            returncode=-1,
            stdout_path=tmp_path / "missing.jsonl",
            stderr_path=tmp_path / "missing.txt",
            timed_out=True,
        )
        assert verdict.kind is StopKind.LEG_TIMEOUT


# ----- launch specs ---------------------------------------------------------------------------


def test_claude_needs_is_sandbox_because_the_container_runs_as_root() -> None:
    assert harness_for("claude_code").base_env()["IS_SANDBOX"] == "1"


def test_every_harness_clears_node_options() -> None:
    for name in ("claude_code", "codex", "prime_agent"):
        assert harness_for(name).base_env()["NODE_OPTIONS"] == ""


def test_codex_opens_the_sandbox_because_exec_defaults_to_read_only(tmp_path: Path) -> None:
    spec = harness_for("codex").launch(
        mcp_url="http://h:1/mcp",
        system_prompt="s",
        user_prompt="u",
        model="m",
        trace_path=tmp_path / "t",
    )
    assert "--dangerously-bypass-approvals-and-sandbox" in spec.argv
    joined = " ".join(spec.argv)
    assert "mcp_servers.shogym.required=true" in joined
    assert 'default_tools_approval_mode="approve"' in joined


def test_codex_resume_puts_the_subcommand_before_the_flags(tmp_path: Path) -> None:
    spec = harness_for("codex").launch(
        mcp_url="http://h:1/mcp",
        system_prompt="s",
        user_prompt="u",
        model="m",
        trace_path=tmp_path / "t",
        session_id="thread-1",
        resume=True,
    )
    assert spec.argv[:4] == ["codex", "exec", "resume", "thread-1"]


def test_prime_agent_raises_every_autonomous_budget(tmp_path: Path) -> None:
    """The defaults are 3 continuations, 12 turns, 80k tokens, 30 minutes. Any one of them
    left alone would end an 8-hour rollout on a cutoff and record it as a stop."""
    spec = harness_for("prime_agent").launch(
        mcp_url="http://h:1/mcp",
        system_prompt="s",
        user_prompt="u",
        model="m",
        trace_path=tmp_path / "t",
        leg_timeout_s=3600,
    )
    joined = spec.argv
    for flag in (
        "--autonomous",
        "--autonomous-max-continuations",
        "--autonomous-max-turns",
        "--autonomous-max-tokens",
        "--autonomous-timeout-ms",
    ):
        assert flag in joined
    assert int(joined[joined.index("--autonomous-max-turns") + 1]) > 12
    assert int(joined[joined.index("--autonomous-timeout-ms") + 1]) > 30 * 60 * 1000


def test_prime_agent_declares_an_http_server_with_a_bearer_token(tmp_path: Path) -> None:
    """Only http entries are honored, and the client refuses a session without a token."""
    spec = harness_for("prime_agent").launch(
        mcp_url="http://h:1/mcp",
        system_prompt="s",
        user_prompt="u",
        model="m",
        trace_path=tmp_path / "t",
    )
    settings = json.loads(spec.home_files[".prime/agent/settings.json"])
    server = settings["mcpServers"]["shogym"]
    assert server["type"] == "http"
    assert server["bearerTokenEnvVar"] in spec.env


def test_the_repo_root_resolves_from_the_package_not_the_cwd() -> None:
    assert (repo_root() / "pyproject.toml").is_file()


# ----- resuming the session the harness actually ran under ------------------------------------


def test_only_claude_code_lets_the_runner_choose_the_session_id() -> None:
    assert harness_for("claude_code").pins_session_id
    assert not harness_for("codex").pins_session_id
    assert not harness_for("prime_agent").pins_session_id


def test_each_harness_reads_its_own_session_id_off_its_trace(tmp_path: Path) -> None:
    """Resuming a runner-chosen id would start a fresh session for the two harnesses that mint
    their own, losing everything the rollout had built up in context."""
    cases = {
        "claude_code": ([{"type": "system", "subtype": "init", "session_id": "cc-1"}], "cc-1"),
        "codex": ([{"type": "thread.started", "thread_id": "cx-1"}], "cx-1"),
        "prime_agent": ([{"type": "session", "version": 3, "id": "pa-1"}], "pa-1"),
    }
    for name, (events, expected) in cases.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        assert harness_for(name).session_id_from_trace(path) == expected


def test_a_trace_with_no_session_line_yields_no_id(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    for name in ("claude_code", "codex", "prime_agent"):
        assert harness_for(name).session_id_from_trace(path) is None


# ----- what counts as the agent's durable self ------------------------------------------------


def test_session_byproducts_are_not_the_durable_self() -> None:
    """These all change on every run whether or not the agent changed itself. A digest that
    included them would answer "did a session happen", which is always yes."""
    for noise in (
        ".cache/claude-cli-nodejs/-work/mcp-logs-shogym/2026-08-10T19-16-37.jsonl",
        ".claude/projects/-work/226276c6-9dd1-4a54-af48-942d543d8c6b.jsonl",
        ".claude/.last-cleanup",
        ".claude/policy-limits.json",
        ".claude/remote-settings.json",
        ".claude/statsig/statsig.cached.evaluations",
        ".codex/sessions/2026/08/10/rollout-abc.jsonl",
        ".codex/logs_2.sqlite-wal",
        ".prime/agent/kernel-venv/lib/python3.11/site-packages/x.py",
        ".prime/agent/sessions/abc.jsonl",
    ):
        assert is_noise(noise), noise


def test_credential_material_is_in_no_record_this_runner_writes() -> None:
    for secret in (
        ".claude/.credentials.json",
        ".claude.json",
        ".codex/auth.json",
        ".prime/agent/auth.json",
    ):
        assert is_noise(secret), secret


def test_what_the_agent_writes_about_itself_is_kept() -> None:
    """Memory and skills are the durable channel the benchmark measures, and they sit beside
    the transcripts rather than among them."""
    for durable in (
        ".claude/projects/-work/memory/MEMORY.md",
        ".claude/projects/-work/memory/a-note.md",
        ".claude/skills/my-skill/SKILL.md",
        ".claude/CLAUDE.md",
        ".claude/settings.json",
        ".codex/skills/thing/SKILL.md",
        ".prime/agent/skills/thing/SKILL.md",
        "notes.md",
    ):
        assert not is_noise(durable), durable


def test_the_digest_ignores_a_new_transcript_and_notices_a_new_note(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".claude" / "projects" / "-work" / "memory").mkdir(parents=True)
    before = home_digest(home, exclude=is_noise)

    (home / ".claude" / "projects" / "-work" / "abc-123.jsonl").write_text("a transcript\n")
    assert home_digest(home, exclude=is_noise) == before

    (home / ".claude" / "projects" / "-work" / "memory" / "note.md").write_text("a lesson\n")
    assert home_digest(home, exclude=is_noise) != before


# ----- no durable artifact leaks an absolute host path ----------------------------------------

# The four shapes an absolute host path takes on the machines that run cells. A durable record
# carrying any of these leaks a username and a machine layout, and is wrong on another checkout.
_ABSOLUTE_MARKERS = ("/Users/", "/home/", "/private/tmp/", "/var/folders/")


def _absolute_path_values(node: object) -> list[str]:
    """Every string in a parsed JSON tree that reads as an absolute filesystem path.

    A leading slash is the general case; the markers catch an absolute path embedded inside a
    longer string. Repo paths, run-internal paths, and hostnames are all relative or bare, so an
    empty result is the invariant the runner's records must hold.
    """
    if isinstance(node, str):
        leaks = node.startswith("/") or any(marker in node for marker in _ABSOLUTE_MARKERS)
        return [node] if leaks else []
    if isinstance(node, dict):
        return [leak for value in node.values() for leak in _absolute_path_values(value)]
    if isinstance(node, list):
        return [leak for value in node for leak in _absolute_path_values(value)]
    return []


def test_no_durable_artifact_the_runner_writes_carries_an_absolute_path(tmp_path: Path) -> None:
    """The manifest and the results JSON, built the way the runner builds them, are free of any
    absolute host path. This drives the real builders on a synthetic run so a newly added path
    field that forgets to relativize fails here rather than shipping in a golden.
    """
    cell = load_cell_by_name("smoke-automationbench-claude-code")
    split = load_split_by_name("smoke-automationbench")
    instruction = load_instruction(cell.instruction_arm)
    run_id = "guard-run-20260101T000000Z"
    run_dir = tmp_path / run_id
    sandbox = CellSandbox(run_id=run_id, home=run_dir / "home", workdir=run_dir / "work")
    ctx = RunContext(
        cell=cell,
        split=split,
        instruction=instruction,
        harness=harness_for(cell.harness),
        run_id=run_id,
        run_dir=run_dir,
        sandbox=sandbox,
    )

    manifest = build_manifest(ctx, probes={"version": "2.1.226 (Claude Code)"})

    # A leg whose trace lives under the run dir, recorded the way a phase records it.
    leg = LegRecord(
        leg=0,
        phase="rollout",
        task_idx=None,
        started_at=0.0,
        ended_at=1.0,
        returncode=0,
        verdict=StopVerdict(StopKind.CHOSEN, "the session ended its turn cleanly"),
        tasks_consumed_before=0,
        tasks_consumed_after=1,
        trace_path=str(run_dir / "rollout" / "traces" / "leg-0000.stream.jsonl"),
        run_dir=run_dir,
    )
    stopping = {"stop_reason": "agent_chose_to_stop", "legs": [leg.to_json()]}
    # The leg keeps its absolute trace in-process for the resume read, but records the relative.
    assert leg.trace_path.startswith("/")
    assert leg.to_json()["trace_path"] == "rollout/traces/leg-0000.stream.jsonl"

    # An egress summary built from a capture at the run dir root.
    tsv = run_dir / "egress.tsv"
    tsv.parent.mkdir(parents=True, exist_ok=True)
    tsv.write_text("1.0\t203.0.113.7\t443\t\tapi.example.com\t\n", encoding="utf-8")
    egress_summary = egress.summarize(tsv)

    manifest_path = write_json(run_dir / "manifest.json", manifest)
    results_path = write_results(
        run_dir / "results.json",
        manifest=manifest,
        phases={"eval_before": [], "rollout": [], "eval_after": []},
        stopping=stopping,
        egress=egress_summary,
    )

    for path in (manifest_path, results_path):
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert _absolute_path_values(doc) == [], f"{path.name} leaks an absolute path"
