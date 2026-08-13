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
from shobench.config import (
    load_all_cells,
    load_cell,
    load_cell_by_name,
    load_instruction,
    repo_root,
)
from shobench.containers import CellSandbox, home_digest, write_json
from shobench.harness import LaunchSpec, StopKind, StopVerdict
from shobench.harnesses import harness_for
from shobench.report import paired_bootstrap, render_table, report_cell
from shobench.results import TaskResult, eval_summary, pair_evals, write_results
from shobench.runner import LegRecord, RunContext, build_manifest, is_noise, write_home_files
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


def test_every_v0_cell_pins_xhigh_effort_and_the_manifest_records_it() -> None:
    """Effort is a cell axis, so the v0 matrix pins one value rather than inheriting each
    CLI's own default; a cell that drifted would make its numbers non-comparable silently."""
    for cell in load_all_cells():
        if cell.name.startswith("smoke"):
            continue
        assert cell.effort == "xhigh", cell.name
        assert cell.to_manifest()["effort"] == "xhigh", cell.name


def test_every_v0_cell_serves_concurrent_tasks_and_the_manifest_records_it() -> None:
    """One-at-a-time serving would make it the only behavior an agent can show; at 8 it is a
    choice the agent makes, which is the thing the benchmark observes."""
    for cell in load_all_cells():
        expected = 2 if cell.name.startswith("smoke") else 8
        assert cell.max_in_flight == expected, cell.name
        assert cell.to_manifest()["max_in_flight"] == expected, cell.name


def test_a_cell_with_no_concurrency_slots_is_refused(tmp_path: Path) -> None:
    source = (
        '[cell]\nname = "bad"\nenv = "automationbench"\nharness = "codex"\nmodel = "m"\n'
        'split = "automationbench"\nmax_in_flight = 0\n[budget]\nrollout_wall_clock_s = 1\n'
    )
    path = tmp_path / "bad.toml"
    path.write_text(source)
    with pytest.raises(ValueError, match="max_in_flight"):
        load_cell(path)


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
    paired, unpaired = pair_evals(
        [_row(1, 0.2), _row(2, 0.4)], [_row(2, 0.9), _row(1, 0.5)], task_ids=[1, 2]
    )
    assert [p["task_idx"] for p in paired] == [1, 2]
    assert [p["reward_delta"] for p in paired] == pytest.approx([0.3, 0.5])
    assert unpaired == []


def test_a_task_scored_in_only_one_phase_is_reported_not_dropped() -> None:
    paired, unpaired = pair_evals(
        [_row(1, 0.2), _row(2, 0.4)], [_row(1, 0.5)], task_ids=[1, 2]
    )
    assert len(paired) == 1
    assert len(unpaired) == 1
    assert unpaired[0]["task_idx"] == 2


def test_a_task_that_produced_no_row_in_either_phase_is_still_paired_against() -> None:
    """The id that used to vanish. It is in neither phase's rows, so a pairing over the union of
    what arrived left it out of both outputs: no pair, no unpaired entry, and a paired mean over
    a subset nobody chose. Walking the committed ids is what makes the loss visible."""
    paired, unpaired = pair_evals([_row(1, 0.2)], [_row(1, 0.5)], task_ids=[1, 2])

    assert [p["task_idx"] for p in paired] == [1]
    assert [u["task_idx"] for u in unpaired] == [2]
    assert unpaired[0] == {"task_idx": 2, "before": None, "after": None}


def test_an_unscored_closure_never_counts_as_a_zero() -> None:
    rows = [_row(1, 0.5), _row(2, None, closure="timeout")]
    summary = eval_summary(rows, task_ids=[1, 2])
    assert summary["n_requested"] == 2
    assert summary["n_scored"] == 1
    assert summary["mean_reward"] == pytest.approx(0.5)
    assert summary["closures"] == {"sealed": 1, "timeout": 1}


def test_the_requested_count_is_the_committed_set_and_not_the_rows_that_arrived() -> None:
    """A phase that lost a task must not publish as a smaller phase that lost nothing."""
    summary = eval_summary([_row(1, 0.5)], task_ids=[1, 2, 3])

    assert summary["n_requested"] == 3
    assert summary["n_scored"] == 1
    assert summary["n_missing"] == 2
    assert summary["complete"] is False


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


def test_the_report_says_which_cell_could_not_account_for_every_held_out_task() -> None:
    """A row whose numbers are over a subset says so where the numbers are, not only in the file
    they came from. Read off the published accounting rather than recomputed, since the ids a
    cell never measured are exactly what its own rows cannot show."""
    doc = {
        "schema": "shobench.results/1",
        "manifest": {"cell": {"name": "c", "env": "e", "harness": "h", "model": "m"}},
        "heldout": {
            "n_requested": 4,
            "complete": False,
            "eval_before": {"missing_task_ids": [3]},
            "eval_after": {"missing_task_ids": [2, 3]},
        },
        "paired": [{"task_idx": 1, "reward_before": 0.2, "reward_after": 0.5, "reward_delta": 0.3}],
        "unpaired": [{"task_idx": 2}, {"task_idx": 3}],
        "eval_before": {"summary": {}},
        "eval_after": {"summary": {}},
        "rollout": {"summary": {}, "stopping": {}},
    }

    out = report_cell(doc, resamples=100, seed=7)

    assert out.complete is False
    assert out.n_requested == 4
    assert out.n_missing == 2, "the ids lost in either phase, counted once each"
    table = render_table([out])
    assert "c *" in table and "INCOMPLETE" in table
    # The denominator the delta is a mean over is in the table, not only in the file.
    assert "1/4" in table


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
        model="claude-opus-5",
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
        model="claude-opus-5",
        trace_path=tmp_path / "t",
        session_id="thread-1",
        resume=True,
    )
    assert spec.argv[:4] == ["codex", "exec", "resume", "thread-1"]


def test_effort_reaches_the_harness_only_when_the_cell_pins_it(tmp_path: Path) -> None:
    """An empty effort must leave the CLI's own default untouched, so the flag appears only
    when a cell pins one. codex takes it as a config override and its prompt stays the last
    positional argument either way."""
    kwargs = dict(
        mcp_url="http://h:1/mcp",
        system_prompt="s",
        user_prompt="u",
        model="claude-opus-5",
        trace_path=tmp_path / "t",
    )
    claude = harness_for("claude_code").launch(effort="xhigh", **kwargs)
    index = claude.argv.index("--effort")
    assert claude.argv[index + 1] == "xhigh"
    assert "--effort" not in harness_for("claude_code").launch(**kwargs).argv

    codex = harness_for("codex").launch(effort="xhigh", **kwargs)
    assert 'model_reasoning_effort="xhigh"' in codex.argv
    assert codex.argv[-1] == "s\n\nu"
    bare = harness_for("codex").launch(**kwargs)
    assert not any("model_reasoning_effort" in arg for arg in bare.argv)
    assert bare.argv[-1] == "s\n\nu"


def test_prime_agent_raises_every_autonomous_budget(tmp_path: Path) -> None:
    """The defaults are 3 continuations, 12 turns, 80k tokens, 30 minutes. Any one of them
    left alone would end an 8-hour rollout on a cutoff and record it as a stop."""
    spec = harness_for("prime_agent").launch(
        mcp_url="http://h:1/mcp",
        system_prompt="s",
        user_prompt="u",
        model="claude-opus-5",
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
        model="claude-opus-5",
        trace_path=tmp_path / "t",
    )
    settings = json.loads(spec.home_files[".prime/agent/settings.json"])
    server = settings["mcpServers"]["shogym"]
    assert server["type"] == "http"
    assert server["bearerTokenEnvVar"] in spec.env


def test_the_repo_root_resolves_from_the_package_not_the_cwd() -> None:
    assert (repo_root() / "pyproject.toml").is_file()


def test_a_seeded_home_file_survives_the_next_leg_and_a_per_leg_one_does_not(
    tmp_path: Path,
) -> None:
    """The two HOME channels differ only in what the second leg does to them.

    Per-leg files carry what the runner alone knows and what moves between legs (the stream
    endpoint is different for every phase and every concurrent eval task), so a leg rewrites
    them even over an edit. A seed is the agent's from the moment it exists: the rollout is a
    measurement of what the agent made durable, and the eval-after home is a copy of the one the
    rollout accumulated, so a leg that restored a seed would erase the improvement it is about
    to measure."""
    home = tmp_path / "home"
    per_leg = {"cfg/endpoint.json": '{"port": 1}'}
    seed = {"skills/thing/SKILL.md": "vendored"}
    write_home_files(home, LaunchSpec(argv=[], env={}, home_files=per_leg, home_seed_files=seed))
    assert (home / "cfg/endpoint.json").read_text() == '{"port": 1}'
    assert (home / "skills/thing/SKILL.md").read_text() == "vendored"

    # The agent edits both between legs; only one of them is its business.
    (home / "cfg/endpoint.json").write_text('{"port": 99}')
    (home / "skills/thing/SKILL.md").write_text("improved by the agent")

    second_leg = {"cfg/endpoint.json": '{"port": 2}'}
    write_home_files(
        home, LaunchSpec(argv=[], env={}, home_files=second_leg, home_seed_files=seed)
    )
    assert (home / "cfg/endpoint.json").read_text() == '{"port": 2}'
    assert (home / "skills/thing/SKILL.md").read_text() == "improved by the agent"


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


def test_the_session_id_is_found_however_long_the_trace_grew(tmp_path: Path) -> None:
    """A long run is exactly the run whose session id matters most.

    Two of the three harnesses mint their own id and announce it in the first event, and the
    runner reads it back to continue a rollout a usage limit suspended. Searching a window of
    the tail would find it in every test trace and lose it in every real one, and a suspension
    with no id is a rollout that cannot be continued at all.
    """
    for name, header, expected in (
        ("codex", {"type": "thread.started", "thread_id": "cx-1"}, "cx-1"),
        ("prime_agent", {"type": "session", "version": 3, "id": "pa-1"}, "pa-1"),
    ):
        path = tmp_path / f"long-{name}.jsonl"
        chatter = (json.dumps({"type": "item.completed", "n": i}) for i in range(5000))
        path.write_text("\n".join([json.dumps(header), *chatter]) + "\n")
        assert harness_for(name).session_id_from_trace(path) == expected


def test_a_trace_with_no_session_line_yields_no_id(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    for name in ("claude_code", "codex", "prime_agent"):
        assert harness_for(name).session_id_from_trace(path) is None


def test_a_resumed_eval_launch_names_the_rollout_session_per_harness(tmp_path: Path) -> None:
    """The argv shape a resumed eval_after fork launches with, per harness.

    Claude Code restores nothing on resume, so the rebuilt argv must still carry the eval
    instruction and never a --session-id beside the --resume. codex takes the subcommand form
    with the prompt as the trailing positional. prime-agent separates the id from the prompt
    with --, so the id can never be read as a message.
    """
    kwargs = dict(
        mcp_url="http://h:1/mcp",
        system_prompt="EVALSYS",
        user_prompt="go",
        model="claude-opus-5",
        trace_path=tmp_path / "t",
        session_id="rollout-sid",
        resume=True,
    )

    claude = harness_for("claude_code").launch(**kwargs).argv
    assert claude[claude.index("--resume") + 1] == "rollout-sid"
    assert "--session-id" not in claude
    assert claude[claude.index("--append-system-prompt") + 1] == "EVALSYS"

    codex = harness_for("codex").launch(**kwargs).argv
    assert codex[:4] == ["codex", "exec", "resume", "rollout-sid"]
    assert codex[-1] == "EVALSYS\n\ngo"

    prime = harness_for("prime_agent").launch(**kwargs).argv
    at = prime.index("--resume")
    assert prime[at + 1] == "rollout-sid"
    assert prime[at + 2 :] == ["--", "EVALSYS\n\ngo"]


# What each harness's preflight requires of a transcript, per case. Every ``valid`` body was
# resumed by its pinned CLI to the auth or transport boundary in a network-off probe, and each
# was found by minimizing a record the CLI itself wrote, so the positive control is a proven
# one rather than a guess. Every degenerate body was refused by that same CLI: empty and
# ``undecodable`` files get the CLIs' own parse refusals ("No conversation found", "failed to
# read session metadata", first-parseable-line anchoring), so a preflight that passed one
# would only move the refusal to after the fan-out was paid for.
_SID = "sid-123"
_TS = "2026-08-12T00:00:00.000Z"
_TRANSCRIPT_CASES = {
    "claude_code": {
        "rel": f".claude/projects/-work/{_SID}.jsonl",
        # The verified floor: a user line with a message (role + content), a timestamp, and
        # the sessionId. Dropping the timestamp or the message is "No conversation found";
        # dropping the content crashes the resume.
        "valid": json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": "hello"},
                "timestamp": _TS,
                "sessionId": _SID,
            }
        )
        + "\n",
        # Identity-bearing but undecodable: the id without a conversation. This exact body
        # was this file's previous positive control, and the CLI refuses it.
        "undecodable": json.dumps({"type": "user", "sessionId": _SID}) + "\n",
        "mismatch": None,  # filled below: the valid body recorded under another session's id
        "substring_rel": f".claude/projects/-work/{_SID}4.jsonl",
    },
    "codex": {
        "rel": f".codex/sessions/2026/08/12/rollout-2026-08-12T00-00-00-{_SID}.jsonl",
        # The verified floor: a first-line session_meta whose payload decodes whole (id,
        # timestamp, cwd, originator, cli_version). No items are required after it.
        "valid": json.dumps(
            {
                "timestamp": _TS,
                "type": "session_meta",
                "payload": {
                    "id": _SID,
                    "timestamp": _TS,
                    "cwd": "/work",
                    "originator": "codex_exec",
                    "cli_version": "0.147.0",
                },
            }
        )
        + "\n",
        # Identity-bearing but undecodable: the previous positive control; the CLI refuses it
        # as unreadable session metadata.
        "undecodable": json.dumps(
            {"type": "session_meta", "payload": {"id": _SID, "cwd": "/work"}}
        )
        + "\n",
        "mismatch": None,
        "substring_rel": f".codex/sessions/2026/08/12/rollout-2026-08-12T00-00-00-{_SID}4.jsonl",
    },
    "prime_agent": {
        # The verified floor is the header alone, and the CLI accepted exactly this body
        # (timestamp absent and all), so unlike the other two it stays. What the CLI does
        # refuse is a header that is not the FIRST parseable line, which is the undecodable
        # case here.
        "rel": f".prime/agent/sessions/{_SID}.jsonl",
        "valid": json.dumps({"type": "session", "version": 3, "id": _SID, "cwd": "/work"})
        + "\n",
        "undecodable": json.dumps(
            {"type": "message", "id": "m1", "message": {"role": "user", "content": "x"}}
        )
        + "\n"
        + json.dumps({"type": "session", "version": 3, "id": _SID, "cwd": "/work"})
        + "\n",
        "mismatch": None,
        "substring_rel": f".prime/agent/sessions/{_SID}4.jsonl",
    },
}
# The mismatch case is the valid body verbatim with the identity swapped, so what it tests is
# identity alone and never a second structural difference.
for _case in _TRANSCRIPT_CASES.values():
    _case["mismatch"] = _case["valid"].replace(_SID, "other-session")


def _home_with(tmp_path: Path, rel: str, body: str) -> Path:
    home = tmp_path
    target = home / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return home


@pytest.mark.parametrize("name", sorted(_TRANSCRIPT_CASES))
def test_a_valid_transcript_resolves_and_lives_in_the_forked_subtrees(
    tmp_path: Path, name: str
) -> None:
    """The positive case per harness: the real layout with the CLI's minimum content resolves,
    and it sits inside a declared session-state subtree, which is what a fork's copy carries."""
    case = _TRANSCRIPT_CASES[name]
    home = _home_with(tmp_path, case["rel"], case["valid"])
    harness = harness_for(name)

    assert harness.session_transcript(home, _SID) == home / case["rel"]
    assert harness.session_transcript(home, "absent-999") is None
    assert any(case["rel"].startswith(prefix + "/") for prefix in harness.session_state_dirs)


@pytest.mark.parametrize("name", sorted(_TRANSCRIPT_CASES))
def test_a_transcript_that_only_wears_the_id_does_not_pass_the_preflight(
    tmp_path: Path, name: str
) -> None:
    """The degenerate files the CLIs refuse must not pass the preflight either.

    Five ways a file can wear the id without being the session: empty (a crashed leg's
    leftover), malformed (a cut file with no parseable record), identity-bearing but
    undecodable (the id is in the right place and the record around it does not decode as a
    session; each harness's case is a body its pinned CLI was observed to refuse), an
    identity mismatch (the file's own metadata names another session), and a filename that
    merely contains the id (a longer id's file). Each gets its own home so one passing case
    cannot mask another.
    """
    case = _TRANSCRIPT_CASES[name]
    harness = harness_for(name)

    empty = _home_with(tmp_path / "empty", case["rel"], "")
    assert harness.session_transcript(empty, _SID) is None

    malformed = _home_with(tmp_path / "malformed", case["rel"], "not json\n{cut mid-")
    assert harness.session_transcript(malformed, _SID) is None

    undecodable = _home_with(tmp_path / "undecodable", case["rel"], case["undecodable"])
    assert harness.session_transcript(undecodable, _SID) is None

    mismatch = _home_with(tmp_path / "mismatch", case["rel"], case["mismatch"])
    assert harness.session_transcript(mismatch, _SID) is None

    # A valid transcript for a LONGER id whose name contains this one: the substring trap.
    longer = _home_with(
        tmp_path / "substring", case["substring_rel"], case["valid"].replace(_SID, f"{_SID}4")
    )
    assert harness.session_transcript(longer, _SID) is None


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
        heldout_ids=(),
        egress=egress_summary,
    )

    for path in (manifest_path, results_path):
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert _absolute_path_values(doc) == [], f"{path.name} leaks an absolute path"


def test_a_cell_leaves_one_results_file_whichever_way_its_run_ended(tmp_path: Path) -> None:
    """A results directory holds one artifact per cell.

    A rerun already replaces what the last run of that cell wrote, so the name the new run did
    not take has to go with it. Two files describing one cell from two runs is how a reader ends
    up quoting whichever of them reads better.
    """
    results = tmp_path / "results"
    ids = [1, 2]

    def publish(rows: list[TaskResult]) -> Path:
        return write_results(
            results / "cell.json",
            manifest={},
            phases={"eval_before": rows, "eval_after": rows, "rollout": []},
            stopping={},
            heldout_ids=ids,
        )

    lost_one = publish([_row(1, 0.5)])
    assert lost_one.name == "cell.incomplete.json"
    assert sorted(p.name for p in results.glob("*.json")) == ["cell.incomplete.json"]
    # The rows went in raw, the way a caller that assembled a phase itself would hand them over,
    # and the id with none is still in the published file: this boundary fills too, so no caller
    # can publish a hole by not filling first.
    doc = json.loads(lost_one.read_text(encoding="utf-8"))
    assert [r["task_idx"] for r in doc["eval_before"]["tasks"]] == ids
    assert doc["eval_before"]["tasks"][1]["closure"] == "missing"

    # The rerun that lost nothing takes the finished name, and the incomplete file it supersedes
    # does not survive beside it.
    complete = publish([_row(1, 0.5), _row(2, 0.5)])
    assert complete.name == "cell.json"
    assert sorted(p.name for p in results.glob("*.json")) == ["cell.json"]

    # And back again, so neither direction is the special case.
    assert publish([_row(2, 0.5)]).name == "cell.incomplete.json"
    assert sorted(p.name for p in results.glob("*.json")) == ["cell.incomplete.json"]


# ----- the feedback arm survives into the published record ------------------------------------


def test_the_row_stamp_and_the_legacy_manifest_both_name_the_arm(tmp_path: Path) -> None:
    """A pre-axis manifest publishes as the never arm, and served rows keep their own stamp.

    The manifest claim and the row stamp are deliberately independent: the manifest says what
    was asked for, the row says what the stream actually served, and a disagreement between
    them must stay visible in the artifact rather than be normalized away.
    """
    served = TaskResult(
        seq=1,
        position=1,
        task_idx=1,
        closure="sealed",
        reward=1.0,
        success=True,
        feedback_regime="never",
    )

    path = write_results(
        tmp_path / "cell.json",
        manifest={"cell": {"name": "c"}},
        phases={"eval_before": [served], "eval_after": [served], "rollout": [served]},
        stopping={},
        heldout_ids=[1],
    )

    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["manifest"]["cell"]["rollout_feedback"] == "never"
    assert doc["rollout"]["tasks"][0]["feedback_regime"] == "never"
    assert doc["eval_before"]["tasks"][0]["feedback_regime"] == "never"


def test_a_manifest_that_names_its_arm_is_not_rewritten(tmp_path: Path) -> None:
    """Backfill is for absence only: an explicit immediate manifest keeps saying immediate."""
    path = write_results(
        tmp_path / "cell.json",
        manifest={"cell": {"name": "c", "rollout_feedback": "immediate"}},
        phases={"eval_before": [], "eval_after": [], "rollout": []},
        stopping={},
        heldout_ids=(),
    )

    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["manifest"]["cell"]["rollout_feedback"] == "immediate"


def test_a_resumed_run_keeps_the_arm_its_record_started_under() -> None:
    """Absence in a pre-axis manifest reads as never, never as today's default."""
    from shobench.runner import recorded_rollout_feedback

    assert recorded_rollout_feedback({"cell": {"name": "c"}}) == "never"
    assert recorded_rollout_feedback({}) == "never"
    assert (
        recorded_rollout_feedback({"cell": {"rollout_feedback": "immediate"}}) == "immediate"
    )


# ----- the eval context survives into the published record ------------------------------------


def test_a_continued_run_keeps_the_eval_context_its_record_started_under() -> None:
    """Absence in a pre-axis manifest reads as cold, never as today's resumed default: every
    pre-axis eval task ran fresh, and a continuation that forked a conversation into that run
    would append to a measurement whose before-bookend never had one."""
    from shobench.runner import recorded_eval_context

    assert recorded_eval_context({"cell": {"name": "c"}}) == "cold"
    assert recorded_eval_context({}) == "cold"
    assert recorded_eval_context({"cell": {"eval_context": "resumed"}}) == "resumed"


def test_a_pre_axis_manifest_publishes_as_the_cold_context(tmp_path: Path) -> None:
    """The published record backfills absence explicitly, and an explicit value is kept."""
    path = write_results(
        tmp_path / "cell.json",
        manifest={"cell": {"name": "c"}},
        phases={"eval_before": [], "eval_after": [], "rollout": []},
        stopping={},
        heldout_ids=(),
    )
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["manifest"]["cell"]["eval_context"] == "cold"

    explicit = write_results(
        tmp_path / "explicit" / "cell.json",
        manifest={"cell": {"name": "c", "eval_context": "resumed"}},
        phases={"eval_before": [], "eval_after": [], "rollout": []},
        stopping={},
        heldout_ids=(),
    )
    doc = json.loads(explicit.read_text(encoding="utf-8"))
    assert doc["manifest"]["cell"]["eval_context"] == "resumed"
