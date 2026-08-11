"""Does the pre-publication gate actually stop a signature, and only a signature?

Two properties matter and they pull against each other. The scrub has to remove every
piece of provider ciphertext, because a published signature can be replayed to
reconstruct the raw chain of thought it came from. It also has to leave the rest of the
trace exactly as it found it, because the runner reads a finished trace back to recover
the session id, the observed models and the stop verdict, and it reads from a bounded
tail. A scrub that dropped lines would slide that window and quietly change what a
re-read concludes.

So these tests pin both sides: what leaves, what stays, and that the verifier is willing
to say no. The verifier is checked against material the scrubber has never seen, since a
verifier that only agrees with the scrubber proves nothing.

Fixtures only. Nothing here needs a network, a container, or a provider.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shobench.scrub import (
    ReasoningSignatureFound,
    assert_publishable,
    find_reasoning_material,
    scrub_file,
    scrub_value,
    verify_run_dir,
)
from shobench.scrub import main as scrub_main

# A signature-shaped value. Real ones are long base64; the length is irrelevant to every
# property under test, and a short one keeps the failure output readable.
FAKE_SIG = "ErUBCkYIBxgCKkBd0not0publish"


def _assistant(*blocks: dict) -> dict:
    return {"type": "assistant", "message": {"role": "assistant", "content": list(blocks)}}


def _write(path: Path, events: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


# ----- what leaves ------------------------------------------------------------------


def test_a_thinking_block_loses_its_signature() -> None:
    block = {"type": "thinking", "thinking": "weighing it up", "signature": FAKE_SIG}
    assert scrub_value(block) == {"type": "thinking", "thinking": "weighing it up"}


def test_a_thinking_block_with_no_text_left_is_dropped_entirely() -> None:
    # The empty-text case is the one that actually ships: the API returns the signature
    # without the reasoning, so the block is pure ciphertext in an envelope.
    record = _assistant({"type": "thinking", "thinking": "", "signature": FAKE_SIG})
    scrubbed = scrub_value(record)
    assert scrubbed["message"]["content"] == []


def test_a_redacted_thinking_block_loses_its_data_payload() -> None:
    # This one hides its ciphertext under `data`, so a scrub keyed only to the field name
    # `signature` would pass it through intact.
    record = _assistant({"type": "redacted_thinking", "data": FAKE_SIG})
    assert scrub_value(record)["message"]["content"] == []


def test_a_streamed_signature_delta_is_stripped_without_declaring_itself_a_thinking_block() -> None:
    event = {"type": "stream_event", "delta": {"type": "signature_delta", "signature": FAKE_SIG}}
    assert scrub_value(event) == {"type": "stream_event", "delta": {"type": "signature_delta"}}


def test_a_signature_serialised_into_free_text_is_blanked_not_deleted() -> None:
    # Captured stderr is somebody's log line. Cutting the field out would corrupt the
    # diagnostic the tail was captured for, so only the value goes.
    tail = f'request failed: {{"type":"thinking","signature":"{FAKE_SIG}"}} (retrying)'
    scrubbed = scrub_value({"stderr_tail": tail})
    assert FAKE_SIG not in scrubbed["stderr_tail"]
    assert scrubbed["stderr_tail"].startswith("request failed:")
    assert scrubbed["stderr_tail"].endswith("(retrying)")


# ----- what stays -------------------------------------------------------------------


def test_reasoning_text_survives_when_the_provider_actually_sent_it() -> None:
    # The goal is publishable reasoning, not absent reasoning. Only the replayable part goes.
    block = {"type": "thinking", "thinking": "the queue is empty", "signature": FAKE_SIG}
    assert scrub_value(block)["thinking"] == "the queue is empty"


def test_the_records_the_runner_reads_back_are_untouched(tmp_path: Path) -> None:
    # session id, observed models and the stop verdict are all recovered by re-reading a
    # finished trace, so these four records are load-bearing for classification.
    events = [
        {"type": "system", "subtype": "init", "session_id": "abc-123"},
        _assistant({"type": "thinking", "thinking": "", "signature": FAKE_SIG}),
        _assistant({"type": "text", "text": "done"}),
        {"type": "result", "is_error": False, "num_turns": 3, "modelUsage": {"claude-opus-5": {}}},
    ]
    trace = _write(tmp_path / "eval_before" / "traces" / "task-00001-leg-0000.stream.jsonl", events)
    before = trace.read_text().splitlines()

    scrub_file(trace)
    after = trace.read_text().splitlines()

    assert len(after) == len(before), "line count must not move; the runner reads a bounded tail"
    assert after[0] == before[0]
    assert after[2] == before[2]
    assert after[3] == before[3]
    assert json.loads(after[3])["modelUsage"] == {"claude-opus-5": {}}
    assert json.loads(after[1])["message"]["content"] == []


def test_a_line_that_was_never_json_is_preserved(tmp_path: Path) -> None:
    # A trace is a captured stdout and can hold harness chatter. The runner's reader
    # tolerates those lines, so the scrub must not drop them.
    trace = tmp_path / "traces" / "leg-0000.stream.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text('warning: thinking budget exceeded\n{"type":"result","is_error":false}\n')
    scrub_file(trace)
    assert trace.read_text().splitlines()[0] == "warning: thinking budget exceeded"


# ----- the verifier says no ---------------------------------------------------------


def test_the_verifier_finds_a_signature_the_scrubber_never_ran_on() -> None:
    record = _assistant({"type": "thinking", "thinking": "", "signature": FAKE_SIG})
    assert find_reasoning_material(record) != []


def test_a_clean_record_is_publishable() -> None:
    assert find_reasoning_material(_assistant({"type": "text", "text": "done"})) == []
    assert_publishable(_assistant({"type": "text", "text": "done"}), "results")


def test_publication_is_refused_and_the_refusal_never_quotes_the_signature() -> None:
    record = _assistant({"type": "thinking", "thinking": "", "signature": FAKE_SIG})
    with pytest.raises(ReasoningSignatureFound) as excinfo:
        assert_publishable(record, "results/smoke.json")
    message = str(excinfo.value)
    assert "results/smoke.json" in message
    # The whole point of the gate is that the value never reaches a log or a tracker.
    assert FAKE_SIG not in message
    assert all(FAKE_SIG not in loc for loc in excinfo.value.locations)


def test_an_already_blanked_signature_is_not_a_finding() -> None:
    # Re-running the gate over its own output must not report the field it just emptied.
    assert find_reasoning_material({"type": "thinking", "text": "hi", "signature": ""}) == []


# ----- the CLI round-trips ----------------------------------------------------------


def _run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run-1"
    _write(
        run / "eval_before" / "traces" / "task-00007-leg-0000.stream.jsonl",
        [
            {"type": "system", "subtype": "init", "session_id": "s-1"},
            _assistant({"type": "thinking", "thinking": "", "signature": FAKE_SIG}),
            {"type": "result", "is_error": False},
        ],
    )
    err = run / "eval_before" / "traces" / "task-00007-leg-0000.err.txt"
    err.write_text(f'boom {{"signature":"{FAKE_SIG}"}}\n')
    return run


def test_check_refuses_a_dirty_run_dir_and_writes_nothing(tmp_path: Path, capsys) -> None:
    run = _run_dir(tmp_path)
    trace = run / "eval_before" / "traces" / "task-00007-leg-0000.stream.jsonl"
    before = trace.read_text()

    assert scrub_main([str(run), "--check"]) == 1
    assert trace.read_text() == before, "--check must not modify anything"


def test_scrub_traces_cleans_the_run_dir_and_then_passes_its_own_check(tmp_path: Path) -> None:
    run = _run_dir(tmp_path)

    assert scrub_main([str(run)]) == 0
    assert verify_run_dir(run) == {}
    traces = run / "eval_before" / "traces"
    # Both file kinds leave together, so the stderr sibling must be clean too.
    assert FAKE_SIG not in (traces / "task-00007-leg-0000.err.txt").read_text()
    lines = (traces / "task-00007-leg-0000.stream.jsonl").read_text()
    assert FAKE_SIG not in lines
    # And the run is still readable as a run.
    assert json.loads(lines.splitlines()[0])["session_id"] == "s-1"
    assert scrub_main([str(run), "--check"]) == 0


def test_scrubbing_is_idempotent(tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    assert scrub_main([str(run)]) == 0
    first = {p: p.read_text() for p in run.rglob("*") if p.is_file()}
    assert scrub_main([str(run)]) == 0
    assert {p: p.read_text() for p in run.rglob("*") if p.is_file()} == first


def test_a_missing_directory_is_an_error_not_a_pass(tmp_path: Path) -> None:
    # A gate that returns success on a path it could not read is worse than no gate.
    assert scrub_main([str(tmp_path / "nope")]) == 2
