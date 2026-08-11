"""Does the pre-publication gate actually stop a signature, and only a signature?

Two properties matter and they pull against each other. The scrub has to remove every
piece of provider ciphertext, because a published signature can be replayed to
reconstruct the raw chain of thought it came from. It also has to leave the rest of the
trace exactly as it found it, because the runner reads a finished trace back to recover
the session id, the observed models and the stop verdict, and it reads from a bounded
tail. A scrub that dropped lines would slide that window and quietly change what a
re-read concludes.

The two properties are not held to the same standard everywhere, and that is the design
rather than an inconsistency. Structured content gets precision: named carriers only,
byte-faithful, everything else verbatim. Free text gets suspicion: any sign of reasoning
material takes the whole region, prose included, because locating a carrier inside
arbitrary text is a contest the carrier keeps winning and the diagnostic it costs is not
the research data. So the free-text tests below pin whole-region replacement, and they
pin it against the spellings that beat a precise matcher: another escaping layer, a line
break, a wrapper string, a duplicate member, a truncated tail.

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
    FREE_TEXT_MARKER,
    ReasoningSignatureFound,
    ScrubReport,
    assert_publishable,
    find_reasoning_material,
    safe_label,
    safe_location_key,
    scrub_file,
    scrub_jsonl_text,
    scrub_text,
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
    # Compact, no spaces: this is the form the harness (node's JSON.stringify) actually
    # writes, and the form the byte-fidelity property is defined against.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e, separators=(",", ":")) for e in events) + "\n")
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


def test_a_signature_serialised_into_free_text_takes_the_whole_string_with_it() -> None:
    # The surrounding log line goes too, and that is the trade this gate makes. Keeping the
    # prose means locating the carrier's exact span inside arbitrary text, and every round
    # of that produced a new spelling that landed outside the span. Replacing the region
    # does not depend on locating anything, so a new spelling changes nothing, and what it
    # costs is a diagnostic rather than the trace's structured content.
    tail = f'request failed: {{"type":"thinking","signature":"{FAKE_SIG}"}} (retrying)'
    scrubbed = scrub_value({"stderr_tail": tail})
    assert scrubbed["stderr_tail"] == FREE_TEXT_MARKER


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
    # The label says which artifact by digest. The caller already knows the string they
    # passed and can hash it to match; nobody else needs to read it out of an exception.
    assert safe_label("results/smoke.json") in message
    # The whole point of the gate is that the value never reaches a log or a tracker.
    assert FAKE_SIG not in message
    assert all(FAKE_SIG not in loc for loc in excinfo.value.locations)


def test_a_caller_supplied_label_is_not_quoted_back_into_the_refusal() -> None:
    # A label is usually a path, a path is named after the task that produced it, and a
    # task name is other people's text. An exception is the most widely copied string in
    # any system, so the label is digested there like every other untrusted component.
    record = _assistant({"type": "thinking", "thinking": "", "signature": FAKE_SIG})
    label = "results/authorization-Bearer-credential-value.json"
    with pytest.raises(ReasoningSignatureFound) as excinfo:
        assert_publishable(record, label)
    assert "credential-value" not in str(excinfo.value)
    assert not hasattr(excinfo.value, "what"), "the raw label is not kept on the exception"
    assert excinfo.value.label == safe_label(label)


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
    # A failed request logged to stderr carries the whole block, type and all, which is what
    # lets the text scrub tell a reasoning signature from an unrelated domain one.
    err.write_text(f'stream error while replaying {{"type":"thinking","signature":"{FAKE_SIG}"}}\n')
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


# ----- a carrier escaped inside an outer JSON string --------------------------------


def test_a_signature_escaped_inside_an_outer_json_string_is_scrubbed(tmp_path: Path) -> None:
    # The carrier rides inside a result's stderr tail as serialised JSON, so its quotes are
    # escaped in the JSONL bytes and no bare `"signature"` needle survives to prefilter on.
    # A needle-gated shortcut would pass the line through for the verifier to reject; the
    # gate must instead parse it, scrub the embedded value, and round-trip the directory.
    inner = json.dumps({"type": "thinking", "thinking": "", "signature": FAKE_SIG})
    record = {
        "type": "result",
        "is_error": True,
        "verdict": {"evidence": {"stderr_tail": f"replay failed: {inner}"}},
    }
    line = json.dumps(record, separators=(",", ":"))
    assert '"signature"' not in line, "the needle is escaped away; that is the whole point"

    report = ScrubReport()
    scrubbed = scrub_jsonl_text(line + "\n", report)
    assert FAKE_SIG not in scrubbed
    assert report.text_redactions == 1

    run = tmp_path / "run"
    trace = run / "eval_before" / "traces" / "task-00001-leg-0000.stream.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(line + "\n")
    assert scrub_main([str(run)]) == 0, "the CLI must round-trip, not report incomplete"
    assert FAKE_SIG not in trace.read_text()
    assert verify_run_dir(run) == {}


def test_an_embedded_signature_is_found_whatever_order_the_keys_were_written_in(
    tmp_path: Path,
) -> None:
    # Nothing says a harness serialises `type` before `signature`. A matcher that scans from
    # the type token to the field only sees one of the two orders, so the other order
    # publishes: the CLI reports success, the token is still in the file, and the gate's own
    # re-read agrees it is clean. Recognition has to come from parsing the object, where key
    # order is not a fact about the object at all.
    inner = f'{{"signature":"{FAKE_SIG}","type":"thinking"}}'
    record = {
        "type": "result",
        "is_error": True,
        "verdict": {"evidence": {"stderr_tail": f"request failed: {inner}"}},
    }
    run = tmp_path / "run"
    trace = run / "eval_before" / "traces" / "task-00003-leg-0000.stream.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(json.dumps(record, separators=(",", ":")) + "\n")

    assert scrub_main([str(run)]) == 0
    assert FAKE_SIG not in trace.read_text(), "reversed keys are the same carrier"
    assert verify_run_dir(run) == {}, "and the gate agrees once, not by never looking"


def test_a_brace_inside_an_earlier_string_value_does_not_hide_a_signature() -> None:
    # `}` is an ordinary character inside a JSON string, so any rule that reasons about
    # where the object ends can be walked out of it early. Suspicion does not reason about
    # object boundaries at all, which is why this case stopped being interesting.
    block = json.dumps(
        {"type": "thinking", "thinking": "weigh } this", "signature": FAKE_SIG},
        separators=(",", ":"),
    )
    assert scrub_text(f"request failed: {block}") == FREE_TEXT_MARKER


# ----- byte fidelity ----------------------------------------------------------------


def test_a_scrubbed_dirty_line_differs_from_its_input_only_in_the_carrier() -> None:
    # The block keeps its text so it is not dropped, which means the sole byte change is the
    # removed `,"signature":"..."`. Pin the complete output to prove nothing else shifted.
    line = f'{{"type":"thinking","thinking":"weighing it up","signature":"{FAKE_SIG}"}}'
    report = ScrubReport()
    out = scrub_jsonl_text(line + "\n", report)
    assert out == '{"type":"thinking","thinking":"weighing it up"}\n'
    assert report.lines_rewritten == 1


def test_an_empty_trace_stays_empty() -> None:
    # A clean zero-byte trace must round-trip to zero bytes, not gain a newline, or a
    # re-read of a bounded tail would see a line that was never written.
    assert scrub_jsonl_text("", ScrubReport()) == ""


def test_scrub_file_leaves_an_empty_trace_at_zero_bytes(tmp_path: Path) -> None:
    trace = tmp_path / "traces" / "leg-0000.stream.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("")
    scrub_file(trace)
    assert trace.read_text() == ""


def test_an_ordinary_json_null_in_a_list_survives_byte_identically() -> None:
    # A null in a content list is the trace's own data. It looked like a dropped block only
    # because the scrub used one value for both "this block is gone" and "the caller wrote a
    # null here", and the list rebuild then filtered the caller's nulls out with it. This
    # line has no reasoning carrier anywhere, so the whole thing must come back unchanged.
    line = '{"type":"tool_result","items":[null],"signature":"domain"}'
    report = ScrubReport()
    assert scrub_jsonl_text(line + "\n", report) == line + "\n"
    assert report.lines_rewritten == 0


def test_a_null_beside_a_dropped_block_survives_the_drop() -> None:
    # The two meanings meet in one list: the emptied thinking block goes, the null stays.
    content = [None, {"type": "thinking", "thinking": "", "signature": FAKE_SIG}]
    record = {"type": "assistant", "message": {"role": "assistant", "content": content}}
    assert scrub_value(record)["message"]["content"] == [None]


def test_crlf_terminators_survive_a_rewrite(tmp_path: Path) -> None:
    # Reading a trace as text translates newlines on the way in and writes the translation
    # back out, so scrubbing line 1 silently rewrote the terminator of every other line in
    # the file. Byte fidelity is a claim about the bytes, and a line ending is bytes.
    trace = tmp_path / "traces" / "leg-0000.stream.jsonl"
    trace.parent.mkdir(parents=True)
    clean = b'{"type":"result","is_error":false}'
    dirty = f'{{"type":"thinking","thinking":"x","signature":"{FAKE_SIG}"}}'.encode()
    trace.write_bytes(dirty + b"\r\n" + clean + b"\r\n")

    scrub_file(trace)

    after = trace.read_bytes()
    assert after == b'{"type":"thinking","thinking":"x"}\r\n' + clean + b"\r\n"
    assert after.split(b"\r\n")[1] == clean, "the untouched line keeps its exact bytes"


def test_bytes_that_are_not_utf8_survive_a_rewrite(tmp_path: Path) -> None:
    # A trace is captured stdout from somebody else's process and can hold bytes that are not
    # UTF-8 at all. Decoding with errors="replace" turns them into replacement characters and
    # writing the result back makes that loss permanent, on a line the scrub never touched.
    trace = tmp_path / "traces" / "leg-0000.stream.jsonl"
    trace.parent.mkdir(parents=True)
    noise = b"warning: \xff\xfe truncated"
    dirty = f'{{"type":"thinking","thinking":"x","signature":"{FAKE_SIG}"}}'.encode()
    trace.write_bytes(dirty + b"\n" + noise + b"\n")

    scrub_file(trace)

    assert trace.read_bytes() == b'{"type":"thinking","thinking":"x"}\n' + noise + b"\n"


# ----- precision: ordinary content outside a reasoning block survives ---------------


def test_a_generic_signature_field_outside_a_reasoning_block_is_kept(tmp_path: Path) -> None:
    # A tool result can legitimately carry its own `signature`. It is not reasoning
    # ciphertext, so neither the scrubber may strip it nor the verifier reject it.
    record = {"type": "tool_result", "name": "verify_webhook", "signature": "sha256=deadbeef"}
    assert scrub_value(record) == record
    assert find_reasoning_material(record) == []
    assert_publishable(record, "results")  # does not raise


def test_a_generic_serialised_data_value_is_kept() -> None:
    # `data` is ciphertext only inside a redacted_thinking block. A base64 `data` in an
    # ordinary observation is domain content and must survive verbatim, structurally and
    # as embedded text, and must not trip the verifier either.
    payload = 'HTTP 200 {"data":"eyJhbGciOiJIUzI1NiJ9"} received'
    assert scrub_text(payload) == payload
    assert scrub_value({"observation": payload}) == {"observation": payload}
    assert find_reasoning_material({"observation": payload}) == []


def test_a_signature_nested_under_another_object_is_not_the_blocks_signature() -> None:
    # The negative case that sits beside the generic top-level one: here the reasoning type
    # and the `signature` really are close together, and structurally they still belong to
    # two different objects, so the `metadata` signature is domain content and survives.
    block = {"type": "thinking", "thinking": "public", "metadata": {"signature": "sha256=domain"}}
    assert scrub_value(block) == block, "structurally, the nested field is not a carrier"
    assert find_reasoning_material(block) == []


def test_the_same_nesting_inside_free_text_is_replaced_because_text_gets_no_precision() -> None:
    # The boundary, stated as a test. The identical block loses its precision guarantee the
    # moment it is somebody's log line rather than the trace's structure: proving the
    # `signature` belongs to `metadata` means parsing that text, and parsing is exactly what
    # a wrapper, an extra escaping layer or a truncation takes away. The clean field is lost
    # with the region, and it was a diagnostic rather than research data.
    block = {"type": "thinking", "thinking": "public", "metadata": {"signature": "sha256=domain"}}
    serialised = json.dumps(block, separators=(",", ":"))
    assert scrub_text(serialised) == FREE_TEXT_MARKER
    assert find_reasoning_material({"stderr_tail": serialised}) != []


# ----- the verifier is independent of the scrubber ----------------------------------


def test_the_verifier_catches_a_reasoning_type_the_scrubber_does_not_name() -> None:
    # The scrubber removes only its exact named types; the verifier re-derives "reasoning"
    # from its own rules and errs broad. A harness that renames the block to
    # `interleaved_thinking` is left by the scrubber yet still refused by the gate, the
    # fail-closed direction, which proves the two sides do not share one blind spot: this
    # test still fails (the verifier still catches it) even though the scrubber's tuple does
    # not mention the drifted type.
    drifted = _assistant({"type": "interleaved_thinking", "thinking": "", "signature": FAKE_SIG})
    assert FAKE_SIG in json.dumps(scrub_value(drifted)), "scrubber's exact tuple walks past it"
    assert find_reasoning_material(drifted) != [], "independent verifier still catches it"
    with pytest.raises(ReasoningSignatureFound):
        assert_publishable(drifted, "results")


def test_a_drifted_reasoning_type_makes_the_cli_fail_closed(tmp_path: Path) -> None:
    # End to end: the scrub leaves the drifted carrier, but the gate's own re-read refuses
    # to call the directory publishable, so the CLI reports incomplete rather than shipping.
    run = tmp_path / "run"
    trace = run / "eval_before" / "traces" / "task-00002-leg-0000.stream.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        json.dumps(
            _assistant({"type": "interleaved_thinking", "thinking": "", "signature": FAKE_SIG}),
            separators=(",", ":"),
        )
        + "\n"
    )
    assert scrub_main([str(run)]) == 1
    assert verify_run_dir(run) != {}


def test_a_truncated_tail_is_replaced_rather_than_left_for_the_verifier(tmp_path: Path) -> None:
    # An stderr tail is a bounded tail, so the request body in it can be cut mid-carrier and
    # the cut normally lands in the ciphertext itself. Nothing parses, so the old scrubber
    # left it and the gate refused forever. Suspicion needs no parse: the region goes, the
    # run is publishable, and the operator is not left holding an artifact no rerun can fix.
    truncated = f'replay failed: {{"type":"thinking","signature":"{FAKE_SIG}'
    assert scrub_text(truncated) == FREE_TEXT_MARKER
    assert find_reasoning_material({"stderr_tail": truncated}) != [], "and it was a finding"

    run = tmp_path / "run"
    trace = run / "eval_before" / "traces" / "task-00004-leg-0000.stream.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(json.dumps({"stderr_tail": truncated}, separators=(",", ":")) + "\n")
    assert scrub_main([str(run)]) == 0
    assert FAKE_SIG not in trace.read_text()
    assert verify_run_dir(run) == {}


# ----- a report never carries somebody else's text ----------------------------------


def test_an_untrusted_mapping_key_never_reaches_the_refusal() -> None:
    # Keys in a trace are other people's data. A harness that keys a map by request URL puts
    # the query string in the key, credentials and all, so a location built by interpolating
    # the key publishes the very class of value the gate exists to withhold, in the exception
    # message and in any tracker that catches it.
    key = "https://service.invalid/?token=credential-value"
    record = {key: {"type": "thinking", "thinking": "", "signature": FAKE_SIG}}

    with pytest.raises(ReasoningSignatureFound) as excinfo:
        assert_publishable(record, "results")

    reported = [str(excinfo.value), *excinfo.value.locations]
    for text in reported:
        assert "credential-value" not in text
        assert "service.invalid" not in text
        assert FAKE_SIG not in text
    # And the location is still a location: the key's position in its object plus a digest an
    # operator can recompute from a candidate key to say which one it was.
    assert any(safe_location_key(key) in loc for loc in excinfo.value.locations)
    assert any(loc.endswith(".signature") for loc in excinfo.value.locations)


def _err_run(tmp_path: Path, body: str) -> tuple[Path, Path]:
    """A run directory whose only artifact is a captured stderr tail with `body` in it."""
    run = tmp_path / "run"
    traces = run / "eval_before" / "traces"
    traces.mkdir(parents=True)
    err = traces / "task-00001-leg-0000.err.txt"
    err.write_text(body)
    return run, err


def _assert_gate_closes(run: Path, artifact: Path) -> None:
    """The gate saw it, the scrub cleared it, and the gate agrees afterwards."""
    assert scrub_main([str(run), "--check"]) == 1, "the gate must see it before the scrub"
    assert scrub_main([str(run)]) == 0, "and the scrub must be able to clear it"
    assert FAKE_SIG not in artifact.read_text()
    assert verify_run_dir(run) == {}


# ----- free text: every spelling that beat a precise matcher ------------------------


def test_a_carrier_inside_a_wrapper_string_is_not_hidden_by_the_extra_escaping(
    tmp_path: Path,
) -> None:
    # The carrier is serialised, then that serialisation is put in a string member of
    # another object, so its quotes are escaped one more time. A pass that parses the outer
    # object and steps over its span never looks inside the member, and a matcher on the raw
    # bytes sees `\"type\"` rather than `"type"`. Suspicion decodes before it matches, so
    # the layer count is not a fact the carrier gets to choose.
    inner = json.dumps({"type": "thinking", "signature": FAKE_SIG}, separators=(",", ":"))
    wrapper = json.dumps({"wrapper": inner}, separators=(",", ":"))
    run, err = _err_run(tmp_path, f"error: {wrapper}\n")
    _assert_gate_closes(run, err)


def test_a_carrier_split_across_two_stderr_lines_is_seen_by_both_sides(tmp_path: Path) -> None:
    # Two writers interleaving on one stream put the type on one line and the ciphertext on
    # the next. The scrubber read the file as one document and the verifier read it line by
    # line, so the relationship existed for one side and not the other, and a drifted type
    # that the scrubber declines to touch then published with the gate reporting success.
    # Both sides now split a file the same way, and a run of non-record lines is one region.
    body = f'stream error {{"type":"interleaved_thinking",\n"signature":"{FAKE_SIG}"}}\n'
    run, err = _err_run(tmp_path, body)
    _assert_gate_closes(run, err)
    assert len(err.read_text().splitlines()) == 2, "line count does not move"


def test_an_escaped_type_key_is_still_a_type_key(tmp_path: Path) -> None:
    # `"t\u0079pe"` is the same key to any JSON reader and a different string to any matcher
    # that did not decode first. This one also has no closing brace, so nothing parses it.
    run, err = _err_run(
        tmp_path, f'request failed: {{"t\\u0079pe":"thinking","signature":"{FAKE_SIG}\n'
    )
    _assert_gate_closes(run, err)


def test_an_escaped_type_value_is_still_a_reasoning_type(tmp_path: Path) -> None:
    # Same trick moved from the key to the value, on a tail the capture also cut, so nothing
    # parses it back into `thinking` on the way past. Either half alone was survivable
    # before; the two together were not.
    run, err = _err_run(
        tmp_path, f'request failed: {{"type":"\\u0074hinking","signature":"{FAKE_SIG}\n'
    )
    _assert_gate_closes(run, err)
    # And the spelling that always parsed still behaves, so the decode did not cost the
    # ordinary path anything.
    whole = f'request failed: {{"type":"\\u0074hinking","signature":"{FAKE_SIG}"}}'
    assert scrub_text(whole) == FREE_TEXT_MARKER


def test_a_parsed_island_inside_a_broken_wrapper_does_not_split_the_suspicion(
    tmp_path: Path,
) -> None:
    # The old residue sweep needed both needles in one leftover stretch of text, so any
    # well-formed object between them cut the leftovers in two and each half looked
    # harmless. Suspicion has no notion of leftovers to be split.
    body = f'broken {{"type":"thinking", {{"harmless":1}}, "signature":"{FAKE_SIG}\n'
    run, err = _err_run(tmp_path, body)
    _assert_gate_closes(run, err)


def test_a_tail_cut_mid_word_is_still_a_tail_that_carries_ciphertext(tmp_path: Path) -> None:
    # The capture stopped inside the word `thinking`, after the signature had already been
    # written: the ciphertext is whole and the sign of it is half there. A word cut at the
    # end of the text counts as that word, and the newline the capture was written out with
    # does not make it stop being the end.
    run, err = _err_run(tmp_path, f'POST body {{"signature":"{FAKE_SIG}","type":"thin\n')
    _assert_gate_closes(run, err)


def test_a_console_dump_with_unquoted_keys_is_the_same_carrier(tmp_path: Path) -> None:
    # A node harness logging the request object writes `{ type: 'thinking', signature: '..' }`
    # with no quotes on the keys and single quotes on the values. It is not JSON and never
    # will be, and it carries the identical replayable value.
    run, err = _err_run(tmp_path, f"request: {{ type: 'thinking', signature: '{FAKE_SIG}' }}\n")
    _assert_gate_closes(run, err)


def test_distance_between_the_two_signs_does_not_buy_a_carrier_anything(tmp_path: Path) -> None:
    # A proximity window would be a documented bypass: padding the gap between the type and
    # the ciphertext is free for whoever writes the text. There is no window, so a thousand
    # lines of filler changes nothing about the verdict.
    filler = "\n".join(f"line {i} of ordinary log output" for i in range(1000))
    body = f'starting replay of a thinking block\n{filler}\n"signature":"{FAKE_SIG}"\n'
    run, err = _err_run(tmp_path, body)
    _assert_gate_closes(run, err)
    # Only the line that actually held ciphertext bytes was replaced, so the collateral is
    # bounded even though the suspicion is not.
    kept = err.read_text().splitlines()
    assert kept[0] == "starting replay of a thinking block"
    assert kept[500] == "line 499 of ordinary log output"
    assert kept[-1] == FREE_TEXT_MARKER


def test_a_scrubbed_stderr_body_is_not_suspicious_to_either_side(tmp_path: Path) -> None:
    # The marker has to be inert or the gate would refuse its own output and a second scrub
    # would keep rewriting the file.
    body = f'replay failed: {{"type":"thinking","signature":"{FAKE_SIG}"}}\n'
    run, err = _err_run(tmp_path, body)
    assert scrub_main([str(run)]) == 0
    once = err.read_text()
    assert once.strip() == FREE_TEXT_MARKER
    assert scrub_main([str(run)]) == 0
    assert err.read_text() == once
    assert scrub_text(FREE_TEXT_MARKER) == FREE_TEXT_MARKER
    assert find_reasoning_material({"stderr_tail": FREE_TEXT_MARKER}) == []


def test_a_carrier_serialised_into_a_mapping_key_is_not_a_blind_spot(tmp_path: Path) -> None:
    # A walk that only ever descends into values never reads a key, and a key is a string
    # like any other. A harness that keys a map by the request body it sent puts a whole
    # serialised block into key position.
    key = json.dumps({"type": "thinking", "signature": FAKE_SIG}, separators=(",", ":"))
    run = tmp_path / "run"
    trace = run / "eval_before" / "traces" / "task-00009-leg-0000.stream.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(json.dumps({key: 1}, separators=(",", ":")) + "\n")
    _assert_gate_closes(run, trace)
    assert json.loads(trace.read_text()) == {FREE_TEXT_MARKER: 1}


# ----- structured content: the parser's blind spot, and the carrier's own value -----


def test_a_duplicate_type_member_does_not_launder_a_carrier(tmp_path: Path) -> None:
    # `json.loads` keeps the last `type`, so a record can declare itself reasoning to
    # whatever reads the bytes and prose to whatever parses them, and a pass that trusts the
    # parsed view calls it clean. The raw members keep both, and every declared type is
    # applied, because which one wins is undefined and undefined is not a safe direction.
    line = f'{{"type":"thinking","signature":"{FAKE_SIG}","type":"text"}}'
    run = tmp_path / "run"
    trace = run / "eval_before" / "traces" / "task-00008-leg-0000.stream.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(line + "\n")
    _assert_gate_closes(run, trace)
    # The bytes around the carrier are untouched, both `type` members included: this line is
    # structure, so it keeps the precision guarantee even while being treated as suspect.
    assert trace.read_text() == '{"type":"thinking","signature":"","type":"text"}\n'


def test_a_blank_carrier_field_is_not_deleted(tmp_path: Path) -> None:
    # The scrubber's own rule says a carrier is a non-empty string, and the structural path
    # has to obey that rule rather than delete by field name: a blank `signature` is a field
    # the trace wrote, no verifier would flag it, and removing it is an edit outside the
    # contract on a line that was already clean.
    line = '{"type":"thinking","thinking":"public","signature":""}'
    report = ScrubReport()
    assert scrub_jsonl_text(line + "\n", report) == line + "\n"
    assert report.lines_rewritten == 0
    assert report.fields_removed == 0
    assert find_reasoning_material(json.loads(line)) == []


def test_a_carrier_field_that_is_not_a_string_is_not_deleted() -> None:
    # A null or an object under `signature` is not ciphertext, so neither name is stripped on
    # sight. The same holds for the streamed delta, which used to lose an empty `signature`.
    nulled = {"type": "thinking", "thinking": "public", "signature": None}
    assert scrub_value(nulled) == nulled
    structured = {"type": "redacted_thinking", "text": "public", "data": {"kept": 1}}
    assert scrub_value(structured) == structured
    blank_delta = {"type": "signature_delta", "signature": ""}
    assert scrub_value(blank_delta) == blank_delta


def test_a_block_keeps_its_other_members_when_the_carrier_leaves() -> None:
    # Emptiness was decided by whether `thinking` or `text` was truthy, so a block whose
    # reasoning text was empty was deleted whole and anything else it carried went with it.
    # A block is dropped only when it holds nothing of its own besides the carrier.
    record = {
        "content": [
            {
                "type": "thinking",
                "thinking": "",
                "signature": FAKE_SIG,
                "metadata": {"domain": "keep-me"},
            }
        ]
    }
    scrubbed = scrub_value(record)
    assert scrubbed == {
        "content": [{"type": "thinking", "thinking": "", "metadata": {"domain": "keep-me"}}]
    }


def test_a_block_that_held_nothing_but_its_carrier_is_still_dropped() -> None:
    # The other half of the same rule, so widening it did not turn into keeping envelopes.
    record = {"content": [{"type": "thinking", "thinking": "", "signature": FAKE_SIG}]}
    assert scrub_value(record) == {"content": []}


def test_a_blank_thinking_block_with_no_carrier_at_all_is_left_alone() -> None:
    # Nothing was removed here, so there is no reason to drop the block: deleting it would
    # be an edit to content that was never a carrier.
    record = {"content": [{"type": "thinking", "thinking": ""}]}
    assert scrub_value(record) == record


def test_a_run_dir_report_names_lines_and_not_keys(tmp_path: Path) -> None:
    # Same rule on the path the CLI actually takes, where the finding comes off disk.
    key = "authorization: Bearer credential-value"
    run = tmp_path / "run"
    trace = run / "eval_before" / "traces" / "task-00005-leg-0000.stream.jsonl"
    trace.parent.mkdir(parents=True)
    record = {key: {"type": "interleaved_thinking", "thinking": "", "signature": FAKE_SIG}}
    trace.write_text(json.dumps(record, separators=(",", ":")) + "\n")

    findings = verify_run_dir(run)
    locations = findings[f"file[0]{safe_label(str(trace))}"]
    assert locations, "the drifted type is still refused"
    assert all("credential-value" not in loc for loc in locations)
    assert all(loc.startswith("line 1 ") for loc in locations)


def test_a_trace_filename_never_reaches_a_report(tmp_path: Path, capsys) -> None:
    # A trace is named after the task that produced it, and a task name is not this
    # module's text to quote. Both the scrub report and the --check report name files by
    # position and digest, so a file called after a credential does not publish it into
    # normal output on the way to telling somebody the run was refused.
    run = tmp_path / "run"
    traces = run / "eval_before" / "traces"
    traces.mkdir(parents=True)
    trace = traces / "authorization-Bearer-credential-value.err.txt"
    trace.write_text(f'replay failed: {{"type":"thinking","signature":"{FAKE_SIG}"}}\n')

    assert scrub_main([str(run), "--check"]) == 1
    checked = capsys.readouterr()
    assert "credential-value" not in checked.out
    assert safe_label(str(trace)) in checked.out

    assert scrub_main([str(run)]) == 0
    scrubbed = capsys.readouterr()
    assert "credential-value" not in scrubbed.out
    assert safe_label(str(trace)) in scrubbed.out
