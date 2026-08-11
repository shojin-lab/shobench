"""Reasoning signatures come out of a trace before the trace is published.

**A thinking block's signature is not inert.** An assistant message carrying reasoning
ships two things: the reasoning text, and a `signature` the provider uses to prove the
block came from its own model. "Stealing Reasoning Traces from Proprietary LLM APIs"
(https://arxiv.org/abs/2608.09867) shows the signature can be replayed into a weaker
sibling model to reconstruct the original model's raw chain of thought verbatim. Doing
that across 6,708 public trajectories recovered 315,320 reasoning blocks and, inside
them, hundreds of credentials and PII artifacts that nobody meant to publish. A
published signature is therefore an attack input, and an empty `thinking` field is no
defence at all: the signature is the part that replays.

**So the run keeps everything and the export keeps less.** A trace under `runs/` is the
operator's own data and stays whole, because that is what debugging a leg needs. The
moment an artifact is destined to leave the machine it goes through here first, and
then through :func:`assert_publishable`, which refuses rather than repairs. Scrubbing
and verifying are deliberately two functions: a scrubber that also certified its own
output would fail silently the day it grows a blind spot, so the verifier re-derives the
answer from the bytes that are actually about to ship.

**What is removed, and only what is removed.** The `signature` off every thinking block,
the ciphertext `data` off every `redacted_thinking` block, the `signature` off a streamed
`signature_delta`, and any block that carries nothing else once those are gone. A carrier
serialised into a free-text string (a request body dumped into an error tail) is blanked in
place, and it is found by parsing the serialised object rather than by matching a pattern
around it, so key order does not matter, a `}` inside a string value does not hide it, and a
nested object is judged as the object it is. Nothing else is touched: a `signature` field on
a tool result, a `data` value in an observation, a `signature` under some block's `metadata`,
and every ordinary JSON null are domain content and are left exactly as found. The goal is to
publish reasoning without publishing the means to forge or extract it, not to rewrite the
trace.

**What is preserved, and why it is not negotiable.** Byte-for-byte the whole trace except
the carrier bytes, and above all line count and line order. The runner reads a finished
trace back to recover the session id, the observed models and the stop classification, and
it does so from a bounded tail of the file. Dropping lines would slide that window and
change what a re-read concludes, so a record whose content empties out is written back with
an empty content list rather than deleted, and a clean line comes through as its exact
input bytes, terminator included, whatever its encoding.

**The contract, stated once.** Scrubbed output is byte-identical to its input except that
the named reasoning carriers are removed or blanked. The scrubber is precise: it touches
only those named carriers, and it decides per object rather than per neighbourhood of text.
The verifier is independent and fail-closed: it re-derives what a carrier looks like from its
own rules, raises rather than repairs, and errs toward flagging more than the scrubber
removes, including material the scrubber declined to rewrite, because a false refusal costs
one rerun while a false pass ships an attack input. No signature value, no credential value
and no untrusted key text ever reaches a report, an exception or a log: those carry counts,
structural positions, and only the field names this module itself defines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Locating serialised JSON inside free text ----------------------------------------
# A carrier arrives either as JSON structure or as JSON serialised into somebody's log line,
# and the second form still has to be recognised object by object. A regex cannot do that: a
# bound like `[^}]*` cannot see that a `}` sits inside a string value, and it cannot tell a
# nested object from its parent, so it is at once order-sensitive (it only matches when the
# type happens to be written before the signature) and boundary-blind. So the real parser
# locates the objects: walk to a `{`, ask `raw_decode` how far it consumes, and read the
# members back with the same decoder to learn where each raw value sits.
#
# This is mechanism, not policy. The carrier RULES stay split between the scrubber and the
# verifier further down, which is the split that stops one blind spot blinding both. What is
# shared here is only the answer to "where does this object begin and end", which the
# language's own parser defines and which holds no rule of ours to get wrong.

_DECODER = json.JSONDecoder()
_WHITESPACE = " \t\n\r"

# Given an enclosing object's `type` plus one of its members, is that member ciphertext?
# Both sides answer this question, and they answer it differently on purpose.
CarrierRule = Callable[[object, str, Any], bool]


def _skip_ws(text: str, index: int) -> int:
    """The next non-whitespace offset. `raw_decode` will not skip leading space itself."""
    while index < len(text) and text[index] in _WHITESPACE:
        index += 1
    return index


def _members(text: str, start: int) -> list[tuple[str, Any, int, int]] | None:
    """Members of the JSON object at `text[start]`, as (key, value, raw start, raw end).

    Keys and values are consumed by `raw_decode`, so strings, nested objects and arrays are
    delimited by the real parser and a brace inside a string value is data rather than a
    boundary. The raw span comes back beside the parsed value because a carrier has to be
    blanked in the original bytes, not in a reserialisation of them. `None` means the text
    does not hold a well-formed object here, a fact the caller must not paper over: for the
    scrubber it means there is nothing safe to repair, and for the verifier it means the
    residue sweep has to take over.
    """
    index = _skip_ws(text, start + 1)
    members: list[tuple[str, Any, int, int]] = []
    if index < len(text) and text[index] == "}":
        return members
    while True:
        try:
            key, index = _DECODER.raw_decode(text, index)
        except ValueError:
            return None
        if not isinstance(key, str):
            return None
        index = _skip_ws(text, index)
        if index >= len(text) or text[index] != ":":
            return None
        value_start = _skip_ws(text, index + 1)
        try:
            value, value_end = _DECODER.raw_decode(text, value_start)
        except ValueError:
            return None
        members.append((key, value, value_start, value_end))
        index = _skip_ws(text, value_end)
        if index >= len(text):
            return None
        if text[index] == ",":
            index = _skip_ws(text, index + 1)
            continue
        if text[index] == "}":
            return members
        return None


def _elements(text: str, start: int) -> list[tuple[Any, int, int]] | None:
    """Elements of the JSON array at `text[start]`, as (value, raw start, raw end).

    Arrays matter because that is where a harness puts a message's content blocks, so a
    walk that only descended through objects would step over every block in a content list.
    """
    index = _skip_ws(text, start + 1)
    elements: list[tuple[Any, int, int]] = []
    if index < len(text) and text[index] == "]":
        return elements
    while True:
        try:
            value, end = _DECODER.raw_decode(text, index)
        except ValueError:
            return None
        elements.append((value, index, end))
        index = _skip_ws(text, end)
        if index >= len(text):
            return None
        if text[index] == ",":
            index = _skip_ws(text, index + 1)
            continue
        if text[index] == "]":
            return elements
        return None


def _carrier_spans(
    text: str, value: Any, start: int, is_carrier: CarrierRule
) -> list[tuple[int, int]]:
    """Raw spans of every carrier value inside the JSON value parsed at `text[start]`.

    Each object is judged on its OWN `type`, and the walk then descends into its members and
    elements, so a `signature` sitting under some block's `metadata` belongs to `metadata`
    and not to the reasoning block above it, however adjacent the two look once serialised.
    """
    spans: list[tuple[int, int]] = []
    if isinstance(value, dict):
        members = _members(text, start)
        if members is None:
            return spans
        block_type = value.get("type")
        for key, member, member_start, member_end in members:
            if is_carrier(block_type, key, member):
                spans.append((member_start, member_end))
            else:
                spans.extend(_carrier_spans(text, member, member_start, is_carrier))
    elif isinstance(value, list):
        elements = _elements(text, start)
        if elements is None:
            return spans
        for element, element_start, _end in elements:
            spans.extend(_carrier_spans(text, element, element_start, is_carrier))
    return spans


def _object_start(text: str, index: int) -> int:
    """The next offset at or after `index` where a JSON object could begin, or -1.

    A `{` on its own is not a candidate: JSON keys are strings, so an object is a `{`
    followed by whitespace and then either `"` or `}`, and by nothing else. Checking that
    before calling the decoder is not a shortcut about carrier names, which is the kind of
    guess that grows a blind spot; it is the grammar, and it is exact. It earns its place
    because a failed `raw_decode` raises an error whose line and column are computed by
    scanning the document from the start, so brace-heavy text (a stack trace, a template, a
    shell heredoc) would otherwise cost one document scan per brace.
    """
    while True:
        position = text.find("{", index)
        if position < 0:
            return -1
        after = _skip_ws(text, position + 1)
        if after < len(text) and text[after] in '"}':
            return position
        index = position + 1


def _serialised_objects(text: str) -> list[tuple[Any, int, int]]:
    """Every serialised JSON object in `text`, as (value, raw start, raw end).

    Start positions are walked rather than guessed: at each candidate the decoder is asked to
    parse, and a candidate that begins nothing well-formed is stepped over one character at a
    time, because a broken outer wrapper can still hold a whole object inside it and skipping
    to where the outer parse died would skip that object with it. A parsed object's span is
    then passed over entire, so its nested objects are reached by the structural walk, which
    knows their parent, rather than rediscovered here as though they stood alone.

    Retrying at every candidate is quadratic on text engineered to look like the start of an
    object over and over without ever being one. That is the safe direction: the cost falls on
    a scrub that then refuses or completes, never on a publication that goes out unchecked.
    """
    found: list[tuple[Any, int, int]] = []
    index = 0
    while True:
        position = _object_start(text, index)
        if position < 0:
            return found
        try:
            value, end = _DECODER.raw_decode(text, position)
        except ValueError:
            index = position + 1
            continue
        found.append((value, position, end))
        index = end


# --- Scrubber-side carrier definitions ------------------------------------------------
# These name exactly what the repairing scrubber removes, and nothing else reads them: the
# verifier further down re-derives its own, so a gap in one definition cannot blind both.

# Blocks whose payload is provider ciphertext rather than research content.
# `redacted_thinking` is included because it carries its ciphertext under `data` rather
# than `signature`, so a scrubber that only knew the field name would pass it through.
THINKING_TYPES = ("thinking", "redacted_thinking")

# The streamed event that carries the same replayable ciphertext under `signature` without
# ever declaring itself a thinking block, so it has to be named on its own.
SIGNATURE_DELTA_TYPE = "signature_delta"

# The fields to take off a thinking block. Named per type would be tidier, but a trace is
# other people's JSON and block shapes drift between harness versions, so both names are
# stripped from both thinking types.
CIPHERTEXT_FIELDS = ("signature", "data")

# What makes a block still worth keeping once the ciphertext is off it.
TEXT_FIELDS = ("thinking", "text")

# Traces are JSONL, and their stderr siblings are not JSON at all. Both leave the machine
# together, so both are scrubbed; a gate that only knew about `*.stream.jsonl` would
# publish the stderr file beside it untouched.
TRACE_GLOBS = ("*.stream.jsonl", "*.err.txt")


def _scrubber_carrier(block_type: object, key: str, value: Any) -> bool:
    """The scrubber's rule: is this member the reasoning ciphertext of its own object?

    Exact type names only, because this side repairs and a repair made on a guess is a
    corruption. An already blank value is not a carrier, so re-running the gate over its own
    output rewrites nothing, and a non-string value is not ciphertext to begin with.
    """
    if not isinstance(value, str) or value == "":
        return False
    if block_type in THINKING_TYPES:
        return key in CIPHERTEXT_FIELDS
    if block_type == SIGNATURE_DELTA_TYPE:
        return key == "signature"
    return False


class _Dropped:
    """The scrubber's private "this block is gone" marker.

    Deliberately not `None`: `None` is also an ordinary JSON null, and a list rebuild that
    filtered every `None` deleted the trace's own nulls along with the emptied blocks, which
    is a silent edit to content the gate promised to leave alone. No JSON document can
    contain this object, so it cannot be confused with data.
    """

    __slots__ = ()


_DROPPED = _Dropped()


def _as_json(value: Any) -> Any:
    """A dropped block in a mapping slot becomes a null rather than vanishing.

    Removing the key would change the record's shape for anything that reads the object back
    by name; the ciphertext is gone either way, and a null says the block was there.
    """
    return None if value is _DROPPED else value


class ReasoningSignatureFound(Exception):
    """Publication was refused because reasoning ciphertext reached a publishable artifact.

    Carries the locations only. The whole point of the check is that this value never
    reaches a log, a terminal or an exception tracker, so the message counts findings and
    names where they sit, and never quotes what it found. The locations are structural
    positions and digested keys, never key text, because a key can itself be a credential.
    """

    def __init__(self, what: str, locations: list[str]) -> None:
        self.what = what
        self.locations = locations
        shown = ", ".join(locations[:5])
        more = f" and {len(locations) - 5} more" if len(locations) > 5 else ""
        super().__init__(
            f"{what}: refusing to publish, {len(locations)} reasoning signature "
            f"location(s) present ({shown}{more})"
        )


@dataclass
class ScrubReport:
    """Counts, never content. Safe to print, log, or fold into a manifest."""

    files: int = 0
    lines_rewritten: int = 0
    blocks_dropped: int = 0
    fields_removed: int = 0
    text_redactions: int = 0
    unparsed_lines: int = 0
    paths: list[str] = field(default_factory=list)

    def merge(self, other: ScrubReport) -> None:
        self.files += other.files
        self.lines_rewritten += other.lines_rewritten
        self.blocks_dropped += other.blocks_dropped
        self.fields_removed += other.fields_removed
        self.text_redactions += other.text_redactions
        self.unparsed_lines += other.unparsed_lines
        self.paths.extend(other.paths)

    def to_json(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "lines_rewritten": self.lines_rewritten,
            "blocks_dropped": self.blocks_dropped,
            "fields_removed": self.fields_removed,
            "text_redactions": self.text_redactions,
            "unparsed_lines": self.unparsed_lines,
            "paths": sorted(self.paths),
        }


def scrub_text(text: str, report: ScrubReport | None = None) -> str:
    """Blank the value of a reasoning carrier serialised into a free-text string.

    The value is replaced rather than the whole field deleted, because this runs on captured
    stderr where the surrounding characters are somebody's log line and cutting into it would
    corrupt the diagnostic the tail was captured for. Which values those are is decided by
    parsing: the serialised objects are located with the decoder and each is judged on its
    own `type`, so `{"signature":...,"type":"thinking"}` is caught as surely as the other key
    order, a `}` inside an earlier string value hides nothing, and a `signature` under a
    nested `metadata` object stays. Text that does not parse is not a serialised object, so
    the scrubber leaves it alone rather than editing bytes it cannot read; the verifier is
    the side that refuses in that case.
    """
    spans = [
        span
        for value, start, _end in _serialised_objects(text)
        for span in _carrier_spans(text, value, start, _scrubber_carrier)
    ]
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        pieces.append(text[cursor:start])
        pieces.append('""')  # the value goes; the field, and the log line around it, stay
        cursor = end
        if report is not None:
            report.text_redactions += 1
    pieces.append(text[cursor:])
    return "".join(pieces)


def scrub_value(value: Any, report: ScrubReport | None = None) -> Any:
    """Return `value` with reasoning ciphertext removed, recursively.

    The record shape is not hardcoded: any dict whose `type` names a thinking block is
    treated as one, wherever it sits. That matters because the same block appears at
    three different depths across the harnesses, and because `--include-partial-messages`
    puts a second copy inside streamed deltas.

    A block that empties out returns the private `_DROPPED` sentinel and its parent list
    leaves it out. Lists are rebuilt rather than mutated so the caller does not need to know
    where content arrays live in a given harness's schema, and the rebuild filters only that
    sentinel, so an ordinary JSON null in the trace survives untouched.
    """
    if isinstance(value, dict):
        block_type = value.get("type")
        if block_type in THINKING_TYPES:
            kept = {k: v for k, v in value.items() if k not in CIPHERTEXT_FIELDS}
            if report is not None:
                report.fields_removed += sum(1 for f in CIPHERTEXT_FIELDS if f in value)
            if not any(kept.get(f) for f in TEXT_FIELDS):
                if report is not None:
                    report.blocks_dropped += 1
                return _DROPPED
            return {k: _as_json(scrub_value(v, report)) for k, v in kept.items()}
        if block_type == SIGNATURE_DELTA_TYPE:
            # A streamed `signature_delta` carries the same replayable ciphertext under
            # `signature` without ever declaring itself a thinking block. Strip that one
            # key and keep the event marker; the strip is scoped to this type so a
            # `signature` on anything else is never mistaken for it.
            kept = {k: v for k, v in value.items() if not (k == "signature" and isinstance(v, str))}
            if report is not None and isinstance(value.get("signature"), str):
                report.fields_removed += 1
            return {k: _as_json(scrub_value(v, report)) for k, v in kept.items()}
        # Any other dict is not a reasoning block. Recurse into it, but do NOT strip a
        # generic `signature` or `data` key: those belong to tool results, observations and
        # diagnostics, and blanking them would corrupt the trace the gate promises to keep.
        return {k: _as_json(scrub_value(v, report)) for k, v in value.items()}
    if isinstance(value, list):
        scrubbed = [scrub_value(v, report) for v in value]
        return [v for v in scrubbed if v is not _DROPPED]
    if isinstance(value, str):
        return scrub_text(value, report)
    return value


# --- Verifier-side carrier definitions, derived independently -------------------------
# The gate does not import the scrubber's tuples or its rule, on purpose. If a carrier name
# drifts and only one side learns the new name, the other must still catch it: that split is
# the whole reason scrubbing and verifying are two functions. So the rules below are written
# from scratch and deliberately err broad, and the gate additionally refuses material the
# scrubber declined to touch at all. They stop short of treating a bare domain
# `data`/`signature` as ciphertext, since rejecting clean tool payloads would be its own
# contract breach; broad here means "recognise a reasoning type the scrubber's exact tuple
# would walk past, and doubt anything unparseable", not "flag every field that shares a name".


def _is_reasoning_block(block_type: object) -> bool:
    """True for any block whose type reads as reasoning ciphertext.

    Independent of `THINKING_TYPES` on purpose, and deliberately broader: any type that
    mentions `thinking` (so a drifted `interleaved_thinking` is caught, not walked past)
    plus the streamed `signature_delta`. The literals are repeated here rather than
    imported so that narrowing the scrubber's tuple cannot quietly narrow the gate.
    """
    if not isinstance(block_type, str):
        return False
    return "thinking" in block_type or block_type == "signature_delta"


def _verifier_carrier(block_type: object, key: str, value: Any) -> bool:
    """The gate's rule: does this member read as ciphertext on the object that holds it?

    Broader than the scrubber's rule (any type that mentions thinking, both field names,
    whatever the scrubber's tuple happens to say today) and still decided per object, because
    flagging a `signature` that belongs to a nested `metadata` object would refuse a clean
    trace and teach an operator to route around the gate.
    """
    return (
        _is_reasoning_block(block_type)
        and key in ("signature", "data")
        and isinstance(value, str)
        and value != ""
    )


# The residue sweep. Whatever the decoder could not consume is text the scrubber declined to
# rewrite: a truncated request body, a line where two writers interleaved. Declining is right
# for a repair and wrong for a verdict, so the gate sweeps exactly that residue with a loose
# pair of needles that match in either order, and refuses when both are present. The sweep is
# confined to the residue: well-formed material is judged only by the structural rule above,
# which is what keeps a nested domain `signature` from being called ciphertext.
#
# Neither needle requires its value to be terminated, because the residue is mostly text that
# was cut: an stderr tail is a bounded tail, so the likeliest way a carrier ends up here is
# with the ciphertext itself running off the end of the capture. An empty value still fails
# the field needle, so the gate's own output does not trip it.
_SUSPECT_TYPE = re.compile(r'"type"\s*:\s*"[^"]*(?:thinking|signature_delta)')
_SUSPECT_FIELD = re.compile(r'"(?:signature|data)"\s*:\s*"[^"]')


def _residue(text: str, consumed: list[tuple[Any, int, int]]) -> list[str]:
    """The stretches of `text` that no serialised object accounted for."""
    segments: list[str] = []
    cursor = 0
    for _value, start, end in consumed:
        segments.append(text[cursor:start])
        cursor = end
    segments.append(text[cursor:])
    return segments


def _has_embedded_reasoning(text: str) -> bool:
    """Does this free text carry, or look like it carries, a serialised reasoning carrier?"""
    objects = _serialised_objects(text)
    for value, start, _end in objects:
        if _carrier_spans(text, value, start, _verifier_carrier):
            return True
    return any(
        _SUSPECT_TYPE.search(part) and _SUSPECT_FIELD.search(part)
        for part in _residue(text, objects)
    )


def safe_location_key(key: object) -> str:
    """A location component for a mapping key that cannot leak what the key says.

    A trace's keys are other people's data. A harness that keys a map by request URL puts
    `?token=...` into a key, and one that keys by API key puts the credential itself there,
    so echoing key text into an exception or a log would publish exactly the class of value
    this module exists to keep out of reports. A location therefore carries the key's
    position in its object plus a digest of its text. The digest is stable across runs and
    reproducible on purpose: an operator holding a candidate key can hash it the same way and
    match it against the location without the gate ever having written the key down.
    """
    raw = key if isinstance(key, str) else repr(key)
    digest = hashlib.blake2s(raw.encode("utf-8", "surrogateescape"), digest_size=4).hexdigest()
    return f"#{digest}"


def find_reasoning_material(value: Any, path: str = "$") -> list[str]:
    """Locate every place ciphertext survives, as paths. Never returns the values.

    This is the gate's evidence, so it is written independently of :func:`scrub_value`
    rather than by calling it and diffing, and it uses the verifier-side rules above rather
    than the scrubber's. A verifier that shares the scrubber's notion of where to look
    inherits the scrubber's blind spots and will certify them.

    The paths are made of things this module chose: list indices, member positions, digested
    keys, and the two field names named right here. Nothing read out of the trace is echoed.
    """
    found: list[str] = []
    if isinstance(value, dict):
        if _is_reasoning_block(value.get("type")):
            for f in ("signature", "data"):
                if isinstance(value.get(f), str) and value[f]:
                    found.append(f"{path}.{f}")
        for position, (key, item) in enumerate(value.items()):
            step = f"{path}.{position}{safe_location_key(key)}"
            found.extend(find_reasoning_material(item, step))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            found.extend(find_reasoning_material(item, f"{path}[{i}]"))
    elif isinstance(value, str) and _has_embedded_reasoning(value):
        found.append(f"{path}<embedded>")
    return sorted(set(found))


def assert_publishable(value: Any, what: str) -> None:
    """Raise unless `value` is free of reasoning ciphertext.

    Called on the way out of the process, not on the way in. Refusing to write is the
    correct failure: a published signature cannot be unpublished, whereas a blocked
    artifact costs one rerun of a scrub.
    """
    locations = find_reasoning_material(value)
    if locations:
        raise ReasoningSignatureFound(what, locations)


# --- Byte-level file handling ---------------------------------------------------------


def _split_lines(body: str) -> list[tuple[str, str]]:
    """Every line as (content, terminator), the terminator being `\\r\\n`, `\\n` or `""` at EOF.

    Deliberately not `str.splitlines()`, and deliberately not `split("\\n")`. `splitlines()`
    also breaks on `\\r`, `\\v`, `\\f`, `\\x1c` and `U+2028`, so a trace line that happens to
    contain one would come back as two lines and the runner's bounded tail re-read would see a
    record nobody wrote. `split("\\n")` loses which terminator a line had, so rejoining turns a
    CRLF trace into an LF one and rewrites bytes on lines the scrub never touched. JSONL is
    defined by `\\n`, so only `\\n` ends a line here, its optional `\\r` travels with it, and
    each line is put back exactly as it arrived.
    """
    lines: list[tuple[str, str]] = []
    cursor = 0
    while cursor < len(body):
        at = body.find("\n", cursor)
        if at < 0:
            lines.append((body[cursor:], ""))
            break
        if at > cursor and body[at - 1] == "\r":
            lines.append((body[cursor : at - 1], "\r\n"))
        else:
            lines.append((body[cursor:at], "\n"))
        cursor = at + 1
    return lines


def scrub_jsonl_text(body: str, report: ScrubReport) -> str:
    """Scrub a JSONL body, changing only carrier bytes and never a line's count or order.

    Every non-empty line is parsed rather than prefiltered on a raw `"signature"` needle,
    because a carrier can ride inside an outer JSON string: a serialised thinking block
    dumped into a result's stderr tail has its quotes escaped, so no bare needle survives to
    catch it, and a needle-gated prefilter would pass it through unscrubbed for the verifier
    to reject. A parsed line is rewritten only when it actually carries a reasoning secret,
    and the rewrite is kept byte-attributable. A clean line is re-emitted as its exact input
    bytes, terminator included. A dirty line is reserialised only when an unmodified parse
    already reserialises to the input verbatim, so the one byte difference is the removed
    carrier; otherwise (a line whose formatting is not the canonical compact form) the carrier
    value is blanked in the raw bytes in place, and anything the surgical pass cannot reach
    (a carrier escaped inside a string of a non-canonical line, a truncated one) is left for
    the verifier to refuse rather than rewritten blind. Lines that are not JSON are captured
    stdout or stderr and are text-scrubbed, never dropped, since the runner's reader tolerates
    them and dropping one would slide the tail it reads back.
    """
    if body == "":
        # An empty trace is a clean zero-byte trace; a lone newline would be a byte change.
        return ""
    out: list[str] = []
    for line, terminator in _split_lines(body):
        out.append(_scrub_jsonl_line(line, report))
        out.append(terminator)
    return "".join(out)


def _scrub_jsonl_line(line: str, report: ScrubReport) -> str:
    """One JSONL line's content, without its terminator, which is never ours to change."""
    if line.strip() == "":
        return line
    try:
        parsed = json.loads(line)
    except (ValueError, TypeError):
        # Not JSON, so treat it as the free text it is rather than dropping it.
        redacted = scrub_text(line, report)
        if redacted != line:
            report.lines_rewritten += 1
        report.unparsed_lines += 1
        return redacted
    scrubbed = _as_json(scrub_value(parsed, report))
    if scrubbed == parsed:
        return line  # nothing to remove: the input bytes are the output bytes
    report.lines_rewritten += 1
    canonical = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    if canonical == line:
        return json.dumps(scrubbed, ensure_ascii=False, separators=(",", ":"))
    # Reserialising a non-canonical line would rewrite bytes we promised to keep, so blank the
    # carrier value in the raw line and let the verifier refuse if a carrier is beyond a
    # surgical reach. No report here: the redaction was already counted on the parsed pass.
    return scrub_text(line)


def _read(path: Path) -> str:
    """A file's bytes as text, losslessly.

    `read_text` would translate newlines and, with `errors="replace"`, silently swap a
    replacement character in for every byte that is not valid UTF-8, so a rewrite anywhere in
    the file would rewrite those bytes too. Traces are captured stdout from other people's
    processes and do contain both, so decoding with `surrogateescape` keeps the undecodable
    bytes intact and reversible, and reading in binary keeps `\\r\\n` as two characters that
    belong to the line rather than to the reader.
    """
    return path.read_bytes().decode("utf-8", errors="surrogateescape")


def scrub_file(path: Path) -> ScrubReport:
    """Scrub one trace file in place. JSONL is parsed; anything else is treated as text.

    The file is written only when something actually changed, and it is written back through
    the same lossless encoding, so every line the scrub did not touch leaves byte-identical,
    its terminator and any non-UTF-8 bytes included.
    """
    report = ScrubReport(files=1, paths=[str(path)])
    body = _read(path)
    if path.name.endswith(".jsonl"):
        scrubbed = scrub_jsonl_text(body, report)
    else:
        scrubbed = scrub_text(body, report)
        if scrubbed != body:
            report.lines_rewritten += 1
    if scrubbed != body:
        path.write_bytes(scrubbed.encode("utf-8", errors="surrogateescape"))
    return report


def iter_traces(root: Path) -> list[Path]:
    """Every trace-shaped file under `root`, in a stable order.

    Recursive rather than keyed to `<phase>/traces/`, so this works on a run directory, on
    a single phase, or on a staging directory somebody assembled by hand for an export.
    """
    found: list[Path] = []
    for pattern in TRACE_GLOBS:
        found.extend(root.rglob(pattern))
    return sorted(set(found))


def scrub_run_dir(root: Path) -> ScrubReport:
    total = ScrubReport()
    for path in iter_traces(root):
        total.merge(scrub_file(path))
    return total


def verify_run_dir(root: Path) -> dict[str, list[str]]:
    """Where ciphertext survives under `root`, per file. Empty means publishable."""
    findings: dict[str, list[str]] = {}
    for path in iter_traces(root):
        locations: list[str] = []
        for number, (line, _terminator) in enumerate(_split_lines(_read(path)), start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except (ValueError, TypeError):
                if _has_embedded_reasoning(line):
                    locations.append(f"line {number}<embedded>")
                continue
            locations.extend(f"line {number} {loc}" for loc in find_reasoning_material(parsed))
        if locations:
            findings[str(path)] = locations
    return findings


def main(argv: list[str] | None = None) -> int:
    """The body of `shobench scrub-traces`, also runnable as `python -m shobench.scrub`."""
    parser = argparse.ArgumentParser(
        prog="shobench scrub-traces",
        description="Strip reasoning signatures out of a run's traces before they are published",
    )
    parser.add_argument("directory", type=Path, help="a run directory, or any tree of traces")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what is present and change nothing; non-zero exit if anything is",
    )
    args = parser.parse_args(argv)

    root: Path = args.directory
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    if args.check:
        findings = verify_run_dir(root)
        counts = {k: len(v) for k, v in findings.items()}
        print(json.dumps({"files_with_findings": counts}, indent=2))
        if findings:
            print(
                f"{len(findings)} file(s) still carry reasoning signatures; not publishable",
                file=sys.stderr,
            )
            return 1
        return 0

    report = scrub_run_dir(root)
    # Re-derive the verdict from the files on disk rather than trusting the scrub's own
    # bookkeeping, so a scrubber blind spot surfaces here instead of at publication.
    findings = verify_run_dir(root)
    print(json.dumps(report.to_json(), indent=2))
    if findings:
        print(
            f"scrub incomplete: {len(findings)} file(s) still carry reasoning signatures",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
