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

**The boundary this module is built on: structured content, and free text.** A trace is
JSONL, so a line that parses is structure, and everything the parser hands back (its
objects, their members, their descendants) is judged precisely, per object, by name.
Every string inside that structure, plus a whole `.err.txt` body, is free text, and free
text gets the opposite treatment. Locating a carrier inside arbitrary text is a game the
carrier wins: it can be wrapped in another string, escaped one more time, spelled
`"t\\u0079pe"`, split across two lines by whichever writer interleaved with it, cut in
half by a bounded tail, or hidden behind a duplicate member. Each of those is a
different spelling of the same secret, and matching more spellings only moves the next
one out of reach. So free text is not searched for a carrier at all. It is checked for
any *sign* of reasoning material, cheaply and broadly, and a region that shows one is
replaced whole. Stderr and diagnostics are not the research data; the trace's structured
content is. Losing a diagnostic line is cheap, and leaking a replayable signature is not.

**What is removed from structured content, and only what is removed.** The `signature`
off every thinking block, the ciphertext `data` off every `redacted_thinking` block, the
`signature` off a streamed `signature_delta`, and any block that carries nothing else
once that carrier is gone. A field is a carrier only when the object that holds it says
so and the value is a non-empty string, so an already blank `signature` is left exactly
as found, a null one is not deleted, and a block that still holds anything of its own
survives with its other members intact. Nothing else is touched: a `signature` field on
a tool result, a `data` value in an observation, a `signature` under some block's
`metadata`, and every ordinary JSON null are domain content and come through verbatim.
Where an object declares its `type` more than once, the parser keeps one and the raw
bytes keep both, so every declared type is applied and the object is treated as a
carrier if any of them says it is.

**What free text gets instead.** A string value that shows a sign of reasoning material
is replaced entirely by :data:`FREE_TEXT_MARKER`, prose and all. A suspicious `.err.txt`
region loses every line that holds a ciphertext-shaped value, marker for content and the
line's own terminator kept, so line count and line order do not move. There is no
precision claim on this side and no promise that the surrounding diagnostic survives.
What there is instead is a guarantee that does not depend on finding the carrier's exact
span: escapes are decoded before matching, matching runs across line boundaries, order
does not matter, truncation is tolerated, and the whole region goes.

**The price of that, said plainly.** Every string is free text, including a thinking
block's own reasoning text, so an agent that quotes a `"signature": "..."` field while
reasoning about an API loses that block's text to the marker. The alternative was to
exempt strings by the name of the field holding them, which is a bypass anyone can use
by putting a carrier in a member called `text`. The replacement is counted in
`text_redactions` rather than done quietly, so the collateral is visible in the report
and an operator can see it rise. If it ever rises far enough to matter, the fix is to
narrow what counts as a ciphertext-shaped value, not to reopen the search for a carrier's
exact span in arbitrary text.

**What is preserved, and why it is not negotiable.** Byte-for-byte the whole trace
except the carrier bytes and the suspicious free-text regions, and above all line count
and line order. The runner reads a finished trace back to recover the session id, the
observed models and the stop classification, and it does so from a bounded tail of the
file. Dropping lines would slide that window and change what a re-read concludes, so a
record whose content empties out is written back with an empty content list rather than
deleted, and a clean line comes through as its exact input bytes, terminator included,
whatever its encoding.

**The verifier is independent, broader, and reads the same shapes.** It re-derives what
a carrier looks like from its own rules rather than importing the scrubber's, it errs
toward flagging more than the scrubber removes, and it raises rather than repairs. It
also splits a file into exactly the same chunks the scrubber worked on, because a
verifier that reads an `.err.txt` line by line while the scrubber read it as one document
cannot see a carrier that spans two lines: the shape has to agree or the two sides are
checking different artifacts.

**The contract, stated once.** Structured content: precise, byte-faithful, named
carriers only. Free text: fail-closed, whole-region replacement on suspicion, no
precision promise. Verifier: independent, broader, same body shape as the scrubber,
refuses ambiguity. No signature value, credential value, untrusted key, or
caller-supplied label text reaches any report, exception or log: those carry counts,
structural positions, and digests an operator can recompute from a candidate they
already hold.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Shared mechanism, part one: where structure begins and ends ----------------------
# Structured content is judged per object, and a carrier has to be edited in the original
# bytes rather than in a reserialisation of them, so both sides need to know where each
# member's raw value sits. The language's own parser answers that, and it holds no rule of
# ours to get wrong. The carrier RULES stay split between the scrubber and the verifier
# further down, which is the split that stops one blind spot blinding both.

_DECODER = json.JSONDecoder()
_WHITESPACE = " \t\n\r"

# Given an enclosing object's declared types plus one of its members, is that member
# ciphertext? Both sides answer this question, and they answer it differently on purpose.
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
    boundary. Members are returned in the order they were written and duplicates are kept,
    because `json.loads` collapses a repeated key to whichever copy came last and that
    collapse is exactly where a carrier can hide. `None` means the text does not hold a
    well-formed object here, a fact the caller must not paper over: for the scrubber it
    means there is nothing safe to repair, and the verifier's free-text rules take over.
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


def _member_spans(
    text: str,
    value: Any,
    start: int,
    end: int,
    is_carrier: CarrierRule,
    suspects_text: Callable[[str], bool] | None = None,
) -> list[tuple[int, int, bool]]:
    """Raw spans worth editing inside the JSON value parsed at `text[start]`.

    Each span comes back as (raw start, raw end, is a named carrier). A carrier span is a
    member the caller's rule claimed; a non-carrier span is a string value the caller's
    free-text predicate found suspicious, and the two are tagged apart because they are
    replaced by different things.

    An object is judged on ALL the types it declares, not on the one a parse kept. Two
    `type` members are undefined JSON and consumers disagree about which one wins, so
    reading only the survivor lets `{"type":"thinking","signature":...,"type":"text"}`
    present as prose to anything that parses it while still shipping the ciphertext in the
    bytes. Applying every declared type is the fail-closed reading of an ambiguity.

    The walk then descends into members and elements, so a `signature` sitting under some
    block's `metadata` belongs to `metadata` and not to the reasoning block above it,
    however adjacent the two look once serialised.
    """
    spans: list[tuple[int, int, bool]] = []
    if isinstance(value, dict):
        members = _members(text, start)
        if members is None:
            return spans
        declared = [member for key, member, _s, _e in members if key == "type"]
        for key, member, member_start, member_end in members:
            if any(is_carrier(block_type, key, member) for block_type in declared):
                spans.append((member_start, member_end, True))
            else:
                spans.extend(
                    _member_spans(text, member, member_start, member_end, is_carrier, suspects_text)
                )
    elif isinstance(value, list):
        elements = _elements(text, start)
        if elements is None:
            return spans
        for element, element_start, element_end in elements:
            spans.extend(
                _member_spans(text, element, element_start, element_end, is_carrier, suspects_text)
            )
    elif isinstance(value, str) and suspects_text is not None and suspects_text(value):
        spans.append((start, end, False))
    return spans


# --- Shared mechanism, part two: reading free text through its escapes ----------------
# A carrier serialised into somebody's log line arrives escaped, sometimes more than once:
# a request body dumped into a stderr tail, then that tail dumped into a JSON field, then
# that field quoted into another. Every layer respells the same bytes, so matching the raw
# spelling only catches the layer count somebody happened to use. These helpers undo the
# layers one at a time and remember, for every decoded character, which raw offset it came
# from, so a decision taken on the readable text maps back to the bytes on disk.
#
# This is mechanism, not policy: what counts as a sign of reasoning material is defined
# twice further down, once by each side.

_ESCAPES = {
    '"': '"',
    "'": "'",
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}

# One layer of decoding can only invent a needle out of an escape sequence, and an escape
# sequence needs a backslash. So text with no backslash and none of the plain words can
# hold no sign at any depth, and skipping it is exact rather than a heuristic. It earns its
# place because this runs over every string in a trace, and traces are large.
_MARK_HINT = re.compile(r"signature|data|thinking|\\")

_MAX_DECODE_LAYERS = 4


def _unescape_once(text: str) -> tuple[str, list[int], list[int]] | None:
    """Undo one layer of string escaping, with a sparse map from decoded back to raw offsets.

    The map is two parallel lists holding the points where decoded and raw offsets stop
    moving together, which is one entry per escape rather than one per character: a trace's
    stderr tail can be megabytes, and a per-character map of it costs more memory than the
    file. Between two of those points the offsets differ by a constant, which is what
    :func:`_to_raw` uses to answer for any offset in between, the end of the text included.

    An escape this does not recognise keeps its backslash and is copied through, because
    dropping it could join two characters that were never adjacent. `None` means there was
    nothing to undo.
    """
    if "\\" not in text:
        return None
    out: list[str] = []
    decoded_marks = [0]
    raw_marks = [0]
    index = 0
    size = len(text)
    while index < size:
        char = text[index]
        width = 0
        if char == "\\" and index + 1 < size:
            nxt = text[index + 1]
            if nxt in _ESCAPES:
                out.append(_ESCAPES[nxt])
                width = 2
            elif nxt == "u" and index + 6 <= size:
                digits = text[index + 2 : index + 6]
                if len(digits) == 4 and all(c in "0123456789abcdefABCDEF" for c in digits):
                    out.append(chr(int(digits, 16)))
                    width = 6
            elif nxt == "x" and index + 4 <= size:
                digits = text[index + 2 : index + 4]
                if len(digits) == 2 and all(c in "0123456789abcdefABCDEF" for c in digits):
                    out.append(chr(int(digits, 16)))
                    width = 4
        if width:
            index += width
            decoded_marks.append(len(out))
            raw_marks.append(index)
            continue
        out.append(char)
        index += 1
    decoded = "".join(out)
    if decoded == text:
        return None
    return decoded, decoded_marks, raw_marks


def _to_raw(maps: list[tuple[list[int], list[int]]], index: int) -> int:
    """Walk one offset back through every decoding layer to the offset in the raw bytes."""
    for decoded_marks, raw_marks in reversed(maps):
        at = bisect.bisect_right(decoded_marks, index) - 1
        index = raw_marks[at] + (index - decoded_marks[at])
    return index


def _decoded_views(text: str) -> list[tuple[str, list[tuple[list[int], list[int]]]]]:
    """`text` itself, then each further layer of escaping undone, each with a map back.

    Every layer is searched, not just the deepest one, because decoding is only additive
    for a needle spelled plainly at this level and a producer may have escaped one field
    and not another. The depth is bounded so a pathological run of backslashes cannot turn
    a scrub into a hang; four layers is already one more than any real trace has shown, and
    text past that depth is unreadable enough that the verifier's breadth is the backstop.
    """
    views: list[tuple[str, list[tuple[list[int], list[int]]]]] = [(text, [])]
    current, maps = views[0]
    for _ in range(_MAX_DECODE_LAYERS):
        step = _unescape_once(current)
        if step is None:
            break
        decoded, decoded_marks, raw_marks = step
        maps = [*maps, (decoded_marks, raw_marks)]
        views.append((decoded, maps))
        current = decoded
    return views


def _quoted_value_end(text: str, index: int, quote: str) -> int:
    """Where the string value opened at `index` ends, or the end of `text` if it never does.

    Running off the end is the normal case rather than an error: an stderr tail is a bounded
    tail, so the likeliest way a carrier lands in free text is with its value still running
    when the capture stopped.
    """
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == quote:
            return index + 1
        index += 1
    return len(text)


def _suspect_spans(
    text: str, reasoning: re.Pattern[str], cipher: re.Pattern[str]
) -> list[tuple[int, int]]:
    """Raw spans of every ciphertext-shaped value in `text`, if `text` shows reasoning at all.

    Two needles, either order, any distance apart, at any escaping depth: one says this text
    is about reasoning, the other says here is a field whose value is shaped like ciphertext.
    Distance is deliberately not part of the test. A proximity window would be a published
    bypass parameter, since padding the gap between the two is free for whoever writes the
    text. Breadth is affordable in the other direction because a span is a whole region that
    gets replaced, not a surgical cut that has to be right.

    The spans returned cover each ciphertext-shaped field and the whole of its value, so a
    value that was wrapped across lines or cut by the capture is covered to the end either
    way. The caller widens each span to whole lines or to a whole string value.
    """
    if not _MARK_HINT.search(text):
        return []
    spans: list[tuple[int, int]] = []
    for view, maps in _decoded_views(text):
        if not reasoning.search(view):
            continue
        for match in cipher.finditer(view):
            end = _quoted_value_end(view, match.end(), match.group("vq"))
            spans.append((_to_raw(maps, match.start()), _to_raw(maps, end)))
    return spans


def _truncation_tolerant(*words: str, shortest: int = 3) -> str:
    """A pattern for `words`, plus any prefix of one of them sitting at the end of the text.

    A bounded tail can cut a word in half, and the half that survives can be the half that
    comes before the carrier: `{"signature":"...","type":"think` still ships the ciphertext
    while spelling the type only partly. So a prefix counts, but only where the text stops,
    which is the only place truncation can be. Trailing whitespace still counts as the end,
    because a capture that was cut mid-word is usually written out with a newline after it.
    """
    prefixes = [
        word[:length] for word in words for length in range(len(word) - 1, shortest - 1, -1)
    ]
    return "|".join(words) + "|(?:" + "|".join(prefixes) + r")\s*\Z"


# --- Scrubber-side definitions: what the repairing side removes -----------------------
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

# What a replaced free-text region says. Deliberately free of every word either side treats
# as a needle, so the gate's own output is not suspicious to itself and a second scrub of a
# scrubbed file changes nothing.
FREE_TEXT_MARKER = "[shobench: suspected replayable carrier removed]"


def _scrubber_carrier(block_type: object, key: str, value: Any) -> bool:
    """The scrubber's rule: is this member the reasoning ciphertext of its own object?

    Exact type names only, because this side repairs and a repair made on a guess is a
    corruption. An already blank value is not a carrier, so re-running the gate over its own
    output rewrites nothing, and a non-string value is not ciphertext to begin with. This
    predicate is the single definition of a carrier value on the scrubbing side: the
    structural path and the raw byte path both ask it, so neither can delete a field the
    other would have called clean.
    """
    if not isinstance(value, str) or value == "":
        return False
    if block_type in THINKING_TYPES:
        return key in CIPHERTEXT_FIELDS
    if block_type == SIGNATURE_DELTA_TYPE:
        return key == "signature"
    return False


# The scrubber's free-text needles. `thinking` and `signature_delta` are looked for as bare
# words rather than as a `"type"` member, because in free text the shape around a word is
# exactly what an extra escaping layer, a line break or a duplicate member can change,
# while the word itself is what a reasoning carrier cannot travel without.
#
# The field needle is written for how a carrier is actually written down rather than for
# JSON alone: the same object reaches a log as a Python repr with single quotes, as a
# Node console dump with the key unquoted, and as `signature=` in a query string, and each
# of those spellings carries the identical value. It does require a non-empty quoted value,
# because a field this module already blanked must not be a needle and a value with no
# delimiters has no end to find.
_SCRUB_REASONING = re.compile(_truncation_tolerant("thinking", "signature_delta"))
_SCRUB_CIPHER = re.compile(
    r"""["']?(?:signature|data)["']?\s*[:=]\s*(?P<vq>["'])(?!(?P=vq))(?=.)""",
    re.DOTALL,
)


def _scrubber_suspect_spans(text: str) -> list[tuple[int, int]]:
    return _suspect_spans(text, _SCRUB_REASONING, _SCRUB_CIPHER)


def _scrubber_suspects(text: str) -> bool:
    return bool(_scrubber_suspect_spans(text))


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
    positions and digests, never key text. The caller's own label is digested too and the
    text of it is not kept on the exception at all: a label is usually a path, a path can
    be named after the credential a task was handed, and an exception is the most widely
    copied string in any system.
    """

    def __init__(self, what: str, locations: list[str]) -> None:
        self.label = safe_label(what)
        self.locations = locations
        shown = ", ".join(locations[:5])
        more = f" and {len(locations) - 5} more" if len(locations) > 5 else ""
        super().__init__(
            f"{self.label}: refusing to publish, {len(locations)} reasoning signature "
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
    # Digests, not paths. A trace's file name is caller-supplied and comes off a task the
    # operator did not write, so it can carry the very kind of value this module exists to
    # keep out of reports.
    file_ids: list[str] = field(default_factory=list)

    def merge(self, other: ScrubReport) -> None:
        self.files += other.files
        self.lines_rewritten += other.lines_rewritten
        self.blocks_dropped += other.blocks_dropped
        self.fields_removed += other.fields_removed
        self.text_redactions += other.text_redactions
        self.unparsed_lines += other.unparsed_lines
        self.file_ids.extend(other.file_ids)

    def to_json(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "lines_rewritten": self.lines_rewritten,
            "blocks_dropped": self.blocks_dropped,
            "fields_removed": self.fields_removed,
            "text_redactions": self.text_redactions,
            "unparsed_lines": self.unparsed_lines,
            "file_ids": sorted(self.file_ids),
        }


def scrub_text(text: str, report: ScrubReport | None = None) -> str:
    """Free text that shows any sign of reasoning material is replaced whole.

    The surrounding prose goes with it, on purpose. Every earlier attempt to keep the prose
    and cut out only the carrier turned into a search for the carrier's exact span inside
    arbitrary text, and that search loses to the next spelling: another escaping layer, a
    line break in the middle, a wrapper string, a duplicate member, a truncated tail. The
    replacement does not need to know where the carrier is, only that this text is about
    reasoning and holds a field shaped like ciphertext, so no new spelling changes the
    outcome. What is given up is the diagnostic, which is not the research data. What is
    bought is that the carrier cannot survive by being spelled differently.
    """
    if not _scrubber_suspects(text):
        return text
    if report is not None:
        report.text_redactions += 1
    return FREE_TEXT_MARKER


def _scrub_key(key: Any) -> Any:
    """Keys are strings too, and a string is free text wherever it sits.

    A harness that keys a map by a serialised request body puts the whole object into the
    key position, where a walk that only ever descends into values never looks at it.
    """
    return scrub_text(key) if isinstance(key, str) else key


def _carries_only_the_carrier(kept: dict[str, Any]) -> bool:
    """Is there nothing left in this block worth keeping once the carrier is gone?

    A block is dropped only when it holds nothing of its own: its `type`, and text fields
    that are empty. Any other member is content the gate never promised to delete, so a
    thinking block with an empty `thinking`, a stripped `signature` and a populated
    `metadata` keeps its metadata rather than being deleted along with the ciphertext.
    """
    for key, value in kept.items():
        if key == "type":
            continue
        if key in TEXT_FIELDS:
            if value:
                return False
            continue
        return False
    return True


def scrub_value(value: Any, report: ScrubReport | None = None) -> Any:
    """Return `value` with reasoning ciphertext removed, recursively.

    The record shape is not hardcoded: any dict whose `type` names a thinking block is
    treated as one, wherever it sits. That matters because the same block appears at
    three different depths across the harnesses, and because `--include-partial-messages`
    puts a second copy inside streamed deltas.

    What leaves a block is decided by :func:`_scrubber_carrier` and by nothing else, so a
    blank `signature`, a null one and an object-valued one are all left alone: they are not
    ciphertext, and deleting a field the module's own rule calls clean is an edit outside
    the contract. A block that empties out returns the private `_DROPPED` sentinel and its
    parent list leaves it out, but only when the carrier was really there and nothing else
    was. Lists are rebuilt rather than mutated so the caller does not need to know where
    content arrays live in a given harness's schema, and the rebuild filters only that
    sentinel, so an ordinary JSON null in the trace survives untouched.

    Every descendant is visited: members of a block that was kept, elements of a list,
    strings in value position and strings in key position.
    """
    if isinstance(value, dict):
        block_type = value.get("type")
        if block_type in THINKING_TYPES or block_type == SIGNATURE_DELTA_TYPE:
            carriers = [k for k, v in value.items() if _scrubber_carrier(block_type, k, v)]
            kept = {k: v for k, v in value.items() if k not in carriers}
            if report is not None:
                report.fields_removed += len(carriers)
            # A streamed `signature_delta` is never dropped, only stripped: the event itself
            # is the trace's record that a delta arrived at that point in the stream, and a
            # re-read reconstructs the stream from the events it finds. A thinking block
            # that gave up its carrier and holds nothing else is a different thing, an
            # envelope whose only content was ciphertext, and it goes.
            if block_type in THINKING_TYPES and carriers and _carries_only_the_carrier(kept):
                if report is not None:
                    report.blocks_dropped += 1
                return _DROPPED
            return {_scrub_key(k): _as_json(scrub_value(v, report)) for k, v in kept.items()}
        # Any other dict is not a reasoning block. Recurse into it, but do NOT strip a
        # generic `signature` or `data` key: those belong to tool results, observations and
        # diagnostics, and blanking them would corrupt the trace the gate promises to keep.
        return {_scrub_key(k): _as_json(scrub_value(v, report)) for k, v in value.items()}
    if isinstance(value, list):
        scrubbed = [scrub_value(v, report) for v in value]
        return [v for v in scrubbed if v is not _DROPPED]
    if isinstance(value, str):
        return scrub_text(value, report)
    return value


# --- Verifier-side definitions, derived independently ---------------------------------
# The gate does not import the scrubber's tuples, its rule or its needles, on purpose. If a
# carrier name drifts and only one side learns the new name, the other must still catch it:
# that split is the whole reason scrubbing and verifying are two functions. So the rules
# below are written from scratch and deliberately err broad, and the gate additionally
# refuses material the scrubber declined to touch at all. They stop short of treating a
# bare domain `data`/`signature` as ciphertext, since rejecting clean tool payloads would be
# its own contract breach; broad here means "recognise a reasoning type the scrubber's exact
# tuple would walk past, and doubt anything ambiguous", not "flag every field that shares a
# name". Where the two sides disagree the run is refused rather than published, which costs
# a rerun and a look from a human, and that is the direction the disagreement should fall.


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


# The gate's free-text needles, written separately from the scrubber's and wider on both
# axes: `reasoning` counts as a word, and a field whose name merely contains `signature` or
# `data` counts as ciphertext-shaped, so `thinking_signature` and `redacted_data` are
# refused even though the scrubber would not have recognised either. Wider here means the
# gate can refuse text the scrubber left, which is the fail-closed half of the split.
_VERIFY_REASONING = re.compile(_truncation_tolerant("thinking", "signature_delta", "reasoning"))
_VERIFY_CIPHER = re.compile(
    r"""["']?[\w.-]*(?:signature|data)[\w.-]*["']?\s*[:=]\s*(?P<vq>["'])(?!(?P=vq))(?=.)""",
    re.DOTALL,
)


def _verifier_suspect_spans(text: str) -> list[tuple[int, int]]:
    return _suspect_spans(text, _VERIFY_REASONING, _VERIFY_CIPHER)


def _verifier_suspects(text: str) -> bool:
    return bool(_verifier_suspect_spans(text))


def safe_label(text: str) -> str:
    """An opaque, reproducible stand-in for a string this module did not choose.

    Paths, artifact names and trace file names are all supplied by whoever launched the run
    or by the task itself, so any of them can quote a credential. A report therefore names
    them by digest. The digest is stable across runs and reproducible on purpose: an
    operator holding a candidate string can hash it the same way and match it against the
    report without the gate ever having written the string down.
    """
    digest = hashlib.blake2s(text.encode("utf-8", "surrogateescape"), digest_size=4).hexdigest()
    return f"#{digest}"


def safe_location_key(key: object) -> str:
    """A location component for a mapping key that cannot leak what the key says.

    A trace's keys are other people's data. A harness that keys a map by request URL puts
    `?token=...` into a key, and one that keys by API key puts the credential itself there,
    so echoing key text into an exception or a log would publish exactly the class of value
    this module exists to keep out of reports. A location therefore carries the key's
    position in its object plus a digest of its text.
    """
    return safe_label(key if isinstance(key, str) else repr(key))


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
            if isinstance(key, str) and _verifier_suspects(key):
                found.append(f"{step}<key>")
            found.extend(find_reasoning_material(item, step))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            found.extend(find_reasoning_material(item, f"{path}[{i}]"))
    elif isinstance(value, str) and _verifier_suspects(value):
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


class _NotJson:
    """ "This line was never a JSON record", which `None` cannot say: `null` is a record."""

    __slots__ = ()


_NOT_JSON = _NotJson()


@dataclass(frozen=True)
class _Chunk:
    """One piece of a file, shaped the same way for whoever is reading it.

    A record chunk is a single line that parsed, and it is judged as structure. A text
    chunk is a run of consecutive lines that did not parse, and it is judged as one free
    text document, terminators included, because two writers interleaving on one stream put
    half a carrier on one line and half on the next. A record breaks the run: a JSON record
    is not part of somebody's stderr dump.

    Both sides call :func:`_chunks`, and that is the point. The scrubber reading an
    `.err.txt` as one document while the verifier read it line by line is not a difference
    of opinion about carriers, it is the two of them checking different artifacts, and a
    carrier that spans a line boundary lives in the gap.
    """

    first_line: int
    lines: tuple[tuple[str, str], ...]
    is_record: bool
    record: Any


def _chunks(body: str, jsonl: bool) -> list[_Chunk]:
    """Split a file body the one way both the scrubber and the verifier read it."""
    chunks: list[_Chunk] = []
    pending: list[tuple[str, str]] = []
    pending_first = 1
    for number, (content, terminator) in enumerate(_split_lines(body), start=1):
        record: Any = _NOT_JSON
        if jsonl and content.strip():
            try:
                record = json.loads(content)
            except (ValueError, TypeError):
                record = _NOT_JSON
        if record is _NOT_JSON:
            if not pending:
                pending_first = number
            pending.append((content, terminator))
            continue
        if pending:
            chunks.append(_Chunk(pending_first, tuple(pending), False, None))
            pending = []
        chunks.append(_Chunk(number, ((content, terminator),), True, record))
    if pending:
        chunks.append(_Chunk(pending_first, tuple(pending), False, None))
    return chunks


def _chunk_body(lines: tuple[tuple[str, str], ...]) -> str:
    """A text chunk's lines rejoined exactly as they arrived, terminators included."""
    return "".join(content + terminator for content, terminator in lines)


def _lines_touched(lines: tuple[tuple[str, str], ...], spans: list[tuple[int, int]]) -> set[int]:
    """Which lines of a chunk a set of body offsets falls in, by index within the chunk."""
    bounds: list[tuple[int, int]] = []
    offset = 0
    for content, terminator in lines:
        start = offset
        offset += len(content) + len(terminator)
        bounds.append((start, offset))
    touched: set[int] = set()
    for low, high in spans:
        for index, (start, end) in enumerate(bounds):
            if low < end and high > start:
                touched.add(index)
    return touched


def _scrub_free_text_lines(
    lines: tuple[tuple[str, str], ...], report: ScrubReport
) -> list[tuple[str, str]]:
    """Replace every line of a suspicious free-text region that holds ciphertext bytes.

    The region is read as one document so a carrier split across two lines is still one
    carrier, and a line is the unit of replacement so the file's line count and line order
    do not move. Only the lines a ciphertext-shaped value actually covers are replaced, its
    own line and any line its value runs onto, so an unrelated line elsewhere in a long
    stderr survives. That is a claim about collateral, not about safety: safety comes from
    the value's whole span being inside a replaced line, at whatever distance the sign of
    reasoning material sat.
    """
    spans = _scrubber_suspect_spans(_chunk_body(lines))
    if not spans:
        return list(lines)
    touched = _lines_touched(lines, spans)
    out: list[tuple[str, str]] = []
    for index, (content, terminator) in enumerate(lines):
        if index in touched:
            out.append((FREE_TEXT_MARKER, terminator))
            report.text_redactions += 1
            report.lines_rewritten += 1
        else:
            out.append((content, terminator))
    return out


def _splice(line: str, record: Any) -> tuple[str, int]:
    """Edit a parsed line's carriers in its raw bytes, returning the line and the edit count.

    This is the path for a line whose exact bytes are not what a reserialisation would
    produce: extra spacing, a different escaping choice, a repeated member. Reserialising
    such a line would rewrite bytes the gate promised to keep, so the carrier value is
    blanked where it sits and everything around it is left alone. Blanking rather than
    removing keeps the edit inside one value's span, which is the only edit that can be made
    without moving anything else, and an empty value carries nothing replayable.
    """
    start = _skip_ws(line, 0)
    try:
        _value, end = _DECODER.raw_decode(line, start)
    except ValueError:  # pragma: no cover - the caller has already parsed this line
        return line, 0
    spans = _member_spans(line, record, start, end, _scrubber_carrier, _scrubber_suspects)
    if not spans:
        return line, 0
    pieces: list[str] = []
    cursor = 0
    marker = json.dumps(FREE_TEXT_MARKER, ensure_ascii=False)
    for span_start, span_end, is_carrier in sorted(spans):
        pieces.append(line[cursor:span_start])
        pieces.append('""' if is_carrier else marker)
        cursor = span_end
    pieces.append(line[cursor:])
    return "".join(pieces), len(spans)


def _scrub_record_line(line: str, record: Any, report: ScrubReport) -> str:
    """One parsed JSONL line's content, without its terminator, which is never ours to change.

    A line that is already exactly what a compact reserialisation would produce can be
    rewritten from the parsed value, and the parsed value is known to be complete for such a
    line: a repeated member, or any other spelling the parser collapses, would have made the
    reserialisation differ. That equality check is what lets the structural pass be trusted
    here and nowhere else. Every other line goes through the raw byte path, which reads the
    members back out of the bytes and so still sees the members a parse dropped.
    """
    scrubbed = _as_json(scrub_value(record, report))
    canonical = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    if canonical == line:
        if scrubbed == record:
            return line  # nothing to remove: the input bytes are the output bytes
        report.lines_rewritten += 1
        return json.dumps(scrubbed, ensure_ascii=False, separators=(",", ":"))
    edited, edits = _splice(line, record)
    if edited != line:
        report.lines_rewritten += 1
        if scrubbed == record:
            # The parsed view saw nothing, so this was a carrier only the raw members show.
            report.fields_removed += edits
    return edited


def scrub_body(body: str, jsonl: bool, report: ScrubReport) -> str:
    """Scrub a file body, changing only carrier bytes and suspicious free-text lines.

    Line count and line order never move. Records are scrubbed as structure, and runs of
    lines that are not records are scrubbed as free text, which is how a trace's harness
    chatter and a whole `.err.txt` are both handled by the same rule.
    """
    if body == "":
        # An empty trace is a clean zero-byte trace; a lone newline would be a byte change.
        return ""
    out: list[str] = []
    for chunk in _chunks(body, jsonl):
        if chunk.is_record:
            content, terminator = chunk.lines[0]
            out.append(_scrub_record_line(content, chunk.record, report))
            out.append(terminator)
            continue
        if jsonl:
            report.unparsed_lines += sum(1 for content, _t in chunk.lines if content.strip())
        for content, terminator in _scrub_free_text_lines(chunk.lines, report):
            out.append(content)
            out.append(terminator)
    return "".join(out)


def scrub_jsonl_text(body: str, report: ScrubReport) -> str:
    """Scrub a JSONL body. Lines that are not JSON are the free text they look like."""
    return scrub_body(body, True, report)


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


def _is_jsonl(path: Path) -> bool:
    """Which shape a file is read in. Both sides ask this, so both read the same shape."""
    return path.name.endswith(".jsonl")


def scrub_file(path: Path) -> ScrubReport:
    """Scrub one trace file in place. JSONL is parsed; anything else is one free text body.

    The file is written only when something actually changed, and it is written back through
    the same lossless encoding, so every line the scrub did not touch leaves byte-identical,
    its terminator and any non-UTF-8 bytes included.
    """
    report = ScrubReport(files=1, file_ids=[safe_label(str(path))])
    body = _read(path)
    scrubbed = scrub_body(body, _is_jsonl(path), report)
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


def verify_body(body: str, jsonl: bool) -> list[str]:
    """Where ciphertext survives in one file body, as locations. Never returns the values.

    The body is split by the same :func:`_chunks` the scrubber used, so the two sides are
    reading the same artifact: a free text run is verified as one document and the offsets
    of what it finds are mapped back to line numbers for reporting, rather than each line
    being verified alone, where a carrier that straddles the boundary is invisible to both
    halves of it.

    A record is checked twice, once through the parsed value and once through its raw
    members. The second pass is not redundant: a parse keeps one member per key, and a
    record that declares `type` twice can present as prose to the first pass while the
    bytes still carry the ciphertext.
    """
    locations: list[str] = []
    for chunk in _chunks(body, jsonl):
        if chunk.is_record:
            content, _terminator = chunk.lines[0]
            locations.extend(
                f"line {chunk.first_line} {location}"
                for location in find_reasoning_material(chunk.record)
            )
            raw = _member_spans(
                content, chunk.record, _skip_ws(content, 0), len(content), _verifier_carrier
            )
            if raw:
                locations.append(f"line {chunk.first_line} <raw>")
            continue
        spans = _verifier_suspect_spans(_chunk_body(chunk.lines))
        for index in sorted(_lines_touched(chunk.lines, spans)):
            locations.append(f"line {chunk.first_line + index} <embedded>")
    return locations


def verify_run_dir(root: Path) -> dict[str, list[str]]:
    """Where ciphertext survives under `root`, per file. Empty means publishable.

    Files are named by position and digest rather than by path, because a trace file is
    named after the task that produced it and a task name is not this module's text to
    quote. The position is the file's index in :func:`iter_traces`, which an operator can
    list for themselves, and the digest matches any candidate path they hash the same way.
    """
    findings: dict[str, list[str]] = {}
    for index, path in enumerate(iter_traces(root)):
        locations = verify_body(_read(path), _is_jsonl(path))
        if locations:
            findings[f"file[{index}]{safe_label(str(path))}"] = locations
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
        # The path is the operator's own argument and is echoed back to them on the error
        # channel only, which is the one place it is not a published artifact.
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
