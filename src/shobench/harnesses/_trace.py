"""Reading a harness's JSONL trace, the two ways more than one harness needs.

Every harness writes a stream-json trace, and classifying how a leg ended means reaching into
it for a particular event: the first of a kind (the session header, read for the id the runner
resumes) or the last of a kind (the terminal turn event, read for how it went). These two
walks are shared; what event types to look for, and what to make of them, stays in each
harness file.

They read from opposite ends, and each reads from its own end all the way. The last-of-a-kind
walk builds on ``jsonl_events``, which keeps the tail of a trace, since a terminal event is at
the end by definition. The first-of-a-kind walk cannot use it: the session header is line one,
and a trace long enough for the difference to matter is exactly the one whose tail no longer
holds it.
"""

from __future__ import annotations

import json
from pathlib import Path

from shobench.harness import jsonl_events


def _refuse_nonstandard_constant(constant: str) -> float:
    """Raised into: the preflight dialect has no NaN, Infinity, or -Infinity."""
    raise ValueError(f"nonstandard JSON constant {constant!r}")


def _strings_utf8_clean(node: object) -> bool:
    """Does every string in this decoded tree survive a strict UTF-8 encode?

    Python's json willingly decodes an escaped lone surrogate into a string no strict UTF-8
    decoder could have produced. serde refuses that escape outright, so a preflight that let
    it through would certify a record the codex reader cannot decode (observed: a meta whose
    timestamp is ``\\ud800`` is refused as unreadable session metadata). Keys are checked
    with the values, since a surrogate is no more decodable for being a key.
    """
    if isinstance(node, str):
        try:
            node.encode("utf-8")
        except UnicodeEncodeError:
            return False
        return True
    if isinstance(node, dict):
        return all(_strings_utf8_clean(k) and _strings_utf8_clean(v) for k, v in node.items())
    if isinstance(node, list):
        return all(_strings_utf8_clean(item) for item in node)
    return True


def _strict_json_object(line: str) -> dict | None:
    """One JSONL line decoded in the dialect the pinned session readers share, or ``None``.

    Python's json is more permissive than the parsers on the other side of the preflight in
    two ways that matter. It accepts the extension constants (NaN, Infinity, -Infinity),
    which JSON.parse refuses: a prime header line carrying one is unparseable to the scanner
    and the session is "No session found matching" (observed). And it decodes escaped lone
    surrogates, which serde refuses: see :func:`_strings_utf8_clean`. A preflight reading the
    lenient dialect would certify exactly those files, so this reader refuses both. The
    intersection is deliberately the STRICTEST of the three CLIs' dialects: JS parsers do
    accept a lone-surrogate escape, but no real session file carries one (their strings are
    cwds, ids, and model text), so the narrowing refuses only fabrications.
    """
    try:
        event = json.loads(line, parse_constant=_refuse_nonstandard_constant)
    except ValueError:
        return None
    if not isinstance(event, dict) or not _strings_utf8_clean(event):
        return None
    return event


def _first_parseable_event(path: Path) -> dict | None:
    """The first line of a JSONL file that strictly decodes as an object, whatever its type.

    This is prime-agent's own anchor, mirrored: its scanner walks lines, skips one its parser
    refuses, and reads the first that decodes; a session header below a NaN-poisoned junk
    line resumed, while a header below a PARSEABLE message line did not (both observed). So
    unparseable lines are skipped here and parseable ones anchor. codex anchors harder, on
    the literal first record, which is :func:`_first_record_strict`.
    """
    if not path.exists():
        return None
    with path.open(encoding="utf-8", errors="ignore") as lines:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            event = _strict_json_object(line)
            if event is not None:
                return event
    return None


def _first_record_strict(path: Path) -> dict | None:
    """The FIRST non-empty line, strictly decoded as an object, or ``None`` when it is not one.

    codex's reader mirror: it parses the first rollout record and refuses the whole file when
    that parse fails ("failed to parse first rollout record", observed with a fully valid
    meta sitting on line two), so nothing is skipped here either.
    """
    if not path.exists():
        return None
    with path.open(encoding="utf-8", errors="ignore") as lines:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            return _strict_json_object(line)
    return None


def _first_event_of_type(path: Path, types: tuple[str, ...]) -> dict | None:
    """The first event of a JSONL trace whose ``type`` is one of ``types``.

    Read forward from line one and stopped at the first match, so a harness that announces its
    session at the top of the stream is found whether the run wrote ten lines or ten million.
    Reading a window of the tail instead is how a long run comes to have no session id, and a
    suspension without one cannot be continued: an eight-hour rollout would be lost to the
    interruption it was meant to survive, and the longer the run, the likelier that became.

    Streamed rather than read whole, because a trace that size is not something to hold in
    memory to find its first line, and tolerant of a malformed line for the same reason the
    tail walk is: the file is written by another process and can be cut anywhere.
    """
    if not path.exists():
        return None
    with path.open(encoding="utf-8", errors="ignore") as lines:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") in types:
                return event
    return None


def _last_event_of_type(path: Path, types: tuple[str, ...]) -> dict | None:
    """The last event of a JSONL trace whose ``type`` is one of ``types``."""
    for event in reversed(jsonl_events(path)):
        if event.get("type") in types:
            return event
    return None
