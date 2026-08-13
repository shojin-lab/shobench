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


def _first_parseable_event(path: Path) -> dict | None:
    """The first line of a JSONL file that parses as an object, whatever its type.

    This is the anchor two of the harnesses' own session readers use: codex refuses a rollout
    that "does not start with session metadata", and prime-agent's session scanner returns
    nothing when the first parseable entry is not the session header (both observed against
    the pinned binaries). A search that skipped past a foreign first line to find the record
    later in the file would accept exactly the files those readers refuse, so the transcript
    preflight anchors the same way they do.
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
            return event if isinstance(event, dict) else None
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
