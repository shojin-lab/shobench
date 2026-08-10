"""Reading a harness's JSONL trace, the two ways more than one harness needs.

Every harness writes a stream-json trace, and classifying how a leg ended means reaching into
it for a particular event: the first of a kind (the session header, read for the id the runner
resumes) or the last of a kind (the terminal turn event, read for how it went). These two
walks are shared; what event types to look for, and what to make of them, stays in each
harness file. Both build on ``jsonl_events`` from the base module, which is where the tolerant
parsing lives.
"""

from __future__ import annotations

from pathlib import Path

from shobench.harness import jsonl_events


def _first_event_of_type(path: Path, types: tuple[str, ...]) -> dict | None:
    """The first event of a JSONL trace whose ``type`` is one of ``types``."""
    for event in jsonl_events(path, limit=50):
        if event.get("type") in types:
            return event
    return None


def _last_event_of_type(path: Path, types: tuple[str, ...]) -> dict | None:
    """The last event of a JSONL trace whose ``type`` is one of ``types``."""
    for event in reversed(jsonl_events(path)):
        if event.get("type") in types:
            return event
    return None
