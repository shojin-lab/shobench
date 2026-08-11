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

**What is removed.** The `signature` off every thinking block, the ciphertext `data` off
every `redacted_thinking` block, and any block that carries nothing else once those are
gone. Everything else survives, including the block's text when it has any: the goal is
to publish reasoning without publishing the means to forge or extract it.

**What is preserved, and why it is not negotiable.** Line count and line order. The
runner reads a finished trace back to recover the session id, the observed models and
the stop classification, and it does so from a bounded tail of the file. Dropping lines
would slide that window and change what a re-read concludes, so a record whose content
empties out is written back with an empty content list rather than deleted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Blocks whose payload is provider ciphertext rather than research content.
# `redacted_thinking` is included because it carries its ciphertext under `data` rather
# than `signature`, so a scrubber that only knew the field name would pass it through.
THINKING_TYPES = ("thinking", "redacted_thinking")

# The fields to take off such a block. Named per type would be tidier, but a trace is
# other people's JSON and block shapes drift between harness versions, so both names are
# stripped from both types.
CIPHERTEXT_FIELDS = ("signature", "data")

# What makes a block still worth keeping once the ciphertext is off it.
TEXT_FIELDS = ("thinking", "text")

# A signature that has been serialised into free text rather than left as JSON structure.
# The stop-classification evidence embeds a tail of raw harness stderr, and a harness that
# logs a failed request logs the request, so structural scrubbing alone would miss it.
# Only a non-empty value matches: an already-scrubbed `"signature":""` is not a finding.
EMBEDDED = re.compile(r'("(?:signature|data)"\s*:\s*")([^"]+)(")')

# Traces are JSONL, and their stderr siblings are not JSON at all. Both leave the machine
# together, so both are scrubbed; a gate that only knew about `*.stream.jsonl` would
# publish the stderr file beside it untouched.
TRACE_GLOBS = ("*.stream.jsonl", "*.err.txt")


class ReasoningSignatureFound(Exception):
    """Publication was refused because reasoning ciphertext reached a publishable artifact.

    Carries the locations only. The whole point of the check is that this value never
    reaches a log, a terminal or an exception tracker, so the message counts findings and
    names the paths that hold them, and never quotes what it found.
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
    """Blank out signature values that were serialised into a free-text string.

    The value is replaced rather than the whole field deleted, because this runs on
    captured stderr where the surrounding characters are somebody's log line and cutting
    into it would corrupt the diagnostic that the tail was captured for.
    """

    def replace(match: re.Match[str]) -> str:
        if report is not None:
            report.text_redactions += 1
        return f"{match.group(1)}{match.group(3)}"

    return EMBEDDED.sub(replace, text)


def scrub_value(value: Any, report: ScrubReport | None = None) -> Any:
    """Return `value` with reasoning ciphertext removed, recursively.

    The record shape is not hardcoded: any dict whose `type` names a thinking block is
    treated as one, wherever it sits. That matters because the same block appears at
    three different depths across the harnesses, and because `--include-partial-messages`
    puts a second copy inside streamed deltas.

    A block that empties out returns `None` and its parent list drops it. Lists are
    rebuilt rather than mutated so the caller does not need to know where content arrays
    live in a given harness's schema.
    """
    if isinstance(value, dict):
        if value.get("type") in THINKING_TYPES:
            kept = {k: v for k, v in value.items() if k not in CIPHERTEXT_FIELDS}
            if report is not None:
                report.fields_removed += sum(1 for f in CIPHERTEXT_FIELDS if f in value)
            if not any(kept.get(f) for f in TEXT_FIELDS):
                if report is not None:
                    report.blocks_dropped += 1
                return None
            return {k: scrub_value(v, report) for k, v in kept.items()}
        # A `signature_delta` in a streamed partial message carries the same ciphertext
        # under the same key without ever declaring itself a thinking block, so the key is
        # also stripped wherever it appears on its own.
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key == "signature" and isinstance(item, str):
                if report is not None:
                    report.fields_removed += 1
                continue
            out[key] = scrub_value(item, report)
        return out
    if isinstance(value, list):
        scrubbed = [scrub_value(v, report) for v in value]
        return [v for v in scrubbed if v is not None]
    if isinstance(value, str):
        return scrub_text(value, report)
    return value


def find_reasoning_material(value: Any, path: str = "$") -> list[str]:
    """Locate every place ciphertext survives, as paths. Never returns the values.

    This is the gate's evidence, so it is written independently of :func:`scrub_value`
    rather than by calling it and diffing. A verifier that shares the scrubber's notion of
    where to look inherits the scrubber's blind spots and will certify them.
    """
    found: list[str] = []
    if isinstance(value, dict):
        if value.get("type") in THINKING_TYPES:
            for f in CIPHERTEXT_FIELDS:
                if value.get(f):
                    found.append(f"{path}.{f}")
        if isinstance(value.get("signature"), str) and value["signature"]:
            found.append(f"{path}.signature")
        for key, item in value.items():
            found.extend(find_reasoning_material(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            found.extend(find_reasoning_material(item, f"{path}[{i}]"))
    elif isinstance(value, str) and EMBEDDED.search(value):
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


def scrub_jsonl_text(body: str, report: ScrubReport) -> str:
    """Scrub a JSONL body, preserving line count and line order exactly.

    Lines that name nothing interesting are never parsed, so their bytes come through
    untouched; that is what keeps the diff attributable to the scrub. Lines that fail to
    parse are passed through too, because a trace is a captured stdout and may hold
    harness chatter that was never JSON, and the runner's own reader tolerates them.
    """
    needles = ('"signature"', '"thinking"', '"redacted_thinking"')
    lines = body.split("\n")
    trailing = lines and lines[-1] == ""
    if trailing:
        lines.pop()
    out: list[str] = []
    for line in lines:
        if not any(n in line for n in needles):
            out.append(line)
            continue
        try:
            parsed = json.loads(line)
        except (ValueError, TypeError):
            # Not JSON, so treat it as the free text it is rather than dropping it.
            redacted = scrub_text(line, report)
            if redacted != line:
                report.lines_rewritten += 1
            report.unparsed_lines += 1
            out.append(redacted)
            continue
        rewritten = json.dumps(
            scrub_value(parsed, report), ensure_ascii=False, separators=(",", ":")
        )
        if rewritten != line:
            report.lines_rewritten += 1
        out.append(rewritten)
    return "\n".join(out) + ("\n" if trailing else "")


def scrub_file(path: Path) -> ScrubReport:
    """Scrub one trace file in place. JSONL is parsed; anything else is treated as text."""
    report = ScrubReport(files=1, paths=[str(path)])
    body = path.read_text(encoding="utf-8", errors="replace")
    if path.name.endswith(".jsonl"):
        scrubbed = scrub_jsonl_text(body, report)
    else:
        scrubbed = scrub_text(body, report)
        if scrubbed != body:
            report.lines_rewritten += 1
    if scrubbed != body:
        path.write_text(scrubbed, encoding="utf-8")
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
        body = path.read_text(encoding="utf-8", errors="replace")
        locations: list[str] = []
        for number, line in enumerate(body.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except (ValueError, TypeError):
                if EMBEDDED.search(line):
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
