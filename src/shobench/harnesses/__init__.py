"""The harnesses the runner knows how to drive, one per file.

Each concrete harness is a subclass of ``shobench.harness.Harness`` and lives in its own
module: what makes it autonomous from its first turn, and how it says a leg ended, sourced in
``docs/harness-autonomy.md``. This package holds only the registry that maps a cell's harness
name to its instance, plus the re-exports that keep ``from shobench.harnesses import ...``
working as it did when the harnesses were one module. To add a harness, copy ``_template.py``
and follow ``docs/adding-a-harness.md``; the last step there is the registry line below.
"""

from __future__ import annotations

from shobench.harness import BASE_ENV, Harness
from shobench.harnesses.claude_code import ClaudeCode
from shobench.harnesses.codex import Codex
from shobench.harnesses.prime_agent import (
    SHOGYM_STREAM_SKILL,
    PrimeAgent,
    shogym_stream_skill_files,
)

_REGISTRY = {h.name: h for h in (ClaudeCode(), Codex(), PrimeAgent())}


def harness_for(name: str) -> Harness:
    if name not in _REGISTRY:
        raise ValueError(f"unknown harness {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


__all__ = [
    "BASE_ENV",
    "SHOGYM_STREAM_SKILL",
    "ClaudeCode",
    "Codex",
    "PrimeAgent",
    "harness_for",
    "shogym_stream_skill_files",
]
