"""A pull at the in-flight limit forfeits the task the agent was holding.

A stream serves ``max_in_flight`` tasks at once. A ``get_task`` made at that limit displaces a
live task: it is sealed, scored as it stands, and the new task takes the position it held.

That price is the point. Holding a task is a commitment an agent can lose by asking for another
one, so keeping the payload somewhere a mortal context cannot drop it is a thing an agent can be
observed learning rather than something the harness does on its behalf. Built through the real
``build_stream``, so a stream handed to the rollout that answers a pull any other way fails here.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

from shobench import serving
from shobench.config import load_cell_by_name
from shobench.results import read_phase
from shobench.splits import Side, Split

_SMOKE_CELL = "smoke-automationbench-claude-code"


def test_a_pull_at_the_in_flight_limit_forfeits_the_task_it_displaces(tmp_path: Path) -> None:
    """One slot and two pulls: the first task is sealed and scored, the second is its own."""
    cell = replace(load_cell_by_name(_SMOKE_CELL), env="wordle_v1", max_in_flight=1)
    split = Split(
        env="wordle_v1",
        heldout=Side(task_ids=("0",)),
        pool=Side(task_ids=("1", "2")),
        provenance={"kind": "adopted"},
        source=tmp_path / "split.json",
    )
    prov = tmp_path / "rollout"
    prov.mkdir()

    async def play() -> dict:
        stream = serving.build_stream(cell, split, "rollout", prov)
        async with stream:
            first = await stream.get_task()
            second = await stream.get_task()
            return {
                "distinct": first is not second,
                "closures": [row.closure for row in read_phase(prov)],
                "consumed": stream.queue_info().consumed,
            }

    facts = asyncio.run(play())

    # The second pull was a new dispense, and the first task paid for it with a scored row. The
    # second reads as an unsealed dispense because it is one: it was still live when this read.
    assert facts["distinct"]
    assert facts["closures"] == ["drained", "broker_abort"]
    assert facts["consumed"] == 2
