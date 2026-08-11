# shōbench

A benchmark of agents from shared initial conditions: what they do, and when they
choose to stop.

Every agent gets the same starting state and the same stream of tasks, served by
[shōgym](https://github.com/shojin-lab/shogym); the record is what the server observed,
never what the agent reports. The quantities of interest are behavioral: what an agent
does with identical affordances, and when it decides it is finished.

Early evidence for the shape of this benchmark came from shōgym's own quickstart
verifications: on the same three-task queue with the same prompt, one model completed
every task, another quit after one and summarized confidently, and a third drained two
tasks it never played. Same initial conditions; the difference was the agent.

This repo will hold the benchmark definitions, the deployment and evaluation code, and
the analysis. The replication data for published results lives in
[shōrep](https://github.com/shojin-lab/shorep).

## Traces are scrubbed before they are published

A run writes full stream traces under `<phase>/traces/*.stream.jsonl`, and those traces
contain the assistant's thinking blocks. A thinking block carries a `signature` next to
its text, and the signature is not inert. ["Stealing Reasoning Traces from Proprietary
LLM APIs"](https://arxiv.org/abs/2608.09867) shows that a published signature can be
replayed into a weaker sibling model to reconstruct the original model's raw chain of
thought verbatim; sweeping public trajectories that way recovered hundreds of
credentials and PII artifacts from traces whose authors did not think they were
publishing anything sensitive. An empty `thinking` field is no protection, because the
signature is the part that replays.

So this benchmark draws a line between the run and the export. **A trace under `runs/`
stays whole**, because that is the operator's own data and debugging a leg needs all of
it. **Anything leaving the machine goes through `shobench.scrub` first**, which removes
the `signature` off every thinking block, the ciphertext `data` off every
`redacted_thinking` block, and any block left carrying nothing, and then re-derives the
verdict from the bytes about to ship. Scrubbing and verifying are separate on purpose: a
scrubber that certified its own output would keep passing on the day it grows a blind
spot.

Scrub a run directory, or check one without touching it:

```
python -m shobench.scrub <run-dir>            # scrub in place, then verify
python -m shobench.scrub <run-dir> --check    # report only, non-zero exit if anything is present
```

`scrub.assert_publishable(value, what)` is the gate itself. It raises rather than
repairs, because a published signature cannot be unpublished whereas a blocked artifact
costs one rerun. Reports and exceptions carry counts and field paths only, never the
value that tripped them.
