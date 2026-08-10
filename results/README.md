# results

One JSON per cell, written by `shobench run`. Each file carries the cell's manifest, the
per-task scores from both eval phases, the rollout's tasks and stopping record, the paired
deltas, and the egress summary. `shobench report results/` renders the summary table from
these, and `--format json` emits the forest-plot input.

Nothing here is written by hand. A file's `manifest` block names the shogym rev, the split
digest, the instruction digests, the resolved harness version and model, and the agent home's
digest before and after, which together say whether two files describe the same experiment.

A results file is evidence, not a claim: rows whose closure carried no score are reported as
unscored rather than averaged as zeros, and tasks that scored in only one eval phase appear
under `unpaired` rather than disappearing.
