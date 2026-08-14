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

This repo holds the benchmark definitions, the deployment and evaluation code, and the
analysis. The replication data for published results lives in
[shōrep](https://github.com/shojin-lab/shorep).

## The v0 runner

`docs/scope-v0.md` is the charter. v0 is 3 environments by 4 harness-model pairs, and each
of those 12 cells runs three phases: the held-out split cold, then an improvement rollout
under the env-agnostic "Get Better" instruction, then the same held-out split again with
whatever the rollout wrote still in place. Eval sessions are fresh by design, so the
measurement isolates the durable-artifact channel.

    uv sync
    uv run shobench cells                       # the matrix, as configured
    uv run shobench doctor                      # what is installed, what is missing
    uv run shobench build                       # the agent, holder, and observer images
    uv run shobench creds --cell <name>         # the credential negative control alone
    uv run shobench run --cell <name>           # a plan, no spend
    uv run shobench run --cell <name> --go      # the cell, for real
    uv run shobench stop --run runs/<run-id>    # end a live run through its normal ending
    uv run shobench report results/             # the summary table

`--go` is the safety story. Every command that spends prints its plan and exits without it,
and cells are run one at a time by name; nothing here launches the matrix.

### What each env needs before its cells can run

`shobench run` refuses to start a cell whose serving-side needs are absent, so these are
checks rather than surprises. All of them live outside the agent container: the agent holds no
dataset, no judge key, and no broker credential.

| env | needs |
|---|---|
| automationbench | nothing; the pinned upstream source is fetched once into shogym's cache |
| tau2_telecom | the tau2-bench `data/` tree at sha `1d244f5d` (about 730 MB), provisioned once with `uv run python tools/provision_tau2_data.py`; the runner then points `TAU2_DATA_DIR` at it. Plus `OPENAI_API_KEY` for the user simulator |
| hle | the gated `cais/hle` dataset, so `HF_TOKEN` unless it is already cached, and `OPENAI_API_KEY` for the judge |

shogym provisions each env's upstream *source* at runtime but does not carry tau2's ~730 MB of
`data/`, so that one subtree is provisioned separately by the command above (idempotent: a tree
that already is the pinned data is verified and skipped, and `--force` re-fetches and replaces
what is there). What "is the pinned data" means is settled by a committed digest manifest, not by
a file list: every file a tau2_telecom run reads has to match the size and sha256 the pinned
commit has, so a stale checkout handed in through `TAU2_DATA_DIR`, or an edited policy or DB, is
refused by name instead of quietly moving the numbers. A refused tree that the operator named
through `TAU2_DATA_DIR` is then left exactly as it was, along with everything else living in it:
only `--force` replaces that one, and a tree that passes is only read, not even annotated, so a
read-only checkout works. The runner sets `TAU2_DATA_DIR` itself and refuses a tau2 cell
whose data is not that tree, naming the command. `HF_TOKEN` is not blocking today: the
`cais/hle` dataset is already cached on the run host, so the gate passes without a fresh token;
it is only needed on a cold cache.

### How the pieces fit

| Piece | Where | What it owns |
|---|---|---|
| cell configs | `cells/*.toml` | one file per cell, so the matrix is data and a new arm is a new file |
| split manifests | `splits/*.json` | the exact ids on each side, and the provenance that produced them |
| instruction arms | `instructions/<arm>/` | the prompts, hashed into every manifest |
| serving | `src/shobench/serving.py` | shogym's `TaskStream` and `EvalStream` behind `build_stream_server` |
| the three phases | `src/shobench/runner.py` | container lifecycle, legs, resume, the manifest |
| harnesses | `src/shobench/harnesses/` | autonomous launch, and how a leg ended, one file per harness |
| credentials | `src/shobench/credentials.py` | isolated HOME plus the negative control |
| egress | `src/shobench/egress.py` | passive per-cell capture that restricts nothing |
| reporting | `src/shobench/report.py` | the paired bootstrap and the summary table |

### A few things worth knowing before reading the code

**One fresh session per eval task is enforced by the server, not requested of the agent.**
Each eval task gets its own single-task `EvalStream`, so a session that ignores its
instruction and pulls a second task is told the stream is done rather than quietly consuming
the next task's measurement.

**Every published number is counted against the committed held-out set.** A held-out task can
produce no row at all: its harness dies, or exits before it ever asks for a task. Counted by
rows, that is not a failure but an absence, and an absence moves the denominator, so the ids
come from the split manifest rather than from whatever arrived. An id with nothing recorded for
it is published as an explicit unscored row saying why, it is paired against and reported as
unpaired, and the cell is written to `results/<cell>.incomplete.json` rather than to
`results/<cell>.json`. Its numbers stay readable and the ids it lost are named under
`heldout.missing_task_ids`, but nothing reaching for a cell's result finds a measurement with a
hole in it standing in for one.

**The rollout is one honest run of the harness against one live stream.** A single invocation
is driven against the pool for the cell's wall clock, and the runner does not relaunch it,
because whether a harness sustains autonomous operation is one of the things being measured. A
run that ends on the agent's own terms while the queue still had tasks is the stop the charter
asks about, and nothing prompts the agent onward.

**Stopping a run is a command, not a kill.** `pkill` plus `docker rm -f` ends the runner before
it can write `legs.json` and `rollout_stopping.json`, and a run without those has no terminus:
`rebookend` refuses it forever and the cell can never produce an `eval_after`, so the cheap way
to stop a wedged run destroys the measurement while leaving it to burn its whole clock preserves
it. `stop` inverts that. It spends nothing, so it takes no `--go`, and it is safe to call twice
and safe to call on a run that has already finished:

    uv run shobench stop --run runs/<run-id> --reason "wedged on a non-terminating tool call"

It writes a one-shot ask into the live run directory (a file, not a signal: a run outlives the
process that started it, and the only pid a run records is in a lock file that is never
unlinked). The runner ends its current leg the way a budget does, records the leg as
`operator_stop`, which is its own kind and neither a chosen stop nor a timeout nor a usage-limit
suspension, starts no further phase, and publishes what it has. An operator-ended rollout keeps a
real terminus, being the agent's state at the moment it was stopped, so `shobench rebookend` can
still give it an `eval_after`; that the treatment was shorter than the cell intended is a fact
the artifact states rather than a record nobody can produce.

**A rollout leg that stops getting anywhere is ended the same way.** The rollout is the one leg
with no other bound, every eval leg being bounded per task by `eval_task_timeout_s`, and a
non-terminating tool call there pegs a core and seals nothing for the rest of an eight-hour
clock. What ends it is an absence of progress from EVERY source, not silence in the trace: trace
records, sealed rows, the harness's session state under the cell HOME (which is where a
delegating agent's children are written), and file changes under `/work` each reset the clock.
Keying on the trace alone would end exactly the legs worth keeping, since a task can take an hour
inside one tool call and an agent that delegates goes quiet in its own trace by design. The bound
is the per-cell `budget.rollout_no_progress_s`, two hours by default and `0` to disable it, and a
leg it ends is recorded as `no_progress`.

**A provider usage limit suspends the cell rather than ending it.** That interruption is not
the agent's doing, so the run stops where it stands and an operator continues it once the
window resets:

    uv run shobench resume --run runs/<run-id>        # the plan: which cell waits, and for how much
    uv run shobench resume --run runs/<run-id> --go   # continue it

A suspended cell writes `runs/<run-id>/suspended.json` (the session to reattach to, how far the
pool got, how much of the rollout clock is spent, and the budget that clock came from), stops
its containers, and keeps every directory. It publishes no results and runs no `eval_after`,
because that measurement belongs after a real rollout terminus and never inside an exhausted
window. A suspending run exits 75, so a script can tell "waiting for a window" from "failed".

The continuation reattaches to the same session, reopens the same provenance record at the
position it was holding, and gets only what remained of the recorded rollout budget; a second
limit suspends again with the time accumulated. It publishes what an uninterrupted cell would:
the eval that ran before the interruption is read back off the run directory and paired with
the one that follows the rollout, and the models and legs of both processes are in the record.

Three things stop a continuation before it spends, all of them for the same reason, which is
that a cell can wait hours and a repository changes in hours. The rollout clock must have time
left on it. The cell's serving-side needs (a dataset, a judge key) must be present in this
shell as they were in the first. And the cell, split, and instruction must still match the
digests the manifest recorded, or the run would publish one run id describing two experiments.
Each of those refusals leaves the suspension record where it is, so the run stays resumable
once the shell or the checkout is put right. `docs/harness-autonomy.md` records how each
harness announces a usage limit, and where each rule came from.
