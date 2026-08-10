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
| tau2_telecom | `TAU2_DATA_DIR` pointing at the tau2-bench `data/` tree at sha `1d244f5d` (about 730 MB), and `OPENAI_API_KEY` for the user simulator |
| hle | the gated `cais/hle` dataset, so `HF_TOKEN` unless it is already cached, and `OPENAI_API_KEY` for the judge |

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

### Two things worth knowing before reading the code

**One fresh session per eval task is enforced by the server, not requested of the agent.**
Each eval task gets its own single-task `EvalStream`, so a session that ignores its
instruction and pulls a second task is told the stream is done rather than quietly consuming
the next task's measurement.

**The rollout is a sequence of bounded legs against one live stream.** No harness runs for
eight hours on its own, so the runner owns the outer loop. A leg the provider cut off at a
usage limit is resumed and does not count as a stop; a leg that ended on the agent's own
terms while the queue still had tasks is the stop the charter asks about, and the runner
stops serving there rather than prompting the agent onward. `docs/harness-autonomy.md`
records how each harness announces which, and where each rule came from.
