# shōbench v0 scope: measuring self-improvement across harnesses

Status: DRAFT for discussion. Nothing here is committed infrastructure.

## The question

For each (environment, harness) cell: given the same initial conditions and the
env-agnostic improvement instruction, does the agent get measurably better on tasks it has
never seen? The deliverable per cell is four numbers and an interval: held-out mean before,
held-out mean after, the paired delta, and its confidence interval.

The full design is, for each env (excluding wordle and the tau2 retail and airline domains),
for each harness: run an improvement rollout with the "Get Better" instruction, then evaluate
on held-out tasks. Display initial and post-improvement means with CIs on the held-out split.
v0 below is a scaled-back cut of that matrix chosen for cost, leakage safety, and statistical
power; the full matrix is the target it grows back toward.

## Protocol per cell

1. **Initial conditions.** A pristine agent home (no memory, no skills, no prior sessions),
   pinned harness version, the harness's own default model (recorded), one sandboxed
   container per cell. Identical container image across harnesses.
2. **Held-out eval, before.** Serve the held-out split cold: plain task serving, no
   improvement instruction. One fresh session per task, per-task scores recorded from the
   trace.
3. **Improvement rollout.** Serve the improvement split with the improvement instruction
   (the env-agnostic "Get Better" prompt plus generic loop mechanics, no task or env
   specifics). The agent is free to write memory, skills, code, whatever its harness
   supports. Fixed budget per cell, identical across harnesses within an env.
4. **Held-out eval, after.** The same held-out split, same order, fresh sessions, but the
   agent home now carries whatever the rollout wrote. Fresh sessions are deliberate: they
   strip conversational context, so the eval isolates the durable-artifact channel, which is
   the honest meaning of "the agent improved" (see prior evidence below).
5. **Report.** Per-task paired deltas; mean before, mean after; 95% CI on the mean delta by
   paired bootstrap; full-solve rate before and after; N. Secondary: when the agent chose to
   stop during the rollout (tasks attempted before stopping, whether it self-checkpointed),
   which is the stopping-behavior question in the shōbench charter.

## What prior evidence already tells us (and why the design is shaped this way)

The shōjin program's first study (packaged in shōrep as `conversation-not-memories`) found,
for one harness on one env: improvement during a session is real and large
(+0.099, p < 0.0001), but the durable artifact (memory) transferred almost nothing on top of
context (+0.007, CI [-0.011, +0.025]), and held-out transfer of a written knowledge base was
weak (+0.088 mean lift, flat full-solve). Three consequences for shōbench:

- **Fresh-session eval is the right test.** Context transfer is established; the open
  question is which harnesses can convert experience into artifacts that survive a session
  boundary. That is exactly what before/after with fresh sessions measures.
- **Expect small effects; power accordingly.** At N=40 paired tasks a +0.088 mean lift was
  detectable but unconvincing; the tight null on the artifact channel was visible at N=120.
  Held-out splits below aim for N >= 60 where the env has the tasks to spare.
- **The rollout needs auto-continue.** Under a minimal instruction, agents self-checkpoint
  and stop early; a harness-level continuation cue changed 11 tasks attempted into 49. The
  rollout budget should be enforced by the runner, with a continuation cue, or the
  measurement confounds "chose to stop" with "ran out of loop". Stopping behavior is then
  reported as its own metric rather than silently truncating the treatment.

## The env matrix

Counts measured against the shōgym registry (2026-08-07). Excluded by decision: `wordle_v1`
(trivial), `tau2_retail`, `tau2_airline`.

| env | tasks | scoring | keys/infra | held-out leakage risk | v0? |
|---|---|---|---|---|---|
| `automationbench` | 600 | pure end-state rubric, offline | none | low (simulated world) | **yes** |
| `yc_bench` | 16 | deterministic sim | none | low (simulated) | **yes** |
| `tau2_telecom` | ~114 (needs data fetch) | tau2 evaluator + user simulator | OPENAI_API_KEY | low (simulated) | **yes** |
| `tau2_banking_knowledge` | TBD (needs data fetch) | tau2 evaluator | OPENAI_API_KEY | low | v1 |
| `frontier_bench` | 5 | container end-state verifier | Docker | low | no: 5 tasks cannot give a held-out split with power |
| `hle` | 1726 | exact match + model judge | OPENAI_API_KEY + gated HF | **high**: public dataset, answers on the web | deferred until network policy |
| `browsecomp_plus` | 664 | model judge + retrieval metrics | OPENAI_API_KEY + Java 21 + gated | medium | deferred until network policy |
| `orca_bench` | 755 | task judge (model) | OPENAI_API_KEY + Docker, ~133 GB/host | medium: the hub RPC returns full ground truth | waits for phase 2 (live backend) |

The leakage column is load-bearing: the sandbox has full internet (harnesses need their
model APIs), and disallowing web tools does not stop a Bash-capable agent from curling
answers. Prior work hit exactly this on held-out runs. v0 dodges the problem by choosing
envs whose answers exist only inside a simulation; hle and browsecomp_plus join when eval
containers get an egress allowlist (model API endpoints only), which is real infrastructure
work and its own line item. orca_bench additionally needs its resolver RPC blocked or the
oracle surface is one curl away.

## The harness matrix

The five shōgym quickstart harnesses: `claude_code`, `codex`, `pi`, `hermes`, `prime_agent`.
Harness-native default models, versions pinned and recorded per run. Rationale: shōbench
benchmarks agents as shipped; pinning one model across harnesses would measure scaffolds,
which is a different (also interesting, later) experiment.

Known per-harness hazards from prior runs:

- `codex`: unreliable over a single long loop (variable yielding plus a queue-draining rush
  mode). Its rollout must be supervised episodically (per-task relaunch or watchdog), not
  launched as one long run.
- `claude_code`: needs `IS_SANDBOX=1` to run bypass-permissions as root in the container;
  memory and skills load only at session start, which is fine here because eval sessions
  are fresh by design.

## v0 = 3 envs x 2 harnesses = 6 cells

- **Envs:** `automationbench`, `yc_bench`, `tau2_telecom`.
- **Harnesses:** `claude_code`, `codex` first (both have operational history); `pi`,
  `hermes`, `prime_agent` join in v1 once the runner is proven.
- **Splits** (seeded, disjoint, published in the repo):
  - automationbench: 100 improvement / 60 held-out (of 600).
  - tau2_telecom: 54 improvement / 60 held-out (of ~114; adjust to the real count).
  - yc_bench: 8 improvement / 8 held-out. Underpowered on its own; it earns its slot
    because a task is a year-long survival sim, so it feeds the stopping-behavior question
    even where its CI is wide.
- **Budget per cell:** fixed wall-clock for the rollout (proposal: 4 hours), auto-continue
  on, hard token ceiling as a safety stop. Both eval phases run every held-out task exactly
  once.

## Display

v0: a `results/` directory in this repo holding one JSON per cell (per-task scores, split
manifest, harness and model pins, home-subtree hash before and after) and a small script
that renders the summary table plus a forest plot of paired deltas with CIs. A styled
results page on shojin.dev consumes the same JSON later; nothing in v0 blocks on web work.

## Infrastructure (what has to exist before cell one runs)

1. A cell runner: container lifecycle, the three phases, trace collection, split
   enforcement. The selfopt run_agent broker pattern is the starting point (wandb sink
   broker-side, key injected at runtime, agent keeps zero broker mounts).
2. Split manifests checked into shōbench (env name, seed, task ids per side).
3. Paired-bootstrap reporting script.
4. Per-harness episodic supervision for the rollout (the codex requirement, useful for all).

## Open decisions

1. **Rollout budget parity:** wall-clock (proposed) vs token vs task-count. Wall-clock is
   the only one comparable across harnesses with different token accounting.
2. **The improvement instruction text:** reuse the established env-agnostic "Get Better"
   prompt verbatim (proposed) or re-derive per harness. Verbatim keeps cells comparable and
   ties back to the registered study.
3. **How many rollout repetitions per cell:** v0 proposes one rollout per cell (cost), with
   the understanding that a single rollout is one draw; repetitions are the v1 upgrade that
   turns per-cell claims from anecdote into estimate.
4. **Egress allowlist work:** whether to build it for v1 (unlocks hle, browsecomp_plus,
   orca_bench) or keep growing the simulated-env column first.
