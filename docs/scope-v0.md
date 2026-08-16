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
   pinned harness version, the cell's pinned model per the harness-model pairs below
   (recorded in the manifest), one sandboxed container per cell. Identical container image
   across harnesses. The improvement instruction text is itself part of the initial
   conditions: the same env-agnostic prompt, byte-identical in every cell.
2. **Held-out eval, before.** Serve the held-out split cold: plain task serving, no
   improvement instruction. One fresh session per task, per-task scores recorded from the
   trace.
3. **Improvement rollout.** Serve the improvement split with the improvement instruction
   (the env-agnostic "Get Better" prompt plus generic loop mechanics, no task or env
   specifics). The agent is free to write memory, skills, code, whatever its harness
   supports. Fixed budget per cell, identical across harnesses within an env.
4. **Held-out eval, after.** The same held-out split, same order, but each task resumes the
   rollout conversation from its terminal state, as an independent fork per task (the per-task
   home copies carry the rollout transcript, so the forks never see each other, only the
   rollout's end). The after-bookend therefore measures the agent WITH its lived rollout
   state: what is still in context and in the compaction summaries, plus whatever the rollout
   wrote to the home. A resumed task launches with the ROLLOUT standing instruction rather
   than the blind eval one: the conversation already carries the objective in its history
   and summaries, and swapping the instruction mid-conversation would measure an agent that
   never existed. This is the recorded `eval_context = "resumed"` default; the `"cold"` arm
   (fresh sessions, blind eval instruction, durable channel only) is the ablation, and it is
   what the pre-axis cells measured.
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

- **The lived context is part of the measurement.** Context transfer is established and
  large, which is exactly why eval_after resumes the rollout conversation by default: an
  after-bookend that strips it measures only the weak artifact channel and misses most of
  what the rollout built. The cold arm (`eval_context = "cold"`) remains the ablation that
  isolates artifacts surviving a session boundary, and it is the arm the pre-axis cells ran.
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
| `yc_bench` | 16 | deterministic sim | none | low (simulated) | v1 |
| `tau2_telecom` | ~114 (needs data fetch) | tau2 evaluator + user simulator | OPENAI_API_KEY | low (simulated) | **yes** |
| `tau2_banking_knowledge` | 97, 87 of them servable offline (needs data fetch) | tau2 evaluator | OPENAI_API_KEY | low | **yes (amended in 2026-08-14)** |
| `frontier_bench` | 5 | container end-state verifier | Docker | low | no: 5 tasks cannot give a held-out split with power |
| `hle` | 1726 | exact match + model judge | OPENAI_API_KEY + gated HF | **high**: public dataset, answers on the web | **yes (leakage observed, not gated)** |
| `browsecomp_plus` | 664 | model judge + retrieval metrics | OPENAI_API_KEY + Java 21 + gated | medium | deferred until network policy |
| `orca_bench` | 755 | task judge (model) | OPENAI_API_KEY + Docker, ~133 GB/host | medium: the hub RPC returns full ground truth | waits for phase 2 (live backend) |

The leakage column is an observable, not a gate. The sandbox has full internet (harnesses
need their model APIs), and disallowing web tools does not stop a Bash-capable agent from
curling answers; prior work hit exactly this on held-out runs. The owner call for v0:
observe rather than pre-empt. If a harness opts to cheat, that is a finding, and the
record should show it. The runner captures per-cell network egress alongside the traces,
and held-out answers fetched from the public internet get documented, not prevented. hle
therefore runs in v0 with no egress allowlist, and its public answers make it the cell
most worth watching. orca_bench, when it joins, keeps its resolver RPC noted as an oracle
surface for the same observability treatment.

## The harness matrix

v0 pins models to make the comparison factorial rather than confounded: `claude_code` with
Opus 5, `codex` with GPT-5.6-terra, and `prime_agent` with each of the two. That 4-way gives
harness-vs-harness at a fixed model in both directions (claude_code vs prime_agent on Opus 5;
codex vs prime_agent on GPT-5.6-terra) and model-vs-model inside one scaffold
(prime_agent x both). `pi` and `hermes` join in v1 once the runner is proven.

Billing: subscription and usage-credit credentials are the PREFERRED mode for every cell,
and the runner must support both modes (subscription/OAuth and API key) per harness. Claude
Code logs in with the Claude subscription, codex with the ChatGPT subscription, and both
prime_agent cells run on the owner's subscription credentials rather than API inference spend
(prime_agent's `/login` stores "a subscription or an API key" per provider). Validation
status: the Anthropic OAuth path through prime_agent is verified end to end; the OpenAI
subscription path through prime_agent is validated so far only with an API key and needs its
subscription variant proven before the cell runs. Before any cell runs, the runner records
`prime-agent model list` (and each harness's version and resolved model) in the cell
manifest, so "which model actually answered" is part of the record, not an assumption.
Credential isolation per cell is mandatory: ambient logins were shown to mask bogus
credentials entirely, so every cell runs under an isolated HOME with a negative control.

Known per-harness hazards from prior runs:

- `codex`: unreliable over a single long loop (variable yielding plus a queue-draining rush
  mode). Its rollout must be supervised episodically (per-task relaunch or watchdog), not
  launched as one long run.
- `claude_code`: needs `IS_SANDBOX=1` to run bypass-permissions as root in the container;
  memory and skills load only at session start, which is fine here because eval sessions
  are fresh by design.
- `prime_agent`: no operational history in this program yet; install is the vendor script,
  not npm (the npm identity in its source tree installs Pi instead, per its own docs). Its
  model and credential resolution must be verified per cell (`prime-agent model list`)
  before any rollout spends budget.
- All three harnesses: research the docs and code for settings that promote autonomy from
  the session's first turn (permission bypass, full-auto modes, auto-continue behavior)
  and record the chosen settings per harness in the runner configuration; no cell may
  depend on a human approving anything mid-run.

## v0 = 4 envs x 4 harness-model pairs = 16 cells

- **Envs:** `automationbench`, `tau2_telecom`, `hle` (yc_bench moves to v1; hle swapped in by owner decision),
  and `tau2_banking_knowledge`, **amended in 2026-08-14 by owner decision** as the offline-eval
  variant of that domain (`bm25_grep` retrieval, `evaluation_type = "env"`) over a split
  restricted to the bases that evaluator honors.
- **Harness-model pairs:** claude_code+Opus 5, codex+GPT-5.6-terra, prime_agent+Opus 5,
  prime_agent+GPT-5.6-terra (the 4-way above).
- **Splits.** Two of the four are not ours to invent, which also answers where the numbers
  come from:
  - automationbench: reuse the published conversation-not-memories split exactly: the same
    **120 held-out tasks** recorded in shōrep, improvement pool drawn from the remaining 480.
    This keeps every v0 number directly comparable to the registered study.
  - tau2_telecom: tau2's own declared split at the pinned upstream sha (1d244f5d):
    **train 74 improvement / test 40 held-out**, served via the port's native
    `task_split_name` support, which refuses unsupported splits rather than silently falling
    back. Counts verified against upstream's `split_tasks.json` at that sha. 40 is below the
    N >= 60 power aim; honoring the benchmark's canonical split wins over inventing a larger
    leaky one, and the paired design still detects the large effects.
  - hle: no canonical split exists upstream (it is an eval set), so ours: seeded,
    disjoint, published: 120 held-out and an improvement pool of 300 as the serving
    ceiling (of 1726). 120 matches the automationbench held-out size, and hle's
    single-turn tasks make the before/after evals cheap relative to the multi-turn envs.
    With yc_bench in v1, the rollout-phase stopping metrics carry the charter's
    when-do-they-stop question in v0.
  - tau2_banking_knowledge: no canonical split exists upstream either, so ours: seeded,
    disjoint, published: **40 held-out / 47 improvement**, drawn over the 87 of 97 tasks whose
    reward basis the offline `env` evaluator scores in full. 40 matches tau2_telecom's held-out
    size. The nine ACTION-only tasks and the one carrying an NL assertion are out of the
    population because that evaluator does not score them, not because they are hard; serving
    them means the keyed full-fidelity cell, which is a v1 follow-up.
- **Replication arms are second readings, not new cells.** A cell whose name ends `-r2` reruns
  a cell of the matrix over an `_order2` split: the parent's membership on both sides, the
  held-out side in the parent's own order, and the improvement pool permuted under a recorded
  seed. The matrix is still the 16 above. An arm exists so an effect measured over one pool
  order can be read again over another, and it changes nothing about which tasks v0 holds.
- **Pool sizes are ceilings, not quotas.** The improvement pool is the maximum the runner
  will serve; the agent may stop on its own long before exhausting it. That early stop is a
  primary reported outcome (tasks attempted before stopping, and how the stop happened), not
  a protocol failure. Nothing re-serves tasks to push an agent to the ceiling.
- **Budget per cell:** fixed wall-clock for the rollout: 8 hours. No token ceiling; the
  cells run on subscription billing. Auto-continue on, and a session ended by provider
  usage limits is resumed automatically, so the stopping metrics count only the agent's
  own choice to stop, never an imposed cutoff. Both eval phases run every held-out task
  exactly once.

## Display

v0: a `results/` directory in this repo holding one JSON per cell (per-task scores, split
manifest, harness and model pins, home-subtree hash before and after) and a small script
that renders the summary table plus a forest plot of paired deltas with CIs. A styled
results page on shojin.dev consumes the same JSON later; nothing in v0 blocks on web work.

## Infrastructure (what has to exist before cell one runs)

1. A cell runner: container lifecycle, the three phases, trace collection, split
   enforcement. Serving uses shōgym's first-class classes (`shogym.serve.TaskStream` for
   the rollout queue, `EvalStream` for the eval phases, stood up via
   `build_stream_server`), not bespoke serving code. The original study's broker pattern
   survives only for what it was good at: container and credential lifecycle (wandb sink
   broker-side, keys injected at runtime, the agent keeps zero broker mounts).
2. Split manifests checked into shōbench: for each env, a committed file listing the
   exact task ids on each side of the split and the seed that produced them, so every
   rerun serves identical splits and what was held out is reviewable.
3. The reporting script. Paired bootstrap: resample the held-out tasks with replacement,
   recompute the mean before-to-after delta on each resample, and report the 2.5th and
   97.5th percentiles as the 95% CI. Paired because both evals score the same tasks,
   which cancels per-task difficulty out of the interval.
4. Per-harness episodic supervision for the rollout (the codex requirement, useful for all).

## Decisions

1. **Rollout budget parity:** wall-clock. Decided.
2. **The improvement instruction text:** the established env-agnostic "Get Better" prompt
   verbatim to start; iterate through PRs against this doc if the runs argue for it.
3. **Rollout repetitions:** one per cell in v0. A single rollout is one draw; repetitions
   are the v1 upgrade that turns per-cell claims from anecdote into estimate.
4. **Egress allowlist:** was a proposed per-cell network restriction (containers reach
   model API endpoints only) to prevent held-out answer fetching on public-answer envs.
   Dissolved by the observe-not-preempt decision above; nothing gates on it, and v1
   revisits it only if observed leakage argues for prevention (browsecomp_plus,
   orca_bench) or keep growing the simulated-env column first.
