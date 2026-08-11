# Adding a harness

A harness is one agent CLI the runner can drive autonomously and read a stop out of. Adding
one is four steps: copy the template, implement the methods, add one registry line, and write
the autonomy note. The code surface is small on purpose. Most of what a harness needs is
already in the base class, so a new harness supplies only what it genuinely does differently:
its launch argv and its stop-classification rules.

## The four steps

1. **Create the file.** Copy `src/shobench/harnesses/_template.py` to
   `src/shobench/harnesses/<your_harness>.py`. The template is an unregistered skeleton with
   every method present and a note on each. Pick a `name` in lower_snake_case; it is both the
   value cells carry and the registry key.

2. **Implement the methods.** Fill in `launch`, `classify`, `version_probe`, and whichever of
   the optional methods your harness needs (see the table below for what each is for). Crib
   from the closest of the three real harnesses: `claude_code.py` if the harness has a system
   prompt channel and takes a caller-chosen session id, `codex.py` if it runs one turn and
   mints its own id, `prime_agent.py` if autonomy is a set of budgets and the MCP wiring lives
   in HOME. Do not reach for shared abstraction the base class does not already give you: the
   launch argv and the stop rules are meant to differ per harness, and forcing them into one
   shape hides the differences that matter.

3. **Add the registry line.** In `src/shobench/harnesses/__init__.py`, import the class and add
   an instance to the `_REGISTRY` tuple. That one line is what makes `harness_for("<name>")`
   resolve, and a cell naming the harness load. Add the class name to `__all__` if anything
   outside the package imports it.

4. **Write the autonomy and stop-classification note.** Add a section to
   `docs/harness-autonomy.md` covering, with a citation for each: what makes the harness
   autonomous from its first turn (the flags, and anything that has to be bypassed), where the
   standing instruction goes, and how a usage limit is told apart from a chosen stop. Every
   flag and every stop signal names where it came from, marked **observed** (produced on this
   machine and captured), **source** (read out of the installed binary), **docs** (official
   documentation), or **unverified** (inferred, and flagged as such). A rule nobody can trace
   is a rule that gets something wrong quietly.

Then prove it changed nothing else: run the suite (`pytest`) and `ruff check`. If the harness
is one of the v0 three, the characterization test in `tests/test_harness_characterization.py`
pins its launch spec and verdicts; a new harness is worth pinning there the same way, so a
later edit to shared machinery cannot move it without a failing test.

## What the runner asks each harness, and why

The runner drives every harness through the interface in `src/shobench/harness.py`. Each row
is a method or attribute the runner reads, what it does with the answer, and why the answer
matters. Required means the base class raises or returns nothing usable, so a harness must
supply it.

| Method or attribute | What the runner does with it | Required |
|---|---|---|
| `name` | Keys the registry and labels the cell manifest. | yes |
| `launch(...)` | Builds the argv, env, and files for one autonomous invocation, returned as a `LaunchSpec`. This is the harness's whole autonomy story. | yes |
| `classify(...)` | Reads how a leg ended into a `StopVerdict`: a `CHOSEN` stop (the only kind the stopping metrics count), a `USAGE_LIMIT` (resumed, not counted), a `LEG_TIMEOUT`, an `ERROR`, or `UNKNOWN`. Its evidence must never quote the leg's raw stderr: a harness can mint a credential and overwrite it inside one leg, so those bytes are ones nothing downstream can name. Call `stderr_evidence(stderr_path)`, which describes the file instead. | yes |
| `version_probe()` | Records the installed harness version in the manifest. | yes |
| `usage_limit_rules` | The evidence-based rules `classify` matches against named artifacts via `_match_usage_limit`. Each names where it read and cites its source. | if the harness has a usage limit |
| `pins_session_id` | Says whether the runner may choose the session id before launch. False means the harness mints its own. | default False |
| `session_id_from_trace(...)` | Reads back the id a leg actually ran under, so a resume targets that session and not a fresh one. Needed when `pins_session_id` is False. | if resumable and self-minting |
| `base_env()` | Environment every invocation needs, credentials excluded. Defaults to a clean NODE_OPTIONS; override only to add to it. | default clean NODE_OPTIONS |
| `model_probe()` | An optional command reporting which model the harness resolved, for the manifest. | optional |
| `observed_models(...)` | Model identifiers the trace shows actually answered, for the manifest's record of which model answered rather than which was requested. | optional |
| `reports_observed_models` | Whether the trace names a model at all. Declared rather than inferred, because an empty list from a harness that reports models means none answered, while an empty list from one that does not means nothing was measured; the manifest publishes which of the two it is. | default False |
| `effort_flag` | How the cell's reasoning effort reaches the harness, empty when it has none. The manifest records the requested effort either way, and records separately whether it was applied, so a cell cannot claim a controlled variable the harness dropped. | default none |
| `runner_owned_home_files` | The HOME paths the runner rewrites on every leg. They are excluded from the durable-self digest, and must be exactly the keys of the `home_files` a launch returns, which a test holds. | default none |
| `home_seed_files()` | The HOME assets a cell starts with. The runner places them once, before the baseline digest, and `launch` returns the same mapping. | default none |

Three `LaunchSpec` fields decide where a harness's config lands, and it is a real choice.
`config_files` go to a read-only mount outside the agent's working directory, so a harness
config never becomes part of what the agent thinks of as itself. `home_files` go into the
cell's isolated HOME, for a harness that reads its config only from there; that is inside what
the agent can edit, which is the cost of the harness having no config flag. Every leg rewrites
them, so what belongs there is what only the runner knows and what moves between legs, the
stream endpoint being the example, and for that reason they are excluded from the durable-self
digest: a file the runner overwrites on every launch cannot carry anything the agent chose to
keep, and counting it made a cell that wrote nothing publish a changed durable self. Declare the
same paths in `runner_owned_home_files` so the digest knows about them before any leg has run.
`home_seed_files` land in the same HOME but are written only when absent, an initial condition
the agent then owns: an asset the agent may improve belongs there, because a rewrite would
restore the shipped bytes over the agent's version in the gap between the rollout and the
evaluation that exists to read it. Return them from `home_seed_files()` as well, because the
runner places them before it takes the baseline digest; seeded on the first leg instead, they
land on the far side of the baseline and read as something the rollout wrote. `stdin` defaults
to `/dev/null`, which is what every v0 harness needs; set it only for a harness that reads its
prompt from stdin.
