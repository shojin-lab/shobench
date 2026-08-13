# Harness autonomy and stop classification

Two questions decide whether a cell is measuring what it claims to. First, is the harness
autonomous from its first turn, since no cell may depend on a human approving anything
mid-run. Second, when a run ends, did the agent choose to stop or did something stop it,
since the scope's stopping metrics count only the agent's own choice.

This document records the answers per harness, the settings the runner uses, and where each
claim came from. Evidence classes appear inline: **observed** means the behavior was produced
on this machine and captured, **source** means it was read out of the installed binary or
bundle, **docs** means official documentation, and **unverified** means it is inferred and
flagged as such.

Versions this was established against: Claude Code 2.1.221 and 2.1.226, codex-cli 0.145.0 and
0.147.0, prime-agent 0.7.0 and 0.7.1. The image pins 2.1.226, 0.147.0, and 0.7.1.

## The shape of the problem

None of the three harnesses runs for eight hours on its own. Claude Code runs its agentic
loop until the model emits a turn with no tool calls, then exits. codex exec runs exactly one
turn. prime-agent runs one turn unless autonomous mode is on, and then runs until one of four
host budgets stops it. Where each one ends is not a gap to paper over: it is the measurement.
So the rollout is one invocation against one live stream and the runner never relaunches a
harness that stopped, because a relaunch would turn "gave up after four tasks" into "worked
through the pool".

The one interruption that is not the agent's own is a provider usage limit, and it gets the
one exception. The cell **suspends**: a record says where the rollout stood, the containers
stop, every directory stays, and `uv run shobench resume --run <run-dir> --go` continues the
same session against the same stream for what is left of the rollout clock. `eval_after` waits
for a real terminus, so no measurement is ever taken inside an exhausted window.

Suspending has to leave the process without unwinding, and that is the one place in the runner
where the tidy thing is the wrong thing. shogym's orderly close drains the task in flight into
a scored row, and its resume replays only queue positions that hold no row, so closing politely
would spend the task the agent was working on and no continuation could serve it again. Ending
hard leaves the claim on disk and the position row-less, which is exactly the state `resume`
exists to reclaim (observed, against the pinned shogym: a stream ended without unwinding leaves
no row and replays the position, while the same stream closed in an orderly way leaves a
`drained` row and the position is gone for good). The exit itself is in a `finally`, so a
console that will not take the message and a daemon that will not stop a container cannot turn
the suspension back into the orderly close it exists to avoid.

What a continuation is allowed to be is the other half of the design, because the point of
resuming is a comparable measurement rather than a finished job. Its clock is the budget the
interrupted rollout was given, read from the record, so a cell file edited while the run waited
cannot lengthen a rollout that is already half spent. Its cell, split, and instruction are
checked against the digests the manifest recorded before anything spent, and a difference is
refused by name rather than reconciled. The phases already measured are read back off the run
directory and published with the new ones, so a continued cell reports the same paired
before-and-after as one that ran straight through. And the suspension record is spent only once
the results are written, so a continuation that fails on a missing dataset variable leaves the
cell exactly as resumable as it found it.

None of the three has a usage-limit exit code. All three need text or event classification,
and the rules below are what the runner implements.

## claude_code

### Invocation

    claude -p <kickoff> --model <model>
      --mcp-config /cfg/claude.mcp.json --strict-mcp-config
      --setting-sources ""
      --permission-mode bypassPermissions
      --append-system-prompt <standing instruction>
      --forward-subagent-text
      --output-format stream-json --verbose --include-partial-messages
      [--session-id <uuid> | --resume <uuid>]

`-p` is the headless switch and it also skips the workspace trust dialog. `--setting-sources ""`
is what keeps the initial conditions honest: without it a settings file the image or the home
happens to carry changes the run invisibly.

The standing instruction goes in `--append-system-prompt`, not the user turn. A long rollout
auto-compacts, and anything left in the user turn can be summarized away, which would silently
drop the objective mid-run. The system prompt survives compaction.

Two flags the runner deliberately does not use. `--bare` never reads OAuth credentials or
`CLAUDE_CODE_OAUTH_TOKEN` (docs), so it cannot run a subscription cell. `--tools ""` strips MCP
tools along with the built-ins, so it would remove the task stream itself.

### Permission bypass

`--permission-mode bypassPermissions` is documented as equivalent to
`--dangerously-skip-permissions`. `dontAsk` is not a substitute: it permits only pre-approved
tools and silently withholds everything else. `auto` is worse in headless mode, where repeated
classifier blocks abort the session outright because there is no human to prompt.

The container runs as root, and the CLI refuses bypass as root unless `IS_SANDBOX=1` is set.
That check is in the binary (source), with the exact stderr line
`--dangerously-skip-permissions cannot be used with root/sudo privileges for security reasons`
and exit code 1. The variable is **undocumented officially**, so it can change in any release;
the runner's credential negative control exercises the containerized bypass path on every cell
start, which is what turns a future removal into a loud failure at the start of a cell rather
than a confusing one in the middle. During development this check earned its place: the first
negative control run failed for exactly this reason and would otherwise have looked like a
credential result.

### Resuming a suspended run

The session id is on the first line of the stream (observed):
`{"type":"system","subtype":"init","session_id":"...","mcp_servers":[...]}`. The runner pins it
with `--session-id <uuid>` before launch instead, so a run that dies immediately is still
resumable, and continues it with `--resume <uuid>`. The id is written into the run's record as
well as the suspension, since the process that knew it is gone by the time anyone resumes.

Resume has a trap worth stating plainly (docs): `bypassPermissions` is never restored on
resume, and a session that depended on `--mcp-config`, `--settings`, or `--add-dir` needs them
passed again. The runner rebuilds the whole argv every time it launches, so every flag is
always re-passed.

### Usage limit versus chosen stop

A clean finish, observed:

    {"is_error": false, "stop_reason": "end_turn", "terminal_reason": "completed",
     "subtype": "success", "api_error_status": null, "result": "..."}

exit 0. A bad token, observed:

    {"is_error": true, "stop_reason": "stop_sequence", "terminal_reason": "api_error",
     "subtype": "success", "api_error_status": 401,
     "result": "Failed to authenticate. API Error: 401 Invalid bearer token"}

exit 1. **`subtype` is `"success"` in both.** It is not an error discriminator and the runner
never branches on it. The fields that matter are `is_error`, `terminal_reason`, and
`api_error_status`, and the last of those carries the HTTP status, which is how a 429 will
appear.

A subscription limit lands as HTTP 429 and produces a message from a known family (source and
docs agree): `You've hit your session limit · resets 3:45pm`, and the same shape for the
weekly, Opus, Sonnet, and usage-credit limits. There is also a mid-stream signal,
`{"type":"system","subtype":"api_retry", ..., "error":"rate_limit"}`.

One correction worth recording because the guess is natural and wrong: `terminal_reason:
"blocking_limit"` is **not** the subscription limit. It is the prompt token limit, so it is a
context problem, and resuming on it would loop. The runner classifies it as an error.

The rule the runner applies, in order:

| Condition | Class |
|---|---|
| `api_error_status == 429`, or `result` matches `you've hit your ... limit` | usage limit, resume |
| exit 0 and `is_error == false` | chosen stop |
| `terminal_reason` in `max_turns`, `budget_exhausted`, `tool_deferred`, `background_requested` | chosen stop, bounded |
| `terminal_reason` in `blocking_limit`, `prompt_too_long` | error, context not quota |
| no result event at all | error |
| anything else | error |

### Credentials

Precedence (docs): Bedrock and Vertex and Foundry environments, then a gateway, then
`ANTHROPIC_API_KEY`, then `apiKeyHelper`, then `CLAUDE_CODE_OAUTH_TOKEN`, then the subscription
OAuth from `/login`. In non-interactive mode an `ANTHROPIC_API_KEY` present in the environment
is always used, which is why a subscription cell must not carry one.

| Mode | Variable | File |
|---|---|---|
| subscription | `CLAUDE_CODE_OAUTH_TOKEN` | Linux `~/.claude/.credentials.json` (0600); macOS keychain |
| api key | `ANTHROPIC_API_KEY` | none |

`CLAUDE_CONFIG_DIR` relocates the whole configuration surface, which is an alternative to the
bind mount the runner uses; the bind mount is what has actually been run in this program.

### MCP

    {"mcpServers": {"shogym": {"type": "http", "url": "http://host.docker.internal:PORT/mcp"}}}

The `type` field is load-bearing (docs): an entry with a `url` and no `type` is read as a stdio
server, skipped, and the run continues toolless and exits cleanly. The `system/init` event
carries `mcp_servers` with a per-server status, which the runner records; during the smoke run
this is exactly what showed a leg had connected to nothing.

## codex

### Invocation

    codex exec [resume <thread-id>] --json -m <model>
      --dangerously-bypass-approvals-and-sandbox
      --skip-git-repo-check
      -c mcp_servers.shogym.url="<url>"
      -c mcp_servers.shogym.default_tools_approval_mode="approve"
      -c mcp_servers.shogym.required=true
      -c mcp_servers.shogym.startup_timeout_sec=60
      -c mcp_servers.shogym.tool_timeout_sec=900
      -c cli_auth_credentials_store="file"
      <prompt>
      </dev/null

codex exec has no separate system-prompt channel, so the standing instruction is prepended to
the turn. The bytes are identical to every other harness's system prompt and the manifest
records the digest, so the difference in placement is visible rather than hidden.

### Approvals and sandbox

Approvals need no flag: exec already defaults to never asking (source,
`// Default to never ask for approvals in headless mode.`). `-a/--ask-for-approval` is not an
exec flag at all; it exists only on the interactive root command.

The sandbox does need a flag. `codex exec` defaults to a read-only sandbox (docs), so without
the bypass the agent cannot write anything durable about itself, which would silently make the
rollout unable to produce the artifact the benchmark measures. In a container the docs endorse
full access explicitly.

Two settings prevent silent failures. `default_tools_approval_mode = "approve"` is mandatory:
without it every MCP call comes back cancelled and the agent concludes its tools are broken;
`"auto"` is not the same thing. `required = true` makes a broker that fails to initialize an
exit rather than a toolless run.

The default MCP timeouts, 10 seconds to start and 60 seconds per call, are far too tight for a
first `get_task` that pays for a cold environment and a dataset load, so both are raised.

Project trust fails silently (source and prior art): an untrusted project's `.codex/config.toml`
is skipped with no error, and `-c projects."<path>".trust_level` does not help because trust is
resolved before `-c` overrides apply. The runner therefore declares MCP inline with `-c` and
never relies on a project config file.

### Resuming a suspended run

One invocation is one turn, so codex is the harness whose rollout ends earliest on its own,
and that ending is the finding rather than something to loop around. A run a usage limit
suspended is continued on the same thread with `codex exec resume <thread-id>`, which takes the
subcommand before the flags and accepts neither `-s/--sandbox` nor `-C/--cd`. That is why the
sandbox is opened with the bypass flag rather than `--sandbox`: the bypass flag is accepted by
both forms.

The thread id is the first event (observed): `{"type":"thread.started","thread_id":"..."}`.

### Usage limit versus chosen stop

A clean turn ends with `{"type":"turn.completed","usage":{...}}` and exit 0 (observed). A
failure ends with `{"type":"turn.failed","error":{"message":"..."}}` and exit 1 (observed
against a bad key).

The important subtlety: intermediate `{"type":"error"}` events are retry chatter, not
terminal. codex retries transient failures internally and only a non-retryable one sets the
failure flag (source, the `!payload.will_retry` guard). A classifier that matched any error
event would false-positive constantly, so the runner reads only the terminal `turn.failed`
event's message.

Usage limits arrive in that message. The invariant substring is `You've hit your usage limit`
(source, `CodexErr::UsageLimitReached`), and the reset time is appended as `Try again at ...`
or `or try again at ...`, so it is parseable. Related arms cover `out of credits` and
`spend cap`. The backend discriminator is `error_type == "usage_limit_reached"` with header
`x-codex-rate-limit-reached-type`. There is no dedicated event type and no distinct exit code.

One more case: an **interrupted** turn emits neither terminal event and still exits 1 (source),
so a missing terminal event is an interruption, not a stop. The runner classifies it as an
error.

**Unverified:** no live usage-limit event has been captured for codex. The rules are source
verified at the pinned tag, and the runner records the full terminal payload unconditionally
so the first real occurrence documents itself.

### Credentials

`CODEX_HOME` relocates the whole state surface, and the directory must already exist (docs).
`cli_auth_credentials_store = "file"` is required in a container, which has no OS keyring.

| Mode | Variable | File |
|---|---|---|
| api key | `CODEX_API_KEY` (exec only, single run) | none needed |
| api key, alternative | `OPENAI_API_KEY` piped into `codex login --with-api-key` | `$CODEX_HOME/auth.json` (0600) |
| subscription | none | `$CODEX_HOME/auth.json`, minted by a browser login and copied in |

`OPENAI_API_KEY` alone does not authenticate codex. The runner uses `CODEX_API_KEY` for api-key
mode because it needs no login step, and keeps the piped login available for a version that
drops the variable. A discipline worth inheriting from the prior study: strip `OPENAI_API_KEY`
from a subscription run's environment so it can never silently fall back to the billed key.

## prime_agent

### Install

The vendor script, never npm. `registry.npmjs.org/prime-agent` returns 404, and
`@earendil-works/pi-coding-agent` is Pi, a genuinely different agent with different tools and a
different repository. The docs state the inherited npm workspace names in the source tree are
implementation details and not the public install path. The script downloads a
checksum-verified release tarball and hands it to `npm install -g`, so the command still lands
in `/usr/local` and survives the isolated HOME being mounted over `/root`.

`PRIME_AGENT_BOOTSTRAP_KERNEL_ON_INSTALL=1` bakes the IPython kernel at build time, and the
image sets it. Without it the first session bootstraps a kernel venv and needs the network to
do it, which would put a package install inside the first measured task. The variable is worth
setting even though a build would bake the kernel anyway: unset, it is a prompt the installer
asks, and a build with no terminal takes the yes branch on its own (source). That is the right
outcome arrived at by default rather than by declaration, so the image declares it.

### Invocation

    prime-agent -p --mode json --provider <provider> --model <model>
      --autonomous
      --autonomous-max-continuations <large>
      --autonomous-max-turns <large>
      --autonomous-max-tokens <large>
      --autonomous-timeout-ms <past the run's own wall clock>
      [--resume <id>] -- <prompt>
      </dev/null

The provider is always explicit, from an exact model-to-provider map in the harness
(`claude-opus-5` -> `anthropic`, `gpt-5.6-terra` -> `openai-codex`); a model outside the map
stops the launch. Observed on 0.7.1: a bare `gpt-5.6-terra` resolved to `azure-openai-responses`
and died with "No API key found" while the openai-codex login sat unused, and the map is exact
rather than by prefix because an explicit provider disables the catalog check, so an absent id
launches through the custom-model fallback instead of refusing. The credential probe passes the
same explicit provider, so it exercises the leg's exact resolution path.

### There is nothing to bypass, and that is the finding

prime-agent has no permission prompt, no approval policy, and no sandbox. Its own docs say
workers are process-isolated for failure containment and are not security-sandboxed. Approval
gating exists only as an optional user-written extension. The container is the only boundary,
which is what this runner already provides.

A consequence worth recording: there is no way to fence affordances either. The MCP stream
lives inside the single `ipython` tool, so removing tools removes the stream. For this
benchmark that is fine, because the scope observes leakage rather than gating it.

### Autonomy is a budget problem

Autonomous mode starts disabled, and enabling it brings four budgets whose defaults are all far
below an 8-hour rollout: 3 continuations, 12 turns, 80,000 tokens, and a 30-minute wall clock.
The 30-minute default alone would end every rollout. The docs are explicit that reaching a
limit "does not imply task success", so a run that ends at one has been cut off rather than
finished. Recording that as the agent's own stop is exactly the confound the scope forbids,
which is why the runner raises all of them and classifies a limit that was still reached as a
cutoff.

Value-taking autonomous flags take a separate argument; `--flag=value` is rejected.

### Usage limit versus chosen stop

**The exit code is not usable.** Every model-level failure sets exit 1 only in text mode
(source), so a `--mode json` run that errored still exits 0. Gate failures and thrown
exceptions do exit 1 in json mode; model errors do not. The runner classifies from the event
stream.

The last assistant message carries `stopReason` in `stop`, `length`, `toolUse`, `error`,
`aborted`, plus a structured diagnostic when a stream failed:

    diagnostics: [{"type": "provider_stream_failure",
                   "details": {"kind": "rate_limit", "status": 429, ...}}]

with `kind` drawn from `refusal`, `safety`, `overloaded`, `rate_limit`, `server_error`, `auth`,
`invalid_request`, `malformed_response`, `unknown`, and a 429 or a provider type containing
`rate_limit` or `throttl` mapping to `rate_limit`. This is the cleanest structured rate-limit
signal of the three harnesses. prime-agent also auto-retries retryable failures up to three
attempts, emitting `auto_retry_start` and then `auto_retry_end` with `success: false` and a
`finalError`, so the runner waits for `agent_end` rather than classifying on a retry.

The rule:

| Condition | Class |
|---|---|
| `diagnostics[].details.kind == "rate_limit"`, or `Provider rate limit exceeded` | usage limit, resume |
| stderr names an autonomous limit (`maxContinuations`, `maxTurns`, `maxTokens`, `timeoutMs`) | cutoff, not a stop |
| last `stopReason` in `stop`, `toolUse` | chosen stop |
| last `stopReason == "length"` | chosen stop, output cap |
| last `stopReason` in `error`, `aborted` | error |
| `agent_end` with no readable stop reason | unknown, reported as itself |
| no `agent_end` | error |

### Credentials, and a finding that touches the scope

Environment variables map per provider: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`PRIME_API_KEY`, and so on. `~/.prime/agent/auth.json` holds either api keys or OAuth tokens
from `/login`, at mode 0600, and **auth file credentials take priority over environment
variables** (docs). A stale auth.json in a cell HOME would therefore shadow an injected key,
which a pristine per-cell HOME prevents by construction.

`PRIME_AGENT_CODING_AGENT_DIR` relocates settings, auth, sessions, skills, prompts, logs, and
models in one move, which is the cleanest isolation knob of the three harnesses.

**The finding the scope should hear, now observed rather than suspected.** A
`CLAUDE_CODE_OAUTH_TOKEN` does not authenticate prime-agent. Run in the cell's isolated HOME
with that token present in the environment, prime-agent answered:

    No API key found for anthropic.
    Use /login to log into a provider via OAuth or API key.

So the Anthropic leg is in the same position as the OpenAI one: it needs an interactive
`/login` on the host, and the resulting `auth.json` copied into each cell's HOME. Until that
login happens, the only credential prime-agent would accept is `ANTHROPIC_API_KEY`, which is
api spend rather than subscription allowance and is therefore not what the scope asked for.
The runner marks both prime_agent legs pending and blocks neither of the other cells on them.

For contrast, the same protocol on codex passed on the first attempt. `~/.codex/auth.json`
already carries `auth_mode: "chatgpt"` on this host, so the ChatGPT subscription leg is
validated end to end today: a bogus auth.json produced repeated 401s against the responses
endpoint, and the real one produced `SHOBENCH-OK` with exactly the clean-completion event
sequence this document predicts, `thread.started` then `turn.started` then `item.completed`
then `turn.completed`.

### MCP: http only, and a bearer token is mandatory

    {"mcpServers": {"shogym": {"type": "http", "url": "<url>",
                               "bearerTokenEnvVar": "SHOBENCH_MCP_TOKEN"}}}

Only `http` servers are honored; a stdio entry is dropped with no error, so it produces an
integration that quietly is not there. The client resolves a bearer token before every
connection and refuses to open a session without one, even against a server that ignores it,
so the variable must be set to some non-empty value in the agent's environment. The runner
sets it.

**The skill, now wired.** Declaring the server is only half of what prime-agent needs. Each
integration also requires a Python skill package under `.prime/agent/skills/<name>/` that is
installed into the kernel venv at session start, and a new Python-backed skill needs a fresh
session to be picked up. The `shogym-stream` skill is vendored under `prime_agent/skills/` and
installed into each prime_agent cell's isolated HOME beside the settings entry
(`harnesses/prime_agent.py`, `shogym_stream_skill_files`). It is vendored rather than copied
verbatim from shogym's `examples/prime_agent` because it has to carry the runner's own token
variable (`SHOBENCH_MCP_TOKEN`), which the settings entry names too; a test asserts the two
agree and that the served stream exposes the tools the skill enumerates.

The two files land on different HOME channels, and the difference is measurement rather than
plumbing. The settings entry is per-leg: the endpoint moves between phases and between
concurrent eval tasks, and an eval task's HOME is a copy of the rollout's, so its inherited url
names a server that is gone. The skill package is seeded once and never rewritten, because a
rollout is free to improve it like any other durable artifact and the eval that follows has to
read what the rollout left. A leg that restored the vendored bytes would delete the improvement
in the moment before the session meant to measure it.

**What a tool call returns, since the natural guess is wrong.** `McpIntegration._parse_result`
prefers a result's `structuredContent` over its text content (source), and shogym's stream
declares an output schema for its own control tools and none for the tools an env publishes
(source). So `get_task()` and `queue_info()` arrive as dicts and a task's tools arrive as JSON
strings (observed, against a live stream over http with the runtime's own parser). The
inherited quickstart wording, "every one of them returns a JSON string", would have made the
first line of the documented loop raise `TypeError`; SKILL.md now documents the split and a
test pins which side of it each tool falls on. A missing skill package is likewise a launch
error rather than an empty mapping, because the alternative is a healthy prime-agent that can
reach nothing and a record that reads as an agent which chose to do no work.

That contract holds only under the MCP version the kernel resolves, which is why the skill pins
one. The bundled runtime reads the 1.x result model's `structuredContent` and `isError`, and
mcp 2.0 renamed both to `structured_content` and `is_error` (observed: the 2.0 model handed to
the pinned runtime's parser returns the text of a structured result, and never raises on a
result flagged as an error, so a failed call comes back looking like an answer). Nothing else in
the kernel venv bounds it, since `prime-agent-runtime` declares no MCP dependency at all
(source), so the skill's own requirement is all that stands between a fresh kernel and the
newest release. It is pinned to the 1.x line, floored where `structuredContent` first appears
(1.9 has no such field, 1.10 does). Bumping the harness pin to a runtime that speaks the 2.x
model is what unpins it.

What that leaves is only the credentialed end-to-end check: prime-agent bootstrapping the
kernel, installing this package, importing it as `shogym_stream`, and pulling a task. That needs
the interactive login this host does not yet have (see Credentials above), so it waits on the
same login as the rest of the prime_agent leg, not on more wiring.

## Forking the rollout into eval_after

`eval_after` measures the agent with its lived rollout state, so under the default
`eval_context = "resumed"` each held-out task resumes the rollout's terminal session rather
than starting cold. The forks are independent by home isolation, not by session machinery:
every task's throwaway HOME copy carries the harness's recorded session state (the
`session_state_dirs` each harness declares) beside the durable self, and each launch resumes
the rollout's session id inside its own copy. The runner refuses the whole phase up front when
the recorded session id or its transcript is missing, because a phase that quietly ran cold
would publish a mislabeled measurement; `eval_context = "cold"` on the cell is the recorded
ablation, and eval_before is always cold since no conversation exists yet.

The standing instruction follows the mode. A resumed fork launches with the ROLLOUT
instruction (`rollout.system.txt`), not the eval one: the rule that the eval instruction
never carries the improvement objective was designed for cold measurement, and a resumed
conversation already carries the objective in its history and its compaction summaries, so
swapping the standing instruction mid-conversation would measure an agent that never
existed. The resumed after measures the agent as it lived, objective included. Every cold
session (eval_before always, cold afters) keeps the blind `eval.system.txt`, and the
manifest's instruction block records which one eval_after launched with
(`instruction.eval_prompt_used`), so the artifact says it rather than leaving a reader to
derive it from the axis.

Runs measured before the resumed default get their bookend through `shobench rebookend`: a
NEW run (fresh unique id, its own lock) that copies the source's accumulated home, inherits
the source's recorded axes with `eval_context` forced to resumed, and runs exactly the
resumed eval_after against the source's terminal session, resolved from the source's own
stopping record and validated by the same preflight inside the copied home. The source run
is read and never written: it is an archived artifact, and the new run names it in a
`rebookend` provenance block and publishes under the incomplete name, since a run with no
eval_before cannot account for the before side; pairing with the source happens post-hoc.
The published artifact is named by the BOOKEND's run id, never the cell name. The cell-name
artifact is the source's own measurement, the one the bookend exists to pair with, and the
one-artifact-per-cell rule (`write_results` replaces the stem's other shape) would have
destroyed it; the run-id stem keeps the source result and every bookend of every source
coexisting, self-paired through the provenance block, and it makes the leaf unpredictable
before the run mints its id, so nothing can pre-occupy it, with the minted names still
bounded against the source before the lock; the stem rides through every reopening (a
resumed or repaired bookend republishes under its own run id, never the cell name). The
marker carries two identities, deliberately: `rebookend_of` is the terminal-state lineage
(the run whose conversation this bookend resumes), and `baseline_run_id` is the pairing
identity (the run whose eval_before rows the report joins with this bookend's after rows).
They coincide only when the source measured its own before-side, which is the stated
default; a rollout-only or after-only source (every v0 source, whose baselines are separate
deferred runs) REQUIRES `--baseline`, validated before anything spends: a real run, not
itself a bookend, the same cell, the same split id digest (the load-bearing check, because
before rows over different held-out ids would pair task numbers that are not the same
tasks), and carrying eval_before provenance of its own. The artifact is SELF-CONTAINED: at
creation the baseline's eval_before rows are read from its run directory (refusing before
any spend when none are readable) and persisted into the bookend's own run directory, and
every publication of the bookend, first or reopened, publishes those rows as its own before
block, labeled `eval_before.source_run_id`, so the paired delta lives inside the artifact
and names both runs it derives from. That matters because baseline artifacts live under the
shared cell stem and are routinely evicted by later same-cell publications; a bookend that
depended on the baseline's artifact stopped assembling the day another run of the cell
published. The reporter therefore needs no join for a self-contained bookend; the join
survives only for LEGACY bookends published before the carry, where a baseline missing from
the loaded files still surfaces explicitly as BASELINE MISSING and a bookend-shaped
baseline as INVALID PROVENANCE, never silently. Every reported row carries its run id, its
arm axes (pre-axis artifacts render their recorded arms, never plus cold, exactly as the
recorded-axis helpers define absence), and its pairing. Chains are refused, not walked: a bookend of a bookend re-measures the same
terminal state as rebookending the original directly (the bookend's home is the source's
terminal home, copied, and its own eval_after advanced no rollout), so the runner and the
plan refuse a source that is itself a rebookend, naming the original to bookend instead, and
the reporter independently labels any artifact whose named source is a bookend (a chain or a
cycle) INVALID PROVENANCE rather than assembled, because a bookend's before-side does not
exist to pair against. The source's lock file is held SHARED for
exactly the duration of the snapshot: shared refuses a live mutator and blocks a would-be
one (every mutating owner takes the lock exclusive) without writing a byte, and it is
released before any spend. A source WITHOUT a lock file is refused as unholdable rather than
copied unheld, because every mutator creates the lock on its way in and would write mid-copy
under an empty hold; creating the lock from the rebookend would itself write into the
archive, so the refusal names the operator's own one-file workaround, and every run since
the lock landed carries one, so only pre-lock-era directories are affected. Under the hold,
everything the plan relied on is re-proven before the copy: a suspension written without a
manifest change refuses, a terminus that stopped naming a terminal session refuses, and a
rewritten manifest refuses, so a mutation that ran whole between the plan read and the hold
cannot be snapshotted under the old definition.
The snapshot is MATERIALIZED, transcripts included: every symlink becomes the bytes it
names, resolved from the link's own parent (a valid relative link, the shape of every link in
the real prime homes, materializes wherever the process happens to run; the stdlib copy read
those as dangling and silently dropped them). Nothing in the new tree references the source,
so no later writer (the credential reseed was the demonstrated case) can reach the archive
through a preserved link; a genuinely dangling link drops, and a link resolving outside the
source home, a cycle, or a special file refuses loudly. Publication is atomic everywhere
(``write_results`` swaps the leaf entry in with ``os.replace``), and a rebookend additionally
refuses up front any output directory, result leaf, or new run path that resolves into the
source, so a link planted at a deterministic name is refused or replaced, never followed.
The drift check a bookend applies compares the recorded cell block FIELD BY FIELD rather than
by the cell file's digest, and it permits no drift at all. A resume and a rerun-eval write more
of a measurement that already exists, under the run id that already names it, so they refuse on
the digest: nothing about that run may change. A rebookend publishes a NEW run over a rollout
that is finished and immutable, so an edit made after that rollout ended cannot reach it, and
the digest is the wrong question anyway, since it moves for a comment as readily as for a
swapped model (the real case: an eval timeout retuned after two rollouts finished refused both
of their bookends). What replaces the digest is not permission but INHERITANCE. Three fields
come from the record rather than from the checkout: `rollout_feedback`, the arm the source's
rollout served; and `eval_task_timeout_s` and `eval_concurrency`, the eval runtime its before
side was scored under. The runtime pair matters as much as the arm, because the timeout is both
the held-out task stream's deadline and the leg's hard stop, so a bookend run under a shorter
bound would force-stop a task that the before side had time to seal, and the paired delta would
measure the bounds as much as the agent; concurrency is not score-neutral either, since
concurrent legs share the host, the network and the provider account while their per-task clocks
run, so contention and throttling become timeouts and unscored rows. Per-task homes isolate task
STATE, not resources. `eval_context` is pinned to resumed, being the one axis a rebookend exists
to change. Everything else refuses on any difference: `env`, `harness`, `model`, `effort`,
`credential_mode` and `env_kwargs`, which are what its eval sessions are made of; `max_in_flight`,
`rollout_wall_clock_s` and `pool_ceiling`, which define the rollout it inherits a home from and
would otherwise be published as numbers nobody ran; `required_env`, a precondition rather than a
measurement, refused because a cell that needs a key it did not need before is a cell whose legs
are produced differently; `name`; and the split's id digest and both system-prompt digests, which
are compared under either scope. The inherited fields refuse too, because the comparison runs
against the cell the bookend will actually run: it is what holds the inheritance to its word, and
a record with no runtime to inherit refuses rather than lending the file's, since an absence is
not a value. The comparison walks the UNION of the recorded and current field sets with an
explicit absence sentinel, so an axis added to the cell after a run was recorded surfaces instead
of passing (comparing only recorded fields let every historical manifest through no matter how a
new axis was classified), and a field the cell has since dropped surfaces the same way; only the
two versioned legacy axes read absence as a value, absent `rollout_feedback` as never and absent
`eval_context` as cold. The uncompared remainder is the bookkeeping that never reaches a session:
the lookup keys `split` and `instruction_arm`, whose identities are the digests, plus
`config_path`, `note` and `config_sha256`. The pairing gets the same rule as the ids: a named
`--baseline` whose recorded eval runtime differs from the source's is refused, because the before
and after sides would be scored under two stopping rules. None of this is absorbed silently. The
bookend's manifest carries `rebookend.eval_runtime_from_record` (the values that bounded its
legs), the source's recorded cell block (`rebookend.source_cell`) beside the block it ran under
(`cell`), and `rebookend.cell_drift` naming every field the checkout's file states differently
with both values, and the plan prints the same before anything spends, so a reader of the numbers
sees which rule scored them and what the file would have run instead without finding two
checkouts to diff. A repaired bookend keeps the whole-digest comparison, deliberately:
`rerun-eval` splices rows beside rows this run already measured, so a definitional edit would put
two runtimes inside one artifact with nothing in a row to say which produced it, and the operator
refused there can rebookend the source again, which measures the whole held-out set under one
definition.

Refused before anything is copied or spent: a source without a terminus, without a
resolvable terminal session, holding a suspension record, or still owned by a live process,
and any output path, result leaf, or new run path at or under the source. A
bookend that a usage limit suspends resumes through the ordinary machinery and publishes the
same shape an uninterrupted bookend does: its copied stopping record serves the preflight
only and is never republished as the new run's rollout. The terminus is taken exactly where
the rollout ended, whatever its stop reason: a harness_error ending mid-task resumes from
that mid-task end, deliberately, because stepping back to a tidier point would resume a
conversation the rollout never had.

The preflight validates rather than globs, because existence is not resumability. Each
harness's `session_transcript` resolves the file the way that CLI's own resume lookup does
(exact naming, never a substring) and then requires the floor that CLI was observed to
require, identity included. The floors were established empirically rather than guessed: for
each harness a real session file was produced by driving the pinned CLI itself in the image
(a credential-less run persists its session before failing auth), then minimized cut by cut
with a re-resume after every cut, so each requirement in the predicate is one whose removal
was seen to flip the CLI to a refusal, and each predicate's exact minimum body was seen to
resume to the auth or transport boundary. Value domains got the same discipline as field
presence: where a decoder checks a value's shape (claude's timestamp and content, prime's
cwd) the predicate checks that shape and each rejected shape is one the CLI was observed to
refuse, and where a decoder checks nothing (codex's payload values) the predicate adds
nothing, so the predicate sits as close to the decoder it fronts as probing can place it.
The decoding dialect itself is part of the discipline, because the preflight parses in
Python and the CLIs do not: Python's json accepts NaN-family constants that JSON.parse
refuses, escaped lone surrogates that serde refuses, and nesting far past serde's recursion
boundary, so the preflight reads a strict dialect (`_strict_json_object`: constants raise,
every decoded string must survive a strict UTF-8 encode, and no container may sit deeper
than the bracketed serde boundary), and each of those has its observation (a NaN-carrying
prime header is "No session found matching"; a surrogate-carrying codex meta is unreadable
session metadata; 126 nested arrays resumed and 127 were refused). The byte layer below the
dialect is mirrored per reader rather than shared: codex reads strict UTF-8 and one raw
`0xFF` before a valid record is "stream did not contain valid UTF-8" (a UTF-8 BOM, valid
bytes and invalid JSON, is refused too), while the Node-family CLIs decode invalid bytes to
U+FFFD and carry on (a replaced byte inside a prime header string and inside a claude
transcript string both resumed), so the codex reader treats a decode failure as the fatal
first-record refusal and the other two decode with replacement, never with erasure, which
could stitch refuse-worthy bytes into acceptable JSON. Anchoring is per CLI too: codex
parses the literal first record and refuses the file when that fails, while prime's scanner
skips lines its parser refuses, both observed with a valid record on line two. The refusals all this preempts would otherwise arrive per
task, after the fan-out's copies, streams, and containers were already paid for. What no
line-level predicate can promise is a file that carries the floor plus additional records
the CLI cannot replay (a poisoned line elsewhere in a crashed file); that residue stays a
per-task failure, and only a credentialed dry run could squeeze it further.

Every claim below was produced against the pinned image (Claude Code 2.1.226, codex-cli
0.147.0, prime-agent 0.7.1) with the network disabled and no credential present, so each probe
stopped at authentication or transport and spent nothing. What that class of probe cannot show
is a model actually continuing the conversation; see the unverified list.

### claude_code (observed)

`--resume <uuid>` resolves the id against the transcripts under
`~/.claude/projects/<cwd-slug>/<uuid>.jsonl` in the HOME the process runs with. Observed, in a
container with a fabricated transcript in a mounted HOME copy: a bogus id fails loudly
(`No conversation found with session ID: ...`, `is_error: true`, exit 1), and a present
transcript proceeds, emitting the normal `system/init` event with the **same** session id and
appending the resumed turn to that same transcript file inside the copy. A resume does not
fork the id, so the fork is the copy, which is exactly what the per-task HOME provides. The
resumed argv carried `--append-system-prompt` with the eval instruction and
`--permission-mode bypassPermissions`, both accepted and reflected in the init event; that
matters because resume restores no flags (docs) and the runner rebuilds the whole argv every
launch. The credential-less run then stopped at `authentication_failed` with zero tokens.

The floor a transcript must carry, minimized from a transcript the CLI itself wrote and
re-resumed after every cut (all observed): one `user` line holding a `message` with a role
and non-empty content, a `timestamp`, and the `sessionId`. Exactly that line resumes to the
auth boundary; without the `message` or without the `timestamp` the CLI reports
`No conversation found with session ID` exactly as if the file were absent, and a `message`
without content crashes the resume (`Failed to resume session: undefined is not an
object`). The `uuid` field, present in every real line, is not required. A line that merely
names the id is not a conversation, and the preflight requires the whole floor.

The value domains are part of the floor, not only the fields (all observed, rest of the line
held at the verified minimum). The timestamp must be the complete, calendar-valid,
timezone-carrying ISO instant in the GRAMMAR the CLI accepts, which is narrower than ISO
8601: the extended form, the space-separated form, and the numeric HH:MM offset each
resumed, while a week date (`2026-W33-3T00:00:00Z`) and the basic form (`20260812T000000Z`)
were refused as `No conversation found` alongside an object, an epoch number, a non-date
string, and an ISO-shaped but calendar-invalid string (`2026-99-99T99:99:99Z`); comma
fractions, arbitrary separators, and offsets carrying seconds are also refused per the
review's probes. So the preflight pairs an end-anchored grammar for the accepted profile
with a real datetime parse: the grammar cannot see that 2026-99-99 is not a date, and the
parse alone certified every profile above that `fromisoformat` accepts and the CLI does not. The content must be a string
or a list: an object resolves the session and then crashes the resume with
`TypeError: e.map is not a function`, whose own text names the CLI's domain (a string is
handled apart; everything else is mapped as an array), and a list of non-block elements
resumed, its entries contributing empty text, so the preflight constrains a list's elements
no further than the CLI does.

### codex (observed for the resume mechanics, source elsewhere)

`codex exec resume` exists at the pinned 0.147.0 as a non-interactive subcommand:
`codex exec resume [SESSION_ID] [PROMPT]`, accepting `--json`, `-m`, `-c` overrides,
`--dangerously-bypass-approvals-and-sandbox`, and `--skip-git-repo-check` (observed:
`codex exec resume --help` in the image). Observed against a fabricated session file: a bogus
id exits 1 with `no rollout found for thread id ... (code -32600)`, and a file at
`~/.codex/sessions/YYYY/MM/DD/rollout-<stamp>-<thread-id>.jsonl` in the mounted HOME is found
by the thread id embedded in its filename. The resumed run re-emits `thread.started` with the
**same** thread id, appends the resumed turn to that file, starts the turn, and only then goes
to the provider (blocked here by the disabled network, zero spend). The prompt-after-flags
argv order the harness builds is the order the probe ran. One edge worth naming: `--last` and
thread-name selection are cwd-filtered and `--all` disables the filter (help text); an
explicit UUID, with rollout and eval both running at `/work`, does not touch that path.

The floor a rollout must carry, minimized from a rollout the CLI itself wrote and re-resumed
after every cut (all observed): the FIRST record must be the session metadata, decoded
whole. A first line carrying the envelope `timestamp` and a `session_meta` payload holding
`id`, `timestamp`, `cwd`, `originator`, and `cli_version`, with no items after it, resumes
to the transport boundary. Dropping payload fields, or the envelope timestamp itself, is
refused as `failed to read session metadata` or `rollout ... is empty`; an empty file the
same; a parseable non-meta first line is refused as `does not start with session metadata`;
and an UNPARSEABLE first line is refused as `failed to parse first rollout record` even with
a fully valid meta on line two, so the preflight anchors on the literal first record the way
the reader does, skipping nothing. Decoding is serde-strict at every layer: an escaped lone
surrogate in any meta value is refused as unreadable session metadata (observed, with and
without the envelope timestamp beside it); the raw bytes must be valid UTF-8 (one `0xFF`
before the verified minimum is `stream did not contain valid UTF-8`, and a UTF-8 BOM, valid
bytes and invalid JSON, is refused too); and nesting stops at serde's recursion boundary,
bracketed exactly (an extra field wrapped in 126 arrays resumed, 127 was refused). The
preflight's strict reader enforces all three.

Beyond presence and decodability the VALUES are not domain-checked at 0.147.0. A bogus
originator, a bogus cli_version, a non-date payload timestamp, a non-date envelope
timestamp, and a relative cwd each resumed to the transport boundary (all observed, one
variant at a time against the verified minimum), so the preflight constrains none of them;
stricter would refuse rollouts the CLI accepts.

### prime_agent (observed for the resume mechanics, source for the layout)

`-r/--resume <path|id>` is a run option of the pinned 0.7.1 (help). Sessions are stored as
`<sessions-dir>/<id>.jsonl` where the sessions dir is `<agent-dir>/sessions`, so
`~/.prime/agent/sessions` in a cell HOME (source: `getSessionsDir` and `getSessionFilePath` in
the installed bundle). The resolver matches the id against cwd-matching sessions first and all
sessions second (source: `resolveSessionPath`), so the `/work` cwd shared by the rollout and
every eval task finds the rollout's session in the local set. Observed against a fabricated
session file: a bogus id exits 1 with `No session found matching '<id>'`, and a present file
is resolved, re-emits the session header with the **same** id (the line
`session_id_from_trace` reads), appends the resumed turn to that file, and stops at
`No API key found for anthropic` with nothing spent. `--fork <path|id>` also exists at 0.7.1;
the runner does not use it, since the per-task copy is the fork.

The floor a session file must carry (all observed): a header naming the id and the resuming
cwd, and it must be the FIRST parseable line. A one-line
`{"type": "session", "version": 3, "id": ..., "cwd": "/work"}` file, timestamp absent and
all, resumed to the auth boundary in a pristine home. Three things the CLI refuses around
that. Anchoring: its scanner gives up on a file whose first parseable entry is not the
header (source: `scanSessionInfo`), and a message line above a fully valid header is
`No session found matching` for that id, the same answer a mismatched header id gets,
because the scanner indexes by the header id and never the filename. The filename is in
fact a DIFFERENT identifier space, and requiring it refused real runs: the daemon mints the
session file under one id and the print run's header carries another, rewritten into that
same file (observed on a session the pinned CLI itself wrote, and the shape of both real
prime rollouts, whose recorded terminal id sits inside a file named for a different id).
The CLI resolves the header id out of the differently named file and appends to it
(observed, network off), so the preflight scans the flat sessions directory for exact
header matches instead of trusting a name, and refuses two files carrying the same header
id, because the resolver refuses an ambiguous selector (source: `resolveUniqueMatch`). The dialect: the
scanner's parser is JSON.parse, so a header line carrying a NaN-family constant is
unparseable to it and the session is `No session found matching`; the skipping cuts the
other way too, since a NaN-poisoned junk line ABOVE a valid header is skipped and the
session resumed (both observed), which is exactly the skip-then-anchor the preflight
mirrors. Bytes are decoded with replacement rather than refused: an invalid raw byte inside
a header string arrived as U+FFFD and the session resumed, and an invalid-byte junk line
above the header was skipped like any other unparseable line (both observed), so the
preflight decodes the same way and never erases. And the cwd, which is a value domain of the resume: a header without one, or
recorded at another directory, resolves as `Session found in different project` and stalls
on an interactive `Fork this session into current directory?` prompt, which under the
runner's closed stdin is exit 13 and no resume (observed for both). Every leg runs at
`/work`, so that is the cwd a forkable header must carry.

The transcript is not the whole persisted session. Its artifact tree at
`<agent-dir>/session-artifacts/<id>/` is part of it (source, in the installed bundle): the
kernel snapshot `kernel-state.dill`/`kernel-state.json` is revived into the fresh kernel
right after start ("Only persistent sessions (which have an artifact dir) get a revivable
snapshot", and `restoreState` runs before the runtime bootstrap), the `sub-*` child session
dirs are where the RLM runtime persists children and reads completed ones back ("completed
children from their persisted session dirs, so the tree survives child disposal"), and the
local harness-state dir and the session's scheduled jobs live beside them. A fork that
carried the transcript alone would reopen the conversation while silently dropping the
learned kernel state and every child conversation, so `session_state_dirs` carries both
subtrees. The operational neighbors stay behind: leases are
`<agent-dir>/session-leases/<hash>.lock` and daemon state is `<agent-dir>/daemon-workers`,
both sibling subtrees the copy's noise rules already exclude (source: `leaseDirectory`).

## Docker checklist

Applies to every harness:

    </dev/null                    # mandatory for codex and prime-agent, cheap insurance for claude
    -e NODE_OPTIONS=              # mandatory for prime-agent; an inherited value crashed it outright
    container name <= 63 chars    # a longer name silently fails DNS and the agent gets no tools
    credentials injected with -e at runtime, never baked into an image
    the config mount is read-only and outside the agent's working directory

The stdin rule is not uniform, and the difference is worth knowing. codex blocks forever on an
open stdin: it reads stdin to end whenever stdin is not a TTY, prompt argument or not, and
`docker run` without `-t` gives exactly that (observed: no output at all in 12 seconds).
Claude Code does not block; it warns after 3 seconds and proceeds. prime-agent merges piped
stdin into the prompt in print mode, which is its own reason to close it. The runner closes
stdin for all three.

## What is still unverified

1. No live usage-limit event has been observed for any of the three. The rules are source
   verified, and the runner records the full terminal payload on every leg so the first real
   occurrence documents itself.
2. prime-agent has never been exercised end to end against a live provider in a container in
   this program. It needs its own smoke run before it joins an 8-hour cell.
3. `IS_SANDBOX` is undocumented and can disappear in any release. The negative control
   exercises the path at every cell start.
4. Whether prime-agent's own OAuth `/login` yields a subscription credential or an api key for
   each provider. That is answerable only after the interactive login, which only the owner can
   perform.
5. No resumed eval_after leg has run against a live provider. The fork mechanics (session
   lookup in a copied HOME, same-id resumption, append-to-copy, loud failure on a missing
   transcript) and every preflight floor above are observed to the authentication or
   transport boundary for all three harnesses. What remains: a model actually continuing the
   resumed conversation, and a transcript that carries the floor plus additional records the
   CLI cannot replay, both verifiable only by a credentialed run.
