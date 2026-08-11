# Smoke evidence

The artifacts of one short automationbench x claude_code cell, run end to end through all
three phases to prove the machinery. It is not a measurement and must never be reported as
one: two held-out tasks and a two-task pool on a fifteen-minute budget say nothing about a
harness.

What it does establish, which is the point:

- All three phases ran and every task sealed. Two held-out tasks before, a rollout, and the
  same two held-out tasks after, eight sealed rows in total with no unscored closure.
- The manifest records what the scope asks it to: the shogym pin, the split digest, the
  instruction digests, the harness version, and which model actually answered, read off the
  traces rather than assumed from the config.
- The rollout stopped because the agent stopped, classified from its own result event, and
  `stopped_with_tasks_available` is false because the pool ceiling was reached first. That
  distinction is the stopping metric working.
- The home digest says the agent wrote nothing durable, which is the honest answer for a run
  whose eval sessions carry no improvement objective and whose rollout was one leg long.
- The egress capture observed 65 requests to two hosts: the model API, and Claude Code's own
  telemetry endpoint. Nothing was blocked, which is the scope's observe-not-preempt call.

`report.txt` is the report row the same JSON produces through `shobench report`.
