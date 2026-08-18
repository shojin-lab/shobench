---
name: shogym-stream
description: Play a stream of shogym evaluation tasks served over MCP. Pull tasks with get_task, complete them with the tools each task lists, and end them. Use whenever you are asked to work through a queue of shogym tasks or a task server.
---

# shogym-stream

A queue of evaluation tasks behind one MCP endpoint. Every tool is an `async` method on the
`shogym_stream` module, and two shapes come back, because the kernel's MCP client returns a
tool's structured content when the server declares an output schema for it and the text content
when it does not:

- the stream's own control tools, `get_task` and `queue_info`, arrive **already parsed**, as
  dicts. Index them directly; `json.loads` on one raises `TypeError`.
- the tools a task publishes arrive as **JSON strings**. Parse those before you index.

```python
import json

task = await shogym_stream.get_task()  # a dict already
result = json.loads(await shogym_stream.some_task_tool(arg="value"))  # a JSON string
```

That is the rule as the server stands today, and one line survives either shape if you would
rather not depend on it: `value if isinstance(value, dict) else json.loads(value)`.

## The loop

1. `get_task()` takes the next task off the queue. It answers either
   `{env, instructions, budget, tools}` or `{done: true, ...}`, and `done` means the queue is
   empty.
2. `tools` lists the tools that task published, by name and schema. They are methods on this
   same module: `await shogym_stream.<name>(**kwargs)`. Nothing else is available for the task,
   and their results are the JSON strings above.
3. One of them ends the episode.
4. The stream is exhausted once the queue is empty and no task you pulled is unfinished.

Do not assume tool names between tasks; read `tools` each time. `queue_info()` reports
`{remaining, consumed, in_flight}` if you want to know how much is left. Every tool the server
publishes carries its own description, including what a call costs:
`await shogym_stream.list_tools()` returns them.

## Failures

- `NotEnabled` means no bearer token. Its message tells you to run `/mcp login shogym`: **that
  is wrong here** and will report an unknown integration. This server authenticates with a
  static token, so tell the user to export `SHOBENCH_MCP_TOKEN` (any non-empty value) and
  restart the agent.
- A connection error means the stream server is not running. Tell the user; do not try to
  start it, and do not go looking for the task's answer on disk.
- `McpToolError` is the server rejecting a call. Read the message and fix the arguments.
