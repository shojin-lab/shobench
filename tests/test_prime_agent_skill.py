"""The prime-agent skill wiring: the served stream is reachable only if the cell HOME carries
the ``shogym-stream`` skill, not just the settings entry.

prime-agent hands the model no MCP tools; it reaches a server by importing a Python-backed
skill in its kernel and calling it. So declaring the HTTP server in ``settings.json`` is half
the wiring and the skill package is the other half. These prove the vendored skill is a
well-formed Python-backed skill, that a prime_agent leg installs it into the isolated HOME with
the same token variable the settings entry names, and that a locally-served stream exposes the
tools the skill would drive and hands them back in the shapes SKILL.md tells the model to
expect. A skill that is not installable at all is a launch error here, since the alternative is
a leg that runs to completion having reached nothing.

The skill's own ``__init__.py`` is the one Python file in this repo written for prime-agent's
kernel interpreter: it imports ``rlm``, which is not a shobench dependency and is not on PyPI at
all. It is read with ``ast``, never imported, the same treatment shogym's quickstart gives its
copy. The credentialed end-to-end check (prime-agent bootstrapping the kernel, installing this
package, importing it as ``shogym_stream``, and pulling a task) waits on the interactive login
this host does not have; see ``docs/harness-autonomy.md``.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from shobench.config import repo_root
from shobench.harnesses import SHOGYM_STREAM_SKILL, PrimeAgent, shogym_stream_skill_files
from shobench.serving import SERVER_NAME

_SKILL = repo_root() / "prime_agent" / "skills" / SHOGYM_STREAM_SKILL
_INIT = _SKILL / "src" / "shogym_stream" / "__init__.py"


def _as_the_kernel_sees_it(result: Any) -> Any:
    """One tool result, normalized the way the kernel's MCP client normalizes it.

    ``McpIntegration._parse_result`` returns a result's structured content when the server sent
    any and its joined text content otherwise, so what a bound method hands the model is decided
    by whether the server declared an output schema for that tool, not by anything the skill
    does. That rule is transcribed here rather than called, because the runtime package is not a
    dependency and is not on PyPI (the same reason the skill module is read with ``ast``); the
    field is the wire's ``structuredContent``, which is the field the runtime prefers.
    """
    if result.structured_content is not None:
        return result.structured_content
    return "\n".join(block.text for block in result.content if hasattr(block, "text"))


def _launch_spec(tmp_path: Path, mcp_url: str = "http://host.docker.internal:12345/mcp"):
    """A prime_agent leg's launch spec. Only the url ever matters to these assertions."""
    return PrimeAgent().launch(
        mcp_url=mcp_url,
        system_prompt="s",
        user_prompt="u",
        model="m",
        trace_path=tmp_path / "t",
    )


def _skill_class_attrs() -> dict[str, str]:
    """The ``McpIntegration`` subclass's string attributes, read out of the source (no import)."""
    module = ast.parse(_INIT.read_text())
    classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
    assert len(classes) == 1, "the skill is one integration class"
    return {
        target.id: node.value.value
        for node in classes[0].body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def test_the_skill_is_discoverable_as_a_python_backed_skill() -> None:
    """prime-agent's discovery rules: a directory with a SKILL.md whose frontmatter ``name``
    matches it, a ``pyproject.toml`` (which is what makes it Python-backed), and
    ``src/<import name>/__init__.py`` where the import name is the directory name with hyphens
    turned into underscores."""
    assert _SKILL.name == "shogym-stream"
    assert (_SKILL / "pyproject.toml").is_file()
    assert _INIT.is_file()
    assert _INIT.parent.name == _SKILL.name.replace("-", "_")

    front = (_SKILL / "SKILL.md").read_text().split("---")[1]
    fields = dict(
        line.split(":", 1) for line in front.strip().splitlines() if ":" in line and line[0] != " "
    )
    assert fields["name"].strip() == _SKILL.name
    assert fields["description"].strip(), "a skill with no description is not loaded at all"


def test_the_skill_names_the_runners_server_and_token_variable() -> None:
    """The skill talks to the server the runner serves and reads the token variable the runner
    sets, not the shogym quickstart's ``SHOGYM_MCP_TOKEN``. The url is only a fallback (the host
    settings entry the runner writes per leg wins), so it is checked for shape, not value."""
    attrs = _skill_class_attrs()
    assert attrs["server"] == SERVER_NAME
    assert attrs["bearer_token_env"] == PrimeAgent.MCP_TOKEN_VAR
    assert attrs["url"].startswith("http://")


def test_launch_installs_the_skill_beside_the_settings_entry(tmp_path: Path) -> None:
    """The wiring: a prime_agent leg's HOME carries the skill package, and the settings entry
    and the skill agree on the token variable, resolved from one constant."""
    spec = _launch_spec(tmp_path)
    seeded = spec.home_seed_files
    prefix = f".prime/agent/skills/{SHOGYM_STREAM_SKILL}"
    for rel in ("SKILL.md", "pyproject.toml", "src/shogym_stream/__init__.py"):
        assert f"{prefix}/{rel}" in seeded, rel

    # The installed bytes are exactly the vendored ones, and the skill goes in as a seed while
    # the endpoint stays per-leg, since one of the two is the agent's to change.
    assert seeded == shogym_stream_skill_files()
    assert set(spec.home_files) == {".prime/agent/settings.json"}

    settings = json.loads(spec.home_files[".prime/agent/settings.json"])
    token_var = settings["mcpServers"]["shogym"]["bearerTokenEnvVar"]
    assert token_var == PrimeAgent.MCP_TOKEN_VAR == _skill_class_attrs()["bearer_token_env"]
    assert token_var in spec.env  # the token itself is set in the launch environment


def test_the_eval_after_session_reads_the_skill_the_rollout_left(tmp_path: Path) -> None:
    """The whole point of seeding, walked with the runner's own two pieces.

    An eval-after task copies the home the rollout accumulated and then runs a leg against that
    copy. A leg that rewrote the skill would restore the vendored bytes in the moment between
    the improvement and the session meant to read it, and the cell would report the durable
    effect of an artifact the runner had just deleted. The endpoint has to be refreshed in the
    same breath, because the task serves on a port of its own that its inherited settings entry
    knows nothing about, which is why the two files ride different channels."""
    from shobench.runner import _copy_task_home, write_home_files

    rollout_home = tmp_path / "rollout-home"
    write_home_files(rollout_home, _launch_spec(tmp_path, "http://host:1/mcp"))
    skill = rollout_home / f".prime/agent/skills/{SHOGYM_STREAM_SKILL}/SKILL.md"
    improved = skill.read_text() + "\n## What worked\n\nGuess a vowel-heavy opener.\n"
    skill.write_text(improved)

    task_home = tmp_path / "eval-task-home"
    _copy_task_home(rollout_home, task_home)
    write_home_files(task_home, _launch_spec(tmp_path, "http://host:2/mcp"))

    kept = task_home / f".prime/agent/skills/{SHOGYM_STREAM_SKILL}/SKILL.md"
    assert kept.read_text() == improved, "the eval session must read what the rollout wrote"
    settings = json.loads((task_home / ".prime/agent/settings.json").read_text())
    assert settings["mcpServers"]["shogym"]["url"] == "http://host:2/mcp"


def test_the_served_stream_exposes_the_tools_the_skill_enumerates(tmp_path: Path) -> None:
    """As much of the end-to-end as needs no credential: a stream stood up exactly as the runner
    stands it up, through ``build_stream_server``, hands back the control tools the skill drives
    (``get_task``, ``queue_info``, the terminal) plus the env's own, and a pulled task carries
    the well-formed ``{env, instructions, budget, tools}`` the SKILL.md documents. ``wordle_v1``
    needs no extra, no key and no download, so this stays offline. prime-agent's own client is
    not installed here (it imports ``rlm``), so the manifest is enumerated with the MCP client
    shogym ships and the pulled task is read through the kernel's normalization rule rather than
    off a text block, which is how the model will read it; what prime-agent adds on top, the
    kernel install and import, is read from docs rather than run."""
    import shogym
    from fastmcp import Client
    from shogym.serve import Immediate, TaskRef, TaskStream, build_stream_server

    async def _enumerate() -> tuple[set[str], dict]:
        stream = TaskStream(
            shogym.make,
            [TaskRef("wordle_v1", 0)],
            prov_dir=tmp_path / "prov",
            feedback=Immediate(),
        )
        async with stream:
            server = build_stream_server(stream, name=SERVER_NAME)
            async with Client(server) as client:
                names = {tool.name for tool in await client.list_tools()}
                pulled = _as_the_kernel_sees_it(await client.call_tool("get_task", {}))
                await client.call_tool("terminate", {})
                return names, pulled

    names, task = asyncio.run(_enumerate())
    assert {"get_task", "queue_info", "terminate"} <= names
    assert set(task) == {"env", "instructions", "budget", "tools"}
    assert task["env"] == "wordle_v1"
    assert task["tools"], "the task must publish the tools the skill will call"


def test_the_control_tools_arrive_parsed_and_a_task_tool_as_a_json_string(tmp_path: Path) -> None:
    """The return contract SKILL.md teaches, checked where the server decides it.

    The kernel's client prefers a result's structured content, so the shape the model gets is
    whatever the server declared an output schema for. This stream declares one for its control
    tools and none for the tools an env publishes, which makes ``get_task`` a dict the model
    indexes directly and a task tool a JSON string the model parses. A doc that says otherwise
    costs a prime_agent cell its first call, so the split is pinned here: if shogym ever changes
    which side a tool falls on, this fails and SKILL.md gets corrected with it."""
    import shogym
    from fastmcp import Client
    from shogym.serve import Immediate, TaskRef, TaskStream, build_stream_server

    async def _shapes() -> tuple[Any, Any]:
        stream = TaskStream(
            shogym.make,
            [TaskRef("wordle_v1", 0)],
            prov_dir=tmp_path / "prov",
            feedback=Immediate(),
        )
        async with stream:
            server = build_stream_server(stream, name=SERVER_NAME)
            async with Client(server) as client:
                pulled = _as_the_kernel_sees_it(await client.call_tool("get_task", {}))
                # `terminate` is a tool of the task, published in its manifest like any other,
                # so it stands in for the env's own tools without assuming an env's tool names.
                ended = _as_the_kernel_sees_it(await client.call_tool("terminate", {}))
                return pulled, ended

    task, ended = asyncio.run(_shapes())
    assert isinstance(task, dict), "get_task is already parsed; the SKILL.md example indexes it"
    with pytest.raises(TypeError):
        json.loads(task)  # what the old "every tool returns a JSON string" wording produced
    assert isinstance(ended, str), "a task's tools are text, so the model parses them"
    assert isinstance(json.loads(ended), dict)


def test_a_missing_or_partial_skill_fails_the_launch_loudly(tmp_path: Path, monkeypatch) -> None:
    """The skill is a runtime asset, so its absence is an error and never an empty mapping.

    A leg launched with the settings entry and no skill package starts a healthy prime-agent
    that can reach nothing, and the record of that run is indistinguishable from an agent that
    chose to do no work. Partial counts as missing: a directory without the pyproject is a
    markdown skill, and the model gets a document instead of a client."""
    from shobench.harnesses import prime_agent

    monkeypatch.setattr(prime_agent, "_vendored_skill_dir", lambda: tmp_path / "not-installed")
    with pytest.raises(FileNotFoundError, match="reaches no tools"):
        _launch_spec(tmp_path)

    partial = tmp_path / "partial"
    (partial / "src" / "shogym_stream").mkdir(parents=True)
    (partial / "SKILL.md").write_text("---\nname: shogym-stream\n---\n")
    (partial / "src" / "shogym_stream" / "__init__.py").write_text("")
    monkeypatch.setattr(prime_agent, "_vendored_skill_dir", lambda: partial)
    with pytest.raises(FileNotFoundError, match="pyproject.toml"):
        _launch_spec(tmp_path)
