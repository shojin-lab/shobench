"""The stream, as prime-agent sees it: an MCP integration the kernel imports.

prime-agent does not hand MCP servers to the model as tools. Each server is a Python-backed
skill that the model imports and calls inside the persistent IPython kernel, so this file is
the whole client. ``McpIntegration`` connects over streamable HTTP, discovers the server's
tools and binds each one as an async method::

    import shogym_stream
    task = json.loads(await shogym_stream.get_task())

Two class attributes are load-bearing:

``url``       where the shobench stream server is listening. The host overrides it from the
              ``mcpServers`` entry in ``.prime/agent/settings.json`` when there is one, and the
              runner always writes that entry (with the leg's real, per-run port), so the value
              here is only the fallback the base class needs when no host entry exists.
``bearer_token_env``
              the reason ``SHOBENCH_MCP_TOKEN`` exists. ``McpIntegration._open_session``
              resolves a token before every connection and raises ``NotEnabled`` without one;
              there is no unauthenticated path through it. The stream server authenticates
              nobody, so the value is a formality -- but it must be set and non-empty, in the
              environment the runner launches prime-agent with, or the kernel cannot connect at
              all. It must name the same variable as the settings entry's ``bearerTokenEnvVar``;
              the runner sets both from one constant and a test asserts they agree.
"""

from contextlib import AsyncExitStack

from rlm import McpIntegration

# prime-agent's own default would make this integration unusable, so the client is built here
# instead. ``McpIntegration._open_session`` hands the MCP SDK a bare
# ``httpx.AsyncClient(headers=...)`` with no ``timeout``, which silently inherits httpx's 5s
# *inactivity* defaults in place of the SDK's own 30s general / 300s SSE-read ones. Any tool
# that goes five seconds without emitting a response byte then fails -- and ``get_task`` does
# exactly that, because building an env takes many seconds and says nothing while it works. The
# surfaced error names none of this: the real ``httpx.ReadTimeout`` is swallowed into a debug
# log, and what reaches the caller is ``SSE stream ended without a response`` inside an
# ExceptionGroup.
#
# Upstream: https://github.com/PrimeIntellect-ai/prime-agent/issues/784
# Delete ``_open_session`` below once that lands; the base class will then do the right thing.
_SDK_TIMEOUT, _SDK_SSE_READ_TIMEOUT = 30.0, 300.0


def _timed_http_client(headers):
    """An HTTP client carrying the MCP SDK's documented timeouts, not httpx's defaults.

    Prefers the SDK's own factory, so each SDK version gets its matching client implementation
    (mcp 2.x builds on httpx2, not httpx); falls back to constructing one directly.
    """
    try:
        from mcp.shared._httpx_utils import create_mcp_http_client
    except ImportError:
        pass
    else:
        return create_mcp_http_client(headers=headers)

    import httpx

    return httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(_SDK_TIMEOUT, read=_SDK_SSE_READ_TIMEOUT),
    )


class ShogymStream(McpIntegration):
    server = "shogym"  # the `mcpServers` key, and the `mcp:shogym` id in auth.json
    # Fallback only: the runner writes an `mcpServers` entry per leg whose url carries the real
    # per-run port, and the host's entry wins. This value just has to be a well-formed http url.
    url = "http://host.docker.internal:8973/mcp"
    bearer_token_env = "SHOBENCH_MCP_TOKEN"

    async def _open_session(self, stack: AsyncExitStack):
        """As the base method, but with a client that will wait for a slow tool.

        Only the ``http_client=`` transport branch is reimplemented, because only that branch
        builds a client. Older SDKs taking ``headers=`` never hit the defect, so they are left
        to the base class rather than re-copied here.
        """
        import inspect

        from mcp import ClientSession
        from rlm.mcp_base import _resolve_streamable_http

        transport = _resolve_streamable_http()
        if "http_client" not in inspect.signature(transport).parameters:
            return await super()._open_session(stack)

        url, extra_headers = await self._resolve_config()
        if not url:
            raise ValueError(f"{type(self).__name__} must set `url` or override `_open_session`")
        token = await self._resolve_token()
        # Extra configured headers first, Authorization last so it always wins.
        auth_header = {**extra_headers, "Authorization": f"Bearer {token}"}

        client = await stack.enter_async_context(_timed_http_client(auth_header))
        read, write, *_ = await stack.enter_async_context(transport(url, http_client=client))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session


shogym_stream = ShogymStream()

# Forward bare module access (`import shogym_stream; await shogym_stream.get_task()`) to the
# instance, but NOT the names the kernel bootstrap probes -- forwarding `run` would make it
# treat the module as a callable skill and break tool dispatch.
_RESERVED = {"run", "__wrapped__", "__call__"}


def __getattr__(name):
    if name.startswith("_") or name in _RESERVED:
        raise AttributeError(name)
    return getattr(shogym_stream, name)
