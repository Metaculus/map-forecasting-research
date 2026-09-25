"""Radiant MCP toolset for pydantic-ai agents (streamable HTTP).

`build_radiant_toolset(api_key)` returns a toolset that opens a fresh MCP session
per agent run and reconnects with jittered backoff when the server rejects a
session (401/429) or the handshake fails. Requires RADIANT_BASE_URL.

Install with the `radiant` extra: pip install "map-forecasting[radiant]".
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from typing import Any

import httpx
from pydantic_ai import RunContext
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset

# Opening many sessions at once makes the handshake fail; cap concurrency.
RADIANT_MCP_LIMITER = asyncio.Semaphore(int(os.environ.get("RADIANT_MCP_MAX_CONCURRENT", "4")))

logger = logging.getLogger(__name__)


class _DropStructuredContentParseError(logging.Filter):
    """Drop fastmcp's benign "Error parsing structured content" ERROR line.

    fastmcp validates each tool result's ``structuredContent`` into a typed
    ``.data`` field (fastmcp/client/mixins/tools.py). Radiant's tool-output
    schema is recursive — a ``Root`` forward-ref — which fastmcp's
    ``json_schema_to_type`` builds without the final ``.rebuild()``, so the
    validation raises ``TypeAdapter[ForwardRef('Root')] is not fully defined``
    and fastmcp logs it at ERROR, once per tool call and several times per
    forecast.

    It is inert: pydantic-ai maps tool results from ``structured_content`` and
    ``content`` and never reads ``.data`` (pydantic_ai/mcp.py
    ``_map_mcp_call_tool_result``), so the forecast gets the full tool output
    regardless. We suppress only this one message and let every other error from
    that logger through.
    """

    _MARKER = "Error parsing structured content"

    def filter(self, record: logging.LogRecord) -> bool:
        return self._MARKER not in record.getMessage()


def _silence_benign_fastmcp_structured_content_error() -> None:
    # Attach to the exact originating logger so the record is dropped in
    # Logger.handle before it reaches any handler. Idempotent across imports.
    fastmcp_tools_logger = logging.getLogger("fastmcp.client.mixins.tools")
    if not any(
        isinstance(existing, _DropStructuredContentParseError)
        for existing in fastmcp_tools_logger.filters
    ):
        fastmcp_tools_logger.addFilter(_DropStructuredContentParseError())


_silence_benign_fastmcp_structured_content_error()

RADIANT_API_KEY_ENV_VAR = "RADIANT_API_KEY"
# Base URL of the Radiant instance, e.g. "https://radiant.example.com".
RADIANT_BASE_URL_ENV_VAR = "RADIANT_BASE_URL"

_ENTER_MAX_ATTEMPTS = 6
_ENTER_RETRY_BASE_SECONDS = 4.0
# Floor so a retry never stampedes the host immediately after a 401.
_ENTER_RETRY_MIN_SECONDS = 1.0
_CALL_MAX_ATTEMPTS = 3

# Radiant MCP emits draft-07 schemas with ``definitions`` +
# ``$ref: #/definitions/...``. pydantic-ai's JsonSchemaTransformer only
# resolves ``$defs`` / ``#/$defs/...``, which raises:
#   UserError: Could not find $ref definition for #/definitions/__schema0
_JSON_VALUE_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {"type": "string"},
        {"type": "number"},
        {"type": "boolean"},
        {"type": "null"},
        {"type": "array", "items": True},
        {"type": "object", "additionalProperties": True},
    ]
}


def radiant_base_url() -> str:
    url = os.getenv(RADIANT_BASE_URL_ENV_VAR)
    if not url:
        raise RuntimeError(f"Set {RADIANT_BASE_URL_ENV_VAR} to your Radiant instance's base URL.")
    return url.rstrip("/")


def radiant_mcp_url() -> str:
    return f"{radiant_base_url()}/mcp"


def radiant_project_url_prefix() -> str:
    return f"{radiant_base_url()}/projects/"


def require_radiant_api_key() -> str:
    api_key = os.getenv(RADIANT_API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(
            f"Missing Radiant API key in env var {RADIANT_API_KEY_ENV_VAR!r}."
        )
    return api_key


def normalize_mcp_json_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Rewrite draft-07 ``definitions`` refs into pydantic-ai-compatible ``$defs``."""
    if not schema:
        return {}
    normalized = deepcopy(schema)
    definitions = normalized.pop("definitions", None)
    if definitions:
        defs = normalized.setdefault("$defs", {})
        for key, value in definitions.items():
            defs.setdefault(key, value)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/definitions/"):
                node["$ref"] = "#/$defs/" + ref.removeprefix("#/definitions/")
            nested_definitions = node.pop("definitions", None)
            if nested_definitions:
                defs = node.setdefault("$defs", {})
                for key, value in nested_definitions.items():
                    defs.setdefault(key, value)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(normalized)

    # Ensure recursive JSON-value defs exist when Radiant refs them.
    defs = normalized.setdefault("$defs", {})
    if "__schema0" not in defs and _schema_refs_key(normalized, "__schema0"):
        defs["__schema0"] = deepcopy(_JSON_VALUE_SCHEMA)
    return normalized


def _schema_refs_key(schema: dict[str, Any], key: str) -> bool:
    target = f"#/$defs/{key}"
    legacy = f"#/definitions/{key}"

    def walk(node: Any) -> bool:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if ref in (target, legacy):
                return True
            return any(walk(value) for value in node.values())
        if isinstance(node, list):
            return any(walk(value) for value in node)
        return False

    return walk(schema)


async def _prepare_radiant_tool_defs(
    _ctx: RunContext[Any],
    tool_defs: list[ToolDefinition],
) -> list[ToolDefinition]:
    prepared: list[ToolDefinition] = []
    for tool_def in tool_defs:
        prepared.append(
            replace(
                tool_def,
                parameters_json_schema=normalize_mcp_json_schema(
                    tool_def.parameters_json_schema
                ),
            )
        )
    return prepared


def _enter_retry_delay(attempt: int) -> float:
    """Randomized backoff for a 401 on session enter.

    Full jitter over ``[min, base * attempt]`` keeps sessions that failed
    together from retrying in lockstep; the floor keeps retries off the host.
    """
    return random.uniform(
        _ENTER_RETRY_MIN_SECONDS,
        max(_ENTER_RETRY_MIN_SECONDS, _ENTER_RETRY_BASE_SECONDS * attempt),
    )


# Throttling can surface as 401 as well as 429, so retry both.
_RETRYABLE_STATUS_CODES = (401, 429)
_RETRYABLE_STATUS_PHRASES = (
    "401 Unauthorized",
    "Status 401",
    "429 Too Many Requests",
    "Status 429",
)

# Session setup degrades under many concurrent connections and fails without an
# HTTP status to match on (the client's multi-step handshake gives up). These
# are worth retrying too.
_RETRYABLE_CONNECT_PHRASES = (
    "Failed to initialize server session",
    "Client failed to connect",
)


def _is_retryable_error(exc: BaseException) -> bool:
    """True for failures that mean "back off and reconnect", not "give up"."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    # MCP wraps transport errors, so fall back to matching the rendered text.
    text = str(exc)
    return any(
        phrase in text
        for phrase in (*_RETRYABLE_STATUS_PHRASES, *_RETRYABLE_CONNECT_PHRASES)
    )


def _build_raw_prepared_mcp(api_key: str) -> AbstractToolset[Any]:
    toolset = MCPToolset(
        radiant_mcp_url(),
        headers={"Authorization": f"Bearer {api_key}"},
    )
    return toolset.prepared(_prepare_radiant_tool_defs)


class PerRunToolset(AbstractToolset[Any]):
    """Toolset placeholder that builds a fresh inner toolset on each agent run.

    A concrete ``MCPToolset`` is unsafe to share across concurrent agent runs.
    pydantic-ai calls ``for_run`` once per ``agent.run`` before enter.
    """

    def __init__(
        self,
        factory: Callable[[], AbstractToolset[Any]],
        *,
        toolset_id: str,
    ) -> None:
        self._factory = factory
        self._toolset_id = toolset_id

    @property
    def id(self) -> str | None:
        return self._toolset_id

    async def for_run(self, ctx: RunContext[Any]) -> AbstractToolset[Any]:
        return self._factory()

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, Any]:
        raise RuntimeError(
            f"{self.label}.for_run() must run before get_tools(); "
            "the agent should call for_run automatically."
        )

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: Any,
    ) -> Any:
        raise RuntimeError(
            f"{self.label}.for_run() must run before call_tool(); "
            "the agent should call for_run automatically."
        )


class ResilientRadiantToolset(AbstractToolset[Any]):
    """One Radiant MCP session per agent run, rebuilt with backoff when the
    server later rejects it (401/429)."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._inner: AbstractToolset[Any] | None = None
        self._slot_held = False

    @property
    def id(self) -> str | None:
        return "radiant-mcp"

    @property
    def label(self) -> str:
        return "ResilientRadiantToolset"

    async def __aenter__(self) -> ResilientRadiantToolset:
        last_exc: BaseException | None = None
        for attempt in range(1, _ENTER_MAX_ATTEMPTS + 1):
            try:
                await self._open_session()
                return self
            except BaseException as exc:
                last_exc = exc
                await self._close_session()
                if not _is_retryable_error(exc) or attempt >= _ENTER_MAX_ATTEMPTS:
                    raise
                wait_seconds = _enter_retry_delay(attempt)
                logger.warning(
                    "Radiant MCP enter rejected (attempt %s/%s); retrying in %.1fs: %s",
                    attempt,
                    _ENTER_MAX_ATTEMPTS,
                    wait_seconds,
                    exc,
                )
                await asyncio.sleep(wait_seconds)
        assert last_exc is not None
        raise last_exc

    async def __aexit__(self, *args: Any) -> bool | None:
        return await self._close_session(*args)

    async def _open_session(self) -> None:
        if self._inner is not None:
            return
        await RADIANT_MCP_LIMITER.__aenter__()
        self._slot_held = True
        try:
            inner = _build_raw_prepared_mcp(self._api_key)
            await inner.__aenter__()
        except BaseException:
            self._slot_held = False
            await RADIANT_MCP_LIMITER.__aexit__(None, None, None)
            raise
        self._inner = inner

    async def _close_session(self, *args: Any) -> bool | None:
        result: bool | None = None
        inner = self._inner
        self._inner = None
        try:
            if inner is not None:
                result = await inner.__aexit__(*args)
        finally:
            if self._slot_held:
                self._slot_held = False
                await RADIANT_MCP_LIMITER.__aexit__(None, None, None)
        return result

    async def _reopen_session(self, attempt: int = 1) -> None:
        # Pause before reconnecting: an immediate re-open on a 401 is the same
        # lockstep stampede the enter path backs off from.
        await self._close_session(None, None, None)
        await asyncio.sleep(_enter_retry_delay(attempt))
        await self._open_session()

    def _require_inner(self) -> AbstractToolset[Any]:
        if self._inner is None:
            raise RuntimeError("Radiant MCP session is not open")
        return self._inner

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, Any]:
        last_exc: BaseException | None = None
        for attempt in range(1, _CALL_MAX_ATTEMPTS + 1):
            try:
                return await self._require_inner().get_tools(ctx)
            except Exception as exc:
                last_exc = exc
                if not _is_retryable_error(exc) or attempt >= _CALL_MAX_ATTEMPTS:
                    raise
                logger.warning(
                    "Radiant MCP get_tools rejected (attempt %s/%s); reconnecting: %s",
                    attempt,
                    _CALL_MAX_ATTEMPTS,
                    exc,
                )
                await self._reopen_session(attempt)
        assert last_exc is not None
        raise last_exc

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: Any,
    ) -> Any:
        last_exc: BaseException | None = None
        for attempt in range(1, _CALL_MAX_ATTEMPTS + 1):
            try:
                return await self._require_inner().call_tool(name, tool_args, ctx, tool)
            except Exception as exc:
                last_exc = exc
                if not _is_retryable_error(exc):
                    raise
                if attempt >= _CALL_MAX_ATTEMPTS:
                    logger.warning(
                        "Radiant MCP tool %r still failing after reconnects: %s",
                        name,
                        exc,
                    )
                    return (
                        f"Radiant tool {name!r} was rejected by the server after "
                        "reconnect attempts. Finish the forecast without further "
                        "Radiant edits and note the map may be incomplete."
                    )
                logger.warning(
                    "Radiant MCP tool %r rejected (attempt %s/%s); reconnecting: %s",
                    name,
                    attempt,
                    _CALL_MAX_ATTEMPTS,
                    exc,
                )
                await self._reopen_session(attempt)
        assert last_exc is not None
        raise last_exc

    async def get_instructions(self, ctx: RunContext[Any]) -> Any:
        return await self._require_inner().get_instructions(ctx)


def build_radiant_toolset(api_key: str) -> AbstractToolset[Any]:
    """A Radiant MCP toolset that is safe to share across concurrent agent runs."""
    return PerRunToolset(
        lambda: ResilientRadiantToolset(api_key),
        toolset_id="radiant-mcp",
    )


def project_url_from_id(project_id: str | int) -> str:
    return f"{radiant_project_url_prefix()}{project_id}"
