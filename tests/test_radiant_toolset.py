"""Tests for Radiant MCP JSON Schema draft-07 -> $defs normalization."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

pytest.importorskip("pydantic_ai")

import httpx  # noqa: E402
from pydantic_ai import RunContext  # noqa: E402
from pydantic_ai.toolsets import AbstractToolset  # noqa: E402

from map_forecasting.radiant.toolset import (  # noqa: E402
    _ENTER_MAX_ATTEMPTS,
    _ENTER_RETRY_BASE_SECONDS,
    _ENTER_RETRY_MIN_SECONDS,
    PerRunToolset,
    _DropStructuredContentParseError,
    _enter_retry_delay,
    _is_retryable_error,
    build_radiant_toolset,
    normalize_mcp_json_schema,
)


def test_normalize_moves_definitions_and_rewrites_refs() -> None:
    schema = {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "additionalProperties": {"$ref": "#/definitions/__schema0"},
            }
        },
        "definitions": {
            "__schema0": {
                "anyOf": [{"type": "string"}, {"type": "number"}],
            }
        },
    }

    normalized = normalize_mcp_json_schema(schema)

    assert "definitions" not in normalized
    assert normalized["$defs"]["__schema0"] == {
        "anyOf": [{"type": "string"}, {"type": "number"}],
    }
    assert (
        normalized["properties"]["data"]["additionalProperties"]["$ref"]
        == "#/$defs/__schema0"
    )


def test_normalize_injects_missing_json_value_def() -> None:
    schema = {
        "type": "object",
        "properties": {
            "data": {"additionalProperties": {"$ref": "#/definitions/__schema0"}},
        },
    }

    normalized = normalize_mcp_json_schema(schema)

    assert "__schema0" in normalized["$defs"]
    assert (
        normalized["properties"]["data"]["additionalProperties"]["$ref"]
        == "#/$defs/__schema0"
    )


def test_build_radiant_toolset_is_per_run() -> None:
    toolset = build_radiant_toolset("test-key")
    assert isinstance(toolset, PerRunToolset)
    assert toolset.id == "radiant-mcp"


def test_enter_retry_delay_stays_within_bounds() -> None:
    for attempt in range(1, _ENTER_MAX_ATTEMPTS + 1):
        ceiling = max(_ENTER_RETRY_MIN_SECONDS, _ENTER_RETRY_BASE_SECONDS * attempt)
        for _ in range(200):
            delay = _enter_retry_delay(attempt)
            assert _ENTER_RETRY_MIN_SECONDS <= delay <= ceiling


def test_enter_retry_delay_is_jittered() -> None:
    """Colliding sessions must not retry in lockstep."""
    delays = {_enter_retry_delay(3) for _ in range(200)}
    assert len(delays) > 1
    # Spread should cover a meaningful share of the window, not cluster on it.
    assert max(delays) - min(delays) > _ENTER_RETRY_BASE_SECONDS


class _StubToolset(AbstractToolset[Any]):
    def __init__(self, stub_id: str) -> None:
        self._stub_id = stub_id

    @property
    def id(self) -> str | None:
        return self._stub_id

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, Any]:
        return {}

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: Any,
    ) -> Any:
        return None


def test_per_run_toolset_for_run_builds_fresh_instances() -> None:
    counter = {"n": 0}

    def factory() -> AbstractToolset[Any]:
        counter["n"] += 1
        return _StubToolset(f"stub-{counter['n']}")

    outer = PerRunToolset(factory, toolset_id="outer")

    async def _run() -> None:
        first = await outer.for_run(None)  # type: ignore[arg-type]
        second = await outer.for_run(None)  # type: ignore[arg-type]
        assert first is not second
        assert first.id == "stub-1"
        assert second.id == "stub-2"

    asyncio.run(_run())


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/mcp")
    return httpx.HTTPStatusError(
        f"Status {code}",
        request=request,
        response=httpx.Response(code, request=request),
    )


@pytest.mark.parametrize("code", [401, 429])
def test_retryable_statuses_are_retried(code: int) -> None:
    assert _is_retryable_error(_status_error(code))


@pytest.mark.parametrize("code", [400, 403, 404, 500, 503])
def test_other_statuses_are_not_retried(code: int) -> None:
    assert not _is_retryable_error(_status_error(code))


@pytest.mark.parametrize(
    "text",
    [
        "Client error '401 Unauthorized' for url 'https://example.test/mcp'",
        "Client error '429 Too Many Requests' for url 'https://example.test/mcp'",
    ],
)
def test_retryable_statuses_detected_when_wrapped(text: str) -> None:
    """MCP wraps transport errors, so detection falls back to the message."""
    assert _is_retryable_error(RuntimeError(text))


def test_unrelated_error_message_is_not_retried() -> None:
    assert not _is_retryable_error(RuntimeError("map has 401 nodes"))


@pytest.mark.parametrize(
    "text",
    [
        "Client failed to connect: Failed to initialize server session",
        "Failed to initialize server session",
    ],
)
def test_session_init_failures_are_retried(text: str) -> None:
    """Connect degrades under load and carries no HTTP status to match on.

    Before this, a failed connect hard-failed the forecast with zero retries —
    24/24 Radiant forecasts died that way in the 2026-08-06 load test.
    """
    assert _is_retryable_error(RuntimeError(text))


def test_unrelated_connect_wording_is_not_retried() -> None:
    assert not _is_retryable_error(RuntimeError("failed to initialize the forecaster"))


def _log_record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="fastmcp.client.mixins.tools",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_benign_structured_content_error_is_filtered() -> None:
    """fastmcp's inert '.data' parse failure must not spam the log."""
    filt = _DropStructuredContentParseError()
    assert not filt.filter(
        _log_record(
            "[Client-71b5] Error parsing structured content: "
            "`TypeAdapter[ForwardRef('Root')]` is not fully defined"
        )
    )


def test_real_fastmcp_errors_still_pass_the_filter() -> None:
    """The filter is surgical — only the one benign message is dropped."""
    filt = _DropStructuredContentParseError()
    assert filt.filter(_log_record("[Client-71b5] connection reset by peer"))
