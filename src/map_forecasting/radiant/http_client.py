"""Minimal synchronous Radiant MCP client over streamable HTTP.

Needs RADIANT_BASE_URL and RADIANT_API_KEY. Supports initialize, tools/list and
tools/call, which is enough to create projects, edit canvases and read them back.

    mcp = RadiantMCP()
    mcp.connect()
    print(mcp.call_tool("whoami"))
    mcp.close()
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

PROTOCOL_VERSION = "2025-03-26"


def _base_url() -> str:
    url = os.getenv("RADIANT_BASE_URL")
    if not url:
        raise RuntimeError("Set RADIANT_BASE_URL to your Radiant instance's base URL.")
    return url.rstrip("/")


def _api_key() -> str:
    key = os.getenv("RADIANT_API_KEY")
    if not key:
        raise RuntimeError("Set RADIANT_API_KEY.")
    return key


class RadiantMCP:
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self._mcp_url = f"{(base_url or _base_url()).rstrip('/')}/mcp"
        self._client = httpx.Client(timeout=120.0)
        self._headers = {
            "Authorization": f"Bearer {api_key or _api_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        self._session_id: str | None = None
        self._next_id = 1

    def _rpc(self, method: str, params: dict[str, Any] | None = None,
             notification: bool = False) -> Any:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        if not notification:
            payload["id"] = self._next_id
            self._next_id += 1
        headers = dict(self._headers)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        resp = self._client.post(self._mcp_url, json=payload, headers=headers)
        resp.raise_for_status()
        if notification:
            return None
        sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
        if sid:
            self._session_id = sid
        ctype = resp.headers.get("content-type", "")
        if "text/event-stream" in ctype:
            data = None
            for line in resp.text.splitlines():
                if line.startswith("data:"):
                    data = json.loads(line[len("data:"):].strip())
            if data is None:
                raise RuntimeError(f"No data event in SSE response for {method}")
            body = data
        else:
            body = resp.json()
        if "error" in body:
            raise RuntimeError(f"MCP error for {method}: {body['error']}")
        return body.get("result")

    def connect(self) -> None:
        self._rpc("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "map-forecasting", "version": "0.1"},
        })
        self._rpc("notifications/initialized", {}, notification=True)

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None,
                  retries: int = 3) -> Any:
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                result = self._rpc("tools/call", {"name": name, "arguments": arguments or {}})
                if result.get("isError"):
                    raise RuntimeError(f"tool {name} error: {result.get('content')}")
                # Prefer structuredContent; fall back to first text content block.
                if "structuredContent" in result and result["structuredContent"] is not None:
                    return result["structuredContent"]
                for block in result.get("content", []):
                    if block.get("type") == "text":
                        try:
                            return json.loads(block["text"])
                        except (json.JSONDecodeError, TypeError):
                            return block["text"]
                return result
            except (httpx.HTTPStatusError, httpx.TransportError, RuntimeError) as exc:
                last = exc
                retryable = isinstance(exc, httpx.TransportError) or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code in (401, 429, 502, 503)
                )
                if not retryable or attempt == retries:
                    raise
                time.sleep(2.0 * attempt)
                self._session_id = None
                self.connect()
        raise last  # type: ignore[misc]

    def list_tools(self) -> list[str]:
        return [t["name"] for t in (self._rpc("tools/list") or {}).get("tools", [])]

    def close(self) -> None:
        self._client.close()


def project_url(project_id: str | int, base_url: str | None = None) -> str:
    return f"{(base_url or _base_url()).rstrip('/')}/projects/{project_id}"
