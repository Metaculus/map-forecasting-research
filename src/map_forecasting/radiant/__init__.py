"""Radiant integration: a synchronous MCP client and URL helpers.

The pydantic-ai toolset lives in `map_forecasting.radiant.toolset` and needs the
`radiant` extra (pip install "map-forecasting[radiant]").
"""
from .http_client import RadiantMCP, project_url
from .urls import extract_radiant_map_url, extract_radiant_map_url_from_messages

__all__ = ["RadiantMCP", "project_url", "extract_radiant_map_url",
           "extract_radiant_map_url_from_messages"]
