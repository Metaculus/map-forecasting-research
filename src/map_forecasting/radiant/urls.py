"""Extract Radiant project URLs from agent text or tool message history."""

from __future__ import annotations

import re
from typing import Any

from .http_client import project_url as project_url_from_id

# Any https host whose name contains "radiant", followed by /projects/<id>.
_URL_RE = re.compile(r"https://[A-Za-z0-9.-]*radiant[A-Za-z0-9.-]*/projects/([A-Za-z0-9_-]+)")
_PROJECT_ID_RE = re.compile(
    r"""["']?project_id["']?\s*[:=]\s*["']?([A-Za-z0-9_-]+)["']?""",
    re.IGNORECASE,
)


def extract_radiant_map_url(text: str | None) -> str | None:
    """Return the first Radiant project URL found in text, if any.

    Preserves whichever host was in the text.
    """
    if not text:
        return None
    match = _URL_RE.search(text)
    if not match:
        return None
    return match.group(0)


def extract_radiant_map_url_from_messages(messages: Any) -> str | None:
    """Scan serialized message history for a Radiant project URL or project_id."""
    if messages is None:
        return None
    try:
        blob = str(messages)
    except Exception:
        return None
    from_url = extract_radiant_map_url(blob)
    if from_url:
        return from_url
    project_id_match = _PROJECT_ID_RE.search(blob)
    if project_id_match:
        return project_url_from_id(project_id_match.group(1))
    return None
