"""Radiant map adapters.

  from_radiant_export   project export with project_nodes / project_edges
  from_radiant_v2       project export v2: a single `canvas` holding nodes and edges
  to_radiant_payload    a MapIR as a payload Radiant can import

In v2 exports, node titles live at data.title, except metaculus-question-node which
carries data.metaculus.title. text-node is a canvas sticky (annotation), not a
factor, so it is dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

from .ir import EDGE_TYPES, Edge, MapIR, slugify

SKIP_TYPES = {"text-node"}


def node_title(n: dict) -> str | None:
    d = n.get("data") or {}
    t = (d.get("title") or "").strip()
    if t:
        return t
    m = d.get("metaculus") or {}
    t = (m.get("title") or m.get("source_title") or "").strip()
    return t or None


def from_radiant_v2(path: Path, name: str | None = None) -> MapIR:
    d = json.loads(Path(path).read_text())
    canvas = d["canvas"]
    taken: set[str] = set()
    id_to_slug: dict[str, str] = {}
    nodes: dict[str, dict] = {}
    for n in canvas.get("nodes", []):
        if n.get("node_type") in SKIP_TYPES:
            continue
        title = node_title(n)
        if not title:
            continue
        data = n.get("data") or {}
        slug = slugify(title, taken)
        id_to_slug[n["id"]] = slug
        nodes[slug] = {
            "title": title,
            "desc": (data.get("description") or "").strip(),
            "is_main": bool(data.get("is_main_node")),
            "node_type": n.get("node_type", "custom-node"),
        }
    edges: list[Edge] = []
    for e in canvas.get("edges", []):
        s = id_to_slug.get(e.get("source_node_id"))
        t = id_to_slug.get(e.get("target_node_id"))
        if not s or not t or s == t:
            continue
        data = e.get("data") or {}
        arrow = data.get("arrow_direction")
        if arrow == "backward":
            s, t = t, s
        edges.append(Edge(source=s, target=t, type="causal",
                          directed=arrow not in ("both", "none"),
                          provenance={"arrow_direction": arrow}))
    proj = d.get("project") or {}
    return MapIR(name=name or proj.get("title") or Path(path).stem,
                 nodes=nodes, edges=edges,
                 meta={"source": "radiant_v2", "source_file": str(path),
                       "project_title": proj.get("title")})


def from_radiant_export(path: Path, name: str | None = None) -> MapIR:
    d = json.loads(path.read_text())
    raw_nodes = d["project_nodes"]
    raw_edges = d["project_edges"]
    taken: set[str] = set()
    id_to_slug: dict[str, str] = {}
    nodes: dict[str, dict] = {}
    for n in raw_nodes:
        data = n.get("data", {})
        title = (data.get("title") or "").strip()
        if not title:
            q = data.get("question")
            if isinstance(q, dict):
                title = (q.get("title") or "").strip()
            elif isinstance(q, str):
                title = q.strip()
        if not title:
            title = f"untitled {n.get('node_type', 'node')}"
        slug = slugify(title, taken)
        id_to_slug[n["node_id"]] = slug
        nodes[slug] = {
            "title": title,
            "desc": (data.get("description") or "").strip(),
            "is_main": bool(data.get("isMainNode")),
            "node_type": n.get("node_type", "custom"),
        }
    edges: list[Edge] = []
    for e in raw_edges:
        s, t = id_to_slug.get(e["source_node_id"]), id_to_slug.get(e["target_node_id"])
        if s is None or t is None or s == t:
            continue
        data = e.get("data", {})
        arrow = data.get("arrowDirection")  # forward | backward | both | none | None
        etype = data.get("type") or "unknown"
        if etype not in EDGE_TYPES:
            etype = "unknown"
        directed = arrow not in ("both", "none")
        if arrow == "backward":
            s, t = t, s
        edges.append(Edge(source=s, target=t, type=etype, directed=directed,
                          provenance={"arrowDirection": arrow, "edge_id": e.get("edge_id")}))
    return MapIR(
        name=name or path.stem,
        nodes=nodes,
        edges=edges,
        meta={"source_file": str(path), "project_title": d.get("project", {}).get("title")},
    )


def to_radiant_payload(ir: MapIR) -> dict:
    """Minimal payload accepted by Radiant's lenient importProjectFromPayload —
    for viewing a generated map in Radiant."""
    nodes = []
    # simple grid layout since we carry no positions
    for i, (slug, n) in enumerate(ir.nodes.items()):
        nodes.append({
            "node_id": slug,
            "node_type": "custom",
            "position_x": (i % 6) * 320,
            "position_y": (i // 6) * 180,
            "data": {"title": n["title"], "description": n.get("desc", ""),
                     "nodeType": "custom", **({"isMainNode": True} if n.get("is_main") else {})},
        })
    edges = []
    for i, e in enumerate(ir.edges):
        edges.append({
            "edge_id": f"edge-{i}",
            "source_node_id": e.source,
            "target_node_id": e.target,
            "edge_type": "custom",
            "data": {"type": e.type if e.type != "unknown" else "causal",
                     "arrowDirection": "forward" if e.directed else "none"},
        })
    return {"project": {"title": ir.name, "project_type": "free_form"},
            "project_nodes": nodes, "project_edges": edges}
