"""Guesstimate (getguesstimate.com) adapter.

Public models are served as JSON from https://api.getguesstimate.com/spaces/<id>
(the www. site is a client-rendered Next.js app; the old
guesstimate.herokuapp.com endpoint is dead). The graph lives in
d["graph"]["metrics"] (nodes) and d["graph"]["guesstimates"] (one cell per
metric: an expression, a distribution, or a point value).

Edges are implicit: a FUNCTION expression references its inputs as
`${metric:<uuid>}`. We emit input -> dependent-metric, so edges point toward
the model's outputs.
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from .ir import Edge, MapIR, slugify

API_URL = "https://api.getguesstimate.com/spaces/{space_id}"
METRIC_REF = re.compile(r"\$\{metric:([0-9a-f-]+)\}")


def fetch_space(space_id: int | str) -> dict:
    req = urllib.request.Request(API_URL.format(space_id=space_id),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def from_guesstimate(d: dict, name: str | None = None) -> MapIR:
    graph = d["graph"]
    cells = {g["metric"]: g for g in graph.get("guesstimates", []) if g.get("metric")}
    taken: set[str] = set()
    id_to_slug: dict[str, str] = {}
    nodes: dict[str, dict] = {}
    for m in graph.get("metrics", []):
        title = (m.get("name") or "").strip() or m.get("readableId") or "untitled"
        slug = slugify(title, taken)
        id_to_slug[m["id"]] = slug
        cell = cells.get(m["id"], {})
        nodes[slug] = {
            "title": title,
            "desc": (cell.get("description") or "").strip(),
            "is_main": False,
            "node_type": "guesstimate_metric",
            "guesstimate": {
                "readable_id": m.get("readableId"),
                "type": cell.get("guesstimateType"),
                "expression": cell.get("expression"),
            },
        }
    def readable(expr: str) -> str:
        return METRIC_REF.sub(lambda m: id_to_slug.get(m.group(1), m.group(1)), expr)

    edges: list[Edge] = []
    seen: set[tuple[str, str]] = set()
    for mid, cell in cells.items():
        target = id_to_slug.get(mid)
        expr = cell.get("expression") or ""
        if target is None:
            continue
        if "${" in expr:
            nodes[target]["guesstimate"]["expression_readable"] = readable(expr)
        for ref in METRIC_REF.findall(expr):
            source = id_to_slug.get(ref)
            if source is None or source == target or (source, target) in seen:
                continue
            seen.add((source, target))
            edges.append(Edge(source=source, target=target, type="causal",
                              directed=True,
                              provenance={"from": "guesstimate_expression",
                                          "expression": readable(expr)}))
    ir = MapIR(
        name=name or d.get("name") or f"guesstimate-{d.get('id')}",
        nodes=nodes,
        edges=edges,
        meta={"source": "guesstimate", "space_id": d.get("id"),
              "space_name": d.get("name"), "description": d.get("description"),
              "updated_at": d.get("updated_at")},
    )
    # mark the model's outputs (sinks) as main nodes — guesstimate has no
    # explicit goal flag, so terminal computed metrics are the best proxy
    for slug in ir.sinks():
        ir.nodes[slug]["is_main"] = True
    return ir


def from_guesstimate_file(path: Path, name: str | None = None) -> MapIR:
    ir = from_guesstimate(json.loads(path.read_text()), name=name)
    ir.meta["source_file"] = str(path)
    return ir
