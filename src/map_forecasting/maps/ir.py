"""Common map format (MapIR) and graph utilities.

Every adapter (Guesstimate, Squiggle Hub, Radiant exports) and every builder
produces a MapIR, and the grading tools compare two of them.

Shape:
{
  "name": "example",
  "nodes": {slug: {"title": ..., "desc": ..., "is_main": bool, "node_type": ...}},
  "edges": [{"source": slug, "target": slug, "type": "causal|necessary|evidential|sufficient|unknown",
             "directed": bool, "provenance": {...}}],
}
Edges point from cause to effect, toward the goal node(s).
No layout, no UUIDs, no timestamps.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

EDGE_TYPES = ("causal", "necessary", "evidential", "sufficient", "unknown")


def slugify(title: str, taken: set[str]) -> str:
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    s = "_".join(s.split("_")[:6]) or "node"
    base, i = s, 2
    while s in taken:
        s = f"{base}_{i}"
        i += 1
    taken.add(s)
    return s


@dataclass
class Edge:
    source: str
    target: str
    type: str = "unknown"
    directed: bool = True
    provenance: dict = field(default_factory=dict)

    def key_undirected(self) -> frozenset:
        return frozenset((self.source, self.target))

    def key_directed(self) -> tuple:
        return (self.source, self.target)


@dataclass
class MapIR:
    name: str
    nodes: dict[str, dict]  # slug -> {title, desc, is_main, node_type}
    edges: list[Edge]
    meta: dict = field(default_factory=dict)

    # ---------- serialization ----------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "meta": self.meta,
            "nodes": self.nodes,
            "edges": [vars(e) for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MapIR":
        return cls(
            name=d["name"],
            nodes=d["nodes"],
            edges=[Edge(**e) for e in d["edges"]],
            meta=d.get("meta", {}),
        )

    def save(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: Path) -> "MapIR":
        return cls.from_dict(json.loads(path.read_text()))

    # ---------- graph utils ----------
    def adjacency(self) -> dict[str, list[str]]:
        adj = defaultdict(list)
        for e in self.edges:
            adj[e.source].append(e.target)
        return adj

    def degrees(self) -> tuple[dict[str, int], dict[str, int]]:
        indeg = {s: 0 for s in self.nodes}
        outdeg = {s: 0 for s in self.nodes}
        for e in self.edges:
            outdeg[e.source] = outdeg.get(e.source, 0) + 1
            indeg[e.target] = indeg.get(e.target, 0) + 1
        return indeg, outdeg

    def sinks(self) -> list[str]:
        """Nodes with no outgoing edges but at least one incoming edge."""
        indeg, outdeg = self.degrees()
        return [s for s in self.nodes if outdeg.get(s, 0) == 0 and indeg.get(s, 0) > 0]

    def sources(self) -> list[str]:
        indeg, outdeg = self.degrees()
        return [s for s in self.nodes if indeg.get(s, 0) == 0 and outdeg.get(s, 0) > 0]

    def cycles(self) -> list[list[str]]:
        """Return one representative cycle per back-edge found via DFS."""
        adj = self.adjacency()
        color: dict[str, int] = {}
        stack: list[str] = []
        found: list[list[str]] = []

        def dfs(u: str):
            color[u] = 1
            stack.append(u)
            for v in adj.get(u, []):
                if color.get(v) == 1:
                    found.append(stack[stack.index(v):] + [v])
                elif color.get(v) is None:
                    dfs(v)
            stack.pop()
            color[u] = 2

        for u in list(self.nodes):
            if color.get(u) is None:
                dfs(u)
        return found

    def layers_from(self, roots: list[str], reverse: bool = True) -> dict[str, int]:
        """BFS layer number for each reachable node, starting at `roots` (layer 0).

        reverse=True walks edges backwards (i.e. from a sink/goal toward its
        ancestors), which is the natural layering when edges point toward the goal.
        """
        adj = defaultdict(list)
        for e in self.edges:
            if reverse:
                adj[e.target].append(e.source)
            else:
                adj[e.source].append(e.target)
        layer = {r: 0 for r in roots}
        frontier = list(roots)
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj.get(u, []):
                    if v not in layer:
                        layer[v] = layer[u] + 1
                        nxt.append(v)
            frontier = nxt
        return layer
