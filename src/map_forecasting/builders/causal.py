"""Build a map's nodes, its edges, or both, from a topic and a goal.

  problem_brief  a short framing of the problem space, from topic and goal only
  build_nodes    the node set; method "single" (one call) or "workshop"
                 (over-generate across opportunities, constraints and
                 externalities, then cluster and dedupe)
  add_edges      propose the edges between a map's existing nodes
  expand_layer   given a partial map, propose the next layer of causes
  build_map      nodes and edges in one call

Edges point from cause to effect, toward the goal node(s).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..grading.scoring import EdgePrediction
from ..llm import call_structured
from ..maps.ir import Edge, MapIR, slugify

NODE_SYSTEM = """You build the NODES of a causal map for a forecasting problem: \
the variables, milestones, enabling technologies, bottlenecks, constraints, and \
externalities that drive the goal. Each node is a specific, ideally falsifiable \
factor, not a vague theme. One concept per node; split compound factors. Aim for \
the requested count."""

EDGE_SYSTEM = """You build the dependency structure of a causal map for a \
forecasting problem. You are given the map's nodes (variables, milestones, goals). \
Propose the DIRECT dependency edges.

Edge convention: source -> target means "source influences / enables / provides \
evidence about target". Goal nodes are sinks. Only propose direct relationships; \
do not add an edge that is merely implied transitively through another node. \
Good maps are sparse: roughly one edge per node, not dense meshes.

Types: causal (source drives target), necessary (target can't happen without \
source), sufficient (source alone brings about target), evidential (source is \
evidence about target).

Give each edge a confidence in [0,1] that it belongs in the map. Include \
lower-confidence candidates too; calibration matters more than precision."""

BRIEF_SYSTEM = """You write the framing a group is given before it builds a causal \
map of the key factors that would help a forecaster understand a TARGET: what \
drives it, what blocks it, what would have to be true. You are given only the topic \
and the target. Sketch the problem space as a well-informed outsider would: the \
time horizon, what is at stake, the kinds of forces likely to matter, and the \
wild cards. Do NOT enumerate specific factors or propose a causal structure. \
About 150 words."""

Category = Literal["opportunity", "constraint", "externality", "milestone", "goal"]


class GenNode(BaseModel):
    title: str
    desc: str = Field(default="", description="one sentence, optional")
    category: Optional[Category] = None


class NodeList(BaseModel):
    nodes: list[GenNode]


class Variables(BaseModel):
    variables: list[GenNode] = Field(description="over-generated; about 1.5x the target")


class ProposedEdge(BaseModel):
    source: str
    target: str
    type: Literal["causal", "necessary", "evidential", "sufficient"]
    confidence: float = Field(ge=0, le=1)


class EdgeProposal(BaseModel):
    edges: list[ProposedEdge]


class SourceEdges(BaseModel):
    edges: list[ProposedEdge] = Field(description="edges where THIS node is the source; may be empty")


class LayerExpansion(BaseModel):
    new_nodes: list[GenNode] = Field(description="the next layer: direct causes of the frontier nodes")
    new_edges: list[ProposedEdge] = Field(description="edges from each new node into the frontier")


class FullMap(BaseModel):
    nodes: list[GenNode]
    edges: list[ProposedEdge]


class Brief(BaseModel):
    text: str


def _tokens(n: int, small: int, large: int, cut: int = 60) -> int:
    return small if n < cut else large


def _header(topic: str, goal: str | None, brief: str | None) -> str:
    parts = [f"TOPIC: {topic}", f"GOAL / TARGET: {goal or '(not specified)'}"]
    if brief:
        parts.append(f"PROBLEM-SPACE BRIEF:\n{brief}")
    return "\n".join(parts)


def _to_ir(gen: list[GenNode], name: str, topic: str | None = None) -> MapIR:
    taken: set[str] = set()
    nodes: dict[str, dict] = {}
    for g in gen:
        title = g.title.strip()
        if not title:
            continue
        nodes[slugify(title, taken)] = {"title": title, "desc": g.desc.strip(),
                                        "is_main": g.category == "goal",
                                        "node_type": "custom", "category": g.category}
    return MapIR(name=name, nodes=nodes, edges=[],
                 meta={"generated": True, "project_title": topic})


def _node_block(ir: MapIR) -> str:
    lines = []
    for slug, n in ir.nodes.items():
        d = f" — {n['desc']}" if n.get("desc") else ""
        m = " [goal]" if n.get("is_main") else ""
        lines.append(f"  {slug}: {n['title']}{m}{d}")
    return "\n".join(lines)


def _validate(edges: list[ProposedEdge], ir: MapIR) -> list[EdgePrediction]:
    out, seen = [], set()
    for e in edges:
        if e.source not in ir.nodes or e.target not in ir.nodes or e.source == e.target:
            continue
        k = frozenset((e.source, e.target))
        if k in seen:
            continue
        seen.add(k)
        out.append(EdgePrediction(source=e.source, target=e.target,
                                  type=e.type, confidence=e.confidence))
    return out


def problem_brief(topic: str, goal: str, llm=None) -> str:
    """A ~150-word framing of the problem space, written from topic and goal only."""
    prompt = f"Topic: {topic}\nTarget: {goal}\n\nWrite the framing for this problem space."
    return call_structured(prompt, Brief, system=BRIEF_SYSTEM, max_tokens=4000,
                           llm=llm).text


def build_nodes(topic: str, goal: str | None = None, n: int = 20,
                brief: str | None = None, method: str = "workshop",
                llm=None) -> MapIR:
    """The node set for a map of `topic` leading to `goal`. No edges."""
    header = _header(topic, goal, brief)
    if method == "single":
        prompt = (f"{header}\n\nBuild about {n} nodes for this map: the goal node(s), "
                  f"enabling factors, bottlenecks, and externalities.")
        res = call_structured(prompt, NodeList, system=NODE_SYSTEM,
                              max_tokens=_tokens(n, 8000, 24000), llm=llm)
        return _to_ir(res.nodes, topic, topic=topic)
    if method != "workshop":
        raise ValueError("method must be 'single' or 'workshop'")
    var_prompt = (
        f"{header}\n\nBrainstorm a broad list of about {int(n * 1.5)} factors that "
        f"would drive this problem toward or away from its goal. Deliberately "
        f"over-generate. Cover all three categories: opportunity-space levers "
        f"(things someone could act on), system constraints (things outside anyone's "
        f"control), and externalities. Tag each with its category.")
    varset = call_structured(var_prompt, Variables, system=NODE_SYSTEM,
                             max_tokens=_tokens(n, 12000, 32000, 40), llm=llm)
    listing = "\n".join(f"  - [{v.category}] {v.title}" for v in varset.variables)
    clust_prompt = (
        f"{header}\n\nRaw brainstormed factors:\n{listing}\n\n"
        f"Cluster and dedupe these into a clean set of about {n} map nodes: merge "
        f"redundant factors, split compound ones, keep the goal node(s). Preserve "
        f"category tags.")
    clustered = call_structured(clust_prompt, NodeList, system=NODE_SYSTEM,
                                max_tokens=_tokens(n, 32000, 64000, 40), llm=llm)
    ir = _to_ir(clustered.nodes, topic, topic=topic)
    ir.meta["n_raw_variables"] = len(varset.variables)
    return ir


def propose_edges(ir: MapIR, method: str = "single_shot", brief: str | None = None,
                  llm=None) -> list[EdgePrediction]:
    """Candidate edges between `ir`'s nodes, each with a confidence."""
    goals = "; ".join(n["title"] for n in ir.nodes.values() if n.get("is_main")) or None
    header = _header(ir.meta.get("project_title") or ir.name, goals, brief)
    block = _node_block(ir)
    if method == "single_shot":
        prompt = (f"{header}\n\nNODES:\n{block}\n\nThe map has {len(ir.nodes)} nodes; "
                  f"expect a similar number of edges. Propose the edge list.")
        res = call_structured(prompt, EdgeProposal, system=EDGE_SYSTEM,
                              max_tokens=_tokens(len(ir.nodes), 16000, 32000), llm=llm)
        return _validate(res.edges, ir)
    if method != "per_source":
        raise ValueError("method must be 'single_shot' or 'per_source'")
    best: dict[frozenset, ProposedEdge] = {}
    for slug, node in ir.nodes.items():
        prompt = (f"{header}\n\nNODES:\n{block}\n\nFocus node: {slug} ({node['title']}).\n"
                  f"Which other nodes does {slug} DIRECTLY influence, enable, or evidence? "
                  f"Return edges with {slug} as source (an empty list is fine).")
        res = call_structured(prompt, SourceEdges, system=EDGE_SYSTEM, max_tokens=8000,
                              llm=llm)
        for e in res.edges:
            k = frozenset((e.source, e.target))
            if e.source == slug and (k not in best or e.confidence > best[k].confidence):
                best[k] = e
    return _validate(list(best.values()), ir)


def add_edges(ir: MapIR, threshold: float = 0.5, **kwargs) -> MapIR:
    """A copy of `ir` with proposed edges at or above `threshold` confidence.
    Other arguments (method, brief, llm) go to `propose_edges`."""
    preds = propose_edges(ir, **kwargs)
    edges = [Edge(source=p.source, target=p.target, type=p.type,
                  provenance={"confidence": p.confidence})
             for p in preds if p.confidence >= threshold]
    return MapIR(name=ir.name, nodes=dict(ir.nodes), edges=list(ir.edges) + edges,
                 meta=dict(ir.meta))


def expand_layer(ir: MapIR, frontier: list[str] | None = None,
                 llm=None) -> MapIR:
    """Propose the direct causes of the frontier nodes (by default, the nodes
    farthest from the goal) and return the map with them added."""
    if frontier is None:
        sinks = ir.sinks() or [s for s, n in ir.nodes.items() if n.get("is_main")]
        layers = ir.layers_from(sinks, reverse=True) if sinks else {}
        deepest = max(layers.values()) if layers else 0
        frontier = [s for s, d in layers.items() if d == deepest] or list(ir.nodes)
    prompt = (
        f"TOPIC: {ir.meta.get('project_title') or ir.name}\n\n"
        f"Map so far (goal first, working outward):\n"
        + "\n".join(f"  {s}: {n['title']}" for s, n in ir.nodes.items())
        + f"\n\nThe current frontier nodes are: {', '.join(sorted(frontier))}.\n"
        f"Propose the NEXT layer of nodes, the factors that directly influence or "
        f"enable the frontier nodes, and the edges from each new node into the frontier.")
    res = call_structured(prompt, LayerExpansion, system=EDGE_SYSTEM, max_tokens=12000,
                          llm=llm)
    taken = set(ir.nodes)
    nodes = {s: dict(n) for s, n in ir.nodes.items()}
    slug_of: dict[str, str] = {}
    for g in res.new_nodes:
        slug = slugify(g.title, taken)
        slug_of[g.title] = slug
        nodes[slug] = {"title": g.title, "desc": g.desc, "is_main": False,
                       "node_type": "custom", "category": g.category}
    edges = list(ir.edges)
    for e in res.new_edges:
        s = slug_of.get(e.source, e.source)
        t = e.target if e.target in nodes else slug_of.get(e.target, e.target)
        if s in nodes and t in nodes and s != t:
            edges.append(Edge(source=s, target=t, type=e.type,
                              provenance={"confidence": e.confidence}))
    return MapIR(name=ir.name, nodes=nodes, edges=edges, meta=dict(ir.meta))


def build_map(topic: str, goal: str | None = None, n: int = 20, brief: str | None = None,
              llm=None) -> MapIR:
    """Nodes and edges in a single call."""
    prompt = (f"{_header(topic, goal, brief)}\n\nBuild the full causal map for this "
              f"problem: about {n} nodes (goal, enablers, bottlenecks, externalities) "
              f"and the edges between them.")
    res = call_structured(prompt, FullMap, system=EDGE_SYSTEM + "\n\n" + NODE_SYSTEM,
                          max_tokens=_tokens(n, 16000, 32000), llm=llm)
    ir = _to_ir(res.nodes, topic, topic=topic)
    by_title = {node["title"]: s for s, node in ir.nodes.items()}
    for e in res.edges:
        s = e.source if e.source in ir.nodes else by_title.get(e.source)
        t = e.target if e.target in ir.nodes else by_title.get(e.target)
        if s and t and s != t:
            ir.edges.append(Edge(source=s, target=t, type=e.type,
                                 provenance={"confidence": e.confidence}))
    return ir
