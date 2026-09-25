"""Map cleaning: find and fix likely errors in a hand-drawn map.

  run_audit    LLM audit: reversed arrows, wrong link types, non-relationships,
               duplicate or vague nodes, each with a severity
  apply_fixes  reverse flagged arrows, fill in types, drop high-severity
               non-relationships; nodes are never deleted
  clean_map    both steps

Cycles are reported, not treated as errors: real systems have feedback loops.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .ir import Edge, MapIR
from ..llm import call_structured

EDGE_ERROR_CLASSES = (
    "reversed_direction",   # arrow points the wrong way given the stated convention
    "wrong_type",           # causal/necessary/evidential/sufficient mislabeled
    "not_a_relationship",   # no meaningful direct dependency between the nodes
    "redundant",            # duplicate of / implied by another edge (e.g. transitive)
    "should_be_undirected", # genuinely mutual influence, arrow is spurious precision
    "missing_type",         # no type recorded
)

NODE_ERROR_CLASSES = (
    "vague_or_untestable",  # not a variable/claim; couldn't be operationalized
    "duplicate_node",       # same concept as another node
    "orphan",               # disconnected but looks like it should link somewhere
    "compound",             # bundles several distinct variables in one node
)


class EdgeFinding(BaseModel):
    source: str = Field(description="slug of the edge's current source node")
    target: str = Field(description="slug of the edge's current target node")
    error_class: Literal[
        "reversed_direction", "wrong_type", "not_a_relationship",
        "redundant", "should_be_undirected", "missing_type"]
    severity: Literal["low", "medium", "high"]
    explanation: str = Field(description="one sentence")
    suggested_type: Optional[Literal["causal", "necessary", "evidential", "sufficient"]] = Field(
        default=None, description="only for wrong_type / missing_type")


class NodeFinding(BaseModel):
    slug: str
    error_class: Literal["vague_or_untestable", "duplicate_node", "orphan", "compound"]
    severity: Literal["low", "medium", "high"]
    explanation: str
    duplicate_of: Optional[str] = None


class MapAudit(BaseModel):
    convention_assessment: str = Field(
        description="1-3 sentences: does this map follow source->target = 'source influences/enables target'? Or was it drawn goal-outward?")
    sink_slugs: list[str] = Field(
        description="the goal / outcome nodes this map is ultimately about (usually 1-3)")
    edge_findings: list[EdgeFinding]
    node_findings: list[NodeFinding]


AUDIT_SYSTEM = """You are auditing causal dependency maps built by people. \
Maps have nodes (variables, milestones, goals) \
and typed edges (causal, necessary, evidential, sufficient).

The INTENDED convention: an edge source -> target means "source influences / \
enables / provides evidence about target". Consequently the map's ultimate goal \
node(s) should be SINKS (arrows flow into them), not sources.

A common error in hand-drawn maps is a reversed \
arrow: the semantic relationship is right but the recorded direction is backwards. \
Judge direction by the meaning of the node titles/descriptions, not by graph \
position. Cycles are NOT automatically errors — real systems have feedback loops.

Report every defect you find, including low-severity ones — a downstream filter \
will rank them. severity: high = the edge/node is actively misleading as recorded; \
medium = clearly wrong but recoverable; low = cosmetic or debatable."""


def audit_prompt(ir: MapIR) -> str:
    lines = [f"MAP: {ir.meta.get('project_title') or ir.name}", "", "NODES:"]
    for slug, n in ir.nodes.items():
        d = f" — {n['desc']}" if n.get("desc") else ""
        m = " [flagged as main node]" if n.get("is_main") else ""
        lines.append(f"  {slug}: {n['title']}{m}{d}")
    lines.append("")
    lines.append("EDGES (source -> target [type, directed?]):")
    for e in ir.edges:
        arrow = "->" if e.directed else "<->"
        lines.append(f"  {e.source} {arrow} {e.target} [{e.type}]")
    lines.append("")
    lines.append(
        "Audit this map: identify the goal/sink node(s), assess whether the "
        "direction convention is followed, and list every edge and node defect.")
    return "\n".join(lines)


def run_audit(ir: MapIR, llm=None) -> MapAudit:
    n_items = len(ir.nodes) + len(ir.edges)
    max_tokens = 16000 if n_items < 120 else 32000
    return call_structured(audit_prompt(ir), MapAudit, system=AUDIT_SYSTEM,
                           max_tokens=max_tokens, llm=llm)


def apply_fixes(ir: MapIR, audit: MapAudit,
                min_severity: str = "medium") -> MapIR:
    """Produce the 'fixed' map: reverse flagged arrows, fill types, drop
    not_a_relationship edges (high severity only). Nodes are never deleted."""
    sev_rank = {"low": 0, "medium": 1, "high": 2}
    threshold = sev_rank[min_severity]
    by_pair: dict[tuple, list[EdgeFinding]] = {}
    for f in audit.edge_findings:
        by_pair.setdefault((f.source, f.target), []).append(f)

    fixed_edges: list[Edge] = []
    applied = []
    for e in ir.edges:
        src, tgt, etype, directed = e.source, e.target, e.type, e.directed
        drop = False
        for f in by_pair.get((e.source, e.target), []):
            if sev_rank[f.severity] < threshold:
                continue
            if f.error_class == "reversed_direction":
                src, tgt = tgt, src
                applied.append(("reverse", e.source, e.target))
            elif f.error_class in ("wrong_type", "missing_type") and f.suggested_type:
                etype = f.suggested_type
                applied.append(("retype", e.source, e.target, f.suggested_type))
            elif f.error_class == "should_be_undirected":
                directed = False
                applied.append(("undirect", e.source, e.target))
            elif f.error_class == "not_a_relationship" and f.severity == "high":
                drop = True
                applied.append(("drop", e.source, e.target))
        if not drop:
            fixed_edges.append(Edge(source=src, target=tgt, type=etype, directed=directed,
                                    provenance={**e.provenance, "fixed": True}))

    fixed = MapIR(name=ir.name, nodes=dict(ir.nodes), edges=fixed_edges,
                  meta={**ir.meta, "fixed": True,
                        "audit_sinks": audit.sink_slugs,
                        "fixes_applied": applied,
                        "cycles_after_fix": None})
    fixed.meta["cycles_after_fix"] = [c for c in fixed.cycles()]
    return fixed


def clean_map(ir: MapIR, llm=None,
              min_severity: str = "medium") -> tuple[MapAudit, MapIR]:
    """Audit a map and return (audit, fixed map)."""
    audit = run_audit(ir, llm=llm)
    return audit, apply_fixes(ir, audit, min_severity=min_severity)
