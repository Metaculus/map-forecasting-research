"""Node alignment: which nodes in one map express the same concept as nodes in another.

Given a GENERATED node set and a REFERENCE node set, produce a correspondence.
Many-to-one is allowed (a generated node may cover several reference nodes and
vice versa) so that granularity mismatch — the one problem tooling can't fix —
is scored as coverage rather than penalized arbitrarily.

Returns Alignment: for each reference slug, the generated slug(s) that express it
(or none). Used by `scoring.score_map` to relabel a generated map into the
reference's slug space before edge metrics.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..maps.ir import MapIR
from ..llm import call_structured

ALIGN_SYSTEM = """You align two lists of concept nodes from expert dependency \
maps: a GENERATED map and a REFERENCE map. For each reference node, identify \
which generated node(s) express the same concept — matching on meaning, not \
wording. A reference node may be matched by one generated node, by several (if \
the generated map split it), or by none (if the generated map missed it). Two \
nodes match if a domain expert would call them the same factor. Give each match \
a confidence in [0,1]."""


class RefMatch(BaseModel):
    ref_slug: str
    generated_slugs: list[str] = Field(
        description="generated slug(s) expressing this reference node; empty if missed")
    confidence: float = Field(ge=0, le=1, default=1.0)


class Alignment(BaseModel):
    matches: list[RefMatch]


def _block(ir: MapIR, label: str) -> str:
    lines = [f"{label} NODES:"]
    for slug, n in ir.nodes.items():
        d = f" — {n['desc']}" if n.get("desc") else ""
        lines.append(f"  {slug}: {n['title']}{d}")
    return "\n".join(lines)


def align_nodes(generated: MapIR, reference: MapIR, threshold: float = 0.5,
                llm=None) -> dict[str, list[str]]:
    """Returns {reference_slug: [generated_slug, ...]} above threshold."""
    prompt = (
        f"{_block(generated, 'GENERATED')}\n\n{_block(reference, 'REFERENCE')}\n\n"
        f"For every REFERENCE node, list the GENERATED node(s) that express it."
    )
    n = len(generated.nodes) + len(reference.nodes)
    max_tokens = 12000 if n < 120 else 32000
    result = call_structured(prompt, Alignment, system=ALIGN_SYSTEM,
                             max_tokens=max_tokens, llm=llm)
    out: dict[str, list[str]] = {}
    for m in result.matches:
        if m.ref_slug not in reference.nodes or m.confidence < threshold:
            continue
        gs = [g for g in m.generated_slugs if g in generated.nodes]
        if gs:
            out[m.ref_slug] = gs
    return out
