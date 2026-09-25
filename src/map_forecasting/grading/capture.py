"""Capture grading: how much of each reference node a matched node really expresses.

Alignment credits a reference node fully as soon as any generated node matches
it. That is generous: a broad generated node ("market size") can be matched to
several narrower reference nodes it only gestures at. Capture grading asks, for
each matched pair, whether the reference node's meaning is

  full    -> 1.0   fully recoverable from the generated node
  partial -> 0.5   partly there; real content would be lost
  absent  -> 0.0   not actually expressed; the match was wrong

Graded coverage is the mean credit over reference nodes, where a reference node
matched by several generated nodes takes its best credit. Binary coverage is the
same with every match worth 1.0, so graded <= binary, and the gap measures how
loose the alignment was.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ..maps.ir import MapIR
from ..llm import call_structured

CAPTURE_SYSTEM = """You grade one node from a GENERATED expert dependency map \
against several nodes from a REFERENCE map that an automatic aligner claims it \
covers. For each reference node, judge how well the generated node expresses \
it, matching on meaning, not wording:
  full    — the reference node's meaning is fully recoverable from the \
generated node (identical scope, or a broader umbrella that clearly includes it)
  partial — the generated node expresses part of it, but a reader of the \
generated map would lose real content the reference node carries
  absent  — the generated node does not actually express this reference node; \
the claimed match is wrong
Judge as a domain expert would."""

CREDIT = {"full": 1.0, "partial": 0.5, "absent": 0.0}


class RefGrade(BaseModel):
    ref_slug: str
    capture: Literal["full", "partial", "absent"]


class CaptureGrades(BaseModel):
    grades: list[RefGrade]


def _text(nd) -> str:
    d = f" — {nd['desc']}" if nd.get("desc") else ""
    return f"{nd['title']}{d}"


def capture_prompt(gen: MapIR, ref: MapIR, gen_slug: str,
                   ref_slugs: list[str]) -> str:
    # the wording is the same for one reference node or five, so one-to-one
    # pairs are graded on the same terms as merges
    lines = [f"GENERATED node:\n  {_text(gen.nodes[gen_slug])}", "",
             "REFERENCE nodes it is claimed to cover:"]
    for r in sorted(ref_slugs):
        lines.append(f"  {r}: {_text(ref.nodes[r])}")
    lines.append("")
    lines.append("Grade each reference node: full, partial, or absent.")
    return "\n".join(lines)


def all_matched(alignment: dict[str, list[str]]) -> dict[str, list[str]]:
    """{gen_slug: [ref_slug, ...]} for every generated node that matched.

    Grade all of these, not only the merges: one-to-one matches can be loose too.
    """
    covers: dict[str, list[str]] = {}
    for r, gs in alignment.items():
        for g in gs:
            covers.setdefault(g, []).append(r)
    return covers


def one_to_many(alignment: dict[str, list[str]]) -> dict[str, list[str]]:
    """{gen_slug: [ref_slug, ...]} restricted to generated nodes covering more than one."""
    return {g: rs for g, rs in all_matched(alignment).items() if len(rs) > 1}


def grade_gen_node(gen: MapIR, ref: MapIR, gen_slug: str, ref_slugs: list[str],
                   llm=None) -> dict[str, str]:
    """{ref_slug: full|partial|absent} for one generated node."""
    result = call_structured(capture_prompt(gen, ref, gen_slug, ref_slugs),
                             CaptureGrades, system=CAPTURE_SYSTEM,
                             max_tokens=8000, llm=llm)
    out = {g.ref_slug: g.capture for g in result.grades if g.ref_slug in ref_slugs}
    # pairs the grader skipped stay ungraded and count as binary credit
    return out


def graded_coverage(ref: MapIR, alignment: dict[str, list[str]],
                    grades: dict[tuple[str, str], str]) -> dict:
    """grades: {(gen_slug, ref_slug): full|partial|absent}.

    Pairs without a grade count as 1.0 (binary credit); `n_graded_pairs` and
    `n_ungraded_pairs` report how many of each went into the number.

    A reference node matched by several generated nodes takes its BEST credit:
    if any one generated node captures it fully, it is not missing from the map.
    """
    credit_sum, graded, ungraded = 0.0, 0, 0
    for r, gs in alignment.items():
        best = 0.0
        for g in gs:
            if (g, r) in grades:
                best = max(best, CREDIT[grades[(g, r)]])
                graded += 1
            else:
                best = 1.0
                ungraded += 1
        credit_sum += best
    n = len(ref.nodes)
    return {
        "binary_coverage": round(len(alignment) / n, 3) if n else 0.0,
        "graded_coverage": round(credit_sum / n, 3) if n else 0.0,
        "n_ref_nodes": n,
        "n_matched": len(alignment),
        "n_graded_pairs": graded,
        "n_ungraded_pairs": ungraded,
    }
