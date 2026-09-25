"""One-call grading of a map against a reference map."""
from __future__ import annotations

from ..maps.ir import MapIR
from .align import align_nodes
from .capture import all_matched, grade_gen_node, graded_coverage
from .scoring import score_map


def grade_map(generated: MapIR, reference: MapIR, capture: bool = True,
              llm=None) -> dict:
    """Align the two maps' nodes, score nodes, edges and reachability, and
    (with `capture=True`) grade how fully each matched node is expressed.

    Returns the `scoring.score_map` result, plus "coverage_graded" when capture
    grading ran and the alignment itself under "alignment".
    """
    alignment = align_nodes(generated, reference, llm=llm)
    result = score_map(generated, reference, alignment)
    result["alignment"] = alignment
    if capture:
        grades: dict[tuple[str, str], str] = {}
        for gen_slug, ref_slugs in all_matched(alignment).items():
            for ref_slug, g in grade_gen_node(generated, reference, gen_slug, ref_slugs,
                                              llm=llm).items():
                grades[(gen_slug, ref_slug)] = g
        result["coverage_graded"] = graded_coverage(reference, alignment, grades)
    return result
