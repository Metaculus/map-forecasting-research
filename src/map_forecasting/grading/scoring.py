"""Scoring a map against a reference map.

  score_edges  edges predicted over the reference's own nodes: undirected and
               directed precision/recall/F1, edge-type accuracy against the
               majority baseline, a PR curve over confidence, and calibration
  score_map    a map with its own nodes: after `align.align_nodes`, reports node P/R/F1 + coverage, edge
metrics on the induced subgraph over matched nodes, and reachability agreement
(a map that routes a dependency through a different intermediate is still
'right' — local edge F1 punishes that; reachability doesn't).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..maps.ir import Edge, MapIR


@dataclass
class EdgePrediction:
    source: str
    target: str
    type: str
    confidence: float  # 0..1
    directed: bool = True


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def score_edges(preds: list[EdgePrediction], ref: MapIR, threshold: float = 0.5) -> dict:
    ref_und = {frozenset((e.source, e.target)) for e in ref.edges}
    ref_dir = {(e.source, e.target) for e in ref.edges if e.directed}
    ref_type = {frozenset((e.source, e.target)): e.type for e in ref.edges
                if e.type != "unknown"}

    kept = [p for p in preds if p.confidence >= threshold]
    pred_und = {frozenset((p.source, p.target)) for p in kept}
    pred_dir = {(p.source, p.target) for p in kept if p.directed}

    und = _prf(len(pred_und & ref_und), len(pred_und - ref_und), len(ref_und - pred_und))
    dir_ = _prf(len(pred_dir & ref_dir), len(pred_dir - ref_dir), len(ref_dir - pred_dir))

    # type accuracy among undirected true positives that have a reference type
    type_hits, type_total = 0, 0
    for p in kept:
        k = frozenset((p.source, p.target))
        if k in ref_type:
            type_total += 1
            if p.type == ref_type[k]:
                type_hits += 1
    majority = Counter(ref_type.values()).most_common(1)
    majority_acc = majority[0][1] / len(ref_type) if ref_type else None

    # PR curve over thresholds
    curve = []
    for t in [i / 10 for i in range(0, 11)]:
        pu = {frozenset((p.source, p.target)) for p in preds if p.confidence >= t}
        curve.append({"threshold": t,
                      **{k: v for k, v in _prf(len(pu & ref_und), len(pu - ref_und),
                                               len(ref_und - pu)).items()
                         if k in ("precision", "recall", "f1")}})
    best = max(curve, key=lambda c: c["f1"])

    # calibration: bucket predictions by confidence, empirical hit rate per bucket
    calib = []
    for lo in [0.0, 0.25, 0.5, 0.75]:
        bucket = [p for p in preds if lo <= p.confidence < lo + 0.25 or
                  (lo == 0.75 and p.confidence == 1.0)]
        if bucket:
            hits = sum(1 for p in bucket if frozenset((p.source, p.target)) in ref_und)
            calib.append({"bin": f"{lo:.2f}-{lo + 0.25:.2f}", "n": len(bucket),
                          "mean_conf": round(sum(p.confidence for p in bucket) / len(bucket), 2),
                          "hit_rate": round(hits / len(bucket), 2)})

    return {
        "map": ref.name,
        "n_ref_edges": len(ref_und),
        "n_pred": len(kept),
        "undirected": und,
        "directed": dir_,
        "type_accuracy": round(type_hits / type_total, 3) if type_total else None,
        "type_majority_baseline": round(majority_acc, 3) if majority_acc else None,
        "type_n": type_total,
        "pr_curve": curve,
        "best_threshold": best,
        "calibration": calib,
    }


# ---------- whole-map scoring (generated vertex set) ----------

def _reachable_pairs(ir: MapIR) -> set[tuple[str, str]]:
    """All (u, v) with a directed path u -> ... -> v (transitive closure)."""
    adj = ir.adjacency()
    closure: set[tuple[str, str]] = set()
    for start in ir.nodes:
        seen, stack = set(), list(adj.get(start, []))
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            closure.add((start, u))
            stack.extend(adj.get(u, []))
    return closure


def score_map(generated: MapIR, reference: MapIR,
              alignment: dict[str, list[str]]) -> dict:
    """alignment: {reference_slug: [generated_slug, ...]} from align.align_nodes."""
    ref_nodes = set(reference.nodes)
    matched_ref = set(alignment) & ref_nodes
    # generated nodes that matched something (for node precision)
    matched_gen = {g for gs in alignment.values() for g in gs}

    node_recall = len(matched_ref) / len(ref_nodes) if ref_nodes else 0.0
    node_prec = len(matched_gen) / len(generated.nodes) if generated.nodes else 0.0
    node_f1 = (2 * node_prec * node_recall / (node_prec + node_recall)
               if node_prec + node_recall else 0.0)

    # Relabel generated edges into reference space. A generated node can legitimately
    # match SEVERAL reference nodes (the human map split a concept the model merged),
    # so map to the full set and expand each edge across the cross-product. Taking
    # only the first match silently dropped real edge correspondences.
    gen_to_refs: dict[str, set[str]] = {}
    for ref_slug, gen_slugs in alignment.items():
        for g in gen_slugs:
            gen_to_refs.setdefault(g, set()).add(ref_slug)
    gen_edges_in_ref = set()
    for e in generated.edges:
        for s in gen_to_refs.get(e.source, ()):
            for t in gen_to_refs.get(e.target, ()):
                if s != t:
                    gen_edges_in_ref.add(frozenset((s, t)))

    # edge metrics on the induced subgraph (both endpoints matched)
    ref_edges_sub = {frozenset((e.source, e.target)) for e in reference.edges
                     if e.source in matched_ref and e.target in matched_ref}
    edge_sub = _prf(len(gen_edges_in_ref & ref_edges_sub),
                    len(gen_edges_in_ref - ref_edges_sub),
                    len(ref_edges_sub - gen_edges_in_ref))
    # global: unmatched reference edges count as misses
    ref_edges_all = {frozenset((e.source, e.target)) for e in reference.edges}
    edge_global = _prf(len(gen_edges_in_ref & ref_edges_all),
                       len(gen_edges_in_ref - ref_edges_all),
                       len(ref_edges_all - gen_edges_in_ref))

    # reachability agreement over matched reference-node pairs
    ref_reach = {p for p in _reachable_pairs(reference)
                 if p[0] in matched_ref and p[1] in matched_ref}
    # build a generated map relabelled to ref space for its closure
    relabelled_edges = [
        Edge(source=s, target=t)
        for e in generated.edges
        for s in gen_to_refs.get(e.source, ())
        for t in gen_to_refs.get(e.target, ())
        if s != t
    ]
    relabelled = MapIR(name="_relabelled",
                       nodes={r: reference.nodes[r] for r in matched_ref},
                       edges=relabelled_edges)
    gen_reach = {p for p in _reachable_pairs(relabelled)
                 if p[0] in matched_ref and p[1] in matched_ref}
    reach = _prf(len(gen_reach & ref_reach), len(gen_reach - ref_reach),
                 len(ref_reach - gen_reach))

    return {
        "map": reference.name,
        "n_ref_nodes": len(ref_nodes), "n_gen_nodes": len(generated.nodes),
        "nodes": {"precision": round(node_prec, 3), "recall": round(node_recall, 3),
                  "f1": round(node_f1, 3), "matched_ref": len(matched_ref)},
        "coverage": round(node_recall, 3),
        "edges_induced": edge_sub,
        "edges_global": edge_global,
        "reachability": reach,
    }
