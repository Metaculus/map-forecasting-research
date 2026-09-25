"""Grading and builders, with the Claude call replaced by canned responses."""
from pathlib import Path

import pytest

from map_forecasting.builders import causal, quantitative
from map_forecasting.grading import (EdgePrediction, align, capture, grade, graded_coverage,
                                     score_edges, score_map)
from map_forecasting.maps import Edge, MapIR

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "maps"
ALIGNMENT = {"goal": ["electric_majority"], "budget": ["funding"],
             "federal_grants": ["funding"], "battery_cost": ["battery_prices"],
             "depot_charging": ["charging_infra"]}


@pytest.fixture
def maps():
    return MapIR.load(EXAMPLES / "generated.json"), MapIR.load(EXAMPLES / "reference.json")


def test_score_map(maps):
    gen, ref = maps
    r = score_map(gen, ref, ALIGNMENT)
    assert r["coverage"] == round(5 / 7, 3)
    assert r["nodes"]["precision"] == round(4 / 5, 3)
    # budget->goal, battery_cost->budget and depot_charging->goal are recovered;
    # the merge also proposes federal_grants->goal, which the reference lacks
    assert r["edges_induced"]["tp"] == 3
    assert r["edges_induced"]["fp"] == 2


def test_score_edges(maps):
    _, ref = maps
    preds = [EdgePrediction("budget", "goal", "necessary", 0.9),
             EdgePrediction("pilot", "goal", "causal", 0.8),
             EdgePrediction("grid_upgrade", "depot_charging", "causal", 0.3)]
    r = score_edges(preds, ref)
    assert r["undirected"]["tp"] == 1 and r["undirected"]["fp"] == 1
    assert r["type_accuracy"] == 1.0


def test_graded_coverage(maps):
    _, ref = maps
    grades = {("funding", "budget"): "full", ("funding", "federal_grants"): "partial"}
    r = graded_coverage(ref, ALIGNMENT, grades)
    assert r["binary_coverage"] == round(5 / 7, 3)
    assert r["graded_coverage"] == round(4.5 / 7, 3)
    assert r["n_graded_pairs"] == 2 and r["n_ungraded_pairs"] == 3


def test_grade_map_with_stubbed_llm(maps, monkeypatch):
    gen, ref = maps

    def fake_align(prompt, schema, **kw):
        return schema(matches=[align.RefMatch(ref_slug=r, generated_slugs=g)
                               for r, g in ALIGNMENT.items()] +
                      [align.RefMatch(ref_slug="pilot", generated_slugs=["nonexistent"])])

    def fake_capture(prompt, schema, **kw):
        refs = [r for r in ref.nodes if f"  {r}:" in prompt]
        return schema(grades=[capture.RefGrade(ref_slug=r, capture="partial"
                                               if r == "federal_grants" else "full")
                              for r in refs])

    monkeypatch.setattr(align, "call_structured", fake_align)
    monkeypatch.setattr(capture, "call_structured", fake_capture)
    r = grade.grade_map(gen, ref)
    assert r["alignment"] == ALIGNMENT
    assert r["coverage_graded"]["graded_coverage"] == round(4.5 / 7, 3)


def test_builders_with_stubbed_llm(monkeypatch):
    def fake(prompt, schema, **kw):
        if schema is causal.Variables:
            return schema(variables=[causal.GenNode(title=t) for t in "abcdef"])
        if schema is causal.NodeList:
            return schema(nodes=[causal.GenNode(title="Goal", category="goal"),
                                 causal.GenNode(title="Budget"), causal.GenNode(title="Chargers")])
        if schema is causal.EdgeProposal:
            return schema(edges=[
                causal.ProposedEdge(source="budget", target="goal", type="necessary", confidence=0.9),
                causal.ProposedEdge(source="chargers", target="goal", type="causal", confidence=0.4),
                causal.ProposedEdge(source="budget", target="missing", type="causal", confidence=0.9)])
        if schema is causal.LayerExpansion:
            return schema(new_nodes=[causal.GenNode(title="Grid upgrade")], new_edges=[
                causal.ProposedEdge(source="Grid upgrade", target="chargers", type="necessary",
                                    confidence=0.8)])
        raise AssertionError(schema)

    monkeypatch.setattr(causal, "call_structured", fake)
    nodes = causal.build_nodes("topic", "Goal", n=3)
    assert nodes.meta["n_raw_variables"] == 6 and nodes.nodes["goal"]["is_main"]
    m = causal.add_edges(nodes, threshold=0.5)
    assert [(e.source, e.target) for e in m.edges] == [("budget", "goal")]
    m.edges.append(Edge("chargers", "goal"))
    grown = causal.expand_layer(m, frontier=["chargers"])
    assert ("grid_upgrade", "chargers") in {(e.source, e.target) for e in grown.edges}


def test_quantitative_validate_and_to_ir():
    Q = quantitative
    good = Q.QuantMap(
        indicators=[Q.Indicator(id="price_now", title="Price today", status="resolved",
                                detail="100 USD", informs=["price"])],
        quant_nodes=[
            Q.QuantNode(kind="distribution", ref="price", title="Price (USD)", unit="USD",
                        description="", family="skewed", p10=90, p50=100, p90=115),
            Q.QuantNode(kind="parameter", ref="threshold", title="Threshold (USD)",
                        unit="USD", description="", value=110),
            Q.QuantNode(kind="formula", ref="target", title="Above threshold", unit="",
                        description="", expression="if(price >= threshold, 1, 0)")],
        target_ref="target", target_output="probability", omissions_note="", method_note="")
    assert Q.validate(good) == []
    ir = Q.to_ir(good)
    assert {(e.source, e.target) for e in ir.edges} == {
        ("price", "target"), ("threshold", "target"), ("ind_price_now", "price")}
    bad = good.model_copy(deep=True)
    bad.quant_nodes[2].expression = "price * missing_ref"
    bad.quant_nodes[0].p50 = 200
    errors = Q.validate(bad)
    assert any("missing_ref" in e for e in errors) and any("skewed" in e for e in errors)
