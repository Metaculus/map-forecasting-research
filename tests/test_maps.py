import json
from pathlib import Path

from map_forecasting.maps import (Edge, MapIR, from_guesstimate, from_radiant_export,
                                  from_radiant_v2, from_squiggle_code, slugify,
                                  to_radiant_payload)
from map_forecasting.maps.cleaning import EdgeFinding, MapAudit, apply_fixes

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "maps"


def small_map():
    return MapIR(name="t", nodes={s: {"title": s} for s in "abcd"},
                 edges=[Edge("a", "b"), Edge("b", "c"), Edge("c", "a"), Edge("c", "d")])


def test_roundtrip(tmp_path):
    m = MapIR.load(EXAMPLES / "reference.json")
    m.save(tmp_path / "m.json")
    again = MapIR.load(tmp_path / "m.json")
    assert again.to_dict() == m.to_dict()


def test_graph_utils():
    m = small_map()
    assert m.sinks() == ["d"]
    assert m.cycles() and set(m.cycles()[0]) == {"a", "b", "c"}
    assert m.layers_from(["d"]) == {"d": 0, "c": 1, "b": 2, "a": 3}


def test_slugify_is_unique():
    taken: set[str] = set()
    assert slugify("Battery cost", taken) == "battery_cost"
    assert slugify("Battery cost", taken) == "battery_cost_2"


def test_guesstimate_adapter():
    d = {"id": 1, "name": "demo", "graph": {
        "metrics": [{"id": "a1", "name": "Buses"}, {"id": "b2", "name": "Cost per bus"},
                    {"id": "c3", "name": "Total cost"}],
        "guesstimates": [{"metric": "a1", "input": "100"}, {"metric": "b2", "input": "5"},
                         {"metric": "c3", "expression": "=${metric:a1}*${metric:b2}"}]}}
    m = from_guesstimate(d)
    assert {(e.source, e.target) for e in m.edges} == {("buses", "total_cost"),
                                                       ("cost_per_bus", "total_cost")}
    assert m.nodes["total_cost"]["is_main"]


def test_squiggle_adapter():
    code = """
buses = 100 to 200
cost_per_bus = 0.5 to 1
@name("Total cost")
total = buses * cost_per_bus
"""
    m = from_squiggle_code(code, name="demo")
    titles = {s: n["title"] for s, n in m.nodes.items()}
    total = next(s for s, t in titles.items() if t == "Total cost")
    assert {e.target for e in m.edges} == {total}
    assert len(m.edges) == 2
    assert m.nodes[total]["is_main"]


def test_radiant_export_adapters(tmp_path):
    v1 = {"project": {"title": "p"},
          "project_nodes": [{"node_id": "n1", "data": {"title": "Cause"}},
                            {"node_id": "n2", "data": {"title": "Effect", "isMainNode": True}}],
          "project_edges": [{"source_node_id": "n2", "target_node_id": "n1",
                             "data": {"arrowDirection": "backward", "type": "causal"}}]}
    (tmp_path / "v1.json").write_text(json.dumps(v1))
    m = from_radiant_export(tmp_path / "v1.json")
    assert [(e.source, e.target) for e in m.edges] == [("cause", "effect")]
    v2 = {"project": {"title": "p"}, "canvas": {
        "nodes": [{"id": "x", "node_type": "custom-node", "data": {"title": "Cause"},
                   "created_by_user_name": "someone"},
                  {"id": "y", "node_type": "custom-node", "data": {"title": "Effect"}},
                  {"id": "z", "node_type": "text-node", "data": {"title": "note"}}],
        "edges": [{"source_node_id": "x", "target_node_id": "y", "data": {}}]}}
    (tmp_path / "v2.json").write_text(json.dumps(v2))
    m2 = from_radiant_v2(tmp_path / "v2.json")
    assert set(m2.nodes) == {"cause", "effect"}
    assert "author" not in m2.meta
    payload = to_radiant_payload(m2)
    assert len(payload["project_nodes"]) == 2 and len(payload["project_edges"]) == 1


def test_apply_fixes():
    m = MapIR(name="t", nodes={s: {"title": s} for s in ["goal", "x", "y"]},
              edges=[Edge("goal", "x"), Edge("y", "goal", type="unknown"), Edge("x", "y")])
    audit = MapAudit(convention_assessment="", sink_slugs=["goal"], node_findings=[],
                     edge_findings=[
                         EdgeFinding(source="goal", target="x", error_class="reversed_direction",
                                     severity="high", explanation=""),
                         EdgeFinding(source="y", target="goal", error_class="missing_type",
                                     severity="medium", explanation="", suggested_type="causal"),
                         EdgeFinding(source="x", target="y", error_class="not_a_relationship",
                                     severity="high", explanation="")])
    fixed = apply_fixes(m, audit)
    assert {(e.source, e.target, e.type) for e in fixed.edges} == {
        ("x", "goal", "unknown"), ("y", "goal", "causal")}
    assert len(fixed.meta["fixes_applied"]) == 3
