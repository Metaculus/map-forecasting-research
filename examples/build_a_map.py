"""Build a small causal map with an LLM, then check it for structural problems.

    MAP_FORECASTING_LLM=openai:<model> OPENAI_API_KEY=... python examples/build_a_map.py
"""
from map_forecasting.builders import add_edges, build_nodes, problem_brief
from map_forecasting.maps.cleaning import clean_map

topic = "City bus fleet electrification"
goal = "Majority of the city's buses are electric by 2030"

brief = problem_brief(topic, goal)
nodes = build_nodes(topic, goal, n=12, brief=brief, method="workshop")
m = add_edges(nodes, threshold=0.5, brief=brief)
audit, fixed = clean_map(m)

print(f"{len(fixed.nodes)} nodes, {len(fixed.edges)} edges")
for e in fixed.edges:
    print(f"  {fixed.nodes[e.source]['title']}  ->  {fixed.nodes[e.target]['title']}  [{e.type}]")
print("fixes applied:", fixed.meta["fixes_applied"])
