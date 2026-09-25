"""Grade a generated map against a reference map.

    python examples/grade_maps.py            # offline: uses a hand-made alignment
    python examples/grade_maps.py --llm      # aligns and capture-grades with an LLM

For --llm, set MAP_FORECASTING_LLM (e.g. "openai:<model>") or call set_default_llm.
"""
import sys
from pathlib import Path

from map_forecasting.grading import grade_map, score_map
from map_forecasting.maps import MapIR

here = Path(__file__).parent / "maps"
reference = MapIR.load(here / "reference.json")
generated = MapIR.load(here / "generated.json")

if "--llm" in sys.argv:
    result = grade_map(generated, reference)
else:
    # {reference node: [generated nodes that express it]}
    alignment = {"goal": ["electric_majority"], "budget": ["funding"],
                 "federal_grants": ["funding"], "battery_cost": ["battery_prices"],
                 "depot_charging": ["charging_infra"]}
    result = score_map(generated, reference, alignment)

for key in ("nodes", "coverage", "edges_induced", "edges_global", "reachability",
            "coverage_graded"):
    if key in result:
        print(f"{key}: {result[key]}")
