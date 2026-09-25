"""Predict how forecasters would rate a question's key factors.

    MAP_FORECASTING_LLM=openai:<model> OPENAI_API_KEY=... python examples/grade_key_factors.py

The anchors below are made up. For real use, pass rated key factors from your own
questions so the grader is calibrated to your raters.
"""
from map_forecasting.grading import Anchor, KeyFactorGrader

anchors = [
    Anchor("Will the city's new tram line open by June 2027?",
           "The main contractor went into administration in March.", n_ratings=9,
           shares={"no_impact": 0.0, "low": 0.1, "medium": 0.2, "high": 0.7}, factor_type="news"),
    Anchor("Will the city's new tram line open by June 2027?",
           "Tram lines in other cities opened on time about 30% of the time.", n_ratings=7,
           shares={"no_impact": 0.1, "low": 0.3, "medium": 0.4, "high": 0.2},
           factor_type="base_rate"),
    Anchor("Will turnout in the next mayoral election exceed 40%?",
           "The mayor's favourite football team won the league.", n_ratings=6,
           shares={"no_impact": 0.7, "low": 0.3, "medium": 0.0, "high": 0.0}),
]

question = {"title": "Will a majority of the city's buses be electric by the end of 2030?",
            "description": "The city runs about 400 buses, 60 of them electric today.",
            "resolution_criteria": "Resolves YES if the transit authority's fleet report for "
                                   "2030 lists more than half of active buses as battery-electric."}
factors = [
    {"id": "budget", "text": "The council approved funding for 150 electric buses in 2026.",
     "type": "news", "direction": "raises the probability"},
    {"id": "depots", "text": "Only one of the four depots has high-power charging.",
     "type": "driver", "direction": "lowers the probability"},
    {"id": "color", "text": "The new buses will be painted green.", "type": "driver"},
]

grades = KeyFactorGrader(anchors=anchors).grade(question, factors)
for fid, g in grades.items():
    print(f"{fid:8s} expected {g.expected():.2f}  "
          f"[{g.no_impact}/{g.low}/{g.medium}/{g.high}]  {g.rationale}")
