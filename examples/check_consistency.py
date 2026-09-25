"""Check a few forecasts for consistency and snap them to the nearest coherent set.

    python examples/check_consistency.py
"""
from map_forecasting.consistency import (WORLD_TABLES, arbitrage_error, bayes,
                                         expected_evidence, negation, project)

# A = the council approves the electric-bus budget by 2027
# B = most of the city's buses are electric by 2030
print("negation:", round(negation(p_a=0.40, p_not_a=0.55), 4))
print("bayes:", round(bayes(p_a=0.40, p_b=0.35, p_a_given_b=0.80, p_b_given_a=0.70), 4))

forecasts = {"A": 0.40, "~A": 0.60, "B": 0.50, "B|A": 0.70, "B|~A": 0.15}
ee = WORLD_TABLES["expected_evidence"]
print("expected evidence:", round(expected_evidence(0.40, 0.60, 0.50, 0.70, 0.15), 4))

coherent = project(forecasts, ee, method="klf")
print("nearest coherent:", {k: round(v, 3) for k, v in coherent.items()})
print("error after projection:", round(arbitrage_error(ee, coherent), 6))
