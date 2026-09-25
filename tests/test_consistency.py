import csv
import math
from pathlib import Path

import pandas as pd
import pytest

from map_forecasting.consistency import (WORLD_TABLES, addition, arbitrage_error, bayes,
                                         conditional, expected_evidence, negation,
                                         partition, project, rounding_floor, shrink)
from map_forecasting.consistency.batch import CONSISTENCY_OUTPUT_COLS, compute_consistency

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def coherent_bundle(pa=0.4, pb_a=0.7, pb_na=0.2):
    pb = pb_a * pa + pb_na * (1 - pa)
    pand = pb_a * pa
    return dict(pa=pa, pb=pb, pand=pand, por=pa + pb - pand, pb_a=pb_a, pb_na=pb_na,
                pa_b=pand / pb)


def test_coherent_forecasts_have_zero_error():
    c = coherent_bundle()
    assert negation(c["pa"], 1 - c["pa"]) == pytest.approx(0, abs=1e-9)
    assert addition(c["pa"], c["pb"], c["por"], c["pand"]) == pytest.approx(0, abs=1e-6)
    assert conditional(c["pa"], c["pb_a"], c["pand"]) == pytest.approx(0, abs=1e-6)
    assert expected_evidence(c["pa"], 1 - c["pa"], c["pb"], c["pb_a"], c["pb_na"]) == \
        pytest.approx(0, abs=1e-6)
    assert bayes(c["pa"], c["pb"], c["pa_b"], c["pb_a"]) == pytest.approx(0, abs=1e-6)
    assert partition([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0, abs=1e-6)


def test_incoherent_forecasts_have_positive_error():
    assert negation(0.3, 0.5) > 0.01
    assert expected_evidence(0.4, 0.6, 0.8, 0.7, 0.2) > 0.05
    assert partition([0.5, 0.5, 0.5], [0.2, 0.3, 0.5]) > 0.01


def test_negation_closed_form_matches_solver():
    for p, q in [(0.3, 0.5), (0.9, 0.3), (0.05, 0.9)]:
        solved = arbitrage_error(WORLD_TABLES["negation"], {"A": p, "~A": q})
        assert negation(p, q) == pytest.approx(solved, rel=1e-3, abs=1e-6)


def test_error_grows_with_incoherence():
    assert negation(0.3, 0.6) < negation(0.3, 0.5) < negation(0.3, 0.3)


def test_missing_member_raises():
    with pytest.raises(ValueError):
        arbitrage_error(WORLD_TABLES["bayes"], {"A": 0.5, "B": 0.5})


@pytest.mark.parametrize("method", ["klf", "klr", "l2", "game"])
def test_projection_is_coherent(method):
    worlds = WORLD_TABLES["expected_evidence"]
    f = {"A": 0.4, "~A": 0.6, "B": 0.8, "B|A": 0.7, "B|~A": 0.2}
    q = project(f, worlds, method=method)
    assert set(q) == set(worlds)
    assert arbitrage_error(worlds, q) < 1e-3


def test_projection_leaves_coherent_forecasts_alone():
    c = coherent_bundle()
    f = {"A": c["pa"], "~A": 1 - c["pa"], "B": c["pb"], "B|A": c["pb_a"], "B|~A": c["pb_na"]}
    q = project(f, WORLD_TABLES["expected_evidence"], method="l2")
    for k in f:
        assert q[k] == pytest.approx(f[k], abs=1e-3)


def test_shrink_interpolates():
    f, q = {"A": 0.2}, {"A": 0.6}
    assert shrink(f, q, 0.0)["A"] == pytest.approx(0.2)
    assert shrink(f, q, 1.0)["A"] == pytest.approx(0.6)
    assert shrink(f, q, 0.5)["A"] == pytest.approx(0.4)
    lo = shrink(f, q, 0.5, space="log_odds")["A"]
    mid = 1 / (1 + math.exp(-(math.log(0.2 / 0.8) + math.log(0.6 / 0.4)) / 2))
    assert lo == pytest.approx(mid)


def test_rounding_floor_is_small_and_positive():
    out = rounding_floor("negation", n_sim=20, workers=1)
    assert 0 < out["floor"] < 0.01
    assert out["p50"] <= out["p90"] <= out["floor"] <= out["max"] + 1e-12


def test_batch_on_example_csv():
    rows = pd.read_csv(EXAMPLES / "forecasts.csv").to_dict("records")
    out = compute_consistency(rows, "example_pmf")
    types = {r["consistency_check_type"] for r in out}
    assert types == {"negation", "addition", "conditional", "expected_evidence", "bayes"}
    by_type = {r["consistency_check_type"]: r["consistency_error"] for r in out}
    assert by_type["negation"] == pytest.approx(negation(0.40, 0.55), rel=1e-6)
    assert by_type["bayes"] == pytest.approx(0, abs=1e-6)  # 0.8 * 0.35 == 0.7 * 0.4
    assert set(out[0]) <= set(CONSISTENCY_OUTPUT_COLS)


def test_example_csv_is_valid():
    with open(EXAMPLES / "forecasts.csv") as f:
        rows = list(csv.DictReader(f))
    assert len({r["question_id"] for r in rows}) == len(rows)
