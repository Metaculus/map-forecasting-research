import json
import re

import pandas as pd
import pytest
from pydantic import BaseModel

from map_forecasting import llm as llm_mod
from map_forecasting.grading.key_factors import (Anchor, FactorGrade, KeyFactorGrader, evaluate,
                                                 noise_ceiling)
from map_forecasting.llm import FunctionLLM, call_structured, get_default_llm, set_default_llm


class Pair(BaseModel):
    a: int
    b: str


@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    monkeypatch.setenv("MAP_FORECASTING_CACHE_DIR", "")
    monkeypatch.delenv("MAP_FORECASTING_LLM", raising=False)
    set_default_llm(None)
    yield
    set_default_llm(None)


def test_generic_structured_output_parses_fenced_json():
    seen = {}

    def fn(system, prompt):
        seen["prompt"] = prompt
        return 'Sure:\n```json\n{"a": 3, "b": "x"}\n```'

    out = call_structured("give a pair", Pair, system="s", llm=FunctionLLM(fn))
    assert out == Pair(a=3, b="x")
    assert '"a"' in seen["prompt"] and "JSON schema" in seen["prompt"]


def test_generic_structured_output_repairs_once():
    replies = iter(['{"a": "not an int", "b": 1}', '{"a": 1, "b": "ok"}'])
    prompts = []

    def fn(system, prompt):
        prompts.append(prompt)
        return next(replies)

    assert call_structured("p", Pair, llm=FunctionLLM(fn)) == Pair(a=1, b="ok")
    assert "could not be used" in prompts[1]


def test_any_object_with_generate_works():
    class Bare:
        name = "bare"

        def generate(self, system, prompt, max_tokens=16000):
            return '{"a": 5, "b": "y"}'

    assert call_structured("p", Pair, llm=Bare()).a == 5


def test_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MAP_FORECASTING_CACHE_DIR", str(tmp_path))
    calls = []

    def fn(system, prompt):
        calls.append(1)
        return '{"a": 1, "b": "c"}'

    m = FunctionLLM(fn, name="cached-test")
    call_structured("same", Pair, llm=m)
    call_structured("same", Pair, llm=m)
    call_structured("same", Pair, llm=m, variant=1)
    assert len(calls) == 2


def test_default_llm_errors_clearly_and_reads_env(monkeypatch):
    with pytest.raises(RuntimeError, match="No LLM configured"):
        get_default_llm()
    monkeypatch.setenv("MAP_FORECASTING_LLM", "openai:some-model")
    m = get_default_llm()
    assert isinstance(m, llm_mod.OpenAICompatibleLLM) and m.model == "some-model"


QUESTION = {"title": "Will most buses be electric by 2030?", "description": "d",
            "resolution_criteria": "r"}
FACTORS = [{"id": "x", "text": "Budget approved", "type": "news"},
           {"id": 7, "text": "Buses painted green"}]


def fake_grader_llm(system, prompt):
    ids = re.findall(r"^\[(\w+)\]", prompt, flags=re.M)
    return json.dumps({"grades": [{"id": i, "rationale": "r", "no_impact": 10, "low": 20,
                                   "medium": 30, "high": 40} for i in ids]})


def test_key_factor_grader():
    anchors = [Anchor("Q?", "A factor", {"no_impact": 1, "low": 1, "medium": 1, "high": 7}, 10)]
    g = KeyFactorGrader(anchors=anchors, base_rates={"no_impact": .1, "low": .2, "medium": .3,
                                                     "high": .4}, llm=FunctionLLM(fake_grader_llm))
    assert "Example 1" in g.system and "high 70%" in g.system and "about 2.8" in g.system
    out = g.grade(QUESTION, FACTORS)
    assert set(out) == {"x", "7"}
    assert out["x"].expected() == pytest.approx(0.2 + 0.6 + 2.0)


def test_key_factor_grader_rejects_missing_ids():
    def drops_one(system, prompt):
        return json.dumps({"grades": [{"id": "x", "rationale": "r", "no_impact": 0, "low": 0,
                                       "medium": 0, "high": 100}]})

    with pytest.raises(RuntimeError, match="omitted"):
        KeyFactorGrader(llm=FunctionLLM(drops_one)).grade(QUESTION, FACTORS)


def test_factor_grade_normalizes():
    g = FactorGrade(id="a", rationale="", no_impact=0, low=0, medium=0, high=50)
    assert g.expected() == pytest.approx(5.0)


def test_evaluate_and_noise_ceiling():
    df = pd.DataFrame({"question_id": [1, 1, 1, 2, 2, 2], "pred": [1, 2, 3, 1, 2, 3],
                       "y_mean": [0.5, 2.0, 4.0, 1.0, 1.5, 4.5]})
    r = evaluate(df)
    assert r["within_question_pair_acc"] == 1.0 and r["within_question_spearman"] == pytest.approx(1.0)
    votes = pd.DataFrame([{"factor_id": f, "question_id": f // 3, "score": s}
                          for f in range(9) for s in ([5, 5, 2, 5, 2, 5] if f % 2 else
                                                      [0, 1, 0, 1, 2, 0])])
    nc = noise_ceiling(votes, n_rep=20)
    assert nc["factors"] == 9 and nc["pearson"] > 0.5
