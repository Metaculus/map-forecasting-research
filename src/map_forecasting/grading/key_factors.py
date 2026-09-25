"""Key-factor grader: predict how forecasters would rate each key factor's strength.

A key factor is a short statement attached to a forecasting question (a driver, a
news item, a base rate) that someone thinks bears on how it resolves. On Metaculus,
readers rate each one with four strength buttons, scored 0 / 1 / 2 / 5:

    no_impact, low, medium, high

The grader rates every key factor on a question in one call, since they share the
question context and the within-question ranking is often what matters. For each
factor it returns a distribution over the four buttons; `expected()` is the
predicted mean score. Raters disagree a lot, so the distribution is the prediction,
not a single verdict.

Calibration comes from anchors you supply: rated key factors from other questions
with the observed share of each button. Optionally pass overall button shares too.

    from map_forecasting.grading.key_factors import Anchor, KeyFactorGrader
    grader = KeyFactorGrader(anchors=[Anchor(...), ...], llm=my_llm)
    grades = grader.grade(question={"title": ..., "description": ...,
                                    "resolution_criteria": ...},
                          factors=[{"id": "a", "text": "...", "type": "driver"}, ...])
    grades["a"].expected()

`evaluate` and `noise_ceiling` measure a grader against observed ratings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats

from ..llm import call_structured

BUTTONS = ("no_impact", "low", "medium", "high")
SCORES = (0, 1, 2, 5)
DESC_CHARS, CRIT_CHARS, FACTOR_CHARS = 3000, 2000, 1200

SYSTEM_TEMPLATE = """\
You predict how the forecasting community will rate *key factors* on a forecasting question.

A key factor is a short statement a forecaster attaches to a question: a driver, a news item or
a base rate that they think bears on how the question resolves. Other forecasters then rate each
key factor's strength with one of four buttons:

  no_impact  - doesn't bear on the forecast (also used for wrong-direction or redundant factors)
  low        - minor influence
  medium     - meaningful influence
  high       - a major driver of the forecast

Raters are mostly people who also forecast the question. A factor usually gets only a handful of
ratings, and individual raters disagree a lot, so predict the *distribution* of ratings, not a
single verdict.

For each question you get the title, description and resolution criteria, and the key factors
posted on it, each with its type, stated direction of impact where the author gave one, and the
date it was posted. Judge each factor as a rater on that date would have: you may know how the
question resolved or what happened later, but the raters did not.

Raters are generous to factors that are relevant at all, and they do separate the factors on one
question: compare the factors with each other, decide which are the strongest and weakest, and
let their distributions differ accordingly. Don't push every factor to an extreme.{base_rates}

{anchors}

For every key factor, give a one-sentence rationale, then integer percentages for the four
buttons that sum to 100. Return every factor id you were given, exactly once."""


@dataclass
class Anchor:
    """A rated key factor used for calibration. `shares` maps each button to its
    observed share of ratings (fractions or percentages)."""
    question_title: str
    factor_text: str
    shares: dict[str, float]
    n_ratings: int
    factor_type: str = "driver"


class FactorGrade(BaseModel):
    id: str
    rationale: str
    no_impact: int
    low: int
    medium: int
    high: int

    def probs(self) -> np.ndarray:
        p = np.clip(np.array([self.no_impact, self.low, self.medium, self.high], float), 0, None)
        return p / p.sum() if p.sum() > 0 else np.full(4, 0.25)

    def expected(self) -> float:
        """Predicted mean score on the 0 / 1 / 2 / 5 scale."""
        return float(self.probs() @ np.array(SCORES, float))


class QuestionGrades(BaseModel):
    grades: list[FactorGrade]


def _clip(s: object, n: int) -> str:
    s = "" if not isinstance(s, str) or s != s else s.strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " [...]"


def render_anchors(anchors: list[Anchor]) -> str:
    if not anchors:
        return ""
    out = ["Here are rated examples from other questions, with the observed share of each button:"]
    for i, a in enumerate(anchors, 1):
        total = sum(a.shares.get(b, 0) for b in BUTTONS) or 1
        shares = ", ".join(f"{b} {round(100 * a.shares.get(b, 0) / total)}%" for b in BUTTONS)
        out.append(f"Example {i}. Question: {a.question_title}\n"
                   f"Key factor ({a.factor_type}): {_clip(a.factor_text, 400)}\n"
                   f"{a.n_ratings} ratings: {shares}")
    return "\n\n".join(out)


def render_question(question: dict, factors: list[dict]) -> str:
    parts = [f"Question: {question['title']}"]
    if question.get("type"):
        parts.append(f"Question type: {question['type']}")
    if _clip(question.get("description"), DESC_CHARS):
        parts.append(f"Description:\n{_clip(question.get('description'), DESC_CHARS)}")
    if _clip(question.get("resolution_criteria"), CRIT_CHARS):
        parts.append(f"Resolution criteria:\n{_clip(question.get('resolution_criteria'), CRIT_CHARS)}")
    blocks = []
    for f in factors:
        bits = [f"type: {f.get('type') or 'driver'}"]
        if f.get("direction"):
            bits.append(f"stated direction: {f['direction']}")
        if f.get("date"):
            bits.append(f"posted: {f['date']}")
        blocks.append(f"[{f['id']}] ({'; '.join(bits)})\n{_clip(f['text'], FACTOR_CHARS)}")
    parts.append("Key factors to rate:\n\n" + "\n\n".join(blocks))
    return "\n\n".join(parts)


class KeyFactorGrader:
    def __init__(self, anchors: list[Anchor] | None = None,
                 base_rates: dict[str, float] | None = None, llm=None):
        """`base_rates`: optional overall share of each button across rated factors."""
        extra = ""
        if base_rates:
            shares = ", ".join(f"{b} {round(100 * base_rates[b])}%" for b in BUTTONS if b in base_rates)
            mean = sum(base_rates.get(b, 0) * s for b, s in zip(BUTTONS, SCORES))
            extra = (f"\n\nAcross all rated key factors the button shares are roughly: {shares}; "
                     f"the mean score is about {mean:.1f}.")
        self.system = SYSTEM_TEMPLATE.format(anchors=render_anchors(anchors or []),
                                             base_rates=extra)
        self.llm = llm

    def grade(self, question: dict, factors: list[dict]) -> dict[str, FactorGrade]:
        """Grades for every factor, keyed by factor id. `factors` items need "id"
        and "text"; "type", "direction" and "date" are optional."""
        factors = [dict(f, id=str(f["id"])) for f in factors]
        res = call_structured(render_question(question, factors), QuestionGrades,
                              system=self.system, llm=self.llm)
        out = {g.id: g for g in res.grades}
        missing = {f["id"] for f in factors} - set(out)
        if missing:
            raise RuntimeError(f"grader omitted factor ids {sorted(missing)}")
        return out


# ---------------------------------------------------------------- evaluation

def _within_question(df: pd.DataFrame, pred: str, y: str, gap: float,
                     min_per_q: int = 3) -> tuple[float, float, int]:
    rhos, correct, total = [], 0.0, 0
    for _, g in df.groupby("question_id"):
        if len(g) >= min_per_q and g[y].nunique() > 1 and g[pred].nunique() > 1:
            rhos.append(stats.spearmanr(g[pred], g[y]).statistic)
        yv, pv = g[y].to_numpy(), g[pred].to_numpy()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if abs(yv[i] - yv[j]) >= gap:  # only pairs raters clearly separated
                    total += 1
                    d = (pv[i] - pv[j]) * (yv[i] - yv[j])
                    correct += 1.0 if d > 0 else 0.5 if d == 0 else 0.0
    return (float(np.mean(rhos)) if rhos else np.nan, correct / total if total else np.nan, total)


def evaluate(df: pd.DataFrame, pred: str = "pred", y: str = "y_mean", gap: float = 1.0) -> dict:
    """Predicted vs observed ratings. `df` has one row per factor with columns
    question_id, `pred` and `y`. Reports agreement across all factors (Pearson,
    Spearman, MAE) and within each question (mean Spearman, and the share of
    factor pairs raters separated by at least `gap` that the grader orders
    correctly)."""
    wq_rho, wq_pair, n_pairs = _within_question(df, pred, y, gap)
    varied = df[pred].nunique() > 1
    return {"n": len(df), "questions": df["question_id"].nunique(),
            "pearson": stats.pearsonr(df[pred], df[y]).statistic if varied else np.nan,
            "spearman": stats.spearmanr(df[pred], df[y]).statistic if varied else np.nan,
            "mae": float(np.mean(np.abs(df[pred] - df[y]))),
            "within_question_spearman": wq_rho, "within_question_pair_acc": wq_pair,
            "within_question_pairs": n_pairs}


def noise_ceiling(votes: pd.DataFrame, n_rep: int = 200, min_votes: int = 4,
                  seed: int = 0) -> dict:
    """How well half of each factor's raters predict the other half: roughly the
    best any grader can do against held-out ratings. `votes` has one row per
    rating with columns factor_id, question_id and score (0/1/2/5)."""
    rng = np.random.default_rng(seed)
    counts = votes.groupby("factor_id")["score"].transform("size")
    v = votes[counts >= min_votes]
    grouped = {k: g["score"].to_numpy() for k, g in v.groupby("factor_id")}
    q_of = v.groupby("factor_id")["question_id"].first()
    reps = []
    for _ in range(n_rep):
        rows = []
        for k, s in grouped.items():
            s = rng.permutation(s)
            h = len(s) // 2
            rows.append({"question_id": q_of[k], "a": s[:h].mean(), "b": s[h:].mean()})
        reps.append(evaluate(pd.DataFrame(rows), pred="a", y="b"))
    reps = pd.DataFrame(reps)
    keys = ["pearson", "spearman", "mae", "within_question_spearman", "within_question_pair_acc"]
    return {k: float(reps[k].mean()) for k in keys} | {"factors": len(grouped)}
