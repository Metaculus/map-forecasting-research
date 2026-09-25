"""Arbitrage-based consistency checks for probability estimates.

A set of forecasts on logically linked questions is incoherent when a bettor can
trade against it and profit in every possible world. The consistency error of a
check is that guaranteed profit, measured in log-score units: 0 means the
forecasts are coherent, and larger values mean a bigger sure loss.

Each check is described by a world table: for every member question, its
outcome in each possible world (1 = yes, 0 = no, None = the question is
annulled in that world, as a conditional is when its condition fails).

Forecasts are passed as probabilities of "yes". Functions that accept a PMF
([p_no, p_yes]) say so.
"""
from __future__ import annotations

import numpy as np
import scipy.optimize

# world tables: member -> outcome per world (None = annulled in that world)
NEGATION = {"A": [1, 0], "~A": [0, 1]}
ADDITION = {"A": [1, 1, 0, 0], "B": [1, 0, 1, 0],
            "AvB": [1, 1, 1, 0], "A&B": [1, 0, 0, 0]}
CONDITIONAL = {"A": [1, 1, 0, 0], "B|A": [1, 0, None, None], "A&B": [1, 0, 0, 0]}
EXPECTED_EVIDENCE = {"A": [1, 1, 0, 0], "~A": [0, 0, 1, 1], "B": [1, 0, 1, 0],
                     "B|A": [1, 0, None, None], "B|~A": [None, None, 1, 0]}
BAYES = {"A": [1, 1, 0, 0], "B": [1, 0, 1, 0],
         "A|B": [1, None, 0, None], "B|A": [1, 0, None, None]}

WORLD_TABLES = {"negation": NEGATION, "addition": ADDITION,
                "conditional": CONDITIONAL, "expected_evidence": EXPECTED_EVIDENCE,
                "bayes": BAYES}

# Differential evolution cost grows fast with dimension; partitions with more
# members than this are refused rather than left running for hours.
MAX_PARTITION_SIZE = 15

_SEED = 0


def _pmf(p: float | list[float]) -> list[float]:
    if isinstance(p, (list, tuple, np.ndarray)):
        return [float(p[0]), float(p[1])]
    return [1.0 - float(p), float(p)]


def _score(outcome: int | None, pmf: list[float]) -> float:
    if outcome is None:
        return 0.0
    return float(np.log(pmf[outcome]))


def arbitrage_error(worlds: dict[str, list[int | None]],
                    forecasts: dict[str, float | list[float]]) -> float:
    """Guaranteed log-score profit against `forecasts` for the given world table.

    `worlds` maps each member to its outcome in every world; `forecasts` maps the
    same members to a probability of yes (or a [p_no, p_yes] PMF).
    """
    members = list(worlds)
    missing = [m for m in members if m not in forecasts]
    if missing:
        raise ValueError(f"missing forecasts for {missing}")
    n = len(members)
    n_worlds = len(next(iter(worlds.values())))
    table = [[worlds[m][w] for m in members] for w in range(n_worlds)]
    given = [_pmf(forecasts[m]) for m in members]

    def neg_guaranteed_profit(bets: np.ndarray) -> float:
        return -min(
            sum(_score(table[w][i], [1 - bets[i], bets[i]]) - _score(table[w][i], given[i])
                for i in range(n))
            for w in range(n_worlds))

    eps = 1e-9
    result = scipy.optimize.differential_evolution(
        neg_guaranteed_profit, [(eps, 1 - eps)] * n, seed=_SEED, tol=1e-10, polish=True)
    return float(max(-result.fun, 0.0))


def negation(p_a: float, p_not_a: float) -> float:
    """P(A) and P(not A) should sum to 1. Closed form, no solver."""
    a0, a1 = _pmf(p_a)
    n0, n1 = _pmf(p_not_a)
    return abs(float(-2 * np.log(np.sqrt(a0 * n1) + np.sqrt(a1 * n0))))


def addition(p_a: float, p_b: float, p_a_or_b: float, p_a_and_b: float) -> float:
    """P(A or B) = P(A) + P(B) - P(A and B)."""
    return arbitrage_error(ADDITION, {"A": p_a, "B": p_b, "AvB": p_a_or_b, "A&B": p_a_and_b})


def conditional(p_a: float, p_b_given_a: float, p_a_and_b: float) -> float:
    """P(A and B) = P(A) * P(B | A)."""
    return arbitrage_error(CONDITIONAL, {"A": p_a, "B|A": p_b_given_a, "A&B": p_a_and_b})


def expected_evidence(p_a: float, p_not_a: float, p_b: float,
                      p_b_given_a: float, p_b_given_not_a: float) -> float:
    """P(B) = P(B | A) P(A) + P(B | not A) P(not A)."""
    return arbitrage_error(EXPECTED_EVIDENCE, {"A": p_a, "~A": p_not_a, "B": p_b,
                                               "B|A": p_b_given_a, "B|~A": p_b_given_not_a})


def bayes(p_a: float, p_b: float, p_a_given_b: float, p_b_given_a: float) -> float:
    """P(A | B) P(B) = P(B | A) P(A)."""
    return arbitrage_error(BAYES, {"A": p_a, "B": p_b, "A|B": p_a_given_b, "B|A": p_b_given_a})


def partition(member_probs: list[float | list[float]],
              implied_probs: list[float | list[float]],
              maxiter: int = 400) -> float:
    """Mutually exclusive, exhaustive outcomes forecast two ways.

    `member_probs[i]` is the forecast that outcome i happens, asked as its own
    yes/no question; `implied_probs[i]` is the probability the parent forecast
    (a distribution or multiple-choice question) gives the same outcome. Covers
    binning a numeric forecast and regrouping multiple-choice options.
    """
    p = [_pmf(x) for x in member_probs]
    q = [_pmf(x) for x in implied_probs]
    n = len(p)
    if n != len(q):
        raise ValueError("member_probs and implied_probs differ in length")
    if n > MAX_PARTITION_SIZE:
        raise ValueError(f"{n} outcomes exceeds MAX_PARTITION_SIZE={MAX_PARTITION_SIZE}")
    popsize = max(5, min(15, 120 // n))

    def neg_guaranteed_profit(bets: np.ndarray) -> float:
        return -min(
            sum(_score(1 if i == w else 0, [1 - bets[i], bets[i]])
                - _score(1 if i == w else 0, p[i])
                + _score(1 if i == w else 0, [1 - bets[i], bets[i]])
                - _score(1 if i == w else 0, q[i])
                for i in range(n)) / n
            for w in range(n))

    eps = 0.001
    rng = np.random.default_rng(_SEED)
    init = rng.uniform(eps, 1 - eps, size=(max(5, popsize * n), n))
    init[0] = np.clip([x[1] for x in p], eps, 1 - eps)
    result = scipy.optimize.differential_evolution(
        neg_guaranteed_profit, [(eps, 1 - eps)] * n, tol=1e-6, popsize=popsize,
        maxiter=maxiter, polish=n <= 20, init=init)
    return float(max(-result.fun, 0.0))
