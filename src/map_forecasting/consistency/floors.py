"""Rounding floors: how much consistency error rounding alone produces.

Forecasts are usually stated on a grid (to the nearest 1%, say). A perfectly
coherent set of beliefs, once each member is rounded, can still show a small
arbitrage error, and a real but tiny inconsistency can round away to zero. Below
the floor, a measured error says nothing.

The floor for a check type is a high percentile of the error measured on
simulated coherent forecasts whose members are each jittered by up to half a
grid step. Treat errors under it as zero (or as left-censored in a model).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import arbitrage

CHECK_TYPES = ("negation", "addition", "conditional", "expected_evidence", "bayes",
               "partition")


def _jitter(p: float, rng: np.random.Generator, grid: float) -> float:
    return float(np.clip(p + rng.uniform(-grid / 2, grid / 2), 0.001, 0.999))


def simulate_error(check: str, rng: np.random.Generator, grid: float = 0.01,
                   n_outcomes: int = 8) -> float:
    """Consistency error of one coherent forecast set after rounding."""
    j = lambda p: _jitter(p, rng, grid)  # noqa: E731
    if check == "negation":
        a = rng.uniform(0.02, 0.98)
        return arbitrage.negation(j(a), j(1 - a))
    if check == "partition":
        w = rng.dirichlet(np.ones(n_outcomes) * rng.uniform(0.5, 2.0))
        implied = [float(np.clip(x, 1e-6, 1 - 1e-6)) for x in w]
        return arbitrage.partition([j(x) for x in w], implied)
    pa = rng.uniform(0.05, 0.95)
    pb_a = rng.uniform(0.02, 0.98)
    pb_na = rng.uniform(0.02, 0.98)
    pb = pb_a * pa + pb_na * (1 - pa)
    pand = pb_a * pa
    if check == "addition":
        return arbitrage.addition(j(pa), j(pb), j(pa + pb - pand), j(pand))
    if check == "conditional":
        return arbitrage.conditional(j(pa), j(pb_a), j(pand))
    if check == "expected_evidence":
        return arbitrage.expected_evidence(j(pa), j(1 - pa), j(pb), j(pb_a), j(pb_na))
    if check == "bayes":
        return arbitrage.bayes(j(pa), j(pb), j(pand / pb), j(pb_a))
    raise ValueError(f"unknown check type {check!r}")


def _one(args):
    check, seed, grid, n_outcomes = args
    return simulate_error(check, np.random.default_rng(seed), grid, n_outcomes)


def rounding_floor(check: str, grid: float = 0.01, n_sim: int = 250,
                   percentile: float = 99, n_outcomes: int = 8,
                   workers: int | None = None, seed: int = 10_000) -> dict:
    """Simulate `n_sim` rounded coherent sets and summarize their errors.

    Returns {"floor": the chosen percentile, "p50", "p90", "max", "n", "grid"}.
    """
    jobs = [(check, seed + i, grid, n_outcomes) for i in range(n_sim)]
    if workers == 1:
        errs = np.array([_one(a) for a in jobs])
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            errs = np.array(list(ex.map(_one, jobs, chunksize=8)))
    return {"floor": float(np.percentile(errs, percentile)),
            "p50": float(np.percentile(errs, 50)),
            "p90": float(np.percentile(errs, 90)),
            "max": float(errs.max()), "n": n_sim, "grid": grid}
