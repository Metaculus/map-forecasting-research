"""Batch consistency checks over a table of linked forecast questions.

For one-off checks on a handful of probabilities, use the functions in
`map_forecasting.consistency.arbitrage` instead. This module is for a table of
questions in which some rows are derived "consistency questions" built from
others, one row per question.

Input columns:
  question_id   unique id
  type          binary | numeric | discrete | date | multiple_choice
  options       JSON list (multiple-choice questions)
  range_min, range_max, zero_point
                scaling of a continuous question (binning checks only)
  metadata      JSON; derived questions carry
                {"format": "consistency_question",
                 "info": {"consistency_type": ..., "parent_ids": [...], ...}}
  <name>_pmf    one or more forecast columns, JSON lists: [p_no, p_yes] for
                binary, one probability per option for multiple choice, and
                for continuous questions a PMF of n+2 values over an n+1-point
                CDF grid (Metaculus uses 201 points)

consistency_type and its extra "info" fields:
  conditional, one parent A          the question "not A" (negation check)
  conditional, two parents A and B   logical_form is one of "P(A∨B)",
                                     "P(A∧B)", "P(A|B)", "P(B|A)", "P(B|¬A)";
                                     together these give the addition,
                                     conditional, expected-evidence and Bayes checks
  binning                            interval_min / interval_max of the bin
                                     (8 bins per parent)
  mc_option_consolidation            options_groups: lists of parent options
                                     merged into each child option
  mc_binary_regrouping               option: the parent option this yes/no
                                     question asks about

Output: one row per check with its consistency_error (0 = coherent) and the
member question ids and PMFs.

Command line:
  python -m map_forecasting.consistency.batch --input-csv forecasts.csv \
      --scored-col-info my_pmf,my_forecaster --output-folder out/
"""

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import arbitrage

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Inlined math helpers (no project dependencies)
# --------------------------------------------------------------------------- #


def _unscale_value(
    scaled: float, range_min: float, range_max: float, zero_point: float | None
) -> float:
    lo, hi, zp = range_min, range_max, zero_point
    if zp is None:
        return (scaled - lo) / (hi - lo)
    deriv_ratio = (hi - zp) / (lo - zp)
    return (
        np.log((scaled - lo) * (deriv_ratio - 1) + (hi - lo)) - np.log(hi - lo)
    ) / np.log(deriv_ratio)


def _get_cdf_at(
    cdf: list[float],
    nominal_value: float,
    range_min: float,
    range_max: float,
    zero_point: float | None,
) -> float:
    n = len(cdf) - 1
    float_index = (n + 1) * _unscale_value(
        nominal_value, range_min, range_max, zero_point
    )
    k = int(float_index)
    frac = float_index - k
    if k < 0:
        return 0.0
    if k > n:
        return 1.0
    if k == n:
        return cdf[n] + frac * (1.0 - cdf[n])
    return cdf[k] + frac * (cdf[k + 1] - cdf[k])


def _pmf_to_cdf(pmf: list[float]) -> list[float]:
    """Convert a continuous PMF (n+2 values) to a CDF (n+1 values)."""
    cdf: list[float] = []
    cumsum = 0.0
    for v in pmf[:-1]:
        cumsum += v
        cdf.append(cumsum)
    return cdf


# --------------------------------------------------------------------------- #
# Row accessors
# --------------------------------------------------------------------------- #


def _qid(row: dict | None) -> str | None:
    return None if row is None else row.get("question_id")


def _meta(row: dict) -> dict:
    raw = row.get("metadata")
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return {}
    return json.loads(raw) if isinstance(raw, str) else raw


def _options(row: dict) -> list[str]:
    raw = row.get("options")
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    return json.loads(raw) if isinstance(raw, str) else list(raw)


def _get_pmf(row: dict | None, pmf_key: str) -> list[float] | None:
    if row is None:
        return None
    val = row.get(pmf_key)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, str):
        return json.loads(val)
    if isinstance(val, list):
        return val
    return None


def _has_forecast(row: dict | None, pmf_key: str) -> bool:
    return _get_pmf(row, pmf_key) is not None


def _scaling(row: dict) -> tuple[float, float, float | None] | None:
    """Return (range_min, range_max, zero_point) or None if unavailable."""
    lo = row.get("range_min")
    hi = row.get("range_max")
    if lo is None or hi is None:
        return None
    if isinstance(lo, float) and math.isnan(lo):
        return None
    if isinstance(hi, float) and math.isnan(hi):
        return None
    zp = row.get("zero_point")
    if isinstance(zp, float) and math.isnan(zp):
        zp = None
    return (float(lo), float(hi), float(zp) if zp is not None else None)


# --------------------------------------------------------------------------- #
# Consistency group dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class NegationGroup:
    parent_id: str
    parent: dict | None = None
    neg_a: dict | None = None


@dataclass
class ConditionalGroup:
    parent_a_id: str
    parent_b_id: str
    parent_a: dict | None = None
    parent_b: dict | None = None
    neg_a: dict | None = None
    or_ab: dict | None = None
    and_ab: dict | None = None
    a_given_b: dict | None = None
    b_given_a: dict | None = None
    b_given_not_a: dict | None = None


@dataclass
class BinningGroup:
    parent_id: str
    parent: dict | None = None
    bins: list[dict] = field(default_factory=list)


@dataclass
class MCOptionConsolidationGroup:
    parent_id: str
    parent: dict | None = None
    mc_children: list[dict] = field(default_factory=list)


@dataclass
class MCBinaryRegroupingGroup:
    parent_id: str
    parent: dict | None = None
    options: list[dict] = field(default_factory=list)


_LOGICAL_FORM_FIELD: dict[str, str] = {
    "P(A∨B)": "or_ab",
    "P(A∧B)": "and_ab",
    "P(A|B)": "a_given_b",
    "P(B|A)": "b_given_a",
    "P(B|¬A)": "b_given_not_a",
}

# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #


def _group_and_wire(
    data: list[dict],
) -> tuple[
    dict[str, NegationGroup],
    dict[tuple[str, str], ConditionalGroup],
    dict[str, BinningGroup],
    dict[str, MCOptionConsolidationGroup],
    dict[str, MCBinaryRegroupingGroup],
]:
    """Group rows and wire parent references; does not filter by forecast presence."""
    negation: dict[str, NegationGroup] = {}
    conditional: dict[tuple[str, str], ConditionalGroup] = {}
    binning: dict[str, BinningGroup] = {}
    mc_consol: dict[str, MCOptionConsolidationGroup] = {}
    mc_binary: dict[str, MCBinaryRegroupingGroup] = {}
    parents: dict[str, dict] = {}

    for row in data:
        meta = _meta(row)
        qid = row.get("question_id", "")
        parents[qid] = row

        if meta.get("format") != "consistency_question":
            continue

        info = meta.get("info", {})
        ctype: str = info.get("consistency_type", "")
        logical_form: str | None = info.get("logical_form")
        parent_ids: list[str] = info.get("parent_ids", [])

        if ctype == "binning":
            pid = parent_ids[0]
            if pid not in binning:
                binning[pid] = BinningGroup(parent_id=pid)
            binning[pid].bins.append(row)
        elif ctype == "mc_option_consolidation":
            pid = parent_ids[0]
            if pid not in mc_consol:
                mc_consol[pid] = MCOptionConsolidationGroup(parent_id=pid)
            mc_consol[pid].mc_children.append(row)
        elif ctype == "mc_binary_regrouping":
            pid = parent_ids[0]
            if pid not in mc_binary:
                mc_binary[pid] = MCBinaryRegroupingGroup(parent_id=pid)
            mc_binary[pid].options.append(row)
        elif ctype == "mc_regrouping":
            pid = parent_ids[0]
            if row.get("type") == "multiple_choice":
                if pid not in mc_consol:
                    mc_consol[pid] = MCOptionConsolidationGroup(parent_id=pid)
                mc_consol[pid].mc_children.append(row)
            else:
                if pid not in mc_binary:
                    mc_binary[pid] = MCBinaryRegroupingGroup(parent_id=pid)
                mc_binary[pid].options.append(row)
        elif ctype == "conditional":
            if len(parent_ids) == 1:
                pid = parent_ids[0]
                if pid not in negation:
                    negation[pid] = NegationGroup(parent_id=pid)
                negation[pid].neg_a = row
            elif len(parent_ids) == 2:
                key = (parent_ids[0], parent_ids[1])
                if key not in conditional:
                    conditional[key] = ConditionalGroup(
                        parent_a_id=parent_ids[0], parent_b_id=parent_ids[1]
                    )
                attr = _LOGICAL_FORM_FIELD.get(logical_form or "")
                if attr:
                    setattr(conditional[key], attr, row)

    for grp in negation.values():
        grp.parent = parents.get(grp.parent_id)
    for grp in conditional.values():
        grp.parent_a = parents.get(grp.parent_a_id)
        grp.parent_b = parents.get(grp.parent_b_id)
        neg = negation.get(grp.parent_a_id)
        if neg:
            grp.neg_a = neg.neg_a
    for grp in binning.values():
        grp.parent = parents.get(grp.parent_id)
    for grp in mc_consol.values():
        grp.parent = parents.get(grp.parent_id)
    for grp in mc_binary.values():
        grp.parent = parents.get(grp.parent_id)

    return negation, conditional, binning, mc_consol, mc_binary


def _group_consistency_rows(
    data: list[dict],
    pmf_key: str,
) -> tuple[
    list[NegationGroup],
    list[ConditionalGroup],
    list[BinningGroup],
    list[MCOptionConsolidationGroup],
    list[MCBinaryRegroupingGroup],
]:
    negation, conditional, binning, mc_consol, mc_binary = _group_and_wire(data)

    valid_negation: list[NegationGroup] = []
    valid_conditional: list[ConditionalGroup] = []
    valid_binning: list[BinningGroup] = []
    valid_mc_consol: list[MCOptionConsolidationGroup] = []
    valid_mc_binary: list[MCBinaryRegroupingGroup] = []

    for grp in negation.values():
        if _has_forecast(grp.parent, pmf_key) and _has_forecast(grp.neg_a, pmf_key):
            valid_negation.append(grp)

    for grp in conditional.values():
        if _has_forecast(grp.parent_a, pmf_key) and _has_forecast(
            grp.parent_b, pmf_key
        ):
            valid_conditional.append(grp)
        else:
            logger.warning("conditional group missing forecast: %s, %s", grp.parent_a_id, grp.parent_b_id)

    for grp in binning.values():
        if (
            _has_forecast(grp.parent, pmf_key)
            and grp.bins
            and all(_has_forecast(b, pmf_key) for b in grp.bins)
        ):
            valid_binning.append(grp)

    for grp in mc_consol.values():
        if (
            _has_forecast(grp.parent, pmf_key)
            and grp.mc_children
            and all(_has_forecast(c, pmf_key) for c in grp.mc_children)
        ):
            valid_mc_consol.append(grp)

    for grp in mc_binary.values():
        if (
            _has_forecast(grp.parent, pmf_key)
            and grp.options
            and all(_has_forecast(o, pmf_key) for o in grp.options)
        ):
            valid_mc_binary.append(grp)

    return (
        valid_negation,
        valid_conditional,
        valid_binning,
        valid_mc_consol,
        valid_mc_binary,
    )


def validate_consistency_structure(rows: list[dict]) -> list[str]:
    """Return error strings for consistency groups with missing structural members.

    Each row must have at minimum ``question_id``, ``type``, and ``metadata``
    keys (metadata may be a JSON string or a plain dict).
    """
    negation, conditional, binning, mc_consol, mc_binary = _group_and_wire(rows)
    errors: list[str] = []
    for grp in negation.values():
        missing = (
            ([f"parent {grp.parent_id}"] if grp.parent is None else [])
            + (["¬A question"] if grp.neg_a is None else [])
        )
        if missing:
            errors.append(
                f"Incomplete negation group {grp.parent_id}: missing {', '.join(missing)}"
            )
    for grp in conditional.values():
        missing = (
            ([f"parent A {grp.parent_a_id}"] if grp.parent_a is None else [])
            + ([f"parent B {grp.parent_b_id}"] if grp.parent_b is None else [])
        )
        if missing:
            errors.append(
                f"Incomplete conditional group ({grp.parent_a_id}, {grp.parent_b_id}): "
                f"missing {', '.join(missing)}"
            )
    for grp in binning.values():
        if grp.parent is None:
            errors.append(
                f"Incomplete binning group {grp.parent_id}: missing parent question"
            )
        elif not grp.bins:
            errors.append(
                f"Incomplete binning group {grp.parent_id}: no bin questions found"
            )
    for grp in mc_consol.values():
        if grp.parent is None:
            errors.append(
                f"Incomplete MC option consolidation group {grp.parent_id}: missing parent question"
            )
        elif not grp.mc_children:
            errors.append(
                f"Incomplete MC option consolidation group {grp.parent_id}: no child questions found"
            )
    for grp in mc_binary.values():
        if grp.parent is None:
            errors.append(
                f"Incomplete MC binary regrouping group {grp.parent_id}: missing parent question"
            )
        elif not grp.options:
            errors.append(
                f"Incomplete MC binary regrouping group {grp.parent_id}: no option questions found"
            )
    return errors


# --------------------------------------------------------------------------- #
# Arbitrage evaluation functions
# --------------------------------------------------------------------------- #

negation_resolutions = arbitrage.NEGATION
addition_resolutions = arbitrage.ADDITION
conditional_resolutions = arbitrage.CONDITIONAL
expected_evidence_resolutions = arbitrage.EXPECTED_EVIDENCE
bayes_resolutions = arbitrage.BAYES

MAX_BINNING_SOLVER_DIM = arbitrage.MAX_PARTITION_SIZE


def _evaluate_negation(p: list[float], q: list[float]) -> float:
    return arbitrage.negation(p, q)


def _arbitrage_solver(resolutions: dict[str, list], forecasts: dict[str, list[float]]):
    return arbitrage.arbitrage_error(resolutions, forecasts), None


def _arbitrage_solver_binning(p_bots: list[list[float]], q_bots: list[list[float]]):
    return arbitrage.partition(p_bots, q_bots), None


def _bin_range_min(row: dict) -> float:
    info = _meta(row).get("info") or {}
    v = info.get("interval_min")
    return v if v is not None else float("-inf")


def _q_for_bin(
    parent_pmf: list[float],
    bin_info: dict,
    range_min: float,
    range_max: float,
    zero_point: float | None,
) -> float:
    b_min: float | None = bin_info.get("interval_min")
    b_max: float | None = bin_info.get("interval_max")
    cdf = _pmf_to_cdf(parent_pmf)
    lo = (
        0.0
        if b_min is None
        else _get_cdf_at(cdf, b_min, range_min, range_max, zero_point)
    )
    hi = (
        1.0
        if b_max is None
        else _get_cdf_at(cdf, b_max, range_min, range_max, zero_point)
    )
    return hi - lo


# --------------------------------------------------------------------------- #
# Output schema
# --------------------------------------------------------------------------- #

QUESTION_SLOTS: list[str] = [
    "A",
    "~A",
    "B",
    "AvB",
    "A&B",
    "B|A",
    "B|~A",
    "A|B",
    *[f"B{i}" for i in range(1, 9)],
]

CONSISTENCY_OUTPUT_COLS: list[str] = (
    ["consistency_check_type", "consistency_error"]
    + [col for s in QUESTION_SLOTS for col in (s, f"{s} pmf")]
    + ["Bs", "Bs pmfs"]
)


def _make_row(
    check_type: str,
    slots: dict[str, str | None],
    pmfs: dict[str, list[float] | None],
    score_val: float | None,
    bs: list[str | None] | None = None,
    bs_pmfs: list[list[float] | None] | None = None,
) -> dict:
    row: dict = {
        "consistency_check_type": check_type,
        "Bs": json.dumps(bs) if bs is not None else None,
        "Bs pmfs": json.dumps(bs_pmfs) if bs_pmfs is not None else None,
    }
    for s in QUESTION_SLOTS:
        row[s] = slots.get(s)
        pmf_val = pmfs.get(s)
        row[f"{s} pmf"] = json.dumps(pmf_val) if pmf_val is not None else None
    row["consistency_error"] = score_val
    return row


# --------------------------------------------------------------------------- #
# Progress printing helpers
# --------------------------------------------------------------------------- #


def _section_start(label: str, n: int) -> float:
    logger.info("%s (%d groups)", label, n)
    return time.time()


def _section_end(t0: float, n: int) -> None:
    if n:
        dt = time.time() - t0
        logger.info("  %.1fs total, %.2fs per group", dt, dt / n)


def _progress(i: int, n: int, detail: str) -> None:
    logger.debug("  %d/%d %s", i + 1, n, detail)


# --------------------------------------------------------------------------- #
# Main consistency evaluation
# --------------------------------------------------------------------------- #


def compute_consistency(data: list[dict], pmf_key: str) -> list[dict]:
    """Evaluate all consistency groups and return flat output rows."""
    neg_groups, cond_groups, bin_groups, mc_consol_groups, mc_binary_groups = (
        _group_consistency_rows(data, pmf_key)
    )
    logger.info("%d negation groups", len(neg_groups))
    logger.info("%d conditional groups", len(cond_groups))
    logger.info("%d binning groups", len(bin_groups))
    logger.info("%d MC option consolidation groups", len(mc_consol_groups))
    logger.info("%d MC binary regrouping groups", len(mc_binary_groups))

    rows: list[dict] = []
    # --- Negation ---
    _t0 = _section_start("Negation", len(neg_groups))
    for i, grp in enumerate(neg_groups):
        _progress(i, len(neg_groups), f"QID: {_qid(grp.parent)}")
        try:
            pa = _get_pmf(grp.parent, pmf_key)
            pna = _get_pmf(grp.neg_a, pmf_key)
            if pa is None or pna is None:
                raise ValueError("missing PMF")
            s = _evaluate_negation(pa, pna)
            rows.append(
                _make_row(
                    "negation",
                    {"A": _qid(grp.parent), "~A": _qid(grp.neg_a)},
                    {"A": pa, "~A": pna},
                    s,
                )
            )
        except Exception as e:
            logger.warning(f"negation error ({grp.parent_id}): {e}")
    _section_end(_t0, len(neg_groups))

    # --- Conditional group checks ---
    _t0 = _section_start("Conditional", len(cond_groups))
    for i, grp in enumerate(cond_groups):
        pa = _get_pmf(grp.parent_a, pmf_key)
        pb = _get_pmf(grp.parent_b, pmf_key)
        pna = _get_pmf(grp.neg_a, pmf_key)
        por = _get_pmf(grp.or_ab, pmf_key)
        pand = _get_pmf(grp.and_ab, pmf_key)
        pagivenb = _get_pmf(grp.a_given_b, pmf_key)
        pbgivena = _get_pmf(grp.b_given_a, pmf_key)
        pbgivenna = _get_pmf(grp.b_given_not_a, pmf_key)

        qid_a = _qid(grp.parent_a)
        qid_b = _qid(grp.parent_b)
        qid_na = _qid(grp.neg_a)
        qid_or = _qid(grp.or_ab)
        qid_and = _qid(grp.and_ab)
        qid_agivenb = _qid(grp.a_given_b)
        qid_bgivena = _qid(grp.b_given_a)
        qid_bgivenna = _qid(grp.b_given_not_a)
        label = f"({qid_a}, {qid_b})"
        _progress(i, len(cond_groups), f"QID A: {qid_a} QID B: {qid_b}")

        if pa is not None and pb is not None and por is not None and pand is not None:
            try:
                s, _ = _arbitrage_solver(
                    addition_resolutions, {"A": pa, "B": pb, "AvB": por, "A&B": pand}
                )
                rows.append(
                    _make_row(
                        "addition",
                        {"A": qid_a, "B": qid_b, "AvB": qid_or, "A&B": qid_and},
                        {"A": pa, "B": pb, "AvB": por, "A&B": pand},
                        s,
                    )
                )
            except Exception as e:
                logger.warning(f"addition error {label}: {e}")

        if pa is not None and pbgivena is not None and pand is not None:
            try:
                s, _ = _arbitrage_solver(
                    conditional_resolutions, {"A": pa, "B|A": pbgivena, "A&B": pand}
                )
                rows.append(
                    _make_row(
                        "conditional",
                        {"A": qid_a, "B|A": qid_bgivena, "A&B": qid_and},
                        {"A": pa, "B|A": pbgivena, "A&B": pand},
                        s,
                    )
                )
            except Exception as e:
                logger.warning(f"conditional error {label}: {e}")

        if (
            pa is not None
            and pna is not None
            and pb is not None
            and pbgivena is not None
            and pbgivenna is not None
        ):
            try:
                s, _ = _arbitrage_solver(
                    expected_evidence_resolutions,
                    {"A": pa, "~A": pna, "B": pb, "B|A": pbgivena, "B|~A": pbgivenna},
                )
                rows.append(
                    _make_row(
                        "expected_evidence",
                        {
                            "A": qid_a,
                            "~A": qid_na,
                            "B": qid_b,
                            "B|A": qid_bgivena,
                            "B|~A": qid_bgivenna,
                        },
                        {
                            "A": pa,
                            "~A": pna,
                            "B": pb,
                            "B|A": pbgivena,
                            "B|~A": pbgivenna,
                        },
                        s,
                    )
                )
            except Exception as e:
                logger.warning(f"expected_evidence error {label}: {e}")

        if (
            pa is not None
            and pb is not None
            and pagivenb is not None
            and pbgivena is not None
        ):
            try:
                s, _ = _arbitrage_solver(
                    bayes_resolutions,
                    {"A": pa, "B": pb, "A|B": pagivenb, "B|A": pbgivena},
                )
                rows.append(
                    _make_row(
                        "bayes",
                        {
                            "A": qid_a,
                            "B": qid_b,
                            "A|B": qid_agivenb,
                            "B|A": qid_bgivena,
                        },
                        {"A": pa, "B": pb, "A|B": pagivenb, "B|A": pbgivena},
                        s,
                    )
                )
            except Exception as e:
                logger.warning(f"bayes error {label}: {e}")
    _section_end(_t0, len(cond_groups))

    # --- Binning ---
    _t0 = _section_start("Binning", len(bin_groups))
    for i, grp in enumerate(bin_groups):
        _progress(i, len(bin_groups), f"QID: {_qid(grp.parent)} Bins: {len(grp.bins)}")
        try:
            if len(grp.bins) < 8:
                raise ValueError(f"not enough bins (found {len(grp.bins)}, expected 8)")
            parent_pmf = _get_pmf(grp.parent, pmf_key)
            if grp.parent is None or parent_pmf is None or not grp.bins:
                raise ValueError("missing parent, PMF, or bins")
            sc = _scaling(grp.parent)
            if sc is None:
                raise ValueError("no scaling")
            range_min, range_max, zero_point = sc
            sorted_bins = sorted(grp.bins, key=_bin_range_min)
            bin_pmfs = [_get_pmf(row, pmf_key) for row in sorted_bins]
            if any(p is None for p in bin_pmfs):
                raise ValueError("missing bin PMF")
            q_bots = [
                [1 - q, q]
                for row in sorted_bins
                for q in [
                    _q_for_bin(
                        parent_pmf,
                        (_meta(row).get("info") or {}),
                        range_min,
                        range_max,
                        zero_point,
                    )
                ]
            ]
            error, _ = _arbitrage_solver_binning(bin_pmfs, q_bots)  # type: ignore[arg-type]
            slots: dict[str, str | None] = {"A": _qid(grp.parent)}
            pmfs: dict[str, list[float] | None] = {"A": parent_pmf}
            for idx, (bin_row, bin_pmf) in enumerate(zip(sorted_bins, bin_pmfs), 1):
                slots[f"B{idx}"] = _qid(bin_row)
                pmfs[f"B{idx}"] = bin_pmf
            rows.append(_make_row("binning", slots, pmfs, error))
        except Exception as e:
            logger.warning(f"binning error ({grp.parent_id}): {e}")
    _section_end(_t0, len(bin_groups))

    # --- MC Option Consolidation ---
    _t0 = _section_start("MC option consolidation", len(mc_consol_groups))
    for i, grp in enumerate(mc_consol_groups):
        if grp.parent is None or not grp.mc_children:
            continue
        parent_pmf = _get_pmf(grp.parent, pmf_key)
        if parent_pmf is None:
            continue
        parent_outcomes = _options(grp.parent)
        outcome_to_idx = {o.strip("'\""): j for j, o in enumerate(parent_outcomes)}
        _progress(
            i,
            len(mc_consol_groups),
            f"QID: {_qid(grp.parent)} Parent options: {len(parent_outcomes)}",
        )

        for mc_row in grp.mc_children:
            try:
                child_pmf = _get_pmf(mc_row, pmf_key)
                if child_pmf is None:
                    raise ValueError("no PMF available")
                child_info = _meta(mc_row).get("info") or {}
                options_groups: list[list[str]] | None = child_info.get(
                    "options_groups"
                )
                if not options_groups:
                    raise ValueError("no options_groups metadata")
                mc_p_bots: list[list[float]] = []
                mc_q_bots: list[list[float]] = []

                if len(options_groups) == 1:
                    if set(options_groups[0]) == set(parent_outcomes):
                        options_groups = [[o] for o in _options(mc_row)]
                    else:
                        raise ValueError(
                            "single group doesn't cover all parent outcomes"
                        )

                for k, group in enumerate(options_groups):
                    if k >= len(child_pmf):
                        break
                    p_k = child_pmf[k]
                    q_k = sum(
                        parent_pmf[outcome_to_idx[opt]]
                        for opt in group
                        if opt in outcome_to_idx
                    )
                    mc_p_bots.append([1 - p_k, p_k])
                    mc_q_bots.append([1 - q_k, q_k])

                if not mc_p_bots:
                    raise ValueError("no valid option groups")
                error, _ = _arbitrage_solver_binning(mc_p_bots, mc_q_bots)
                rows.append(
                    _make_row(
                        "mc_option_consolidation",
                        {"A": _qid(grp.parent), "B1": _qid(mc_row)},
                        {"A": parent_pmf, "B1": child_pmf},
                        error,
                    )
                )
            except Exception as e:
                logger.warning(f"mc_option_consolidation error ({_qid(mc_row)}): {e}")
    _section_end(_t0, len(mc_consol_groups))

    # --- MC Binary Regrouping ---
    _t0 = _section_start("MC binary regrouping", len(mc_binary_groups))
    for i, grp in enumerate(mc_binary_groups):
        try:
            if grp.parent is None or not grp.options:
                raise ValueError("missing parent or options")
            parent_pmf = _get_pmf(grp.parent, pmf_key)
            if parent_pmf is None:
                raise ValueError("no PMF available")
            parent_outcomes = _options(grp.parent)
            outcome_to_idx = {o.strip("'\""): j for j, o in enumerate(parent_outcomes)}
            _progress(
                i,
                len(mc_binary_groups),
                f"QID: {_qid(grp.parent)} Parent options: {len(parent_outcomes)}",
            )
            if len(parent_outcomes) > MAX_BINNING_SOLVER_DIM:
                raise ValueError(
                    f"skipping high-cardinality mc_binary group ({len(parent_outcomes)} options)"
                )
            if len(parent_outcomes) != len(grp.options):
                raise ValueError(
                    f"number of options ({len(parent_outcomes)}) doesn't match number of "
                    f"binary questions ({len(grp.options)})"
                )

            def _outcome_order(
                row: dict, _oti=outcome_to_idx, _n=len(parent_outcomes)
            ) -> int:
                info = _meta(row).get("info") or {}
                opt = info.get("option") or []
                return _oti.get(opt, _n) if opt else _n

            p_bots: list[list[float]] = []
            q_bots: list[list[float]] = []
            valid: list[dict] = []
            for row in sorted(grp.options, key=_outcome_order):
                info = _meta(row).get("info") or {}
                opt = info.get("option") or []
                if not opt:
                    raise ValueError(f"option {_qid(row)} has no option metadata")
                idx = outcome_to_idx.get(opt)
                if idx is None:
                    raise ValueError(
                        f"option '{opt}' ({_qid(row)}) not found in parent outcomes"
                    )
                child_pmf = _get_pmf(row, pmf_key)
                if child_pmf is None:
                    raise ValueError(f"option {_qid(row)} has no PMF available")
                q_val = parent_pmf[idx]
                p_bots.append(child_pmf)
                q_bots.append([1 - q_val, q_val])
                valid.append(row)

            if valid:
                error, _ = _arbitrage_solver_binning(p_bots, q_bots)
                if len(valid) <= 8:
                    b_slots: dict[str, str | None] = {"A": _qid(grp.parent)}
                    b_pmfs: dict[str, list[float] | None] = {"A": parent_pmf}
                    for j, (vrow, vpmf) in enumerate(zip(valid, p_bots), 1):
                        b_slots[f"B{j}"] = _qid(vrow)
                        b_pmfs[f"B{j}"] = vpmf
                    rows.append(
                        _make_row("mc_binary_regrouping", b_slots, b_pmfs, error)
                    )
                else:
                    rows.append(
                        _make_row(
                            "mc_binary_regrouping",
                            {"A": _qid(grp.parent)},
                            {"A": parent_pmf},
                            error,
                            bs=[_qid(vrow) for vrow in valid],
                            bs_pmfs=list(p_bots),
                        )
                    )
        except Exception as e:
            logger.warning(f"mc_binary_regrouping error ({grp.parent_id}): {e}")
    _section_end(_t0, len(mc_binary_groups))

    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input-csv",
        dest="input_csvs",
        action="append",
        required=True,
        metavar="FILE",
        help="Input CSV file (repeatable).",
    )
    ap.add_argument(
        "--scored-col-info",
        dest="scored_col_infos",
        action="append",
        required=True,
        metavar="PMF_COL,FORECASTER_NAME",
        help="'<pmf_col>,<forecaster_name>' pair to evaluate (repeatable).",
    )
    ap.add_argument("--output-folder", default="consistency_output")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Parse --scored-col-info → [(display_name, pmf_col), ...]
    forecasters_spec: list[tuple[str, str]] = []
    for info in args.scored_col_infos:
        parts = info.split(",", 1)
        if len(parts) != 2:
            sys.exit(
                f"--scored-col-info must be '<pmf_col>,<forecaster_name>', got: {info!r}"
            )
        pmf_col, display_name = parts
        forecasters_spec.append((display_name, pmf_col))

    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading {len(args.input_csvs)} file(s)...")
    all_rows: list[dict] = []
    for file_str in args.input_csvs:
        file = Path(file_str)
        logger.info(f"  {file.name}")
        df = pd.read_csv(file, low_memory=False)
        all_rows.extend(df.to_dict("records"))
    logger.info(f"Read {len(all_rows)} rows from {len(args.input_csvs)} file(s)")

    for display_name, pmf_col in forecasters_spec:
        col_prefix = pmf_col.removesuffix("_pmf")
        logger.info(f"Forecaster: {display_name} ({pmf_col})")
        consistency_rows = compute_consistency(all_rows, pmf_col)
        out_df = pd.DataFrame(consistency_rows, columns=CONSISTENCY_OUTPUT_COLS)
        out = output_path / f"{col_prefix}_consistency.csv"
        out_df.to_csv(out, index=False)
        logger.info(f"Wrote {out} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
