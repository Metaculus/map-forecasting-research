"""Quantitative map builder: a computable forecasting DAG, built the way a
professional in the question's field would model it.

Stage 1 (`elicit_method`): ask how quantitative professionals in the relevant
field model questions like this: the units, the canonical algebra, the base
process, which factors shift the mean versus widen the uncertainty, what
current data already reflects, and what experts deliberately leave out.

Stage 2 (`instantiate`): turn that method, plus any evidence you supply, into a
DAG of distributions, parameters and formulas in real-world units, validated
for missing references and malformed distributions, with one repair round.

`build_quantitative_map` runs both and returns the DAG and a MapIR view of it.
"""
from __future__ import annotations

import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..llm import call_structured
from ..maps.ir import Edge, MapIR


class Quantity(BaseModel):
    name: str
    unit: str
    role: Literal["base", "mean_effect", "uncertainty_effect", "conversion"]
    why_tracked: str


class Omission(BaseModel):
    factor: str
    why_omitted: Literal["unpredictable", "negligible_variance", "implied_by_data",
                         "double_counts"]


class MethodSpec(BaseModel):
    profession: str = Field(description="who does this for a living")
    canonical_algebra: str = Field(description="the whiteboard decomposition: what is "
                                   "added, multiplied, thresholded, in what order")
    core_quantities: list[Quantity] = Field(description="the 3-6 quantities the field tracks")
    base_process: str = Field(description="the field's model of normal variation for "
                              "this target, stated so it can become explicit nodes")
    information_principle: str = Field(description="what current data already reflects "
                                       "and should not be re-added as a factor, or 'none'")
    deliberate_omissions: list[Omission]
    update_rules: str = Field(description="how the model changes as new data lands, "
                              "including time passing")


class Indicator(BaseModel):
    id: str
    title: str = Field(description="a fact, or a pending proxy question")
    status: Literal["resolved", "pending"]
    detail: str = Field(description="source; value or resolution criteria")
    informs: list[str] = Field(description="refs of the quant nodes this evidence sets")


class QuantNode(BaseModel):
    kind: Literal["distribution", "parameter", "formula"]
    ref: str = Field(description="snake_case identifier")
    title: str = Field(description="title including the unit")
    unit: str
    description: str = Field(description="evidence basis and update rule")
    family: Optional[Literal["skewed", "bell_curve", "uniform"]] = Field(
        default=None, description="distributions only")
    p10: Optional[float] = None
    p50: Optional[float] = None
    p90: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    value: Optional[float] = Field(default=None, description="parameters only")
    expression: Optional[str] = Field(default=None, description="formulas only; uses other refs")


class QuantMap(BaseModel):
    indicators: list[Indicator]
    quant_nodes: list[QuantNode]
    target_ref: str = Field(description="ref of the node that answers the question")
    target_output: Literal["probability", "numeric"]
    omissions_note: str
    method_note: str = Field(description="5-8 sentences: the field method as implemented")


METHOD_PROMPT = """FORECAST QUESTION: {question}
Resolution criteria: {criteria}

Before any model is built: describe how quantitative professionals in the relevant
field (name the profession) actually model questions like this when advising clients.
Examples of an information principle: market prices embed public information;
current case counts embed current transmission; polls embed current opinion."""

INSTANTIATE_PROMPT = """Build a forecasting map as a computable DAG, following the field
method below exactly. This is a formalization of how {profession} would model it.

QUESTION: {question}
Resolution criteria: {criteria}
{as_of}
FIELD METHOD (follow its algebra, units and omissions):
{method}

EVIDENCE:
{evidence}

LEGIBILITY CONTRACT:
- Every node has a real-world unit from the field method. No unitless weights or
  scores; no logit unless the field itself uses log-odds.
- The base process is explicit nodes, one per term. Mean-effect factors enter the
  algebra where the field puts them; uncertainty-effect factors scale noise terms
  and never enter as level shifts.
- Respect the information principle: do not re-add what current data embeds.
- Honor the deliberate omissions in the omissions note, not as nodes.
- Every distribution's description says what evidence sets it and what would
  change it.

Expression language: + - * / ^ (x^0.5 for sqrt), comparisons, if(cond, a, b), min,
max, abs, clamp(x, lo, hi), and no other functions. skewed takes p10 <= p50 <= p90;
bell_curve takes p10 and p90; uniform takes min and max. If target_output is
probability, the target expression must be a comparison or if(<value> >= <threshold>, 1, 0).
8-20 quant nodes in total. Every ref a formula uses must exist."""

_IDENT = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_RESERVED = {"if", "min", "max", "abs", "round", "floor", "ceil", "clamp", "logit",
             "inv_logit", "odds", "cdf", "prob_lt", "prob_gt", "quantile"}


def _refs_in(expr: str) -> set[str]:
    return set(_IDENT.findall(expr)) - _RESERVED


def validate(qm: QuantMap) -> list[str]:
    """Structural problems in a quantitative map; empty if none."""
    errors = []
    refs = {n.ref for n in qm.quant_nodes}
    for n in qm.quant_nodes:
        if n.kind == "distribution":
            if n.family == "skewed" and not (None not in (n.p10, n.p50, n.p90)
                                             and n.p10 <= n.p50 <= n.p90):
                errors.append(f"{n.ref}: skewed needs p10 <= p50 <= p90")
            if n.family == "bell_curve" and None in (n.p10, n.p90):
                errors.append(f"{n.ref}: bell_curve needs p10 and p90")
            if n.family == "uniform" and None in (n.min, n.max):
                errors.append(f"{n.ref}: uniform needs min and max")
            if n.family is None:
                errors.append(f"{n.ref}: distribution has no family")
        if n.kind == "parameter" and n.value is None:
            errors.append(f"{n.ref}: parameter has no value")
        if n.kind == "formula":
            missing = {u for u in _refs_in(n.expression or "") if u not in refs}
            if not n.expression:
                errors.append(f"{n.ref}: formula has no expression")
            elif missing:
                errors.append(f"{n.ref}: unknown refs {sorted(missing)}")
    if qm.target_ref not in refs:
        errors.append(f"target_ref {qm.target_ref} is not a quant node")
    return errors


def _fix_families(qm: QuantMap) -> None:
    for n in qm.quant_nodes:
        if n.kind == "distribution" and n.family == "uniform" and n.min is None:
            n.min, n.max = n.p10, n.p90


def elicit_method(question: str, criteria: str = "", llm=None) -> MethodSpec:
    return call_structured(METHOD_PROMPT.format(question=question, criteria=criteria),
                           MethodSpec, max_tokens=6000, llm=llm)


def instantiate(method: MethodSpec, question: str, criteria: str = "",
                evidence: list[str] | None = None, as_of: str | None = None,
                llm=None) -> QuantMap:
    prompt = INSTANTIATE_PROMPT.format(
        profession=method.profession, question=question, criteria=criteria,
        as_of=f"As-of date: {as_of} (use only evidence up to this date).\n" if as_of else "",
        method=json.dumps(method.model_dump(), indent=1),
        evidence="\n".join(f"- {e}" for e in evidence) if evidence else "(none supplied)")
    qm = call_structured(prompt, QuantMap, max_tokens=14000, llm=llm)
    _fix_families(qm)
    errors = validate(qm)
    if errors:
        repair = (f"{prompt}\n\nA previous attempt had these validation errors; return a "
                  f"corrected map that avoids them: " + "; ".join(errors))
        qm = call_structured(repair, QuantMap, max_tokens=14000, llm=llm)
        _fix_families(qm)
        errors = validate(qm)
        if errors:
            raise ValueError(f"map still invalid after repair: {errors}")
    return qm


def to_ir(qm: QuantMap, name: str = "quantitative map") -> MapIR:
    """Nodes for quant nodes and indicators; edges from formula inputs and from
    each indicator to the nodes it informs."""
    nodes: dict[str, dict] = {}
    for n in qm.quant_nodes:
        nodes[n.ref] = {"title": n.title, "desc": n.description,
                        "is_main": n.ref == qm.target_ref,
                        "node_type": n.kind, "quant": n.model_dump(exclude_none=True)}
    for ind in qm.indicators:
        nodes[f"ind_{ind.id}"] = {"title": ind.title, "desc": ind.detail, "is_main": False,
                                  "node_type": f"indicator_{ind.status}"}
    edges = []
    for n in qm.quant_nodes:
        if n.kind == "formula":
            for ref in sorted(_refs_in(n.expression or "")):
                if ref in nodes and ref != n.ref:
                    edges.append(Edge(source=ref, target=n.ref, type="causal",
                                      provenance={"from": "formula"}))
    for ind in qm.indicators:
        for ref in ind.informs:
            if ref in nodes:
                edges.append(Edge(source=f"ind_{ind.id}", target=ref, type="evidential"))
    return MapIR(name=name, nodes=nodes, edges=edges,
                 meta={"generated": True, "target_ref": qm.target_ref,
                       "target_output": qm.target_output,
                       "method_note": qm.method_note, "omissions_note": qm.omissions_note})


def build_quantitative_map(question: str, criteria: str = "",
                           evidence: list[str] | None = None, as_of: str | None = None,
                           llm=None) -> tuple[MethodSpec, QuantMap, MapIR]:
    method = elicit_method(question, criteria, llm=llm)
    qm = instantiate(method, question, criteria, evidence, as_of, llm=llm)
    return method, qm, to_ir(qm, name=question)
