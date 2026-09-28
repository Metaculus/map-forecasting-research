# map-forecasting

Tools for forecasting with causal maps:

- **Consistency checks** for probability estimates on linked questions: how much a
  bettor could win for sure against a set of forecasts, and the nearest coherent set.
- **Map adapters and cleaning**: read Guesstimate, Squiggle Hub and Radiant maps into
  one format, and find reversed arrows and other likely errors in hand-drawn maps.
- **Map grading**: compare a map with a reference map, node by node and edge by edge.
- **Key-factor grading**: predict how forecasters would rate the key factors on a question.
- **Map builders**: build causal or quantitative maps with an LLM.

The LLM-backed tools work with any model: OpenAI, Anthropic, OpenRouter, a local
server, or your own function.

## Install

```bash
pip install -e .                 # core: consistency checks, adapters, scoring
pip install -e ".[openai]"       # client for OpenAI-compatible endpoints
pip install -e ".[anthropic]"    # client for Anthropic
pip install -e ".[radiant]"      # the Radiant MCP toolset for pydantic-ai
pip install -e ".[dev]"          # pytest
```

Python 3.10+. The consistency checks, adapters and scoring need no model or API key.

## Choosing an LLM

Alignment, capture grading, cleaning, key-factor grading and the builders take an
`llm` argument, or use a default you set once:

```python
from map_forecasting.llm import (AnthropicLLM, FunctionLLM, OpenAICompatibleLLM,
                                 set_default_llm)

set_default_llm(OpenAICompatibleLLM("gpt-4.1"))                          # OpenAI
set_default_llm(OpenAICompatibleLLM("google/gemini-3.5-flash-lite",
                                    base_url="https://openrouter.ai/api/v1",
                                    api_key_env="OPENROUTER_API_KEY"))    # OpenRouter
set_default_llm(OpenAICompatibleLLM("llama3", base_url="http://localhost:11434/v1"))
set_default_llm(AnthropicLLM("claude-opus-5"))
set_default_llm(FunctionLLM(lambda system, prompt: my_model(system, prompt)))
```

Or set `MAP_FORECASTING_LLM=openai:<model>` (honours `OPENAI_BASE_URL`) or
`anthropic:<model>`. Any object with a `generate(system, prompt, max_tokens) -> str`
method works too. Structured output is handled for you: the JSON schema goes into the
prompt, and the reply is validated, with one repair round if needed. Responses are
cached on disk (`MAP_FORECASTING_CACHE_DIR`), so re-running on the same inputs is free.

The tools were developed with strong frontier models. Small models run them, but
check their output: in testing, a small model's cleaning audit flagged correct arrows
as reversed.

## Consistency checks

Each check takes probabilities of "yes" and returns the guaranteed log-score profit
available against them: 0 means coherent.

```python
from map_forecasting.consistency import negation, expected_evidence, WORLD_TABLES, project

negation(p_a=0.40, p_not_a=0.55)                           # 0.0026
expected_evidence(p_a=0.4, p_not_a=0.6, p_b=0.5,
                  p_b_given_a=0.7, p_b_given_not_a=0.15)   # 0.0188

forecasts = {"A": 0.4, "~A": 0.6, "B": 0.5, "B|A": 0.7, "B|~A": 0.15}
project(forecasts, WORLD_TABLES["expected_evidence"])      # nearest coherent set
```

| Function | Checks |
|---|---|
| `negation` | P(A) + P(not A) = 1 |
| `addition` | P(A or B) = P(A) + P(B) − P(A and B) |
| `conditional` | P(A and B) = P(A) P(B given A) |
| `expected_evidence` | P(B) = P(B given A) P(A) + P(B given not A) P(not A) |
| `bayes` | P(A given B) P(B) = P(B given A) P(A) |
| `partition` | yes/no forecasts on exclusive outcomes agree with the parent distribution (bins of a numeric forecast, regrouped multiple-choice options) |
| `arbitrage_error` | any check you describe with your own world table |

Also:
- `project` / `shrink`: move forecasts all or part of the way to the nearest coherent
  set, under KL, L2 or the bettor's own objective.
- `rounding_floor`: the error that rounding to a grid (1% by default) produces on its
  own. Treat measured errors below it as zero.
- `consistency.batch`: runs every check over a CSV of linked questions in the
  Metaculus consistency-question format; the format is documented in the module.

  ```bash
  python -m map_forecasting.consistency.batch --input-csv examples/forecasts.csv \
      --scored-col-info example_pmf,example --output-folder out/
  ```

## Maps

Every adapter and builder produces a `MapIR`: nodes keyed by slug, and edges pointing
from cause to effect, toward the goal.

```python
from map_forecasting.maps import MapIR, fetch_space, from_guesstimate
from map_forecasting.maps.cleaning import clean_map

m = from_guesstimate(fetch_space(12345))   # a public Guesstimate model
audit, fixed = clean_map(m)                # LLM audit + reversed-arrow fixes (review them)
m.sinks(), m.cycles(), m.save(path)
```

Adapters: `from_guesstimate`, `from_squiggle_code` (with `fetch_model_source`),
`from_radiant_export`, `from_radiant_v2`, and `to_radiant_payload` for the reverse direction.

## Grading

```python
from map_forecasting.grading import grade_map, score_map

result = grade_map(generated, reference)   # align nodes, score, capture-grade
result["coverage"], result["coverage_graded"], result["edges_induced"], result["reachability"]
```

- **Alignment** (`align_nodes`): which generated nodes express each reference node, on
  meaning rather than wording. Many-to-one matches are allowed.
- **Coverage**: the share of reference nodes found. Extra generated nodes don't count
  against it; node precision reports them separately.
- **Capture grading**: whether each matched reference node is fully, partly or not
  really expressed (1, ½, 0). It is always at or below binary coverage.
- **Edges**: precision, recall and F1 on the edges between matched nodes, and over all
  reference edges.
- **Reachability**: agreement on which nodes lead to which, so a map that routes a
  dependency through a different intermediate still gets credit.
- `score_edges` scores edges predicted over the reference's own nodes, with type
  accuracy, a PR curve over confidence, and calibration.

## Key-factor grading

A key factor is a short statement attached to a question: a driver, a news item or a
base rate. The grader predicts how forecasters would rate each one on Metaculus's four
strength buttons (no impact, low, medium, high, scored 0/1/2/5). It returns a
distribution per factor, since raters disagree.

```python
from map_forecasting.grading import Anchor, KeyFactorGrader

grader = KeyFactorGrader(anchors=[Anchor(question_title, factor_text, shares, n_ratings), ...])
grades = grader.grade(question={"title": ..., "description": ..., "resolution_criteria": ...},
                      factors=[{"id": "a", "text": ..., "type": "news"}, ...])
grades["a"].expected()        # predicted mean score
```

Calibrate it with anchors: rated key factors from other questions, with the share of
each button. Optionally pass overall button shares as `base_rates`.
`grading.key_factors.evaluate` scores predictions against observed ratings, across
factors and within each question. `noise_ceiling` estimates how well half of the
raters predict the other half, which is roughly the best any grader can do.

## Builders

```python
from map_forecasting.builders import problem_brief, build_nodes, add_edges, expand_layer

brief = problem_brief("City bus electrification", "Most city buses are electric by 2030")
m = build_nodes("City bus electrification", "Most city buses are electric by 2030",
                n=15, brief=brief, method="workshop")   # brainstorm, then cluster
m = add_edges(m, brief=brief)
m = expand_layer(m)                                     # the next layer of causes
```

`build_map` produces nodes and edges in one call. `build_quantitative_map(question,
criteria, evidence)` works differently: it first asks how professionals in the field
model the question, then builds a validated DAG of distributions and formulas in
real-world units.

## Radiant

`map_forecasting.radiant.RadiantMCP` is a small synchronous MCP client.
`map_forecasting.radiant.toolset.build_radiant_toolset` is a reconnecting toolset for
pydantic-ai agents. Both need `RADIANT_BASE_URL` and `RADIANT_API_KEY`.

## Examples and tests

`examples/` has runnable scripts on small made-up inputs:
- `check_consistency.py` and `grade_maps.py` run offline.
- `grade_maps.py --llm`, `grade_key_factors.py` and `build_a_map.py` need an LLM.

`pytest` runs the tests offline; the LLM calls are stubbed.

## License

Not yet chosen.
