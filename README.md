# usability-metrics-kit

**Calculate, benchmark and narrate the whole usability-metrics family — from one toolkit, with one consistent output shape.**

Most teams already collect SUS, a task-completion rate, maybe an NPS. The numbers
then live in three different spreadsheets, get interpreted three different ways,
and arrive in the readout stripped of their confidence intervals. `usability-metrics-kit`
scores fifteen standard metrics through a single interface, attaches the published
benchmarks (grades, adjectives, norms), keeps a confidence interval on every
headline number, and — optionally — writes the plain-language readout for you.

It also ships an **AI-native flows** module for evaluating generative-AI
interfaces, where deterministic usability metrics fall short: retention of
AI-generated content, prompt iterations, human fallback, suggestion acceptance,
the Jian et al. trust-in-automation scale, perceived output quality, and a
composite AI Collaboration Score — each with a statistically appropriate interval.

The rule-based core runs fully offline with zero API calls. LLM narrative
enrichment is **on by default** across three backends and degrades silently to
the offline scores when no API key is present.

---

## What it scores

| Metric | What you get | Input |
| --- | --- | --- |
| **SUS** | 0–100 score, **Curved Grading Scale** grade, Bangor adjective, acceptability band, approx. percentile, 95% CI | 10 items, 1–5 |
| **UMUX** | 0–100 SUS surrogate | 4 items, 1–7 |
| **UMUX-LITE** | Raw + SUS-corrected estimate | 2 items, 1–7 |
| **SUPR-Q** | Overall + usability / trust / appearance / loyalty sub-scales | 8 items |
| **PSSUQ** | Overall + 3 sub-scales vs. Sauro–Lewis norms | 16 items, 1–7 |
| **CSUQ** | IBM computer-system variant | 16 items, 1–7 |
| **SEQ** | Single-task ease with a benchmark flag | 1 item, 1–7 |
| **UEQ-S** | Pragmatic + hedonic quality with benchmark bands | 8 items, 1–7 |
| **NASA-TLX** | Raw or weighted workload + per-dimension breakdown | 6 items, 0–100 |
| **Task Completion Rate** | % complete with an **adjusted-Wald** interval | 0/1 flags |
| **Time on Task** | **Geometric mean** + log interval, plus mean & median | seconds |
| **Error Rate** | Errors per user / per opportunity | counts |
| **NPS** | Score + promoter/passive/detractor split | 0–10 |
| **CSAT** | Top-two-box % + mean | 1–5 |
| **CES** | Effort agreement + easy % | 1–7 |
| **Usability Composite** | Equal-weight SUM-style roll-up | normalised sub-scores |

Every metric returns the same `MetricResult` shape: `score`, `scale`,
`direction`, `n`, a confidence `interval`, plain-language `interpretation`,
structured `components`, `benchmark` detail, `warnings`, and an optional
`llm_summary`.

---

## AI-native flows

Generative-AI interfaces break the assumptions traditional metrics rely on:
output is non-deterministic, "acceptance" is graded rather than binary, and the
sharpest failure signal is often *abandonment* rather than task failure. This
module adds two layers, each metric carrying the interval its data type demands.

| Metric | What you get | Interval |
| --- | --- | --- |
| **AI Content Retention Rate** | % of AI-generated content surviving into the final submission (mean of per-session fractions) | t-interval, clamped to [0,1] — *not* binomial |
| **Prompt Iterations** | Mean turns to a satisfactory output, with an overdispersion check | **Poisson exact (Garwood)** |
| **Human Fallback Rate** | Rate of abandoning the AI to work manually | adjusted-Wald |
| **AI Acceptance Rate** | Share of suggestions accepted vs. dismissed (+ clustering & automation-bias caveats) | adjusted-Wald |
| **Trust in Automation** | Jian et al. (2000) 12-item scale; distrust items reverse-coded; Trust & Distrust subscales | t-interval |
| **Perceived Output Quality** | 3-item accuracy / relevance / helpfulness scale | t-interval |
| **AI Collaboration Score** | Heuristic equal-weight roll-up of acceptance, retention & trust, with SD | t-interval (aligned per-unit data) |

Two design notes worth stating plainly. **Retention is a continuous proportion,
not a binomial** — each session contributes a *fraction*, so it gets a t-interval
on the mean rather than an adjusted-Wald count interval. **Iteration counts are
usually overdispersed**, so the Poisson interval can be too narrow; the kit
computes the index of dispersion and warns when a bootstrap or negative-binomial
model would be the more honest choice.

Unlike SUS or PSSUQ, none of these have an established normed database yet, so
their benchmark bands are explicitly orientation heuristics, surfaced as such in
each result's `warnings` — not validated norms.

```python
from usability_metrics_kit import (
    retention_rate, prompt_iterations, ai_acceptance_rate,
    trust_automation, ai_collaboration_score,
)

retention_rate([0.82, 0.61, 0.74, 0.90])          # 76.8%, t (clamped)
prompt_iterations([2, 3, 1, 4, 2])                 # 2.4 turns, Poisson exact
ai_acceptance_rate(accepted=8, offered=10)         # 80%, adjusted-Wald

# Composite from aligned per-participant data (honest joint SD):
ai_collaboration_score(
    acceptance=[0.79, 0.71, 0.83],
    retention=[0.82, 0.61, 0.74],
    trust=[5.2, 4.8, 5.5],
)
```

---

## Install

```bash
pip install usability-metrics-kit          # core (offline)
pip install "usability-metrics-kit[ui]"     # + Streamlit app
pip install "usability-metrics-kit[llm]"    # + DSPy/LiteLLM narrative layer
```

## Command line

```bash
umk list                                    # every metric, grouped by family

umk score -m sus responses.csv              # rich readout (+ narrative if a key is set)
umk score -m sus responses.csv --no-llm     # offline only
umk score -m nps ratings.csv --json-output  # machine-readable, for CI

# Score a whole study from a {metric: csv} spec:
umk scorecard study.json --study "Checkout v2"
```

`study.json`:

```json
{ "sus": "data/sus.csv", "task_success": "data/tasks.csv", "nps": "data/nps.csv" }
```

## Streamlit app

```bash
streamlit run usability_metrics_kit/app.py
```

Pick a metric, paste or upload data, and read the scored result with its
interval, benchmark chips, sub-scale chart and (optionally) an LLM narrative.

## Python

```python
from usability_metrics_kit import score_sus, task_success_rate

sus = score_sus([[4, 2, 4, 2, 4, 1, 4, 2, 4, 2], ...])
print(sus.score, sus.benchmark["grade"], sus.interval.as_tuple())

rate = task_success_rate([1, 1, 1, 0, 1])
print(rate.score, rate.interval.as_tuple())
```

---

## LLM narratives (on by default)

Set any one of these and narratives turn on automatically:

```bash
export ANTHROPIC_API_KEY=...   # claude opus      -> anthropic/claude-opus-4-7
export GEMINI_API_KEY=...      # gemini 4 argon    -> gemini/gemini-4-argon
export OPENAI_API_KEY=...      # gpt 5.5           -> openai/gpt-5.5
```

Model strings live in `usability_metrics_kit/llm.py` and are configurable. The
narrative layer is strictly additive: it never changes a score, and with no key
present the kit returns the rule-based result untouched.

---

## What this is — and what it is not

**It is** a calculator and interpretation aid. It applies published scoring
formulas, attaches the benchmarks those authors published alongside them, and
keeps the uncertainty visible so a sample of eight is read as a sample of eight.

**It is not** a verdict. A SUS grade is a comparison against a reference average,
not a measure of whether your product is good. Confidence intervals from small
samples are wide on purpose. Where a "percentile" depends on a proprietary normed
database (SUS, SUPR-Q), the kit uses a transparent Normal approximation and says
so in the result's `warnings` — it does not pretend to the original norms. The
LLM narrative is a drafting aid for your readout, not an analyst.

Treat the numbers as signal for your judgement, not a substitute for it.

---

## Part of the CPR Ecosystem

`usability-metrics-kit` is one of the **CPR Orbital Series** — open-source tools
for AI-native product research, shipped under the discipline of **Computational
Product Research (CPR)**.

Related quantitative & mixed-methods tools in the series:

- **max-diff-analyzer** — LLM-assisted MaxDiff preference scoring and segmentation
- **survey-designer** — survey question generation with bias detection
- **segment-finder** — behavioural segmentation with LLM-generated profiles
- **quant-qual-bridge** — links quantitative segments to qualitative insight threads

See the rest of the series and the broader CPR work at
[github.com/kgkazakos](https://github.com/kgkazakos).

## License

MIT — see [LICENSE](LICENSE).
