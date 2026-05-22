"""Central registry of every metric the kit knows how to score.

The registry is the single source of truth for the CLI help text, the
Streamlit picker and the test suite, so adding a metric in one place lights it
up everywhere.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from . import ai_native as ai
from . import questionnaires as q
from . import task_metrics as t
from .models import MetricFamily, MetricResult


@dataclass(frozen=True)
class MetricSpec:
    key: str
    name: str
    family: MetricFamily
    input_kind: str  # "matrix" | "binary" | "values" | "ratings_0_10" | ...
    n_items: int | None  # columns expected for matrix inputs
    item_scale: str
    blurb: str
    scorer: Callable[..., MetricResult]


REGISTRY: dict[str, MetricSpec] = {
    "sus": MetricSpec(
        "sus", "System Usability Scale", MetricFamily.QUESTIONNAIRE,
        "matrix", 10, "1-5",
        "The 10-item industry standard; grade, adjective and acceptability.",
        q.score_sus,
    ),
    "umux": MetricSpec(
        "umux", "UMUX", MetricFamily.QUESTIONNAIRE, "matrix", 4, "1-7",
        "Four-item SUS surrogate.", q.score_umux,
    ),
    "umux_lite": MetricSpec(
        "umux_lite", "UMUX-LITE", MetricFamily.QUESTIONNAIRE, "matrix", 2, "1-7",
        "Two-item micro-survey with a SUS-corrected estimate.", q.score_umux_lite,
    ),
    "suprq": MetricSpec(
        "suprq", "SUPR-Q", MetricFamily.QUESTIONNAIRE, "matrix", 8,
        "items 1-7: 1-5, item 8: 0-10",
        "Website quality across usability, trust, appearance, loyalty.",
        q.score_suprq,
    ),
    "pssuq": MetricSpec(
        "pssuq", "PSSUQ", MetricFamily.QUESTIONNAIRE, "matrix", 16, "1-7 (low=good)",
        "IBM post-study questionnaire with three sub-scales.", q.score_pssuq,
    ),
    "csuq": MetricSpec(
        "csuq", "CSUQ", MetricFamily.QUESTIONNAIRE, "matrix", 16, "1-7 (low=good)",
        "IBM computer-system usability questionnaire.", q.score_csuq,
    ),
    "seq": MetricSpec(
        "seq", "Single Ease Question", MetricFamily.TASK, "matrix", 1, "1-7",
        "Per-task ease rating with benchmark flag.", q.score_seq,
    ),
    "ueq_s": MetricSpec(
        "ueq_s", "UEQ-S", MetricFamily.QUESTIONNAIRE, "matrix", 8, "1-7",
        "Short UX questionnaire: pragmatic + hedonic quality.", q.score_ueq_s,
    ),
    "nasa_tlx": MetricSpec(
        "nasa_tlx", "NASA-TLX", MetricFamily.WORKLOAD, "matrix", 6, "0-100",
        "Six-dimension perceived workload (raw TLX).", q.score_nasa_tlx,
    ),
    "task_success": MetricSpec(
        "task_success", "Task Completion Rate", MetricFamily.TASK, "binary", 1,
        "0/1", "Completion rate with an adjusted-Wald interval.",
        t.task_success_rate,
    ),
    "time_on_task": MetricSpec(
        "time_on_task", "Time on Task", MetricFamily.TASK, "values", 1, "seconds",
        "Geometric-mean completion time with a log interval.", t.time_on_task,
    ),
    "error_rate": MetricSpec(
        "error_rate", "Error Rate", MetricFamily.TASK, "values", 1, "count",
        "Errors per user / per opportunity.", t.error_rate,
    ),
    "nps": MetricSpec(
        "nps", "Net Promoter Score", MetricFamily.LOYALTY, "ratings_0_10", 1,
        "0-10", "Promoters minus detractors.", t.net_promoter_score,
    ),
    "csat": MetricSpec(
        "csat", "CSAT", MetricFamily.LOYALTY, "ratings", 1, "1-5",
        "Top-two-box satisfaction.", t.csat,
    ),
    "ces": MetricSpec(
        "ces", "Customer Effort Score", MetricFamily.LOYALTY, "ratings", 1, "1-7",
        "Effort agreement, higher = easier.", t.ces,
    ),
    "retention_rate": MetricSpec(
        "retention_rate", "AI Content Retention Rate", MetricFamily.AI_NATIVE,
        "values", 1, "0-1 fraction",
        "Share of AI-generated content surviving into the final submission.",
        ai.retention_rate,
    ),
    "prompt_iterations": MetricSpec(
        "prompt_iterations", "Prompt Iterations to Satisfaction", MetricFamily.AI_NATIVE,
        "values", 1, "count",
        "Turns to a satisfactory output (Poisson exact interval).",
        ai.prompt_iterations,
    ),
    "human_fallback": MetricSpec(
        "human_fallback", "Human Fallback Rate", MetricFamily.AI_NATIVE,
        "binary", 1, "0/1",
        "Rate of abandoning the AI to do the task manually.",
        ai.human_fallback_rate,
    ),
    "ai_acceptance": MetricSpec(
        "ai_acceptance", "AI Acceptance Rate", MetricFamily.AI_NATIVE,
        "binary", 1, "0/1",
        "Share of AI suggestions accepted vs dismissed.",
        ai.ai_acceptance_rate,
    ),
    "trust_automation": MetricSpec(
        "trust_automation", "Trust in Automation (Jian et al.)", MetricFamily.AI_NATIVE,
        "matrix", 12, "1-7",
        "12-item trust scale; distrust items reverse-coded, two subscales.",
        ai.trust_automation,
    ),
    "poq": MetricSpec(
        "poq", "Perceived Output Quality", MetricFamily.AI_NATIVE,
        "matrix", 3, "1-7",
        "3-item accuracy / relevance / helpfulness scale.",
        ai.perceived_output_quality,
    ),
    "appropriate_reliance": MetricSpec(
        "appropriate_reliance", "Appropriate Reliance", MetricFamily.AI_NATIVE,
        "matrix", 2, "0/1 [ai_correct, accepted]",
        "Over- / under-reliance against ground-truth AI correctness.",
        ai.appropriate_reliance,
    ),
    "relative_reliance": MetricSpec(
        "relative_reliance", "Relative Reliance (RAIR / RSR)", MetricFamily.AI_NATIVE,
        "matrix", 3, "0/1 [initial, ai_correct, followed_ai]",
        "Schemmer et al. RAIR / RSR appropriate-reliance measures.",
        ai.relative_reliance,
    ),
    "productivity_delta": MetricSpec(
        "productivity_delta", "AI Productivity Delta", MetricFamily.AI_NATIVE,
        "matrix", 2, "[with_ai, without_ai]",
        "Improvement vs a no-AI baseline (paired t by default).",
        ai.productivity_delta,
    ),
}


FAMILY_LABELS: dict[MetricFamily, str] = {
    MetricFamily.QUESTIONNAIRE: "Questionnaire",
    MetricFamily.TASK: "Task",
    MetricFamily.LOYALTY: "Loyalty",
    MetricFamily.WORKLOAD: "Workload",
    MetricFamily.AI_NATIVE: "AI-Native Flows",
    MetricFamily.COMPOSITE: "Composite",
}


def family_label(family: MetricFamily) -> str:
    return FAMILY_LABELS.get(family, family.value.replace("_", " ").title())


def get(key: str) -> MetricSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown metric '{key}'. Known: {', '.join(REGISTRY)}")
    return REGISTRY[key]


def by_family() -> dict[MetricFamily, list[MetricSpec]]:
    grouped: dict[MetricFamily, list[MetricSpec]] = {}
    for spec in REGISTRY.values():
        grouped.setdefault(spec.family, []).append(spec)
    return grouped
