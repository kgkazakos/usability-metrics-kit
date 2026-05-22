"""usability-metrics-kit — calculate, benchmark and narrate usability metrics.

Part of the CPR Orbital Series of open-source product-research tools.
"""

from __future__ import annotations

from .ai_native import (
    ai_acceptance_rate,
    ai_collaboration_score,
    appropriate_reliance,
    human_fallback_rate,
    perceived_output_quality,
    productivity_delta,
    prompt_iterations,
    relative_reliance,
    retention_from_edits,
    retention_rate,
    trust_automation,
)
from .models import Direction, Interval, MetricFamily, MetricResult, Scorecard
from .questionnaires import (
    score_csuq,
    score_nasa_tlx,
    score_pssuq,
    score_seq,
    score_suprq,
    score_sus,
    score_ueq_s,
    score_umux,
    score_umux_lite,
)
from .registry import REGISTRY, MetricSpec, by_family, get
from .task_metrics import (
    ces,
    composite_score,
    csat,
    error_rate,
    net_promoter_score,
    task_success_rate,
    time_on_task,
)

__version__ = "0.1.0"

__all__ = [
    "Direction",
    "Interval",
    "MetricFamily",
    "MetricResult",
    "Scorecard",
    "MetricSpec",
    "REGISTRY",
    "by_family",
    "get",
    "score_sus",
    "score_umux",
    "score_umux_lite",
    "score_suprq",
    "score_pssuq",
    "score_csuq",
    "score_seq",
    "score_ueq_s",
    "score_nasa_tlx",
    "task_success_rate",
    "time_on_task",
    "error_rate",
    "net_promoter_score",
    "csat",
    "ces",
    "composite_score",
    "retention_rate",
    "retention_from_edits",
    "prompt_iterations",
    "human_fallback_rate",
    "ai_acceptance_rate",
    "trust_automation",
    "perceived_output_quality",
    "ai_collaboration_score",
    "appropriate_reliance",
    "relative_reliance",
    "productivity_delta",
    "__version__",
]
