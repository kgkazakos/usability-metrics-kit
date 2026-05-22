"""Core data models for usability-metrics-kit.

Everything a metric produces is funnelled through :class:`MetricResult` so the
CLI, the Streamlit app and the JSON serialiser can treat every metric the same
way regardless of how it was scored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Direction(str, Enum):
    """Whether a higher raw score is better or worse for the user."""

    HIGHER_BETTER = "higher_is_better"
    LOWER_BETTER = "lower_is_better"


class MetricFamily(str, Enum):
    """Coarse grouping used for filtering and display."""

    QUESTIONNAIRE = "questionnaire"
    TASK = "task"
    LOYALTY = "loyalty"
    WORKLOAD = "workload"
    AI_NATIVE = "ai_native"
    COMPOSITE = "composite"


@dataclass
class Interval:
    """A two-sided confidence interval."""

    low: float
    high: float
    confidence: float = 0.95
    method: str = ""

    def as_tuple(self) -> tuple[float, float]:
        return (self.low, self.high)


@dataclass
class MetricResult:
    """The single shape every metric returns.

    Attributes
    ----------
    key:
        Stable machine identifier (e.g. ``"sus"``).
    name:
        Human readable name (e.g. ``"System Usability Scale"``).
    family:
        Which :class:`MetricFamily` the metric belongs to.
    score:
        The headline number for the metric, on ``scale``.
    scale:
        Short description of the score's range (e.g. ``"0-100"``).
    direction:
        Whether higher or lower is better.
    n:
        Number of respondents / observations the score was computed from.
    interpretation:
        Plain-language reading of the score (rule-based, always present).
    components:
        Sub-scores, item means, or other structured detail.
    benchmark:
        External comparison points (percentile, grade, adjective, norms).
    interval:
        Optional confidence interval on the headline score.
    warnings:
        Data-quality or interpretation caveats surfaced for the analyst.
    llm_summary:
        Optional narrative produced by the LLM layer (``None`` when offline).
    """

    key: str
    name: str
    family: MetricFamily
    score: float
    scale: str
    direction: Direction
    n: int
    interpretation: str = ""
    components: dict[str, Any] = field(default_factory=dict)
    benchmark: dict[str, Any] = field(default_factory=dict)
    interval: Interval | None = None
    warnings: list[str] = field(default_factory=list)
    llm_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["family"] = self.family.value
        data["direction"] = self.direction.value
        if self.interval is not None:
            data["interval"] = asdict(self.interval)
        return data


@dataclass
class Scorecard:
    """A bundle of metric results scored together for one study."""

    study: str
    results: list[MetricResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "study": self.study,
            "n_metrics": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }
