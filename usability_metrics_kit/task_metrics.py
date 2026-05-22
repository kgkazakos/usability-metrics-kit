"""Behavioural and loyalty metrics plus a simplified composite.

These take raw observations (binary success flags, times in seconds, 0-10
recommend ratings, ...) rather than questionnaire matrices.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from . import benchmarks as bm
from .models import Direction, Interval, MetricFamily, MetricResult
from .stats import adjusted_wald_ci, geometric_mean_ci, mean_ci


def task_success_rate(successes: Sequence[int]) -> MetricResult:
    """Binary task completion. ``successes`` is a sequence of 1/0 flags."""
    if not successes:
        raise ValueError("task_success_rate: no observations")
    total = len(successes)
    n_success = sum(1 for s in successes if s)
    point, low, high = adjusted_wald_ci(n_success, total)
    pct = round(point * 100, 1)
    return MetricResult(
        key="task_success",
        name="Task Completion Rate",
        family=MetricFamily.TASK,
        score=pct,
        scale="0-100%",
        direction=Direction.HIGHER_BETTER,
        n=total,
        interpretation=(
            f"{n_success}/{total} completed. Small samples make the interval "
            "wide; report the range, not just the point estimate."
        ),
        benchmark={"successes": n_success, "failures": total - n_success},
        interval=Interval(round(low * 100, 1), round(high * 100, 1), method="adjusted-Wald"),
    )


def time_on_task(times: Sequence[float]) -> MetricResult:
    """Completion times in seconds. Reports the geometric mean (the right centre
    for skewed times) alongside arithmetic mean and median."""
    positive = [t for t in times if t > 0]
    if not positive:
        raise ValueError("time_on_task: no positive times")
    geo, low, high = geometric_mean_ci(positive)
    arith = statistics.mean(positive)
    median = statistics.median(positive)
    return MetricResult(
        key="time_on_task",
        name="Time on Task",
        family=MetricFamily.TASK,
        score=round(geo, 1),
        scale="seconds (lower is better)",
        direction=Direction.LOWER_BETTER,
        n=len(positive),
        interpretation=(
            f"Geometric mean {round(geo, 1)}s (median {round(median, 1)}s, "
            f"arithmetic mean {round(arith, 1)}s). The geometric mean resists "
            "the long right tail typical of task times."
        ),
        components={
            "geometric_mean": round(geo, 1),
            "arithmetic_mean": round(arith, 1),
            "median": round(median, 1),
        },
        interval=Interval(round(low, 1), round(high, 1), method="log-t"),
    )


def error_rate(errors: Sequence[float], opportunities: int | None = None) -> MetricResult:
    """Errors per user (or per opportunity when ``opportunities`` is given)."""
    if not errors:
        raise ValueError("error_rate: no observations")
    mean, low, high = mean_ci(errors)
    per_opp = None
    if opportunities and opportunities > 0:
        per_opp = round(mean / opportunities, 3)
    return MetricResult(
        key="error_rate",
        name="Error Rate",
        family=MetricFamily.TASK,
        score=round(mean, 2),
        scale="errors per user (lower is better)",
        direction=Direction.LOWER_BETTER,
        n=len(errors),
        interpretation=(
            f"Average {round(mean, 2)} errors per user"
            + (f" ({per_opp} per opportunity)." if per_opp is not None else ".")
        ),
        components={"errors_per_opportunity": per_opp},
        interval=Interval(round(max(0.0, low), 2), round(high, 2), method="t"),
    )


def net_promoter_score(ratings: Sequence[int]) -> MetricResult:
    """NPS from 0-10 likelihood-to-recommend ratings."""
    if not ratings:
        raise ValueError("net_promoter_score: no ratings")
    total = len(ratings)
    promoters = sum(1 for r in ratings if r >= 9)
    detractors = sum(1 for r in ratings if r <= 6)
    passives = total - promoters - detractors
    nps = round((promoters - detractors) / total * 100, 1)
    return MetricResult(
        key="nps",
        name="Net Promoter Score",
        family=MetricFamily.LOYALTY,
        score=nps,
        scale="-100 to +100",
        direction=Direction.HIGHER_BETTER,
        n=total,
        interpretation=(
            f"NPS {nps} ({bm.nps_band(nps)}): {promoters} promoters, "
            f"{passives} passives, {detractors} detractors."
        ),
        components={
            "promoters": promoters,
            "passives": passives,
            "detractors": detractors,
            "promoter_pct": round(promoters / total * 100, 1),
            "detractor_pct": round(detractors / total * 100, 1),
        },
        benchmark={"band": bm.nps_band(nps)},
    )


def csat(ratings: Sequence[int], scale_max: int = 5) -> MetricResult:
    """Customer Satisfaction. Top-two-box percentage on a 1..scale_max scale."""
    if not ratings:
        raise ValueError("csat: no ratings")
    total = len(ratings)
    satisfied = sum(1 for r in ratings if r >= scale_max - 1)
    mean, low, high = mean_ci(ratings)
    pct = round(satisfied / total * 100, 1)
    return MetricResult(
        key="csat",
        name="Customer Satisfaction (CSAT)",
        family=MetricFamily.LOYALTY,
        score=pct,
        scale="0-100% (top-2-box)",
        direction=Direction.HIGHER_BETTER,
        n=total,
        interpretation=(
            f"{pct}% rated in the top two boxes (mean {round(mean, 2)} on a "
            f"1-{scale_max} scale)."
        ),
        components={"mean": round(mean, 2)},
        interval=Interval(round(max(0.0, low), 2), round(high, 2), method="t"),
    )


def ces(ratings: Sequence[int], scale_max: int = 7) -> MetricResult:
    """Customer Effort Score (CES 2.0). 1..scale_max, higher = easier."""
    if not ratings:
        raise ValueError("ces: no ratings")
    total = len(ratings)
    mean, low, high = mean_ci(ratings)
    easy = sum(1 for r in ratings if r >= scale_max - 1)
    return MetricResult(
        key="ces",
        name="Customer Effort Score",
        family=MetricFamily.LOYALTY,
        score=round(mean, 2),
        scale=f"1-{scale_max} (higher = lower effort)",
        direction=Direction.HIGHER_BETTER,
        n=total,
        interpretation=(
            f"Mean effort agreement {round(mean, 2)}; "
            f"{round(easy / total * 100, 1)}% found it easy (top-2-box)."
        ),
        components={"easy_pct": round(easy / total * 100, 1)},
        interval=Interval(round(max(0.0, low), 2), round(high, 2), method="t"),
    )


def composite_score(components: dict[str, float], study: str = "study") -> MetricResult:
    """A simplified Single Usability Metric-style composite.

    ``components`` are sub-scores already normalised to 0-100 where higher is
    better (e.g. completion %, normalised satisfaction, error-free %, time
    efficiency %). The composite is their unweighted mean. This is a directional
    roll-up, not the fully standardised Sauro-Kindlund SUM.
    """
    if not components:
        raise ValueError("composite_score: no components")
    values = list(components.values())
    score = round(sum(values) / len(values), 1)
    return MetricResult(
        key="sum_composite",
        name="Usability Composite (SUM-style)",
        family=MetricFamily.COMPOSITE,
        score=score,
        scale="0-100",
        direction=Direction.HIGHER_BETTER,
        n=len(components),
        interpretation=(
            f"Unweighted mean of {len(components)} normalised sub-scores: {score}/100."
        ),
        components={"inputs": {k: round(v, 1) for k, v in components.items()}},
        warnings=[
            "Simplified composite — an equal-weight roll-up, not the fully "
            "standardised Single Usability Metric (SUM)."
        ],
    )
