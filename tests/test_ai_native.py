"""Tests for the AI-native flow metrics, with known-value and small-n checks."""

from __future__ import annotations

import pytest

from usability_metrics_kit import ai_native as ai
from usability_metrics_kit.models import Direction, MetricFamily

# --- retention --------------------------------------------------------------


def test_retention_perfect_and_half():
    assert ai.retention_rate([1.0, 1.0, 1.0]).score == 100.0
    assert ai.retention_rate([0.5, 0.5]).score == 50.0


def test_retention_accepts_percentages():
    # values > 1 are treated as percentages
    assert ai.retention_rate([80, 80, 80]).score == 80.0


def test_retention_is_continuous_not_binomial():
    r = ai.retention_rate([0.7, 0.6, 0.8])
    assert r.family == MetricFamily.AI_NATIVE
    assert r.direction == Direction.HIGHER_BETTER
    assert "t" in r.interval.method  # t-interval, not adjusted-Wald
    assert r.interval.low >= 0.0 and r.interval.high <= 100.0


def test_retention_from_edits():
    out = ai.retention_from_edits([100, 100, 50], [20, 0, 50])
    assert out == [0.8, 1.0, 0.0]


def test_retention_small_n_warns():
    r = ai.retention_rate([0.5, 0.6])
    assert any("n <" in w for w in r.warnings)


# --- prompt iterations ------------------------------------------------------


def test_prompt_iterations_mean_and_method():
    r = ai.prompt_iterations([2, 2, 2])
    assert r.score == 2.0
    assert "Poisson" in r.interval.method
    assert r.direction == Direction.LOWER_BETTER


def test_prompt_iterations_interval_positive_when_events():
    r = ai.prompt_iterations([3, 4, 5, 2, 3])
    assert r.interval.low > 0.0
    assert r.interval.high > r.score


def test_prompt_iterations_overdispersion_warning():
    r = ai.prompt_iterations([1, 1, 1, 1, 20])
    assert any("overdispers" in w.lower() for w in r.warnings)


# --- human fallback ---------------------------------------------------------


def test_fallback_none_and_all():
    assert ai.human_fallback_rate([0, 0, 0, 0]).score == 0.0
    assert ai.human_fallback_rate([1, 1]).score == 100.0


def test_fallback_lower_is_better_adjusted_wald():
    r = ai.human_fallback_rate([1, 0, 0, 0, 0])
    assert r.direction == Direction.LOWER_BETTER
    assert r.interval.method == "adjusted-Wald"


# --- acceptance -------------------------------------------------------------


def test_acceptance_from_decisions():
    r = ai.ai_acceptance_rate([1, 1, 1, 1, 0, 0])
    assert r.score == pytest.approx(66.7, abs=0.1)


def test_acceptance_from_aggregate():
    r = ai.ai_acceptance_rate(accepted=8, offered=10)
    assert r.score == 80.0


def test_acceptance_clustering_warning_present():
    r = ai.ai_acceptance_rate([1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1])
    assert any("cluster" in w.lower() for w in r.warnings)


def test_acceptance_requires_input():
    with pytest.raises(ValueError):
        ai.ai_acceptance_rate(offered=0, accepted=0)


# --- trust (Jian et al.) ----------------------------------------------------


def test_trust_reverse_coding_extremes():
    # distrust items (1-5) all 1, trust items (6-12) all 7 -> max trust = 7.0
    high = [[1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7]]
    assert ai.trust_automation(high).score == 7.0
    # distrust items all 7, trust items all 1 -> min trust = 1.0
    low = [[7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 1, 1]]
    assert ai.trust_automation(low).score == 1.0


def test_trust_exposes_two_subscales():
    r = ai.trust_automation([[1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7]])
    subs = r.components["subscales"]
    assert subs["trust"] == 7.0
    assert subs["distrust_raw"] == 1.0


def test_trust_rejects_wrong_item_count():
    with pytest.raises(ValueError):
        ai.trust_automation([[1, 2, 3]])


# --- perceived output quality ----------------------------------------------


def test_poq_known_value_and_items():
    r = ai.perceived_output_quality([[7, 7, 7], [5, 5, 5]])
    assert r.score == 6.0
    assert r.components["items"]["accuracy"] == 6.0


def test_poq_flags_unvalidated():
    r = ai.perceived_output_quality([[6, 6, 6]])
    assert any("validated" in w.lower() for w in r.warnings)


# --- composite --------------------------------------------------------------


def test_collaboration_aligned_perfect():
    r = ai.ai_collaboration_score([1, 1], [1, 1], [7, 7])
    assert r.score == 100.0
    assert r.components["std_dev"] == 0.0
    assert r.family == MetricFamily.COMPOSITE


def test_collaboration_aligned_midpoint():
    r = ai.ai_collaboration_score([0.5, 0.5], [0.5, 0.5], [4, 4])
    assert r.score == 50.0


def test_collaboration_reports_sd_with_spread():
    r = ai.ai_collaboration_score([1.0, 0.0], [1.0, 0.0], [7, 1])
    assert r.components["std_dev"] > 0.0


def test_collaboration_unaligned_warns():
    r = ai.ai_collaboration_score([1, 1], [1, 1], [7, 7], aligned=False)
    assert any("unaligned" in w.lower() for w in r.warnings)


def test_collaboration_truncates_unequal_aligned():
    r = ai.ai_collaboration_score([1, 1, 1], [1, 1], [7, 7], aligned=True)
    assert r.n == 2
    assert any("truncated" in w.lower() for w in r.warnings)


# --- registry integration ---------------------------------------------------


def test_ai_native_metrics_registered():
    from usability_metrics_kit.registry import REGISTRY

    for key in (
        "retention_rate",
        "prompt_iterations",
        "human_fallback",
        "ai_acceptance",
        "trust_automation",
        "poq",
        "appropriate_reliance",
        "relative_reliance",
        "productivity_delta",
    ):
        assert key in REGISTRY
        assert REGISTRY[key].family == MetricFamily.AI_NATIVE


def test_ai_native_scorers_dispatch_through_registry():
    from usability_metrics_kit.registry import get

    assert get("human_fallback").scorer([0, 1, 0]).key == "human_fallback"
    assert get("retention_rate").scorer([0.8, 0.9]).key == "retention_rate"


# --- appropriate reliance ---------------------------------------------------


def test_appropriate_reliance_all_correct():
    # accept when right, reject when wrong = perfectly appropriate
    r = ai.appropriate_reliance([[1, 1], [1, 1], [0, 0], [0, 0]])
    assert r.score == 100.0
    assert r.components["over_reliance"]["rate"] == 0.0
    assert r.components["under_reliance"]["rate"] == 0.0


def test_appropriate_reliance_over_reliance():
    # AI wrong but always accepted -> over-reliance 100%, appropriate 0%
    r = ai.appropriate_reliance([[0, 1], [0, 1], [0, 1]])
    assert r.score == 0.0
    assert r.components["over_reliance"]["rate"] == 100.0
    assert r.components["under_reliance"] is None  # no AI-correct cases


def test_appropriate_reliance_under_reliance():
    # AI right but always rejected -> under-reliance 100%
    r = ai.appropriate_reliance([[1, 0], [1, 0]])
    assert r.components["under_reliance"]["rate"] == 100.0
    assert r.components["over_reliance"] is None


def test_appropriate_reliance_flags_ground_truth():
    r = ai.appropriate_reliance([[1, 1], [0, 0]])
    assert any("ground-truth" in w.lower() for w in r.warnings)


def test_appropriate_reliance_rejects_one_column():
    with pytest.raises(ValueError):
        ai.appropriate_reliance([[1]])


# --- relative reliance (RAIR / RSR) -----------------------------------------


def test_relative_reliance_perfect():
    # initially wrong + AI right + switched (RAIR=1); initially right + AI wrong + kept (RSR=1)
    rows = [[0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]
    r = ai.relative_reliance(rows)
    assert r.components["rair"]["rate"] == 100.0
    assert r.components["rsr"]["rate"] == 100.0
    assert r.score == 100.0


def test_relative_reliance_worst():
    rows = [[0, 1, 0], [1, 0, 1]]  # missed good advice; followed bad advice
    r = ai.relative_reliance(rows)
    assert r.components["rair"]["rate"] == 0.0
    assert r.components["rsr"]["rate"] == 0.0
    assert r.score == 0.0


def test_relative_reliance_rejects_two_columns():
    with pytest.raises(ValueError):
        ai.relative_reliance([[1, 1]])


# --- productivity delta -----------------------------------------------------


def test_productivity_delta_paired_time_saved():
    # with-AI faster than baseline -> positive improvement
    r = ai.productivity_delta([[80, 100], [90, 110], [70, 95]])
    assert r.score > 0
    assert r.direction == Direction.HIGHER_BETTER
    assert r.interval.method == "paired t"


def test_productivity_delta_no_effect_warns():
    r = ai.productivity_delta([[100, 100], [100, 100], [100, 100]])
    assert r.score == 0.0
    assert any("zero" in w.lower() for w in r.warnings)


def test_productivity_delta_two_sample_welch():
    r = ai.productivity_delta(
        [80, 90, 70, 85], [100, 110, 95, 108], paired=False, lower_is_better=True
    )
    assert r.score > 0
    assert "Welch" in r.interval.method


def test_productivity_delta_quality_higher_better():
    # quality scores: with-AI higher than baseline
    r = ai.productivity_delta([[8, 6], [9, 7], [7, 5]], lower_is_better=False)
    assert r.score > 0
