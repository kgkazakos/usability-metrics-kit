"""Tests for task, loyalty and composite metrics."""

from __future__ import annotations

import pytest

from usability_metrics_kit import task_metrics as t


def test_task_success_rate_and_ci():
    res = t.task_success_rate([1, 1, 1, 0])
    assert res.score == 75.0
    assert res.interval is not None
    assert res.interval.low <= 75.0 <= res.interval.high
    assert res.interval.method == "adjusted-Wald"


def test_time_on_task_geometric_mean():
    res = t.time_on_task([10, 20, 40])
    assert res.score == pytest.approx(20.0, abs=0.1)
    assert res.components["median"] == 20.0
    assert res.components["arithmetic_mean"] == pytest.approx(23.3, abs=0.1)


def test_error_rate_per_opportunity():
    res = t.error_rate([0, 1, 2, 1], opportunities=4)
    assert res.score == 1.0
    assert res.components["errors_per_opportunity"] == 0.25


def test_nps_segments():
    res = t.net_promoter_score([10, 10, 9, 8, 7, 6, 0])
    assert res.components["promoters"] == 3
    assert res.components["detractors"] == 2
    assert res.score == pytest.approx(14.3, abs=0.1)


def test_csat_top_two_box():
    res = t.csat([5, 5, 4, 3, 1])
    assert res.score == 60.0
    assert res.components["mean"] == pytest.approx(3.6, abs=0.01)


def test_ces_easy_pct():
    res = t.ces([7, 6, 5])
    assert res.score == pytest.approx(6.0, abs=0.01)
    assert res.components["easy_pct"] == pytest.approx(66.7, abs=0.1)


def test_composite_mean():
    res = t.composite_score({"completion": 80.0, "satisfaction": 60.0})
    assert res.score == 70.0
    assert "Simplified" in res.warnings[0]


def test_empty_inputs_raise():
    for fn in (t.task_success_rate, t.time_on_task, t.net_promoter_score):
        with pytest.raises(ValueError):
            fn([])
