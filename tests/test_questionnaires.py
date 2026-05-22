"""Tests for questionnaire scorers."""

from __future__ import annotations

import pytest

from usability_metrics_kit import questionnaires as q


def test_sus_perfect_and_floor():
    perfect = [[5, 1, 5, 1, 5, 1, 5, 1, 5, 1]] * 5
    floor = [[1, 5, 1, 5, 1, 5, 1, 5, 1, 5]] * 5
    assert q.score_sus(perfect).score == 100.0
    assert q.score_sus(floor).score == 0.0


def test_sus_all_neutral_is_50():
    neutral = [[5] * 10] * 6
    assert q.score_sus(neutral).score == 50.0


def test_sus_benchmark_fields():
    res = q.score_sus([[5, 1, 5, 1, 5, 1, 5, 1, 5, 1]] * 6)
    assert res.benchmark["grade"] == "A+"
    assert res.benchmark["adjective"] == "Best Imaginable"
    assert res.benchmark["acceptability"] == "Acceptable"


def test_sus_small_sample_warns():
    res = q.score_sus([[3, 3, 3, 3, 3, 3, 3, 3, 3, 3]] * 3)
    assert any("respondents" in w for w in res.warnings)


def test_umux_perfect():
    res = q.score_umux([[7, 1, 7, 1]] * 5)
    assert res.score == 100.0


def test_umux_lite_corrected():
    res = q.score_umux_lite([[7, 7]] * 5)
    assert res.score == 100.0
    assert res.components["sus_corrected"] == pytest.approx(87.9, abs=0.1)


def test_suprq_subscales_and_overall():
    res = q.score_suprq([[5, 5, 5, 5, 5, 5, 5, 10]] * 5)
    assert res.score == 100.0
    assert res.components["subscales"]["usability"] == 5.0
    assert set(res.components["subscales"]) == {
        "usability", "trust", "appearance", "loyalty"
    }


def test_pssuq_lower_is_better():
    best = q.score_pssuq([[1] * 16] * 5)
    worst = q.score_pssuq([[7] * 16] * 5)
    assert best.score == 1.0
    assert worst.score == 7.0
    assert best.direction.value == "lower_is_better"
    assert "system_usefulness" in best.components["subscales"]


def test_csuq_matches_ibm_scoring():
    res = q.score_csuq([[3] * 16] * 5)
    assert res.score == 3.0
    assert res.key == "csuq"


def test_seq_flags_difficult_task():
    hard = q.score_seq([[3], [4], [3], [4], [2]])
    easy = q.score_seq([[6], [7], [6], [7], [6]])
    assert any("difficult" in res for res in [hard.interpretation])
    assert not any("flags" in w for w in easy.warnings)


def test_ueq_s_pragmatic_hedonic():
    res = q.score_ueq_s([[7, 7, 7, 7, 7, 7, 7, 7]] * 5)
    assert res.score == 3.0
    assert res.components["pragmatic"] == 3.0
    assert res.components["hedonic"] == 3.0


def test_nasa_tlx_raw_and_weighted():
    rows = [[60, 20, 50, 30, 55, 40]] * 4
    raw = q.score_nasa_tlx(rows)
    assert raw.score == pytest.approx(42.5, abs=0.1)
    weighted = q.score_nasa_tlx(rows, weights=[5, 0, 4, 1, 3, 2])
    assert weighted.score != raw.score
    assert "weighted" in weighted.name


def test_wrong_item_count_raises():
    with pytest.raises(ValueError):
        q.score_sus([[5, 1, 5]])


def test_empty_raises():
    with pytest.raises(ValueError):
        q.score_sus([])
