"""Tests for stats helpers, registry and loaders."""

from __future__ import annotations

import pytest

from usability_metrics_kit import loaders, registry
from usability_metrics_kit import stats as st


def test_percentile_from_normal_center():
    assert st.percentile_from_normal(68, 68, 12.5) == 50.0


def test_adjusted_wald_bounds_within_unit():
    point, low, high = st.adjusted_wald_ci(4, 4)
    assert point == 1.0
    assert 0.0 <= low <= high <= 1.0


def test_geometric_mean_ci():
    geo, low, high = st.geometric_mean_ci([10, 20, 40])
    assert geo == pytest.approx(20.0, abs=0.1)
    assert low <= geo <= high


def test_mean_ci_single_value():
    mean, low, high = st.mean_ci([5.0])
    assert mean == low == high == 5.0


def test_registry_every_spec_resolves():
    for key, spec in registry.REGISTRY.items():
        assert spec.key == key
        assert callable(spec.scorer)
    assert registry.get("sus").n_items == 10
    with pytest.raises(KeyError):
        registry.get("nope")


def test_by_family_groups():
    grouped = registry.by_family()
    keys = {s.key for specs in grouped.values() for s in specs}
    assert keys == set(registry.REGISTRY)


def test_loader_detects_header(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a,b,c\n1,2,3\n4,5,6\n")
    matrix = loaders.load_matrix(p)
    assert matrix == [[1, 2, 3], [4, 5, 6]]


def test_loader_values(tmp_path):
    p = tmp_path / "v.csv"
    p.write_text("score\n10\n9\n8\n")
    assert loaders.load_values(p) == [10, 9, 8]


def test_loader_empty_raises(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("header\n")
    with pytest.raises(ValueError):
        loaders.load_matrix(p)
