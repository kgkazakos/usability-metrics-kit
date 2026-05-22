"""LLM layer tests — mock-based, no network or API keys required.

Covers backend auto-selection across all three providers, graceful offline
behaviour, and error degradation.
"""

from __future__ import annotations

import pytest

from usability_metrics_kit import llm
from usability_metrics_kit.questionnaires import score_sus

_ALL_KEYS = ["ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"]


@pytest.fixture(autouse=True)
def clear_keys(monkeypatch):
    for k in _ALL_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_offline_returns_none():
    assert llm.default_backend() is None
    assert llm.is_available() is False
    result = score_sus([[4, 2, 4, 2, 4, 2, 4, 2, 4, 2]] * 6).to_dict()
    assert llm.narrate(result) is None


@pytest.mark.parametrize(
    "env,expected_backend,expected_model",
    [
        ("ANTHROPIC_API_KEY", "anthropic", "anthropic/claude-opus-4-7"),
        ("GEMINI_API_KEY", "gemini", "gemini/gemini-3.1-pro-preview"),
        ("OPENAI_API_KEY", "openai", "openai/gpt-5.5"),
    ],
)
def test_each_backend(monkeypatch, env, expected_backend, expected_model):
    monkeypatch.setenv(env, "test-key")
    captured = {}

    def fake_generate(payload, model):
        captured["model"] = model
        return "Solid result; ship it."

    monkeypatch.setattr(llm, "_generate", fake_generate)
    assert llm.default_backend() == expected_backend
    result = score_sus([[5, 1, 5, 1, 5, 1, 5, 1, 5, 1]] * 6).to_dict()
    summary = llm.narrate(result)
    assert summary == "Solid result; ship it."
    assert captured["model"] == expected_model


def test_priority_prefers_anthropic(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert llm.default_backend() == "anthropic"
    assert set(llm.available_backends()) == {"anthropic", "openai"}


def test_enrich_attaches_summary(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(llm, "_generate", lambda payload, model: "narrative here")
    result = score_sus([[5, 1, 5, 1, 5, 1, 5, 1, 5, 1]] * 6).to_dict()
    enriched = llm.enrich(result)
    assert enriched["llm_summary"] == "narrative here"


def test_llm_error_degrades(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    def boom(payload, model):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(llm, "_generate", boom)
    result = score_sus([[5, 1, 5, 1, 5, 1, 5, 1, 5, 1]] * 6).to_dict()
    summary = llm.narrate(result)
    assert "unavailable" in summary
