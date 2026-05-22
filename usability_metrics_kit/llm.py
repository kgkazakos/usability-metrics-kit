"""Optional LLM narrative layer.

This kit ships with LLM enrichment *on by default* across three backends:

    * Anthropic  -> claude opus
    * Google     -> gemini 3.1 pro
    * OpenAI     -> gpt 5.5

The model strings below are configurable — swap them for whatever your
account exposes. Enrichment is strictly additive: if no API key is present the
kit detects that, skips the call and returns the rule-based result untouched,
so every metric still works fully offline with zero API dependencies.
"""

from __future__ import annotations

import os
from typing import Any

# Configurable model strings (LiteLLM-style provider/model identifiers).
BACKENDS: dict[str, str] = {
    "anthropic": "anthropic/claude-opus-4-7",
    "gemini": "gemini/gemini-3.1-pro-preview",
    "openai": "openai/gpt-5.5",
}

# Which environment variable(s) hold each backend's key.
_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
}

# Auto-selection order when the caller does not name a backend.
_PRIORITY = ("anthropic", "gemini", "openai")


def available_backends() -> list[str]:
    """Backends whose API key is present in the environment."""
    found = []
    for name, keys in _ENV_KEYS.items():
        if any(os.environ.get(k) for k in keys):
            found.append(name)
    return found


def default_backend() -> str | None:
    available = available_backends()
    for name in _PRIORITY:
        if name in available:
            return name
    return None


def is_available(backend: str | None = None) -> bool:
    if backend is None:
        return default_backend() is not None
    return backend in available_backends()


def _generate(payload: dict[str, Any], model: str) -> str:
    """Run a single DSPy ChainOfThought call. Isolated so tests can monkeypatch."""
    import dspy  # imported lazily so the package works without dspy installed

    class MetricNarrative(dspy.Signature):
        """Write a 2-3 sentence analyst readout of a usability metric result.
        Be concrete, name the score and what it means for the product team, and
        respect any caveats. Do not invent precision the data does not support."""

        metric: str = dspy.InputField(desc="metric name and scale")
        score: str = dspy.InputField(desc="headline score and interval")
        detail: str = dspy.InputField(desc="components, benchmark and warnings")
        summary: str = dspy.OutputField(desc="2-3 sentence narrative")

    lm = dspy.LM(model)
    with dspy.context(lm=lm):
        predictor = dspy.ChainOfThought(MetricNarrative)
        out = predictor(
            metric=payload["metric"],
            score=payload["score"],
            detail=payload["detail"],
        )
    return out.summary.strip()


def narrate(result: dict[str, Any], backend: str | None = None) -> str | None:
    """Produce a narrative for one metric result dict. ``None`` if unavailable."""
    backend = backend or default_backend()
    if backend is None:
        return None
    model = BACKENDS[backend]
    payload = {
        "metric": f"{result['name']} ({result['scale']})",
        "score": f"{result['score']} (CI {result.get('interval')})",
        "detail": (
            f"interpretation: {result.get('interpretation', '')}; "
            f"components: {result.get('components', {})}; "
            f"benchmark: {result.get('benchmark', {})}; "
            f"warnings: {result.get('warnings', [])}"
        ),
    }
    try:
        return _generate(payload, model)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on any LLM error
        return f"[LLM enrichment unavailable: {exc}]"


def enrich(result: dict[str, Any], backend: str | None = None) -> dict[str, Any]:
    """Attach an ``llm_summary`` to a result dict in place and return it."""
    result["llm_summary"] = narrate(result, backend)
    return result
