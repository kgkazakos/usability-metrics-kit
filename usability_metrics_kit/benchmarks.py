"""Published interpretation aids: grades, adjectives, acceptability and norms.

These tables let a raw score become a sentence an analyst can put in a readout.
Where a band depends on a proprietary normed database (SUS percentiles,
SUPR-Q percentiles) we use a transparent Normal approximation and say so in the
metric's ``warnings`` rather than implying access to the original norms.
"""

from __future__ import annotations

# --- SUS -------------------------------------------------------------------

# Curved Grading Scale, Sauro & Lewis (2016). (lower_bound, grade)
SUS_CGS: list[tuple[float, str]] = [
    (84.1, "A+"),
    (80.8, "A"),
    (78.9, "A-"),
    (77.2, "B+"),
    (74.1, "B"),
    (72.6, "B-"),
    (71.1, "C+"),
    (65.0, "C"),
    (62.7, "C-"),
    (51.7, "D"),
    (0.0, "F"),
]

# Bangor et al. (2009) adjective anchors (midpoint score, label).
SUS_ADJECTIVES: list[tuple[float, str]] = [
    (12.5, "Worst Imaginable"),
    (20.3, "Awful"),
    (35.7, "Poor"),
    (50.9, "OK"),
    (71.4, "Good"),
    (85.5, "Excellent"),
    (90.9, "Best Imaginable"),
]

# Reference distribution for the SUS percentile approximation.
SUS_REF_MEAN = 68.0
SUS_REF_SD = 12.5


def sus_grade(score: float) -> str:
    for lower, grade in SUS_CGS:
        if score >= lower:
            return grade
    return "F"


def sus_adjective(score: float) -> str:
    """Nearest-anchor adjective for a SUS score."""
    return min(SUS_ADJECTIVES, key=lambda pair: abs(pair[0] - score))[1]


def sus_acceptability(score: float) -> str:
    if score > 70.0:
        return "Acceptable"
    if score >= 50.0:
        return "Marginal"
    return "Not acceptable"


# --- PSSUQ / CSUQ (7-point, lower is better) -------------------------------

# Sauro & Lewis (2016) norms: (overall, system_usefulness, info_quality,
# interface_quality).
PSSUQ_NORMS = {
    "overall": 2.82,
    "system_usefulness": 2.80,
    "information_quality": 3.02,
    "interface_quality": 2.49,
}


# --- SEQ (7-point, higher is better) ---------------------------------------

SEQ_BENCHMARK = 5.5  # commonly cited average single-task ease.


# --- UEQ-S benchmark interpretation (mean per scale, -3..+3) ----------------


def ueq_band(mean: float) -> str:
    """Crude UEQ benchmark band for a scale mean."""
    if mean >= 1.5:
        return "Excellent"
    if mean >= 0.8:
        return "Good"
    if mean >= 0.0:
        return "Above average / neutral"
    return "Below average"


# --- NPS interpretation -----------------------------------------------------


def nps_band(nps: float) -> str:
    if nps >= 50:
        return "Excellent"
    if nps >= 0:
        return "Good"
    if nps >= -50:
        return "Needs improvement"
    return "Critical"


# --- AI-native flow bands ---------------------------------------------------
#
# These are pragmatic orientation bands, NOT validated industry norms. Unlike
# SUS or PSSUQ there is no established normed database for generative-AI
# collaboration metrics yet, so every band below is a starting heuristic for
# reading a result, surfaced as such in the metric's warnings.


def acceptance_band(rate: float) -> str:
    """Orientation band for AI suggestion acceptance (0-1)."""
    if rate >= 0.9:
        return "Very high — check for automation bias"
    if rate >= 0.6:
        return "High"
    if rate >= 0.35:
        return "Moderate"
    return "Low — suggestions largely dismissed"


def trust_band(mean: float) -> str:
    """Orientation band for the Jian trust overall mean (1-7)."""
    if mean >= 5.5:
        return "High trust"
    if mean >= 4.0:
        return "Moderate trust"
    if mean >= 2.5:
        return "Low trust"
    return "Distrust"


def collaboration_band(score: float) -> str:
    """Orientation band for the composite AI Collaboration Score (0-100)."""
    if score >= 75:
        return "Strong collaboration"
    if score >= 55:
        return "Workable collaboration"
    if score >= 40:
        return "Fragile collaboration"
    return "Poor collaboration"


def reliance_band(rate: float) -> str:
    """Orientation band for the appropriate-reliance rate (0-1)."""
    if rate >= 0.85:
        return "Well-calibrated reliance"
    if rate >= 0.65:
        return "Reasonable reliance"
    if rate >= 0.5:
        return "Mixed reliance"
    return "Poorly calibrated reliance"
