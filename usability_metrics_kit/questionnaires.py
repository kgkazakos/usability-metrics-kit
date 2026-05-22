"""Scoring for standardised usability questionnaires.

Every scorer takes a ``responses`` matrix (rows = respondents, columns = items
in canonical order) and returns a :class:`MetricResult`. Scores are computed
per respondent first and then summarised, so each questionnaire carries a
confidence interval on its headline number rather than a single point estimate.
"""

from __future__ import annotations

from collections.abc import Sequence

from . import benchmarks as bm
from .models import Direction, Interval, MetricFamily, MetricResult
from .stats import mean_ci, percentile_from_normal

Matrix = Sequence[Sequence[float]]


def _check(responses: Matrix, n_items: int, name: str) -> list[str]:
    """Validate shape and return any data-quality warnings."""
    if not responses:
        raise ValueError(f"{name}: no responses provided")
    warnings: list[str] = []
    for i, row in enumerate(responses):
        if len(row) != n_items:
            raise ValueError(
                f"{name}: respondent {i} has {len(row)} items, expected {n_items}"
            )
    if len(responses) < 5:
        warnings.append(
            f"Only {len(responses)} respondents; the interval is wide and the "
            "score should be read as directional, not definitive."
        )
    return warnings


def _interval(values: Sequence[float]) -> tuple[float, Interval]:
    mean, low, high = mean_ci(values)
    return round(mean, 2), Interval(round(low, 2), round(high, 2), method="t")


# --- SUS -------------------------------------------------------------------


def score_sus(responses: Matrix) -> MetricResult:
    """System Usability Scale (Brooke, 1996). 10 items, 1-5, 0-100 output."""
    warnings = _check(responses, 10, "SUS")
    per: list[float] = []
    for row in responses:
        total = 0.0
        for idx, val in enumerate(row):
            item = idx + 1
            total += (val - 1) if item % 2 == 1 else (5 - val)
        per.append(total * 2.5)
    score, interval = _interval(per)
    return MetricResult(
        key="sus",
        name="System Usability Scale",
        family=MetricFamily.QUESTIONNAIRE,
        score=score,
        scale="0-100",
        direction=Direction.HIGHER_BETTER,
        n=len(responses),
        interpretation=(
            f"Grade {bm.sus_grade(score)} — '{bm.sus_adjective(score)}', "
            f"{bm.sus_acceptability(score).lower()}. The benchmark average is 68."
        ),
        components={"per_respondent": [round(p, 1) for p in per]},
        benchmark={
            "grade": bm.sus_grade(score),
            "adjective": bm.sus_adjective(score),
            "acceptability": bm.sus_acceptability(score),
            "percentile_approx": percentile_from_normal(
                score, bm.SUS_REF_MEAN, bm.SUS_REF_SD
            ),
            "reference_mean": bm.SUS_REF_MEAN,
        },
        interval=interval,
        warnings=warnings
        + [
            "Percentile is a Normal approximation (mean 68, SD 12.5), not the "
            "proprietary Sauro-Lewis norms."
        ],
    )


# --- UMUX / UMUX-LITE -------------------------------------------------------


def score_umux(responses: Matrix) -> MetricResult:
    """UMUX (Finstad, 2010). 4 items, 1-7. Items 1,3 positive; 2,4 negative."""
    warnings = _check(responses, 4, "UMUX")
    per: list[float] = []
    for row in responses:
        converted = [
            row[0] - 1,
            7 - row[1],
            row[2] - 1,
            7 - row[3],
        ]
        per.append(sum(converted) / 24.0 * 100.0)
    score, interval = _interval(per)
    return MetricResult(
        key="umux",
        name="Usability Metric for User Experience",
        family=MetricFamily.QUESTIONNAIRE,
        score=score,
        scale="0-100",
        direction=Direction.HIGHER_BETTER,
        n=len(responses),
        interpretation=(
            "Designed to track close to SUS from only four items; read on the "
            "same 0-100 mental model where ~68 is average."
        ),
        components={"per_respondent": [round(p, 1) for p in per]},
        interval=interval,
        warnings=warnings,
    )


def score_umux_lite(responses: Matrix) -> MetricResult:
    """UMUX-LITE (Lewis et al., 2013). 2 positive items, 1-7.

    Reports the raw score and the regression-corrected SUS estimate
    (0.65 * raw + 22.9).
    """
    warnings = _check(responses, 2, "UMUX-LITE")
    raw_per: list[float] = []
    for row in responses:
        raw_per.append((row[0] + row[1] - 2) / 12.0 * 100.0)
    raw, raw_interval = _interval(raw_per)
    corrected = round(0.65 * raw + 22.9, 2)
    return MetricResult(
        key="umux_lite",
        name="UMUX-LITE",
        family=MetricFamily.QUESTIONNAIRE,
        score=raw,
        scale="0-100",
        direction=Direction.HIGHER_BETTER,
        n=len(responses),
        interpretation=(
            f"Raw {raw}; SUS-corrected estimate ~{corrected}. Two items make it "
            "ideal for in-product micro-surveys."
        ),
        components={
            "raw": raw,
            "sus_corrected": corrected,
            "per_respondent": [round(p, 1) for p in raw_per],
        },
        interval=raw_interval,
        warnings=warnings,
    )


# --- SUPR-Q ----------------------------------------------------------------

_SUPRQ_SUBSCALES = {
    "usability": (0, 1),
    "trust": (2, 3),
    "appearance": (4, 5),
    "loyalty": (6, 7),
}


def score_suprq(responses: Matrix) -> MetricResult:
    """SUPR-Q 8-item (Sauro, 2015).

    Items 1-7 are 1-5 agreement; item 8 is a 0-10 likelihood-to-recommend that
    is rescaled to 1-5 before averaging. Overall is reported 0-100.
    """
    warnings = _check(responses, 8, "SUPR-Q")

    def rescaled_row(row: Sequence[float]) -> list[float]:
        scaled = list(row[:7])
        scaled.append(1 + (row[7] / 10.0) * 4)  # 0-10 -> 1-5
        return scaled

    per_overall: list[float] = []
    sub_per: dict[str, list[float]] = {k: [] for k in _SUPRQ_SUBSCALES}
    for row in responses:
        scaled = rescaled_row(row)
        per_overall.append((sum(scaled) / 8.0 - 1) / 4.0 * 100.0)
        for name, (a, b) in _SUPRQ_SUBSCALES.items():
            sub_per[name].append((scaled[a] + scaled[b]) / 2.0)
    score, interval = _interval(per_overall)
    subscales = {
        name: round(sum(vals) / len(vals), 2) for name, vals in sub_per.items()
    }
    return MetricResult(
        key="suprq",
        name="SUPR-Q",
        family=MetricFamily.QUESTIONNAIRE,
        score=score,
        scale="0-100",
        direction=Direction.HIGHER_BETTER,
        n=len(responses),
        interpretation=(
            "Overall website-quality score with usability, trust, appearance "
            "and loyalty sub-scales (sub-scales on the 1-5 item scale)."
        ),
        components={"subscales": subscales},
        benchmark={
            "percentile_approx": percentile_from_normal(score, 50.0, 20.0),
        },
        interval=interval,
        warnings=warnings
        + [
            "True SUPR-Q percentiles require Sauro's normed database; the value "
            "shown is a rough Normal approximation only."
        ],
    )


# --- PSSUQ / CSUQ ----------------------------------------------------------

_PSSUQ_SUBSCALES = {
    "system_usefulness": range(0, 6),
    "information_quality": range(6, 12),
    "interface_quality": range(12, 15),
}


def _score_ibm(responses: Matrix, key: str, name: str) -> MetricResult:
    """Shared scoring for the IBM 16-item PSSUQ / CSUQ family (1-7, lower better)."""
    warnings = _check(responses, 16, name)
    per_overall: list[float] = []
    sub_per: dict[str, list[float]] = {k: [] for k in _PSSUQ_SUBSCALES}
    for row in responses:
        per_overall.append(sum(row) / 16.0)
        for sub, cols in _PSSUQ_SUBSCALES.items():
            vals = [row[c] for c in cols]
            sub_per[sub].append(sum(vals) / len(vals))
    score, interval = _interval(per_overall)
    subscales = {
        sub: round(sum(vals) / len(vals), 2) for sub, vals in sub_per.items()
    }
    return MetricResult(
        key=key,
        name=name,
        family=MetricFamily.QUESTIONNAIRE,
        score=score,
        scale="1-7 (lower is better)",
        direction=Direction.LOWER_BETTER,
        n=len(responses),
        interpretation=(
            f"Overall {score} vs. norm {bm.PSSUQ_NORMS['overall']}; lower means "
            "fewer reported problems with usefulness, information and interface."
        ),
        components={"subscales": subscales},
        benchmark={"norms": bm.PSSUQ_NORMS},
        interval=interval,
        warnings=warnings,
    )


def score_pssuq(responses: Matrix) -> MetricResult:
    """Post-Study System Usability Questionnaire (Lewis). 16 items, 1-7."""
    return _score_ibm(responses, "pssuq", "Post-Study System Usability Questionnaire")


def score_csuq(responses: Matrix) -> MetricResult:
    """Computer System Usability Questionnaire (IBM). 16 items, 1-7."""
    return _score_ibm(responses, "csuq", "Computer System Usability Questionnaire")


# --- SEQ -------------------------------------------------------------------


def score_seq(responses: Matrix) -> MetricResult:
    """Single Ease Question. 1 item, 1-7 (very difficult -> very easy)."""
    warnings = _check(responses, 1, "SEQ")
    per = [row[0] for row in responses]
    score, interval = _interval(per)
    flag = score < 5.0
    return MetricResult(
        key="seq",
        name="Single Ease Question",
        family=MetricFamily.TASK,
        score=score,
        scale="1-7",
        direction=Direction.HIGHER_BETTER,
        n=len(responses),
        interpretation=(
            f"Mean ease {score} vs. typical {bm.SEQ_BENCHMARK}. "
            + ("Below average — likely a difficult task." if flag else "Around or above the usual single-task benchmark.")
        ),
        benchmark={"benchmark": bm.SEQ_BENCHMARK},
        interval=interval,
        warnings=warnings + (["Mean ease under 5 flags a struggling task."] if flag else []),
    )


# --- UEQ-S -----------------------------------------------------------------


def score_ueq_s(responses: Matrix) -> MetricResult:
    """UEQ-S short form (Schrepp et al., 2017). 8 items, 1-7 -> -3..+3.

    Items 1-4 = pragmatic quality, items 5-8 = hedonic quality.
    """
    warnings = _check(responses, 8, "UEQ-S")
    prag_per, hed_per, all_per = [], [], []
    for row in responses:
        centred = [v - 4 for v in row]  # 1-7 -> -3..+3
        prag = sum(centred[0:4]) / 4.0
        hed = sum(centred[4:8]) / 4.0
        prag_per.append(prag)
        hed_per.append(hed)
        all_per.append((prag + hed) / 2.0)
    score, interval = _interval(all_per)
    prag_mean = round(sum(prag_per) / len(prag_per), 2)
    hed_mean = round(sum(hed_per) / len(hed_per), 2)
    return MetricResult(
        key="ueq_s",
        name="UEQ-S (short)",
        family=MetricFamily.QUESTIONNAIRE,
        score=score,
        scale="-3 to +3",
        direction=Direction.HIGHER_BETTER,
        n=len(responses),
        interpretation=(
            f"Overall {score} ({bm.ueq_band(score)}). Pragmatic {prag_mean}, "
            f"hedonic {hed_mean}."
        ),
        components={
            "pragmatic": prag_mean,
            "hedonic": hed_mean,
            "pragmatic_band": bm.ueq_band(prag_mean),
            "hedonic_band": bm.ueq_band(hed_mean),
        },
        interval=interval,
        warnings=warnings,
    )


# --- NASA-TLX ---------------------------------------------------------------

_TLX_DIMS = [
    "mental_demand",
    "physical_demand",
    "temporal_demand",
    "performance",
    "effort",
    "frustration",
]


def score_nasa_tlx(
    responses: Matrix, weights: Sequence[float] | None = None
) -> MetricResult:
    """NASA-TLX workload. 6 items 0-100. Raw TLX by default; weighted if given.

    Lower means less workload. ``weights`` (length 6, summing to 15) switch the
    output to the classic weighted Task Load Index.
    """
    warnings = _check(responses, 6, "NASA-TLX")
    per: list[float] = []
    for row in responses:
        if weights is not None:
            if len(weights) != 6:
                raise ValueError("NASA-TLX weights must have length 6")
            denom = sum(weights)
            per.append(sum(r * w for r, w in zip(row, weights, strict=False)) / denom)
        else:
            per.append(sum(row) / 6.0)
    score, interval = _interval(per)
    dim_means = {
        dim: round(sum(row[i] for row in responses) / len(responses), 1)
        for i, dim in enumerate(_TLX_DIMS)
    }
    return MetricResult(
        key="nasa_tlx",
        name="NASA-TLX" + (" (weighted)" if weights else " (raw)"),
        family=MetricFamily.WORKLOAD,
        score=score,
        scale="0-100 (lower is better)",
        direction=Direction.LOWER_BETTER,
        n=len(responses),
        interpretation=(
            f"Overall perceived workload {score}/100. Highest driver: "
            f"{max(dim_means, key=dim_means.get).replace('_', ' ')}."
        ),
        components={"dimensions": dim_means},
        interval=interval,
        warnings=warnings,
    )
