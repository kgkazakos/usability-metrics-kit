"""AI-native flow metrics for evaluating generative-AI interfaces.

Traditional usability metrics assume a deterministic interface: the same action
produces the same result, and "success" is binary. Generative-AI flows break
that assumption. The user and the model co-produce an output over several turns,
"acceptance" is graded rather than binary, and the interesting failure mode is
often *abandonment* (the user gives up on the AI) rather than task failure.

This module adds two layers:

* **Behavioural (collaboration layer)** — how the human and the model actually
  worked together: how much AI content survived, how many turns it took, how
  often the user fell back to doing it manually, and how often suggestions were
  accepted.
* **Attitudinal (trust & quality layer)** — the validated Jian et al. (2000)
  trust-in-automation scale and a lightweight perceived-output-quality scale.

Every metric returns the same :class:`MetricResult` shape as the rest of the
kit, with a *mathematically appropriate* interval for its data type:

* binomial proportions (fallback, acceptance) -> adjusted-Wald
* continuous per-session proportions (retention) -> t-interval, clamped to [0,1]
* count data (prompt iterations) -> exact Poisson (Garwood), with an
  overdispersion guard
* Likert means (trust, perceived quality) -> t-interval

None of these have an established normed database the way SUS does, so the
benchmark bands are explicitly framed as orientation heuristics, not norms.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from . import benchmarks as bm
from .models import Direction, Interval, MetricFamily, MetricResult
from .stats import (
    adjusted_wald_ci,
    mean_ci,
    poisson_exact_ci,
    proportion_mean_ci,
    welch_ci,
)

SMALL_N = 10


def _small_n_warning(n: int, kind: str = "observations") -> list[str]:
    if n < SMALL_N:
        return [
            f"Only {n} {kind} (n < {SMALL_N}); the interval is wide on purpose. "
            "Treat the point estimate as directional, not settled."
        ]
    return []


# ---------------------------------------------------------------------------
# Behavioural / collaboration layer
# ---------------------------------------------------------------------------


def retention_rate(fractions: Sequence[float]) -> MetricResult:
    """Share of AI-generated content that survives into the final submission.

    ``fractions`` is one *continuous* value per session in [0, 1] (or [0, 100];
    values > 1 are treated as percentages). This is deliberately NOT modelled as
    a binomial proportion: each session contributes a fraction, not a
    success/failure, so the right summary is the mean retention with a t-interval
    (clamped to the [0, 1] bound), not an adjusted-Wald count interval.

    Use :func:`retention_from_edits` to derive these fractions from raw edit
    distances if you only have before/after lengths.
    """
    if not fractions:
        raise ValueError("retention_rate: no sessions")
    norm = [(f / 100.0 if f > 1.0 else f) for f in fractions]
    norm = [min(1.0, max(0.0, f)) for f in norm]
    mean, low, high = proportion_mean_ci(norm)
    pct = round(mean * 100, 1)
    return MetricResult(
        key="retention_rate",
        name="AI Content Retention Rate",
        family=MetricFamily.AI_NATIVE,
        score=pct,
        scale="0-100% (higher is better)",
        direction=Direction.HIGHER_BETTER,
        n=len(norm),
        interpretation=(
            f"On average {pct}% of the AI-generated content survived into the "
            "user's final submission (mean of per-session retention). Higher "
            "means the draft needed less rework; very low retention can mean the "
            "AI missed, or that heavy editing was the intended workflow."
        ),
        components={
            "mean_retention": pct,
            "median_retention": round(statistics.median(norm) * 100, 1),
            "per_session": [round(f * 100, 1) for f in norm],
        },
        interval=Interval(round(low * 100, 1), round(high * 100, 1), method="t (clamped)"),
        warnings=_small_n_warning(len(norm), "sessions"),
    )


def retention_from_edits(
    original_lengths: Sequence[float], edit_distances: Sequence[float]
) -> list[float]:
    """Convert raw edit distances into per-session retention fractions.

    Retention is ``1 - edit_distance / original_length`` per session, floored at
    0. ``edit_distance`` can be a character/token Levenshtein distance computed
    upstream; this keeps the numeric-input convention of the kit while letting
    you feed edit-distance data in.
    """
    if len(original_lengths) != len(edit_distances):
        raise ValueError("retention_from_edits: length mismatch")
    out = []
    for orig, dist in zip(original_lengths, edit_distances, strict=False):
        if orig <= 0:
            out.append(0.0)
        else:
            out.append(max(0.0, 1.0 - dist / orig))
    return out


def prompt_iterations(counts: Sequence[float]) -> MetricResult:
    """Number of back-and-forth turns to reach a satisfactory output.

    ``counts`` is one count per session (>= 1). Counts are discrete, positive
    and right-skewed, so the headline mean carries an exact Poisson (Garwood)
    interval rather than a symmetric t-interval. Iteration counts are frequently
    *overdispersed* (variance > mean), where Poisson understates the interval;
    the index of dispersion is computed and a warning points to a bootstrap or
    negative-binomial model when it is badly violated.
    """
    if not counts:
        raise ValueError("prompt_iterations: no sessions")
    mean, low, high, dispersion = poisson_exact_ci(counts)
    warnings = _small_n_warning(len(counts), "sessions")
    if dispersion > 2.0 and len(counts) > 1:
        warnings.append(
            f"Index of dispersion {round(dispersion, 1)} >> 1: the counts are "
            "overdispersed, so the Poisson interval is too narrow. Prefer a "
            "bootstrap or negative-binomial interval for a headline figure."
        )
    return MetricResult(
        key="prompt_iterations",
        name="Prompt Iterations to Satisfaction",
        family=MetricFamily.AI_NATIVE,
        score=round(mean, 2),
        scale="turns per session (lower is better)",
        direction=Direction.LOWER_BETTER,
        n=len(counts),
        interpretation=(
            f"Users needed {round(mean, 2)} turns on average to reach a "
            f"satisfactory output (median {round(statistics.median(counts), 1)}). "
            "Fewer turns usually means the first response landed closer to intent."
        ),
        components={
            "mean": round(mean, 2),
            "median": round(statistics.median(counts), 1),
            "max": round(max(counts), 1),
            "index_of_dispersion": round(dispersion, 2),
        },
        interval=Interval(round(low, 2), round(high, 2), method="Poisson exact (Garwood)"),
        warnings=warnings,
    )


def human_fallback_rate(fallbacks: Sequence[int]) -> MetricResult:
    """Rate at which users abandoned the AI to do the task manually.

    ``fallbacks`` is a sequence of 1/0 flags (1 = fell back to manual). This is a
    genuine binomial proportion, so it uses an adjusted-Wald interval like the
    task-completion rate. Lower is better.
    """
    if not fallbacks:
        raise ValueError("human_fallback_rate: no sessions")
    total = len(fallbacks)
    n_fall = sum(1 for f in fallbacks if f)
    point, low, high = adjusted_wald_ci(n_fall, total)
    pct = round(point * 100, 1)
    return MetricResult(
        key="human_fallback",
        name="Human Fallback Rate",
        family=MetricFamily.AI_NATIVE,
        score=pct,
        scale="0-100% (lower is better)",
        direction=Direction.LOWER_BETTER,
        n=total,
        interpretation=(
            f"{n_fall}/{total} sessions abandoned the AI and completed the task "
            "manually. A high fallback rate is the clearest signal that the AI "
            "flow is not earning its place in the workflow."
        ),
        benchmark={"fell_back": n_fall, "stayed_with_ai": total - n_fall},
        interval=Interval(round(low * 100, 1), round(high * 100, 1), method="adjusted-Wald"),
        warnings=_small_n_warning(total, "sessions"),
    )


def ai_acceptance_rate(
    decisions: Sequence[int] | None = None,
    *,
    accepted: int | None = None,
    offered: int | None = None,
) -> MetricResult:
    """Share of AI suggestions accepted rather than dismissed.

    Provide either a sequence of 1/0 accept decisions, or aggregate
    ``accepted``/``offered`` counts. Modelled as a binomial proportion with an
    adjusted-Wald interval.

    Caveat surfaced in ``warnings``: suggestions cluster within users, so pooling
    every suggestion as independent overstates precision. For a clustered design,
    feed per-user acceptance fractions to :func:`retention_rate`-style averaging
    instead. Note too that a very high acceptance rate is not unambiguously good
    — it can indicate automation bias rather than good suggestions.
    """
    if decisions is not None:
        offered = len(decisions)
        accepted = sum(1 for d in decisions if d)
    if accepted is None or offered is None:
        raise ValueError("ai_acceptance_rate: provide decisions or accepted/offered")
    if offered == 0:
        raise ValueError("ai_acceptance_rate: no suggestions offered")
    point, low, high = adjusted_wald_ci(accepted, offered)
    pct = round(point * 100, 1)
    band = bm.acceptance_band(point)
    warnings = _small_n_warning(offered, "suggestions")
    warnings.append(
        "Suggestions cluster within users; pooled counts treat them as "
        "independent and can overstate precision. Very high acceptance may "
        "reflect automation bias, not suggestion quality."
    )
    return MetricResult(
        key="ai_acceptance",
        name="AI Acceptance Rate",
        family=MetricFamily.AI_NATIVE,
        score=pct,
        scale="0-100% (higher is better, with caveats)",
        direction=Direction.HIGHER_BETTER,
        n=offered,
        interpretation=(
            f"{accepted}/{offered} suggestions accepted ({band}). Read alongside "
            "fallback and retention: high acceptance with low retention means "
            "users accept then heavily rewrite."
        ),
        benchmark={"accepted": accepted, "dismissed": offered - accepted, "band": band},
        interval=Interval(round(low * 100, 1), round(high * 100, 1), method="adjusted-Wald"),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Attitudinal / trust & quality layer
# ---------------------------------------------------------------------------

# Jian, Bisantz & Drury (2000) "Trust between people and automated systems".
# 12 items on a 1-7 scale. Items 1-5 are distrust-framed and reverse-scored so
# that a high corrected score means more trust; items 6-12 are trust-framed.
# The two-factor (Distrust / Trust) structure is the common empirical finding
# (e.g. Spain et al. 2008), so both subscales are exposed alongside the overall.
JIAN_DISTRUST_ITEMS = (0, 1, 2, 3, 4)  # 0-indexed items 1-5
JIAN_TRUST_ITEMS = (5, 6, 7, 8, 9, 10, 11)  # 0-indexed items 6-12
JIAN_SCALE_MAX = 7


def trust_automation(matrix: Sequence[Sequence[float]]) -> MetricResult:
    """Jian et al. (2000) 12-item trust-in-automation scale.

    ``matrix`` is one row per respondent, each with 12 values on a 1-7 scale.
    Items 1-5 (distrust) are reverse-coded (``8 - x``); the overall is the mean
    of all 12 corrected items. Higher overall = more trust. The Distrust and
    Trust subscales are reported separately because the scale is widely treated
    as two-factor rather than unidimensional.
    """
    if not matrix:
        raise ValueError("trust_automation: no respondents")
    for row in matrix:
        if len(row) != 12:
            raise ValueError("trust_automation: expected 12 items per respondent")

    per_respondent_overall: list[float] = []
    distrust_means: list[float] = []
    trust_means: list[float] = []
    for row in matrix:
        corrected = list(row)
        for i in JIAN_DISTRUST_ITEMS:
            corrected[i] = (JIAN_SCALE_MAX + 1) - row[i]  # reverse on 1-7
        per_respondent_overall.append(sum(corrected) / 12.0)
        distrust_means.append(sum(row[i] for i in JIAN_DISTRUST_ITEMS) / 5.0)
        trust_means.append(sum(row[i] for i in JIAN_TRUST_ITEMS) / 7.0)

    mean, low, high = mean_ci(per_respondent_overall)
    low, high = max(1.0, low), min(float(JIAN_SCALE_MAX), high)
    band = bm.trust_band(mean)
    return MetricResult(
        key="trust_automation",
        name="Trust in Automation (Jian et al.)",
        family=MetricFamily.AI_NATIVE,
        score=round(mean, 2),
        scale="1-7 (higher is better)",
        direction=Direction.HIGHER_BETTER,
        n=len(matrix),
        interpretation=(
            f"Overall trust {round(mean, 2)}/7 ({band}). Items 1-5 (distrust) are "
            "reverse-coded; the Trust and Distrust subscales are reported "
            "separately because the scale is commonly treated as two-factor."
        ),
        components={
            "overall_mean": round(mean, 2),
            "subscales": {
                "trust": round(statistics.mean(trust_means), 2),
                "distrust_raw": round(statistics.mean(distrust_means), 2),
            },
            "per_respondent_overall": [round(v, 2) for v in per_respondent_overall],
        },
        benchmark={"band": band},
        interval=Interval(round(low, 2), round(high, 2), method="t"),
        warnings=_small_n_warning(len(matrix), "respondents"),
    )


def perceived_output_quality(matrix: Sequence[Sequence[float]]) -> MetricResult:
    """Perceived Output Quality (POQ): a 3-item Likert scale.

    Three items — Accuracy, Relevance, Helpfulness — on a 1-7 scale (one row per
    respondent). Reports the overall mean with a t-interval plus the three item
    means. This is a lightweight custom scale, not a validated instrument; it is
    framed as such in ``warnings``.
    """
    if not matrix:
        raise ValueError("perceived_output_quality: no respondents")
    for row in matrix:
        if len(row) != 3:
            raise ValueError("perceived_output_quality: expected 3 items per respondent")
    overall = [sum(row) / 3.0 for row in matrix]
    mean, low, high = mean_ci(overall)
    low, high = max(1.0, low), min(7.0, high)
    labels = ("accuracy", "relevance", "helpfulness")
    item_means = {
        labels[i]: round(statistics.mean(row[i] for row in matrix), 2) for i in range(3)
    }
    return MetricResult(
        key="poq",
        name="Perceived Output Quality",
        family=MetricFamily.AI_NATIVE,
        score=round(mean, 2),
        scale="1-7 (higher is better)",
        direction=Direction.HIGHER_BETTER,
        n=len(matrix),
        interpretation=(
            f"Perceived output quality {round(mean, 2)}/7 across accuracy, "
            "relevance and helpfulness. Useful as a fast subjective read on "
            "generation quality between flows."
        ),
        components={"overall_mean": round(mean, 2), "items": item_means},
        interval=Interval(round(low, 2), round(high, 2), method="t"),
        warnings=(
            _small_n_warning(len(matrix), "respondents")
            + ["POQ is a lightweight custom scale, not a psychometrically validated instrument."]
        ),
    )


# ---------------------------------------------------------------------------
# Composite roll-up
# ---------------------------------------------------------------------------


def ai_collaboration_score(
    acceptance: Sequence[float],
    retention: Sequence[float],
    trust: Sequence[float],
    *,
    aligned: bool = True,
) -> MetricResult:
    """Heuristic AI Collaboration Score rolling up acceptance, retention and trust.

    Each input is normalised to a common 0-1 directional scale (higher = better
    collaboration) and combined with equal weights into a 0-100 score:

    * ``acceptance`` — per-session accept fractions in [0, 1]
    * ``retention``  — per-session retention fractions in [0, 1] (>1 = percent)
    * ``trust``      — per-respondent Jian overall trust on 1-7, mapped (x-1)/6

    With ``aligned=True`` (the default) the three sequences are assumed to be the
    *same units in the same order* (e.g. the same participants), so a per-unit
    composite is computed and the standard deviation and t-interval come straight
    from that distribution — an honest joint SD. With ``aligned=False`` the three
    point estimates are averaged and the SD is propagated from each component's
    spread; this is looser and a warning says so.

    The score is explicitly a *heuristic* roll-up: equal weights, no validated
    weighting, intended as a single directional signal, not a verdict.
    """
    def _norm_prop(v: float) -> float:
        v = v / 100.0 if v > 1.0 else v
        return min(1.0, max(0.0, v))

    def _norm_trust(v: float) -> float:
        return min(1.0, max(0.0, (v - 1.0) / 6.0))

    warnings: list[str] = [
        "Heuristic equal-weight roll-up of three different constructs; a single "
        "directional signal, not a validated composite or a verdict."
    ]

    if aligned:
        n = min(len(acceptance), len(retention), len(trust))
        if n == 0:
            raise ValueError("ai_collaboration_score: empty input")
        if not (len(acceptance) == len(retention) == len(trust)):
            warnings.append(
                f"Aligned inputs had unequal lengths; truncated to the first {n} units."
            )
        per_unit = [
            (_norm_prop(acceptance[i]) + _norm_prop(retention[i]) + _norm_trust(trust[i])) / 3.0
            for i in range(n)
        ]
        composite = [v * 100.0 for v in per_unit]
        mean, low, high = mean_ci(composite)
        sd = statistics.stdev(composite) if n > 1 else 0.0
        method = "t (per-unit composite)"
        n_out = n
    else:
        comps = [
            statistics.mean(_norm_prop(v) for v in acceptance),
            statistics.mean(_norm_prop(v) for v in retention),
            statistics.mean(_norm_trust(v) for v in trust),
        ]
        mean = statistics.mean(comps) * 100.0
        # propagate component spread (SD of the three component means, scaled)
        sd = statistics.stdev([c * 100.0 for c in comps]) if len(comps) > 1 else 0.0
        margin = 1.96 * sd / (len(comps) ** 0.5)
        low, high = mean - margin, mean + margin
        method = "propagated (unaligned components)"
        n_out = min(len(acceptance), len(retention), len(trust))
        warnings.append(
            "Inputs treated as unaligned: the SD is propagated from component "
            "means, not a joint per-unit distribution. Provide aligned per-unit "
            "data for an honest interval."
        )

    score = round(mean, 1)
    band = bm.collaboration_band(score)
    return MetricResult(
        key="ai_collaboration",
        name="AI Collaboration Score",
        family=MetricFamily.COMPOSITE,
        score=score,
        scale="0-100 (higher is better)",
        direction=Direction.HIGHER_BETTER,
        n=n_out,
        interpretation=(
            f"AI Collaboration Score {score}/100 ({band}), equal-weight roll-up of "
            f"acceptance, content retention and trust (SD {round(sd, 1)})."
        ),
        components={
            "weights": {"acceptance": 1 / 3, "retention": 1 / 3, "trust": 1 / 3},
            "std_dev": round(sd, 2),
            "band": band,
        },
        benchmark={"band": band},
        interval=Interval(round(max(0.0, low), 1), round(min(100.0, high), 1), method=method),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Reliance & value — require ground-truth labels or a no-AI baseline
# ---------------------------------------------------------------------------
#
# These close the gap that raw acceptance leaves open: was the user's reliance
# *correct*, and did the AI actually help? They need an extra input the other
# metrics don't — per-decision correctness labels, or a without-AI baseline —
# so they form a natural second sub-group.


def _subset_rate(numerator: int, denom: int) -> dict | None:
    if denom == 0:
        return None
    p, lo, hi = adjusted_wald_ci(numerator, denom)
    return {
        "rate": round(float(p) * 100, 1),
        "ci": [round(float(lo) * 100, 1), round(float(hi) * 100, 1)],
        "n": denom,
    }


def appropriate_reliance(rows: Sequence[Sequence[float]]) -> MetricResult:
    """Appropriate reliance from per-decision (ai_correct, accepted) labels.

    ``rows`` is an N x 2 matrix: column 0 = whether the AI's advice was correct
    (1/0), column 1 = whether the user accepted/followed it (1/0). Requires a
    ground-truth correctness label for each AI suggestion.

    Reliance is *appropriate* when the user accepts correct advice and rejects
    incorrect advice. The two failure modes are reported separately, each as an
    adjusted-Wald rate on its own subset:

    * over-reliance  = accepting the AI when it was wrong   (lower is better)
    * under-reliance = rejecting the AI when it was right   (lower is better)

    This is the metric the raw acceptance rate cannot give you: it tells you not
    just how often suggestions were taken, but whether they *should* have been.
    """
    if not rows:
        raise ValueError("appropriate_reliance: no decisions")
    pairs = []
    for r in rows:
        if len(r) < 2:
            raise ValueError(
                "appropriate_reliance: expected 2 columns [ai_correct, accepted]"
            )
        pairs.append((int(round(r[0])), int(round(r[1]))))
    n = len(pairs)
    appropriate = sum(1 for c, a in pairs if (a == 1 and c == 1) or (a == 0 and c == 0))
    ar_point, ar_low, ar_high = adjusted_wald_ci(appropriate, n)

    n_wrong = sum(1 for c, _ in pairs if c == 0)
    n_right = sum(1 for c, _ in pairs if c == 1)
    over = _subset_rate(sum(1 for c, a in pairs if c == 0 and a == 1), n_wrong)
    under = _subset_rate(sum(1 for c, a in pairs if c == 1 and a == 0), n_right)

    band = bm.reliance_band(ar_point)
    warnings = _small_n_warning(n, "decisions")
    warnings.append(
        "Requires a ground-truth correctness label for each AI suggestion; over- "
        "and under-reliance are estimated on subsets, so their intervals widen fast."
    )
    if 0 < n_wrong < SMALL_N:
        warnings.append(f"Only {n_wrong} AI-incorrect cases — the over-reliance estimate is noisy.")
    if 0 < n_right < SMALL_N:
        warnings.append(f"Only {n_right} AI-correct cases — the under-reliance estimate is noisy.")
    if n_wrong == 0:
        warnings.append("No AI-incorrect cases, so over-reliance cannot be estimated.")
    if n_right == 0:
        warnings.append("No AI-correct cases, so under-reliance cannot be estimated.")

    over_txt = f"{over['rate']}%" if over else "n/a"
    under_txt = f"{under['rate']}%" if under else "n/a"
    return MetricResult(
        key="appropriate_reliance",
        name="Appropriate Reliance",
        family=MetricFamily.AI_NATIVE,
        score=round(ar_point * 100, 1),
        scale="0-100% (higher is better)",
        direction=Direction.HIGHER_BETTER,
        n=n,
        interpretation=(
            f"Users relied appropriately on {appropriate}/{n} decisions ({band}) — "
            "taking correct advice and rejecting incorrect advice. Over-reliance "
            f"(accepting wrong AI): {over_txt}; under-reliance (rejecting right AI): "
            f"{under_txt}."
        ),
        components={
            "appropriate": appropriate,
            "over_reliance": over,
            "under_reliance": under,
        },
        benchmark={"band": band},
        interval=Interval(round(ar_low * 100, 1), round(ar_high * 100, 1), method="adjusted-Wald"),
        warnings=warnings,
    )


def relative_reliance(rows: Sequence[Sequence[float]]) -> MetricResult:
    """RAIR / RSR appropriate-reliance measures (Schemmer et al., 2023).

    ``rows`` is an N x 3 matrix: column 0 = the user's *initial* decision correct
    (1/0, before seeing the AI), column 1 = AI advice correct (1/0), column 2 =
    final decision followed the AI (1/0).

    * **RAIR** (Relative positive AI Reliance) — of the cases where the user was
      initially wrong and the AI was right, the share who switched to the AI
      (rightfully changed their mind).
    * **RSR** (Relative positive Self-Reliance) — of the cases where the user was
      initially right and the AI was wrong, the share who kept their own answer
      (rightfully resisted the AI).

    Both higher is better. The headline is their balanced mean; each is reported
    with its own adjusted-Wald interval and subset size, because these subsets
    are typically small.
    """
    if not rows:
        raise ValueError("relative_reliance: no decisions")
    triples = []
    for r in rows:
        if len(r) < 3:
            raise ValueError(
                "relative_reliance: expected 3 columns "
                "[initial_correct, ai_correct, final_followed_ai]"
            )
        triples.append((int(round(r[0])), int(round(r[1])), int(round(r[2]))))
    n = len(triples)

    rair_denom = [(i, c, f) for i, c, f in triples if i == 0 and c == 1]
    rair = _subset_rate(sum(1 for _, _, f in rair_denom if f == 1), len(rair_denom))
    rsr_denom = [(i, c, f) for i, c, f in triples if i == 1 and c == 0]
    rsr = _subset_rate(sum(1 for _, _, f in rsr_denom if f == 0), len(rsr_denom))

    available = [m["rate"] for m in (rair, rsr) if m]
    score = round(sum(available) / len(available), 1) if available else 0.0

    warnings = _small_n_warning(n, "decisions")
    warnings.append(
        "Needs initial decision, AI correctness and final choice per instance "
        "(ground truth). RAIR/RSR are computed on small conditional subsets."
    )
    if rair is None:
        warnings.append("No 'initially-wrong, AI-right' cases, so RAIR cannot be estimated.")
    if rsr is None:
        warnings.append("No 'initially-right, AI-wrong' cases, so RSR cannot be estimated.")

    rair_txt = f"{rair['rate']}% (n={rair['n']})" if rair else "n/a"
    rsr_txt = f"{rsr['rate']}% (n={rsr['n']})" if rsr else "n/a"
    return MetricResult(
        key="relative_reliance",
        name="Relative Reliance (RAIR / RSR)",
        family=MetricFamily.AI_NATIVE,
        score=score,
        scale="0-100% balanced (higher is better)",
        direction=Direction.HIGHER_BETTER,
        n=n,
        interpretation=(
            f"Balanced appropriate reliance {score}% — RAIR (rightfully adopting "
            f"correct AI advice the user would otherwise have missed): {rair_txt}; "
            f"RSR (rightfully keeping a correct answer against wrong AI advice): {rsr_txt}."
        ),
        components={"rair": rair, "rsr": rsr},
        interval=None,
        warnings=warnings,
    )


def productivity_delta(
    with_ai: Sequence[float],
    without_ai: Sequence[float] | None = None,
    *,
    paired: bool = True,
    lower_is_better: bool = True,
) -> MetricResult:
    """Value/productivity delta: did the AI actually help, vs a no-AI baseline?

    Compares a measure (task time by default, or a quality score) *with* the AI
    against *without* it. The headline is the improvement attributable to the AI,
    in the metric's own units, with the interval the design warrants:

    * **paired** (same participants in both conditions) -> paired t-interval on
      the per-pair improvements.
    * **two-sample** (independent groups) -> Welch interval on the difference of
      means.

    ``lower_is_better=True`` treats the input as a cost like time (improvement =
    baseline - with-AI, i.e. time saved); set it ``False`` for a quality score
    (improvement = with-AI - baseline). When called through the registry with a
    single 2-column matrix, columns are read as ``[with_ai, without_ai]`` (paired).

    If the interval crosses zero, the tool says plainly that there is no reliable
    evidence the AI changed the outcome.
    """
    if without_ai is None:
        rows = [r for r in with_ai if len(r) >= 2]
        if not rows:
            raise ValueError("productivity_delta: expected 2 columns [with_ai, without_ai]")
        with_vals = [float(r[0]) for r in rows]
        without_vals = [float(r[1]) for r in rows]
        paired = True
    else:
        with_vals = [float(v) for v in with_ai]
        without_vals = [float(v) for v in without_ai]
    if not with_vals or not without_vals:
        raise ValueError("productivity_delta: empty input")

    if paired:
        if len(with_vals) != len(without_vals):
            raise ValueError("productivity_delta: paired data must be equal length")
        improvements = [
            (wo - w) if lower_is_better else (w - wo)
            for w, wo in zip(with_vals, without_vals, strict=True)
        ]
        mean, low, high = mean_ci(improvements)
        method = "paired t"
        n = len(improvements)
    else:
        if lower_is_better:
            mean, low, high = welch_ci(without_vals, with_vals)
        else:
            mean, low, high = welch_ci(with_vals, without_vals)
        method = "Welch t (two-sample)"
        n = len(with_vals) + len(without_vals)

    baseline = sum(without_vals) / len(without_vals)
    pct = round(mean / baseline * 100, 1) if baseline else 0.0
    unit = "units"
    crosses_zero = low <= 0 <= high

    warnings = _small_n_warning(n, "observations")
    if crosses_zero:
        warnings.append(
            "The interval includes zero — no reliable evidence the AI changed the "
            "outcome at this sample size."
        )

    direction_note = "faster" if lower_is_better else "higher"
    return MetricResult(
        key="productivity_delta",
        name="AI Productivity Delta",
        family=MetricFamily.AI_NATIVE,
        score=round(mean, 2),
        scale=f"{unit} improved vs no-AI baseline (higher is better)",
        direction=Direction.HIGHER_BETTER,
        n=n,
        interpretation=(
            f"With the AI, the outcome was {round(mean, 2)} {unit} {direction_note} on "
            f"average ({pct}% vs the no-AI baseline of {round(baseline, 2)})."
            + (" The interval crosses zero, so treat this as no measured effect."
               if crosses_zero else "")
        ),
        components={
            "mean_improvement": round(mean, 2),
            "pct_vs_baseline": pct,
            "with_ai_mean": round(sum(with_vals) / len(with_vals), 2),
            "without_ai_mean": round(baseline, 2),
            "paired": paired,
        },
        interval=Interval(round(low, 2), round(high, 2), method=method),
        warnings=warnings,
    )
