"""Small statistics toolkit used across the metrics.

The functions here intentionally avoid pulling in heavyweight machinery: a
usability practitioner running this on a laptop with a sample of eight should
get sensible small-sample behaviour (adjusted-Wald proportions, log-transformed
time intervals) rather than naive Wald intervals that fall off the edge of the
probability scale.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from scipy import stats


def normal_cdf(z: float) -> float:
    """Standard normal CDF via the error function (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def percentile_from_normal(value: float, mean: float, sd: float) -> float:
    """Percentile (0-100) of ``value`` against a Normal(mean, sd) reference."""
    if sd <= 0:
        return 50.0
    return round(normal_cdf((value - mean) / sd) * 100.0, 1)


def mean_ci(
    values: Sequence[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) using a t-distribution."""
    n = len(values)
    mean = sum(values) / n
    if n < 2 or max(values) == min(values):
        return mean, mean, mean
    sd = stats.tstd(values)
    se = sd / math.sqrt(n)
    t = stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, df=n - 1)
    margin = t * se
    return mean, mean - margin, mean + margin


def adjusted_wald_ci(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Adjusted-Wald interval for a proportion (Agresti-Coull).

    Recommended by Sauro & Lewis for small-sample completion rates because the
    standard Wald interval is badly miscalibrated when ``total`` is small or the
    rate is near 0 or 1.
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    n_adj = total + z * z
    p_adj = (successes + z * z / 2.0) / n_adj
    margin = z * math.sqrt(p_adj * (1.0 - p_adj) / n_adj)
    low = max(0.0, p_adj - margin)
    high = min(1.0, p_adj + margin)
    point = successes / total
    return point, low, high


def poisson_exact_ci(
    counts: Sequence[float], confidence: float = 0.95
) -> tuple[float, float, float, float]:
    """Exact (Garwood) Poisson interval for the *mean count per observation*.

    Count data such as the number of prompt iterations per session is discrete,
    non-negative and right-skewed, so a symmetric t-interval misrepresents it.
    The Garwood interval inverts the chi-square relationship for the total event
    count and is exact for any sample size, including ``n < 10``.

    Returns ``(mean, low, high, dispersion)`` where ``dispersion`` is the index
    of dispersion (variance / mean). A value far above 1 signals overdispersion,
    under which Poisson understates the interval and a bootstrap or
    negative-binomial model is the better choice -- the caller surfaces this as
    a warning rather than silently trusting the Poisson interval.
    """
    n = len(counts)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    total = sum(counts)
    mean = total / n
    alpha = 1.0 - confidence
    # Garwood exact limits for the total, then divide by n for the per-session mean.
    low_total = 0.0 if total == 0 else stats.chi2.ppf(alpha / 2.0, 2 * total) / 2.0
    high_total = stats.chi2.ppf(1.0 - alpha / 2.0, 2 * total + 2) / 2.0
    var = stats.tvar(counts) if n > 1 and max(counts) != min(counts) else 0.0
    dispersion = (var / mean) if mean > 0 else 0.0
    return mean, low_total / n, high_total / n, dispersion


def welch_ci(
    a: Sequence[float], b: Sequence[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    """Welch (unequal-variance) interval for the difference of two means, mean(a) - mean(b).

    Used for the two-sample value/productivity delta (independent with-AI vs without-AI
    groups). Welch's approximation is the safe default because the two conditions rarely
    share a variance, and it degrades gracefully at small, unequal sample sizes.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.0, 0.0, 0.0
    ma, mb = sum(a) / na, sum(b) / nb
    diff = ma - mb
    if na < 2 or nb < 2:
        return diff, diff, diff
    va, vb = stats.tvar(a), stats.tvar(b)
    se2 = va / na + vb / nb
    if se2 <= 0:
        return diff, diff, diff
    se = math.sqrt(se2)
    df = se2**2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t = stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, df=df)
    return diff, diff - t * se, diff + t * se


def proportion_mean_ci(
    fractions: Sequence[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    """t-interval for the mean of *continuous* per-unit proportions in [0, 1].

    Used for things like content-retention rate, where each session contributes
    a fraction (not a binary success), so an adjusted-Wald binomial interval
    would be the wrong model. The interval is computed on the mean and clamped
    to the [0, 1] bound.
    """
    mean, low, high = mean_ci(fractions, confidence)
    return mean, max(0.0, low), min(1.0, high)


def geometric_mean_ci(
    values: Sequence[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    """Geometric mean and CI for right-skewed data such as task times.

    The interval is computed on the log scale (where times are roughly normal)
    and exponentiated back, following the Sauro & Lewis recommendation for
    summarising completion times from small samples.
    """
    positive = [v for v in values if v > 0]
    if not positive:
        return 0.0, 0.0, 0.0
    logs = [math.log(v) for v in positive]
    log_mean, log_low, log_high = mean_ci(logs, confidence)
    return math.exp(log_mean), math.exp(log_low), math.exp(log_high)
