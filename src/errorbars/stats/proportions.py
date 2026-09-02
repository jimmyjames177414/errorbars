"""Confidence intervals for a single binomial proportion.

Wilson score interval only. The Wald ("normal approximation") interval is deliberately
not offered: at the proportions evals actually live at (0.9+) it produces intervals that
run past 1.0 and has coverage well below nominal for small n. Wilson is the default
recommendation in Brown, Cai & DasGupta (2001) and in Newcombe (1998).

References:
    Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical
        inference." JASA 22(158):209-212.
    Newcombe, R. G. (1998). "Two-sided confidence intervals for the single proportion:
        comparison of seven methods." Statistics in Medicine 17:857-872.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from .normal import z_critical

__all__ = ["Interval", "wilson_interval"]


class Interval(NamedTuple):
    """A closed confidence interval and the point estimate it surrounds."""

    point: float
    low: float
    high: float
    se: float = math.nan
    """Standard error, where the method has one.

    The bootstrap does: it is the standard deviation of the resampled statistics, and
    :mod:`errorbars.stats.effects` uses it to work out what effect this run's precision
    would actually have detected. Wilson does not, because its interval is deliberately
    asymmetric and no single number summarises it.
    """

    @property
    def width(self) -> float:
        return self.high - self.low

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


def wilson_interval(successes: float, n: int, *, alpha: float = 0.05) -> Interval:
    """Wilson score interval for a binomial proportion.

    The interval solves ``|p_hat - p| = z * sqrt(p(1-p)/n)`` for p, rather than
    substituting ``p_hat`` for ``p`` in the standard error (which is what Wald does).

        centre    = (p_hat + z^2/2n) / (1 + z^2/n)
        half-width= z/(1 + z^2/n) * sqrt( p_hat(1-p_hat)/n + z^2/(4n^2) )

    Args:
        successes: number of successes (may be fractional if scores are not 0/1).
        n: number of trials, must be positive.
        alpha: two-sided significance level; the interval has nominal coverage 1-alpha.

    Returns:
        Interval with ``point`` the raw proportion and bounds clipped to [0, 1].

    Raises:
        ValueError: if ``n <= 0`` or ``successes`` is outside ``[0, n]``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n!r}")
    if not 0.0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes!r}")

    z = z_critical(alpha)
    z2 = z * z
    p_hat = successes / n

    denominator = 1.0 + z2 / n
    centre = (p_hat + z2 / (2.0 * n)) / denominator
    half = (z / denominator) * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))

    return Interval(
        point=p_hat,
        low=max(0.0, centre - half),
        high=min(1.0, centre + half),
    )
