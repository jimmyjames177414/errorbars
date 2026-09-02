"""Normal distribution helpers, stdlib only.

scipy is forbidden in this package (see pyproject.toml). Everything here comes from
:class:`statistics.NormalDist`, which is stdlib since Python 3.8 and implements the
Wichura AS241 inverse-CDF algorithm to roughly 1e-15 absolute accuracy -- more than
enough for sample-size arithmetic.
"""

from __future__ import annotations

from statistics import NormalDist

__all__ = ["normal_cdf", "z_critical", "z_quantile"]

_STANDARD_NORMAL = NormalDist(0.0, 1.0)


def z_quantile(p: float) -> float:
    """Return the standard-normal quantile z such that P(Z <= z) == ``p``.

    Raises:
        ValueError: if ``p`` is not strictly inside (0, 1).
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"quantile probability must be in (0, 1), got {p!r}")
    return _STANDARD_NORMAL.inv_cdf(p)


def z_critical(alpha: float, *, two_sided: bool = True) -> float:
    """Critical value for a test at significance ``alpha``.

    With ``two_sided=True`` this is z_{1 - alpha/2} (1.959964... at alpha=0.05);
    otherwise z_{1 - alpha}.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    tail = alpha / 2.0 if two_sided else alpha
    return z_quantile(1.0 - tail)


def normal_cdf(z: float) -> float:
    """Standard-normal CDF."""
    return _STANDARD_NORMAL.cdf(z)
