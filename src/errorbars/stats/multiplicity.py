"""Holm-Bonferroni step-down multiplicity correction.

Testing eight interventions at alpha = 0.05 gives roughly a one-in-three chance that at
least one of them looks significant when none of them is. Holm fixes that without the
brutality of plain Bonferroni: it controls the family-wise error rate under *any*
dependence structure, and it is uniformly more powerful than Bonferroni.

The procedure, in adjusted-p-value form. Sort the m raw p-values ascending, multiply the
i-th by (m - i + 1), then take a running maximum so the sequence is non-decreasing (the
"step-down" enforcement -- without it you could reject a large p while failing to reject
a smaller one), and clip at 1.

This matches R's ``p.adjust(p, method = "holm")`` exactly; the test suite checks it
against hand-worked examples.

Reference:
    Holm, S. (1979). "A simple sequentially rejective multiple test procedure."
        Scandinavian Journal of Statistics 6(2):65-70.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

__all__ = ["bonferroni_adjust", "holm_adjust", "holm_reject"]


def _validate(p_values: Sequence[float] | npt.ArrayLike) -> npt.NDArray[np.float64]:
    array = np.asarray(p_values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"p_values must be one-dimensional, got shape {array.shape}")
    if array.size:
        out_of_range = bool(np.any(array < 0.0) or np.any(array > 1.0))
        if out_of_range or not np.all(np.isfinite(array)):
            raise ValueError("p_values must all be finite and in [0, 1]")
    return array


def holm_adjust(p_values: Sequence[float] | npt.ArrayLike) -> list[float]:
    """Holm-adjusted p-values, returned in the input order.

    Comparing every adjusted value against a single alpha is equivalent to running the
    sequential procedure, which is why adjusted p-values are the friendlier reporting
    form: the reader does not have to know the family size to interpret them.
    """
    raw = _validate(p_values)
    m = raw.size
    if m == 0:
        return []

    order = np.argsort(raw, kind="stable")
    ranked = raw[order]
    multipliers = np.arange(m, 0, -1, dtype=np.float64)
    stepped = np.maximum.accumulate(ranked * multipliers)

    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = np.minimum(stepped, 1.0)
    return [float(value) for value in adjusted]


def bonferroni_adjust(p_values: Sequence[float] | npt.ArrayLike) -> list[float]:
    """Bonferroni-adjusted p-values (``min(1, m * p)``), for comparison only.

    Holm dominates this: it never rejects less and often rejects more, at the same FWER.
    Kept because "Bonferroni would have said X" is a question people ask.
    """
    raw = _validate(p_values)
    m = raw.size
    return [float(min(1.0, m * value)) for value in raw]


def holm_reject(
    p_values: Sequence[float] | npt.ArrayLike,
    alpha: float = 0.05,
) -> list[bool]:
    """Reject/retain decisions at family-wise error rate ``alpha``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    return [value <= alpha for value in holm_adjust(p_values)]
