"""Estimating the two correlation parameters that power analysis needs.

``errorbars power`` has to assume values for the intraclass correlation of repeats
(``icc``) and the across-arm correlation of item difficulty (``pair_corr``). These
functions measure both from a run you have already done, which closes the loop: analyse
a pilot, feed the measured values back into ``power``, size the real run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

__all__ = ["intraclass_correlation", "pair_correlation"]


def intraclass_correlation(groups: Sequence[npt.ArrayLike]) -> float:
    """One-way random-effects ICC(1) from repeated observations per item.

        ICC = (MSB - MSW) / (MSB + (m0 - 1) * MSW)

    where MSB and MSW are the between- and within-item mean squares and ``m0`` is the
    Satterthwaite-style average group size that makes the estimator unbiased for an
    unbalanced design.

    Args:
        groups: one sequence of observations per item. Groups of size 1 contribute to
            the between-item term but carry no within-item information.

    Returns:
        The estimate clipped to [0, 1], or ``nan`` when it is not identifiable --
        which is the case whenever there are fewer than two items, no item has more
        than one observation (i.e. the run had no repeats), or every observation is
        identical. ``nan`` here means "this run cannot tell you", not "zero".

    Reference:
        Shrout, P. E. & Fleiss, J. L. (1979). "Intraclass correlations: uses in
            assessing rater reliability." Psychological Bulletin 86(2):420-428.
    """
    arrays = [np.asarray(group, dtype=np.float64).ravel() for group in groups]
    arrays = [array for array in arrays if array.size > 0]

    k = len(arrays)
    if k < 2:
        return math.nan

    sizes = np.array([array.size for array in arrays], dtype=np.float64)
    total = float(sizes.sum())
    if total <= k:  # every group has exactly one observation: no within-item variance
        return math.nan

    group_means = np.array([float(array.mean()) for array in arrays], dtype=np.float64)
    grand_mean = float(sum(float(array.sum()) for array in arrays) / total)

    ss_between = float(np.sum(sizes * (group_means - grand_mean) ** 2))
    ss_within = float(sum(float(np.sum((array - array.mean()) ** 2)) for array in arrays))

    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (total - k)

    m0 = (total - float(np.sum(sizes**2)) / total) / (k - 1)
    denominator = ms_between + (m0 - 1.0) * ms_within
    if denominator <= 0.0:
        # Happens when every observation is identical (both mean squares are zero).
        return math.nan

    return float(min(1.0, max(0.0, (ms_between - ms_within) / denominator)))


def pair_correlation(
    control_item_means: npt.ArrayLike,
    treatment_item_means: npt.ArrayLike,
) -> float:
    """Pearson correlation of per-item scores across the two arms.

    This is an *attenuated* estimate of the quantity ``power(pair_corr=...)`` wants.
    Per-item means are noisy estimates of an item's true difficulty, and measurement
    noise pulls a correlation toward zero, so the observed value is a lower bound on
    the true across-arm difficulty correlation. Feeding it into ``errorbars power``
    therefore understates how much pairing helps, which is the safe direction.
    ``docs/statistics.md`` gives the disattenuation formula if you want the other one.

    Returns:
        The correlation, or ``nan`` if either arm has no variance across items (which
        happens when every item scores the same, e.g. a run that got everything right).
    """
    control = np.asarray(control_item_means, dtype=np.float64).ravel()
    treatment = np.asarray(treatment_item_means, dtype=np.float64).ravel()
    if control.shape != treatment.shape:
        raise ValueError(
            f"arms must be aligned per item; got {control.shape} and {treatment.shape}"
        )
    if control.size < 2:
        return math.nan
    if control.std() == 0.0 or treatment.std() == 0.0:
        return math.nan
    return float(np.corrcoef(control, treatment)[0, 1])
