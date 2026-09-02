"""Bootstrap confidence intervals, resampling **items**.

Why items and not trials
------------------------
Items are the independent unit of an eval. Trials on the same item are correlated --
they share whatever makes that item easy or hard -- so resampling trials treats
correlated observations as independent and produces an interval that is too narrow.
Resample items, and let each resampled item carry all of its repeats with it.

Why the percentile method
-------------------------
The percentile interval is the one that survives contact with the shapes an eval
produces: bounded scores, heavy mass at 0 and 1, and paired differences with a spike at
exactly zero. BCa would be a refinement, but its acceleration term is estimated from a
jackknife that is itself unstable on those spiky distributions, and the added machinery
would be harder to test than it is worth at v0.1. The percentile interval's coverage is
verified by simulation in ``tests/test_bootstrap.py``.

Reference:
    Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*, ch. 13.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from .proportions import Interval

__all__ = ["DEFAULT_RESAMPLES", "bootstrap_mean_ci", "paired_bootstrap_ci"]

DEFAULT_RESAMPLES = 10_000

# Cap on index elements held at once, so a large item set does not allocate a
# (resamples x n_items) integer array all in one go.
_MAX_INDEX_ELEMENTS = 4_000_000


def _as_1d(values: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _bootstrap_means(
    values: npt.NDArray[np.float64],
    resamples: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Means of ``resamples`` bootstrap resamples, computed in memory-bounded chunks."""
    n = values.size
    chunk = max(1, min(resamples, _MAX_INDEX_ELEMENTS // max(n, 1)))
    out = np.empty(resamples, dtype=np.float64)
    done = 0
    while done < resamples:
        size = min(chunk, resamples - done)
        idx = rng.integers(0, n, size=(size, n))
        out[done : done + size] = values[idx].mean(axis=1)
        done += size
    return out


def bootstrap_mean_ci(
    values: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int | None = None,
) -> Interval:
    """Percentile bootstrap CI for the mean of ``values`` (one value per item).

    Args:
        values: per-item scores. For a repeated design, pass each item's mean over
            its repeats -- that keeps the item as the resampling unit.
        alpha: two-sided level; a 95% interval is ``alpha=0.05``.
        resamples: number of bootstrap resamples.
        seed: RNG seed. Pass one if you want the interval to be reproducible; the
            bootstrap is a Monte-Carlo procedure and will otherwise move slightly
            between runs.
    """
    array = _as_1d(values, "values")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples!r}")

    rng = np.random.default_rng(seed)
    means = _bootstrap_means(array, resamples, rng)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    # The spread of the bootstrap distribution is the bootstrap standard error. It is
    # returned alongside the interval because the verdict logic needs to know what
    # precision this run actually achieved, not what a design model predicted.
    se = float(means.std(ddof=1)) if resamples > 1 else math.nan
    return Interval(point=float(array.mean()), low=float(low), high=float(high), se=se)


def paired_bootstrap_ci(
    control: npt.ArrayLike,
    treatment: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int | None = None,
) -> Interval:
    """Percentile bootstrap CI for the paired effect ``mean(treatment - control)``.

    ``control`` and ``treatment`` must be aligned item-by-item: element ``i`` of each is
    the same item's score under the two arms. The resampling unit is the *pair*, which
    is what makes this a paired analysis -- an item is either in a resample with both of
    its arms, or not in it at all.

    Raises:
        ValueError: if the two arrays differ in length.
    """
    control_array = _as_1d(control, "control")
    treatment_array = _as_1d(treatment, "treatment")
    if control_array.size != treatment_array.size:
        raise ValueError(
            "control and treatment must have one entry per item and be aligned; "
            f"got {control_array.size} and {treatment_array.size}"
        )
    return bootstrap_mean_ci(
        treatment_array - control_array,
        alpha=alpha,
        resamples=resamples,
        seed=seed,
    )
