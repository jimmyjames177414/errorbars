"""Power and minimum-detectable-effect analysis for paired, repeated LLM evals.

This is the flagship calculation: it answers "what can a run of this size actually see?"
before any money is spent.

The variance model
------------------
For arm ``a`` (control or treatment), item ``i``, repeat ``j``, the scored outcome is

    X[a,i,j] ~ Bernoulli(pi[a,i])

where ``pi[a,i]`` is that item's true success probability in that arm. Two nuisance
parameters describe the correlation structure:

``icc`` (rho_w)
    Intraclass correlation of repeats *within* an item, i.e. the share of outcome
    variance that is between-item rather than within-item::

        icc = Var(pi[a,i]) / (p * (1 - p))

    ``icc = 0`` means every item is equally hard and repeats are independent coin
    flips, so repeats buy full extra sample size. ``icc = 1`` means every item is
    deterministic, so repeats buy nothing at all. Real evals sit in between and
    usually high, because most items are either always-right or always-wrong.

``pair_corr`` (rho_pair)
    Correlation between an item's difficulty in the two arms, ``corr(pi[t,i], pi[c,i])``.
    This is what pairing exploits: if hard items are hard in both arms, the paired
    difference cancels that shared difficulty out.

Working through the algebra for the paired difference of arm means
``D = mean(y[t]) - mean(y[c])`` where ``y[a,i]`` is item ``i``'s mean over ``m`` repeats::

    Var(D) = (p_c*q_c + p_t*q_t) * k / n_items

    k(m, icc, pair_corr) = icc * (1 - pair_corr) + (1 - icc) / m

``k`` is the *design factor*. Note the two boundary cases:

* ``m = 1, pair_corr = 0`` gives ``k = 1``, and the required-N expression below
  collapses **exactly** to the textbook unpaired two-proportion sample size formula.
  :func:`required_items` is tested against hand-computed values of that formula.
* ``pair_corr = 0`` gives ``k = (1 + (m-1)*icc) / m = design_effect / m``, the classical
  cluster design effect for ``m`` correlated observations per cluster.

Required N then follows the standard structure (null-hypothesis variance pooled,
alternative-hypothesis variance unpooled -- the Fleiss form without the continuity
correction)::

    n_items = k * ( z_{1-alpha/2}*sqrt(2*p_bar*q_bar)
                    + z_{1-beta}*sqrt(p_c*q_c + p_t*q_t) )^2 / delta^2

What this model does NOT do
---------------------------
* It is a normal approximation. At very small n, or baselines within a couple of
  standard errors of 0 or 1, treat the answers as indicative.
* It is not a mixed-effects model. ``icc`` and ``pair_corr`` are two scalars standing
  in for a full random-effects structure.
* It assumes a balanced design: every item gets the same number of repeats in both arms.

Defaults are deliberately conservative: ``pair_corr = 0.0`` assumes pairing buys
*nothing*, so the reported MDE is never optimistic. Measure both parameters from a real
run with ``errorbars analyze`` and feed them back in.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from .normal import z_critical, z_quantile

__all__ = [
    "PowerResult",
    "design_effect",
    "design_factor",
    "effective_n",
    "minimum_detectable_effect",
    "power_analysis",
    "required_items",
    "required_n_two_proportion",
]

# Effects below this are treated as unreachable; searching further wastes time and the
# answer ("you need an implausible number of items") is the same either way.
_MIN_SEARCHABLE_EFFECT = 1e-6


def _check_proportion(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be strictly inside (0, 1), got {value!r}")


def _check_unit(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def design_effect(repeats: int, icc: float) -> float:
    """Classical cluster design effect for ``repeats`` correlated observations per item.

        DEFF = 1 + (repeats - 1) * icc

    Always >= 1: correlated repeats can never carry *more* information than the same
    number of independent observations.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats!r}")
    _check_unit(icc, "icc")
    return 1.0 + (repeats - 1) * icc


def design_factor(repeats: int, icc: float, pair_corr: float) -> float:
    """Variance multiplier ``k`` for the paired difference (see module docstring).

    Smaller is better: ``k`` is the factor by which per-item variance is multiplied,
    so ``n_items / k`` is the effective per-arm sample size.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats!r}")
    _check_unit(icc, "icc")
    _check_unit(pair_corr, "pair_corr")
    return icc * (1.0 - pair_corr) + (1.0 - icc) / repeats


def effective_n(n_items: int, repeats: int, icc: float, pair_corr: float = 0.0) -> float:
    """Independent-equivalent observations per arm.

    A run of ``n_items`` items at ``repeats`` repeats carries as much information about
    the paired difference as an unpaired run of this many independent observations per
    arm. With ``pair_corr > 0`` it can exceed ``n_items * repeats``, because pairing
    removes between-item variance that an unpaired run would have to pay for.
    """
    if n_items < 1:
        raise ValueError(f"n_items must be >= 1, got {n_items!r}")
    return n_items / design_factor(repeats, icc, pair_corr)


def required_items(
    baseline: float,
    effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    repeats: int = 1,
    icc: float = 0.0,
    pair_corr: float = 0.0,
) -> int:
    """Items needed per arm to detect ``effect`` against ``baseline``.

    The test is two-sided, but **the sign of ``effect`` still matters**, because the
    alternative-hypothesis variance is evaluated at ``baseline + effect``. Detecting a
    5-point gain from 80% and a 5-point loss from 80% are not the same problem: 85% has
    less binomial variance than 75%, so the gain needs a smaller sample. Passing a
    magnitude and hoping is how a sample-size calculation ends up 20% wrong.

    Args:
        baseline: control-arm success rate, in (0, 1).
        effect: **signed** difference to detect, as a proportion (0.02 == a 2 percentage
            point gain, -0.02 == a 2 point loss).
        alpha: two-sided significance level.
        power: target power (1 - beta).
        repeats: repeats per item per arm.
        icc: intraclass correlation of repeats within an item.
        pair_corr: correlation of item difficulty across arms.

    Returns:
        Number of items, rounded up. Both arms use these same items (that is what
        "paired" means), so this is the size of the item set, not double it.

    Raises:
        ValueError: on out-of-range inputs, or if ``baseline + effect`` leaves (0, 1).
    """
    _check_proportion(baseline, "baseline")
    magnitude = abs(effect)
    if magnitude <= 0.0:
        raise ValueError(
            "effect must be non-zero; detecting an effect of exactly 0 needs infinite N"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power!r}")

    p_c = baseline
    p_t = baseline + effect
    if not 0.0 < p_t < 1.0:
        raise ValueError(
            f"baseline {baseline!r} with effect {effect!r} implies a treatment rate of "
            f"{p_t!r}, which is outside (0, 1); no such alternative exists"
        )

    z_alpha = z_critical(alpha, two_sided=True)
    z_beta = z_quantile(power)

    p_bar = (p_c + p_t) / 2.0
    var_null = 2.0 * p_bar * (1.0 - p_bar)
    var_alt = p_c * (1.0 - p_c) + p_t * (1.0 - p_t)

    numerator = (z_alpha * math.sqrt(var_null) + z_beta * math.sqrt(var_alt)) ** 2
    k = design_factor(repeats, icc, pair_corr)

    return math.ceil(k * numerator / (magnitude * magnitude))


def required_n_two_proportion(
    p1: float,
    p2: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Classical unpaired two-proportion sample size, **per group**.

        n = ( z_{1-a/2}*sqrt(2*p_bar*q_bar) + z_{1-b}*sqrt(p1*q1 + p2*q2) )^2 / (p2-p1)^2

    This is the textbook formula every online calculator implements; it exists here as
    the reference case that :func:`required_items` must reproduce at ``repeats=1,
    pair_corr=0``, and it is what the test suite checks against hand computation.
    """
    _check_proportion(p1, "p1")
    _check_proportion(p2, "p2")
    if p1 == p2:
        raise ValueError("p1 and p2 must differ")
    return required_items(p1, p2 - p1, alpha=alpha, power=power, repeats=1, icc=0.0, pair_corr=0.0)


def _mde_one_direction(
    baseline: float,
    n_items: int,
    sign: float,
    *,
    alpha: float,
    power: float,
    repeats: int,
    icc: float,
    pair_corr: float,
) -> float:
    """Bisection for the MDE in a single direction. See :func:`minimum_detectable_effect`."""

    def needed(magnitude: float) -> int:
        return required_items(
            baseline,
            sign * magnitude,
            alpha=alpha,
            power=power,
            repeats=repeats,
            icc=icc,
            pair_corr=pair_corr,
        )

    # Largest magnitude that keeps the alternative inside (0, 1) in this direction.
    reachable = (1.0 - baseline) if sign > 0 else baseline
    high = reachable * 0.999999
    if needed(high) > n_items:
        return math.nan

    low = _MIN_SEARCHABLE_EFFECT
    if needed(low) <= n_items:
        return low

    # Invariant: needed(low) > n_items >= needed(high).
    for _ in range(200):
        mid = (low + high) / 2.0
        if needed(mid) <= n_items:
            high = mid
        else:
            low = mid
        if high - low < 1e-12:
            break
    return high


def minimum_detectable_effect(
    baseline: float,
    n_items: int,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    repeats: int = 1,
    icc: float = 0.0,
    pair_corr: float = 0.0,
    direction: str = "either",
) -> float:
    """Smallest absolute effect this design can detect, as a proportion.

    Inverts :func:`required_items` by bisection -- monotonically decreasing in the
    effect size, so the inversion is well-posed, and there is no closed form because
    the alternative-hypothesis variance itself depends on the effect.

    Args:
        direction: ``"up"``, ``"down"``, or ``"either"`` (default). The two directions
            genuinely differ, because the alternative's binomial variance differs: from
            a baseline of 80%, a gain to 85% is easier to detect than a loss to 75%.
            ``"either"`` returns the **larger** of the two, i.e. the smallest effect
            that this design would catch *whichever way it went*. That is the number
            you want when you do not know the sign in advance, which is almost always.

    Returns:
        The MDE as a proportion (0.028 == 2.8 percentage points), or ``nan`` if no
        effect is detectable at this N -- which means the design is hopeless, not that
        the answer is small.
    """
    _check_proportion(baseline, "baseline")
    if n_items < 1:
        raise ValueError(f"n_items must be >= 1, got {n_items!r}")
    if direction not in ("up", "down", "either"):
        raise ValueError(f"direction must be 'up', 'down' or 'either', got {direction!r}")

    kwargs = {
        "alpha": alpha,
        "power": power,
        "repeats": repeats,
        "icc": icc,
        "pair_corr": pair_corr,
    }
    if direction == "up":
        return _mde_one_direction(baseline, n_items, 1.0, **kwargs)  # type: ignore[arg-type]
    if direction == "down":
        return _mde_one_direction(baseline, n_items, -1.0, **kwargs)  # type: ignore[arg-type]

    up = _mde_one_direction(baseline, n_items, 1.0, **kwargs)  # type: ignore[arg-type]
    down = _mde_one_direction(baseline, n_items, -1.0, **kwargs)  # type: ignore[arg-type]
    if math.isnan(up) or math.isnan(down):
        return math.nan
    return max(up, down)


class PowerResult(NamedTuple):
    """Everything ``errorbars power`` reports, all computed from the inputs."""

    baseline: float
    n_items: int
    repeats: int
    icc: float
    pair_corr: float
    alpha: float
    power: float
    design_effect: float
    design_factor: float
    effective_n: float
    mde: float
    target_effect: float | None
    items_for_target: int | None


def power_analysis(
    baseline: float,
    n_items: int,
    *,
    repeats: int = 1,
    icc: float = 0.0,
    pair_corr: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.80,
    target_effect: float | None = None,
) -> PowerResult:
    """Full power picture for one design. Pure arithmetic, no I/O, no network."""
    mde = minimum_detectable_effect(
        baseline,
        n_items,
        alpha=alpha,
        power=power,
        repeats=repeats,
        icc=icc,
        pair_corr=pair_corr,
    )
    items_for_target: int | None = None
    if target_effect is not None:
        # Report the harder direction, for the same reason `direction="either"` is the
        # MDE default: you rarely know the sign before you run, and being told you need
        # fewer items than you do is the expensive way to be wrong.
        magnitude = abs(target_effect)
        candidates = []
        for signed in (magnitude, -magnitude):
            if 0.0 < baseline + signed < 1.0:
                candidates.append(
                    required_items(
                        baseline,
                        signed,
                        alpha=alpha,
                        power=power,
                        repeats=repeats,
                        icc=icc,
                        pair_corr=pair_corr,
                    )
                )
        if not candidates:
            raise ValueError(
                f"an effect of {magnitude!r} in either direction leaves (0, 1) from a "
                f"baseline of {baseline!r}"
            )
        items_for_target = max(candidates)

    return PowerResult(
        baseline=baseline,
        n_items=n_items,
        repeats=repeats,
        icc=icc,
        pair_corr=pair_corr,
        alpha=alpha,
        power=power,
        design_effect=design_effect(repeats, icc),
        design_factor=design_factor(repeats, icc, pair_corr),
        effective_n=effective_n(n_items, repeats, icc, pair_corr),
        mde=mde,
        target_effect=target_effect,
        items_for_target=items_for_target,
    )
