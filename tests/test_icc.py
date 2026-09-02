"""Estimating the correlation parameters that power analysis has to assume.

The ICC estimator is checked two ways: against a hand-computable case where the answer
is forced by construction, and by recovering a known value from simulated data. The
boundary behaviour matters as much as the middle -- an ICC of nan means "this run cannot
tell you", and returning 0.0 instead would quietly credit repeats with information they
never provided.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from errorbars.stats.icc import intraclass_correlation, pair_correlation


def test_identical_repeats_within_each_item_give_icc_one() -> None:
    """Deterministic items: all variance is between items, so repeats add nothing."""
    groups = [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    assert intraclass_correlation(groups) == pytest.approx(1.0)


def test_no_between_item_variance_gives_icc_zero() -> None:
    """Every item equally hard: repeats are independent coin flips and worth full value."""
    rng = np.random.default_rng(0)
    groups = [list(rng.binomial(1, 0.5, size=30).astype(float)) for _ in range(200)]
    assert intraclass_correlation(groups) < 0.05


@pytest.mark.slow
def test_recovers_a_known_icc_from_simulated_data() -> None:
    """Draw item difficulties with a known between-item variance and read the ICC back.

    For Bernoulli outcomes, icc = Var(pi_i) / (p * (1 - p)) where pi_i is an item's true
    success probability. Both quantities are known here because the difficulties were
    drawn from a distribution we chose.
    """
    rng = np.random.default_rng(42)
    n_items, repeats = 4000, 8
    difficulty = rng.beta(2.0, 2.0, size=n_items)  # mean 0.5, variance 0.05

    groups = [list((rng.random(repeats) < probability).astype(float)) for probability in difficulty]
    estimate = intraclass_correlation(groups)

    p = float(difficulty.mean())
    expected = float(difficulty.var()) / (p * (1.0 - p))
    print(f"\nICC estimate {estimate:.4f}, population value {expected:.4f}")
    assert estimate == pytest.approx(expected, abs=0.02)


def test_single_repeat_per_item_is_not_identifiable() -> None:
    """With one observation per item there is no within-item variance to measure."""
    groups = [[1.0], [0.0], [1.0], [1.0], [0.0]]
    assert math.isnan(intraclass_correlation(groups))


def test_fewer_than_two_items_is_not_identifiable() -> None:
    assert math.isnan(intraclass_correlation([]))
    assert math.isnan(intraclass_correlation([[1.0, 0.0, 1.0]]))


def test_constant_data_is_not_identifiable() -> None:
    """Everything identical: there is no variance to partition, so the answer is nan."""
    assert math.isnan(intraclass_correlation([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]))


def test_result_is_always_inside_the_unit_interval() -> None:
    rng = np.random.default_rng(8)
    for _ in range(50):
        groups = [list(rng.random(size=int(rng.integers(2, 6)))) for _ in range(20)]
        value = intraclass_correlation(groups)
        assert math.isnan(value) or 0.0 <= value <= 1.0


def test_unbalanced_groups_are_handled() -> None:
    """Ragged repeat counts must not crash or silently drop items."""
    groups = [[1.0, 1.0], [0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    value = intraclass_correlation(groups)
    assert not math.isnan(value)
    assert 0.0 <= value <= 1.0


def test_pair_correlation_recovers_a_perfect_relationship() -> None:
    control = [0.1, 0.4, 0.6, 0.9, 0.3]
    assert pair_correlation(control, control) == pytest.approx(1.0)


def test_pair_correlation_is_zero_for_unrelated_arms() -> None:
    rng = np.random.default_rng(21)
    control = rng.random(size=5000)
    treatment = rng.random(size=5000)
    assert abs(pair_correlation(control, treatment)) < 0.05


def test_pair_correlation_is_nan_without_variance() -> None:
    """A run that got everything right has no across-item variance to correlate."""
    assert math.isnan(pair_correlation([1.0] * 10, [0.5, 0.6] * 5))


def test_pair_correlation_rejects_misaligned_arms() -> None:
    with pytest.raises(ValueError, match="aligned"):
        pair_correlation([1.0, 2.0, 3.0], [1.0, 2.0])
