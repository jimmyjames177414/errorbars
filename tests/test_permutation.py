"""Does the test reject a true null exactly as often as it promises?

A test at alpha = 0.05 promises to produce a false positive 5% of the time when nothing
is going on. That is checkable the same way coverage is: generate data with no effect,
run the test, count rejections.

Two null cases, because they fail differently:

* **Continuous differences.** No ties, so the permutation p-value is exactly uniform on
  its grid and the false-positive rate should land on alpha almost exactly. This is the
  case that catches an off-by-one in the p-value formula -- drop the ``+1`` from
  ``(b+1)/(B+1)`` and the rate creeps above alpha.
* **Binary scores.** Ties everywhere, so the test is *conservative* and rejects less
  often than alpha. That is the safe direction, and asserting it stays on the safe side
  is the real check.
"""

from __future__ import annotations

import numpy as np
import pytest

from errorbars.stats.permutation import paired_permutation_test

ALPHA = 0.05


@pytest.mark.slow
def test_false_positive_rate_matches_alpha_on_continuous_nulls() -> None:
    """2000 simulations, SE = sqrt(.05*.95/2000) = 0.0049, so +/-3 SE is [0.035, 0.065]."""
    simulations = 2000
    n_items = 40
    permutations = 499
    rng = np.random.default_rng(101)
    rejections = 0

    for index in range(simulations):
        control = rng.normal(0.0, 1.0, size=n_items)
        # Same distribution, independent draw: the true effect is exactly zero.
        treatment = rng.normal(0.0, 1.0, size=n_items)
        result = paired_permutation_test(control, treatment, permutations=permutations, seed=index)
        rejections += result.p_value <= ALPHA

    rate = rejections / simulations
    print(f"\ncontinuous-null false positive rate: {rate:.3f} over {simulations} simulations")
    assert 0.035 <= rate <= 0.065, f"false positive rate {rate:.3f} is not close to alpha={ALPHA}"


@pytest.mark.slow
def test_false_positive_rate_stays_at_or_below_alpha_on_binary_nulls() -> None:
    """Paired 0/1 scores with shared item difficulty and no treatment effect.

    Ties make the test conservative here. Conservative is acceptable; anti-conservative
    is not, so the upper bound is the assertion that matters.
    """
    simulations = 1000
    n_items = 60
    repeats = 3
    permutations = 499
    rng = np.random.default_rng(202)
    rejections = 0

    for index in range(simulations):
        difficulty = rng.uniform(0.2, 0.95, size=n_items)
        control = (rng.random((n_items, repeats)) < difficulty[:, None]).mean(axis=1)
        treatment = (rng.random((n_items, repeats)) < difficulty[:, None]).mean(axis=1)
        result = paired_permutation_test(control, treatment, permutations=permutations, seed=index)
        rejections += result.p_value <= ALPHA

    rate = rejections / simulations
    print(f"\nbinary-null false positive rate: {rate:.3f} over {simulations} simulations")
    assert rate <= 0.065, f"false positive rate {rate:.3f} exceeds alpha={ALPHA}"
    assert rate >= 0.005, (
        f"false positive rate {rate:.3f} is so far below alpha that the test is probably "
        "not rejecting at all"
    )


def test_p_value_can_never_be_zero() -> None:
    """A Monte-Carlo p-value of 0 is not a valid p-value (Phipson & Smyth 2010)."""
    control = np.zeros(50)
    treatment = np.ones(50)  # about as extreme as paired data gets
    result = paired_permutation_test(control, treatment, permutations=200, seed=1)
    assert result.p_value > 0.0
    assert result.p_value == pytest.approx(1 / 201)


def test_detects_a_large_real_effect() -> None:
    rng = np.random.default_rng(9)
    control = rng.normal(0.0, 1.0, size=80)
    treatment = control + 0.8
    result = paired_permutation_test(control, treatment, permutations=2000, seed=3)
    assert result.p_value < 0.001
    assert result.observed == pytest.approx(0.8, abs=1e-9)


def test_small_item_sets_are_enumerated_exactly() -> None:
    """With 2**n <= permutations the answer is exact, not sampled."""
    control = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    treatment = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    result = paired_permutation_test(control, treatment, permutations=1000, seed=0)
    assert result.exact
    assert result.permutations == 2**6
    # Only the all-plus and all-minus sign vectors are as extreme as the observed one.
    assert result.p_value == pytest.approx(2 / 64)


def test_exact_result_does_not_depend_on_the_seed() -> None:
    control = np.array([0.2, 0.4, 0.1, 0.9, 0.5])
    treatment = np.array([0.5, 0.4, 0.3, 0.8, 0.9])
    first = paired_permutation_test(control, treatment, permutations=64, seed=1)
    second = paired_permutation_test(control, treatment, permutations=64, seed=99)
    assert first.exact and second.exact
    assert first.p_value == second.p_value


def test_one_sided_alternatives_split_the_evidence() -> None:
    rng = np.random.default_rng(13)
    control = rng.normal(0.0, 1.0, size=60)
    treatment = control + 0.5

    greater = paired_permutation_test(
        control, treatment, permutations=2000, alternative="greater", seed=4
    )
    less = paired_permutation_test(
        control, treatment, permutations=2000, alternative="less", seed=4
    )
    assert greater.p_value < 0.01
    assert less.p_value > 0.99


def test_all_zero_differences_give_a_p_value_of_one() -> None:
    """Identical arms are the strongest possible evidence of nothing happening."""
    values = np.array(
        [
            0.0,
            1.0,
            0.5,
            0.25,
            1.0,
            0.0,
            0.75,
            0.5,
            0.0,
            1.0,
            0.5,
            0.25,
            0.0,
            1.0,
            0.5,
            0.25,
            1.0,
            0.0,
            0.75,
            0.5,
        ]
    )
    result = paired_permutation_test(values, values, permutations=500, seed=0)
    assert result.p_value == pytest.approx(1.0)


def test_seed_makes_the_sampled_test_reproducible() -> None:
    rng = np.random.default_rng(77)
    control = rng.normal(size=100)
    treatment = rng.normal(size=100)
    first = paired_permutation_test(control, treatment, permutations=500, seed=5)
    second = paired_permutation_test(control, treatment, permutations=500, seed=5)
    assert first.p_value == second.p_value


def test_rejects_misaligned_arms() -> None:
    with pytest.raises(ValueError, match="aligned"):
        paired_permutation_test([1.0, 2.0, 3.0], [1.0, 2.0])


def test_rejects_unknown_alternative() -> None:
    with pytest.raises(ValueError, match="alternative"):
        paired_permutation_test([1.0, 2.0], [2.0, 3.0], alternative="sideways")  # type: ignore[arg-type]
