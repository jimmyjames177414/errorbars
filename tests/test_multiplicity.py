"""Holm-Bonferroni, checked against worked examples from outside this repository.

Two published references:

* R's ``stats::p.adjust(c(0.01, 0.04, 0.03, 0.005), method = "holm")`` returns
  ``0.030 0.060 0.060 0.020``. R's implementation is the de-facto reference for
  multiplicity adjustment, so agreeing with it is worth more than any internal check.
* Wikipedia's Holm-Bonferroni worked example uses p = 0.01, 0.04, 0.03 at alpha = 0.05
  and rejects only the first hypothesis.

Plus a simulation that the family-wise error rate is actually controlled, which is the
thing the procedure exists to do.
"""

from __future__ import annotations

import numpy as np
import pytest

from errorbars.stats.multiplicity import bonferroni_adjust, holm_adjust, holm_reject


def test_matches_r_p_adjust_holm() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_matches_the_wikipedia_worked_example() -> None:
    """p = 0.01, 0.04, 0.03 at alpha = 0.05: only the first hypothesis is rejected."""
    raw = [0.01, 0.04, 0.03]
    assert holm_adjust(raw) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_reject(raw, alpha=0.05) == [True, False, False]


def test_returns_values_in_the_input_order() -> None:
    """The smallest p is last here; the adjustment must not silently sort the output.

    By hand: sorted 0.001, 0.5, 0.9 -> multipliers 3, 2, 1 -> 0.003, 1.0, 0.9 -> running
    maximum 0.003, 1.0, 1.0 -> back in the original order [1.0, 1.0, 0.003]. The two
    large values are both capped at 1, which is exactly why the position of 0.003 is
    what proves the ordering was preserved.
    """
    assert holm_adjust([0.9, 0.5, 0.001]) == pytest.approx([1.0, 1.0, 0.003])


def test_adjusted_is_never_smaller_than_raw() -> None:
    rng = np.random.default_rng(3)
    for _ in range(200):
        raw = list(rng.random(size=int(rng.integers(1, 12))))
        for original, adjusted in zip(raw, holm_adjust(raw), strict=True):
            assert adjusted >= original - 1e-12


def test_adjusted_is_never_larger_than_bonferroni() -> None:
    """Holm dominates Bonferroni; if it ever did not, there would be no reason to use it."""
    rng = np.random.default_rng(4)
    for _ in range(200):
        raw = list(rng.random(size=int(rng.integers(2, 12))))
        for holm, bonferroni in zip(holm_adjust(raw), bonferroni_adjust(raw), strict=True):
            assert holm <= bonferroni + 1e-12


def test_adjusted_values_are_monotone_in_the_raw_ordering() -> None:
    """Step-down enforcement: a larger raw p can never get a smaller adjusted p."""
    rng = np.random.default_rng(5)
    raw = list(rng.random(size=10))
    pairs = sorted(zip(raw, holm_adjust(raw), strict=True))
    adjusted_in_order = [adjusted for _, adjusted in pairs]
    assert adjusted_in_order == sorted(adjusted_in_order)


def test_single_hypothesis_is_a_no_op() -> None:
    assert holm_adjust([0.023]) == pytest.approx([0.023])


def test_empty_family_is_empty() -> None:
    assert holm_adjust([]) == []


def test_adjusted_values_are_capped_at_one() -> None:
    assert holm_adjust([0.5, 0.6, 0.7]) == pytest.approx([1.0, 1.0, 1.0])


def test_identical_p_values_all_get_the_largest_multiplier() -> None:
    """Ties must all be treated as the least significant, not silently ordered."""
    assert holm_adjust([0.02, 0.02, 0.02]) == pytest.approx([0.06, 0.06, 0.06])


@pytest.mark.slow
def test_family_wise_error_rate_is_controlled_under_the_global_null() -> None:
    """Eight independent true nulls: at least one raw p under 0.05 about a third of the
    time, but the Holm-adjusted family should reject at most 5% of the time."""
    simulations = 4000
    family_size = 8
    rng = np.random.default_rng(6)

    raw_any = 0
    holm_any = 0
    for _ in range(simulations):
        p_values = rng.random(size=family_size)  # uniform under the null
        raw_any += bool(np.any(p_values <= 0.05))
        holm_any += any(holm_reject(list(p_values), alpha=0.05))

    raw_rate = raw_any / simulations
    holm_rate = holm_any / simulations
    print(f"\nuncorrected family-wise error: {raw_rate:.3f}; Holm: {holm_rate:.3f}")
    assert raw_rate > 0.25, "sanity: uncorrected testing of 8 nulls should fire often"
    assert holm_rate <= 0.065, f"Holm did not control FWER: {holm_rate:.3f}"


def test_rejects_p_values_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_adjust([0.5, 1.5])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_adjust([-0.1, 0.5])


def test_rejects_a_bad_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        holm_reject([0.01], alpha=1.5)
