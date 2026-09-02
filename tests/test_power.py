"""Power arithmetic, checked against values computed by hand.

The three reference sample sizes below were worked out with a calculator from

    n = ( z_{1-a/2}*sqrt(2*p_bar*q_bar) + z_{1-b}*sqrt(p1*q1 + p2*q2) )^2 / (p2-p1)^2

which is the standard two-proportion formula (pooled variance under the null, unpooled
under the alternative; Fleiss, without the continuity correction). They also match the
values published in standard sample-size tables, which is the point: a power calculation
that agrees with nothing outside its own repository is not checked.

The rest of the file pins the properties that make the answers *usable* rather than
merely present: more items detect smaller effects, correlated repeats never help more
than independent ones, and the MDE is a real inverse of the required-N function.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from errorbars.stats.power import (
    design_effect,
    design_factor,
    effective_n,
    minimum_detectable_effect,
    power_analysis,
    required_items,
    required_n_two_proportion,
)

# (p1, p2, expected n per group) at alpha=0.05 two-sided, power=0.80.
HAND_COMPUTED = [
    (0.80, 0.85, 906),
    (0.25, 0.40, 152),
    (0.50, 0.60, 388),
]


@pytest.mark.parametrize(("p1", "p2", "expected"), HAND_COMPUTED)
def test_two_proportion_n_matches_hand_computation(p1: float, p2: float, expected: int) -> None:
    assert required_n_two_proportion(p1, p2) == expected


def test_direction_of_the_effect_changes_the_answer() -> None:
    """80% -> 85% and 80% -> 75% are not the same problem.

    85% carries less binomial variance than 75%, so the gain needs fewer items. A
    calculator that takes the magnitude and throws the sign away gets one of these
    wrong by about 20%.
    """
    gain = required_n_two_proportion(0.80, 0.85)
    loss = required_n_two_proportion(0.80, 0.75)
    assert gain == 906
    assert loss == 1094
    assert loss > gain


def test_required_items_reduces_to_the_two_proportion_formula() -> None:
    """One repeat and no pairing benefit is the classical unpaired design."""
    for p1, p2, expected in HAND_COMPUTED:
        assert required_items(p1, p2 - p1, repeats=1, icc=0.0, pair_corr=0.0) == expected


def test_required_items_decreases_as_the_effect_grows() -> None:
    sizes = [required_items(0.80, effect) for effect in (0.01, 0.02, 0.05, 0.10, 0.15)]
    assert sizes == sorted(sizes, reverse=True)


def test_required_items_grows_when_power_is_raised() -> None:
    assert required_items(0.80, 0.05, power=0.95) > required_items(0.80, 0.05, power=0.80)


def test_required_items_grows_when_alpha_is_tightened() -> None:
    assert required_items(0.80, 0.05, alpha=0.01) > required_items(0.80, 0.05, alpha=0.05)


def test_mde_decreases_monotonically_in_n() -> None:
    """The headline promise of the whole command: bigger runs see smaller effects."""
    mdes = [
        minimum_detectable_effect(0.80, n, repeats=1, icc=0.0)
        for n in (50, 100, 200, 400, 800, 1600, 3200)
    ]
    assert all(math.isfinite(value) for value in mdes)
    assert mdes == sorted(mdes, reverse=True)
    assert all(later < earlier for earlier, later in pairwise(mdes))


def test_a_high_baseline_puts_a_ceiling_on_detectable_improvement() -> None:
    """At 80% there is only 20 points of headroom, and a tiny run cannot even use that.

    25 items cannot detect *any* upward effect from an 80% baseline, because the largest
    one that exists still needs 35 items. ``direction="either"`` reports nan rather than
    quoting the downward answer and letting a reader assume it holds both ways.
    """
    assert math.isnan(minimum_detectable_effect(0.80, 25, direction="up"))
    assert math.isnan(minimum_detectable_effect(0.80, 25, direction="either"))
    assert math.isfinite(minimum_detectable_effect(0.80, 25, direction="down"))
    assert math.isfinite(minimum_detectable_effect(0.80, 50, direction="either"))


def test_mde_decreases_monotonically_in_repeats() -> None:
    mdes = [minimum_detectable_effect(0.80, 200, repeats=m, icc=0.4) for m in (1, 2, 3, 5, 10)]
    assert mdes == sorted(mdes, reverse=True)


def test_mde_inverts_required_items() -> None:
    """The MDE must be exactly the boundary: detectable at N, not detectable just below."""
    n_items = 500
    mde = minimum_detectable_effect(0.80, n_items, repeats=3, icc=0.4, pair_corr=0.2)
    assert required_items(0.80, -mde, repeats=3, icc=0.4, pair_corr=0.2) <= n_items
    just_smaller = mde * 0.95
    assert required_items(0.80, -just_smaller, repeats=3, icc=0.4, pair_corr=0.2) > n_items


def test_mde_either_direction_is_the_harder_of_the_two() -> None:
    kwargs = {"repeats": 2, "icc": 0.3}
    up = minimum_detectable_effect(0.80, 300, direction="up", **kwargs)  # type: ignore[arg-type]
    down = minimum_detectable_effect(0.80, 300, direction="down", **kwargs)  # type: ignore[arg-type]
    either = minimum_detectable_effect(0.80, 300, direction="either", **kwargs)  # type: ignore[arg-type]
    assert down > up, "a loss from 80% should be harder to detect than a gain"
    assert either == pytest.approx(down)


def test_mde_is_nan_when_nothing_is_detectable() -> None:
    """Four items cannot detect anything; saying so beats returning a small number."""
    assert math.isnan(minimum_detectable_effect(0.80, 2))


def test_design_effect_is_never_below_one() -> None:
    """Correlated repeats cannot carry more information than independent ones."""
    for repeats in (1, 2, 3, 5, 10, 50):
        for icc in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0):
            assert design_effect(repeats, icc) >= 1.0


def test_design_effect_boundaries() -> None:
    assert design_effect(1, 0.7) == 1.0, "one repeat cannot be inflated by correlation"
    assert design_effect(5, 0.0) == 1.0, "uncorrelated repeats are free"
    assert design_effect(5, 1.0) == 5.0, "identical repeats are worth one observation"


def test_effective_n_is_bounded_by_the_observation_count_without_pairing() -> None:
    """With no pairing benefit, effective n lies between n_items and n_items*repeats."""
    n_items, repeats = 200, 5
    for icc in (0.0, 0.2, 0.5, 0.8, 1.0):
        value = effective_n(n_items, repeats, icc, pair_corr=0.0)
        assert n_items - 1e-9 <= value <= n_items * repeats + 1e-9

    assert effective_n(n_items, repeats, 1.0, 0.0) == pytest.approx(n_items)
    assert effective_n(n_items, repeats, 0.0, 0.0) == pytest.approx(n_items * repeats)


def test_pairing_can_push_effective_n_past_the_observation_count() -> None:
    """That is the point of pairing: it removes variance an unpaired run would pay for."""
    unpaired = effective_n(200, 5, icc=0.5, pair_corr=0.0)
    paired = effective_n(200, 5, icc=0.5, pair_corr=0.9)
    assert paired > unpaired
    assert paired > 200 * 5


def test_design_factor_never_exceeds_one() -> None:
    for repeats in (1, 3, 8):
        for icc in (0.0, 0.3, 0.7, 1.0):
            for pair_corr in (0.0, 0.5, 1.0):
                assert 0.0 <= design_factor(repeats, icc, pair_corr) <= 1.0


def test_power_analysis_is_internally_consistent() -> None:
    result = power_analysis(0.80, 200, repeats=5, icc=0.5, pair_corr=0.0, target_effect=0.01)
    assert result.design_effect == pytest.approx(3.0)
    assert result.design_factor == pytest.approx(0.6)
    assert result.effective_n == pytest.approx(200 / 0.6)
    assert result.items_for_target is not None
    assert result.items_for_target > result.n_items, (
        "detecting 1pp should need more items than the run has, at an MDE of ~9pp"
    )
    assert math.isfinite(result.mde)


def test_power_analysis_needs_no_network_or_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flagship command must work with nothing available but the standard library."""
    import socket

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("power analysis attempted a network call")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    result = power_analysis(0.75, 500, repeats=3, icc=0.6)
    assert math.isfinite(result.mde)


@pytest.mark.parametrize(
    ("baseline", "effect"),
    [(0.0, 0.05), (1.0, 0.05), (0.5, 0.0), (0.95, 0.10)],
)
def test_rejects_impossible_designs(baseline: float, effect: float) -> None:
    with pytest.raises(ValueError):
        required_items(baseline, effect)


def test_rejects_out_of_range_correlations() -> None:
    with pytest.raises(ValueError, match="icc"):
        required_items(0.8, 0.05, icc=1.5)
    with pytest.raises(ValueError, match="pair_corr"):
        required_items(0.8, 0.05, pair_corr=-0.2)
