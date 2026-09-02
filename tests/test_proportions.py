"""Wilson intervals, checked against published reference values.

The four cases are Newcombe (1998), "Two-sided confidence intervals for the single
proportion: comparison of seven methods", Statistics in Medicine 17:857-872. His method
3 is the Wilson score interval, and the paper tabulates it for r/n = 81/263, 15/148,
0/20 and 1/29. Those numbers are the point of this file: an implementation that agrees
with a published table is checked, and one that agrees with its own output is not.
"""

from __future__ import annotations

import math

import pytest

from errorbars.stats.proportions import wilson_interval

# (successes, n, expected low, expected high) -- Newcombe 1998, 95% two-sided.
NEWCOMBE_1998 = [
    (81, 263, 0.2553, 0.3662),
    (15, 148, 0.0624, 0.1605),
    (0, 20, 0.0000, 0.1611),
    (1, 29, 0.0061, 0.1718),
]


@pytest.mark.parametrize(("successes", "n", "low", "high"), NEWCOMBE_1998)
def test_matches_newcombe_1998_table(successes: int, n: int, low: float, high: float) -> None:
    interval = wilson_interval(successes, n)
    assert interval.low == pytest.approx(low, abs=5e-5)
    assert interval.high == pytest.approx(high, abs=5e-5)


def test_zero_successes_has_lower_bound_of_exactly_zero() -> None:
    """A Wald interval would run negative here. Wilson does not."""
    interval = wilson_interval(0, 20)
    assert interval.low == 0.0
    assert 0.16 < interval.high < 0.17


def test_all_successes_has_upper_bound_of_exactly_one() -> None:
    interval = wilson_interval(20, 20)
    assert interval.high == 1.0
    assert 0.83 < interval.low < 0.84


def test_interval_is_symmetric_under_relabelling() -> None:
    """Swapping success for failure should mirror the interval about 0.5."""
    forward = wilson_interval(30, 100)
    backward = wilson_interval(70, 100)
    assert forward.low == pytest.approx(1.0 - backward.high, abs=1e-12)
    assert forward.high == pytest.approx(1.0 - backward.low, abs=1e-12)


def test_narrower_alpha_gives_wider_interval() -> None:
    ninety_five = wilson_interval(50, 200, alpha=0.05)
    ninety_nine = wilson_interval(50, 200, alpha=0.01)
    assert ninety_nine.width > ninety_five.width


def test_width_shrinks_with_n() -> None:
    widths = [wilson_interval(int(0.3 * n), n).width for n in (25, 100, 400, 1600)]
    assert widths == sorted(widths, reverse=True)


def test_point_estimate_is_inside_the_interval() -> None:
    for successes, n, _, _ in NEWCOMBE_1998:
        interval = wilson_interval(successes, n)
        assert interval.contains(interval.point)


def test_se_is_nan_because_wilson_is_asymmetric() -> None:
    assert math.isnan(wilson_interval(81, 263).se)


@pytest.mark.parametrize(
    ("successes", "n"),
    [(-1, 10), (11, 10), (1, 0), (1, -5)],
)
def test_rejects_impossible_inputs(successes: int, n: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, n)
