"""The most important test in the repo: does the 95% interval actually cover 95%?

A confidence interval is a *frequency* claim. It says that if you repeated the whole
experiment many times, 95% of the intervals you built would contain the true value.
There is exactly one honest way to check that, and it is to repeat the whole experiment
many times and count. Everything else -- eyeballing an interval, comparing against a
snapshot, checking it looks about the right width -- tests that the code does what it
did last time, not that it is right.

So: simulate a thousand experiments from a distribution whose true value we know because
we chose it, build an interval for each, and count how many contain the truth. If the
answer is not near 95%, the interval is lying and no amount of green elsewhere in this
suite makes up for it.

Both cases that matter are covered: the mean of a single arm, and the paired difference
between two arms with correlated item difficulty, which is what the tool actually
reports.
"""

from __future__ import annotations

import numpy as np
import pytest

from errorbars.stats.bootstrap import bootstrap_mean_ci, paired_bootstrap_ci

# 1000 simulations puts the Monte-Carlo standard error of a 95% coverage estimate at
# sqrt(0.95 * 0.05 / 1000) = 0.0069, so a correct implementation lands inside
# [0.93, 0.97] with room to spare, and a broken one does not sneak through.
N_SIMULATIONS = 1000
N_ITEMS = 200
N_RESAMPLES = 1000
COVERAGE_LOW = 0.93
COVERAGE_HIGH = 0.97


@pytest.mark.slow
def test_percentile_interval_covers_a_known_mean_about_95_percent_of_the_time() -> None:
    """Bernoulli(0.7) items; the true mean is 0.7 by construction."""
    true_mean = 0.7
    rng = np.random.default_rng(11)
    covered = 0

    for index in range(N_SIMULATIONS):
        sample = rng.binomial(1, true_mean, size=N_ITEMS).astype(float)
        interval = bootstrap_mean_ci(sample, alpha=0.05, resamples=N_RESAMPLES, seed=index)
        covered += interval.contains(true_mean)

    coverage = covered / N_SIMULATIONS
    # Printed so `pytest -s` reports the measured figure rather than only pass/fail.
    print(f"\nsingle-arm mean coverage: {coverage:.1%} over {N_SIMULATIONS} simulations")
    assert COVERAGE_LOW <= coverage <= COVERAGE_HIGH, (
        f"95% interval covered the true mean {coverage:.1%} of the time over "
        f"{N_SIMULATIONS} simulations; expected about 95%"
    )


@pytest.mark.slow
def test_paired_interval_covers_a_known_effect_about_95_percent_of_the_time() -> None:
    """The case the tool actually reports: a paired effect with correlated items.

    Item difficulty is drawn from Uniform(0.30, 0.85) and the treatment shifts every
    item by exactly +0.05, which stays inside (0, 1) for every draw. So the population
    effect is 0.05 exactly, with no clipping to blur it, and each arm is measured with
    three noisy repeats -- the same structure a real paired eval has.
    """
    true_effect = 0.05
    repeats = 3
    rng = np.random.default_rng(23)
    covered = 0

    for index in range(N_SIMULATIONS):
        difficulty = rng.uniform(0.30, 0.85, size=N_ITEMS)
        control = (rng.random((N_ITEMS, repeats)) < difficulty[:, None]).mean(axis=1)
        treatment = (rng.random((N_ITEMS, repeats)) < (difficulty + true_effect)[:, None]).mean(
            axis=1
        )
        interval = paired_bootstrap_ci(
            control, treatment, alpha=0.05, resamples=N_RESAMPLES, seed=index
        )
        covered += interval.contains(true_effect)

    coverage = covered / N_SIMULATIONS
    print(f"\npaired effect coverage: {coverage:.1%} over {N_SIMULATIONS} simulations")
    assert COVERAGE_LOW <= coverage <= COVERAGE_HIGH, (
        f"95% paired interval covered the true effect {coverage:.1%} of the time over "
        f"{N_SIMULATIONS} simulations; expected about 95%"
    )


def test_ninety_percent_interval_is_narrower_than_ninety_five() -> None:
    rng = np.random.default_rng(5)
    sample = rng.binomial(1, 0.6, size=300).astype(float)
    ninety = bootstrap_mean_ci(sample, alpha=0.10, resamples=4000, seed=1)
    ninety_five = bootstrap_mean_ci(sample, alpha=0.05, resamples=4000, seed=1)
    assert ninety.width < ninety_five.width


def test_interval_narrows_as_items_are_added() -> None:
    rng = np.random.default_rng(7)
    widths = []
    for n in (25, 100, 400, 1600):
        sample = rng.binomial(1, 0.5, size=n).astype(float)
        widths.append(bootstrap_mean_ci(sample, resamples=2000, seed=3).width)
    assert widths == sorted(widths, reverse=True)


def test_point_estimate_is_the_sample_mean_not_a_bootstrap_average() -> None:
    """The point estimate must be the data's mean; resampling only builds the interval."""
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    interval = bootstrap_mean_ci(values, resamples=500, seed=0)
    assert interval.point == pytest.approx(float(np.mean(values)))


def test_paired_effect_is_the_mean_difference() -> None:
    control = [1.0, 0.0, 1.0, 1.0, 0.0]
    treatment = [0.0, 0.0, 1.0, 1.0, 0.0]
    interval = paired_bootstrap_ci(control, treatment, resamples=500, seed=0)
    assert interval.point == pytest.approx(-0.2)


def test_bootstrap_se_tracks_the_analytic_standard_error() -> None:
    """For a plain mean, the bootstrap SE should land near sd/sqrt(n)."""
    rng = np.random.default_rng(31)
    sample = rng.normal(0.0, 1.0, size=500)
    interval = bootstrap_mean_ci(sample, resamples=8000, seed=2)
    analytic = float(sample.std(ddof=1) / np.sqrt(sample.size))
    assert interval.se == pytest.approx(analytic, rel=0.10)


def test_zero_variance_data_gives_a_degenerate_interval() -> None:
    """Every item identical: the interval is a point, which is the right answer."""
    interval = bootstrap_mean_ci([1.0] * 50, resamples=500, seed=0)
    assert interval.low == interval.high == 1.0
    assert interval.se == pytest.approx(0.0)


def test_seed_makes_the_interval_reproducible() -> None:
    values = [0.0, 1.0] * 40
    first = bootstrap_mean_ci(values, resamples=1000, seed=99)
    second = bootstrap_mean_ci(values, resamples=1000, seed=99)
    assert first == second


def test_chunking_does_not_change_the_result() -> None:
    """Large item sets are resampled in memory-bounded chunks; that must be invisible."""
    from errorbars.stats import bootstrap

    values = list(np.random.default_rng(2).binomial(1, 0.4, size=400).astype(float))
    unchunked = bootstrap_mean_ci(values, resamples=500, seed=17)

    original = bootstrap._MAX_INDEX_ELEMENTS
    try:
        bootstrap._MAX_INDEX_ELEMENTS = 800  # forces several chunks
        chunked = bootstrap_mean_ci(values, resamples=500, seed=17)
    finally:
        bootstrap._MAX_INDEX_ELEMENTS = original

    assert chunked.point == pytest.approx(unchunked.point)
    # Different chunk boundaries consume the RNG stream differently, so the intervals
    # are not bit-identical; they must still agree to Monte-Carlo error.
    assert chunked.low == pytest.approx(unchunked.low, abs=0.02)
    assert chunked.high == pytest.approx(unchunked.high, abs=0.02)


def test_mismatched_arm_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="aligned"):
        paired_bootstrap_ci([1.0, 0.0, 1.0], [1.0, 0.0])


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_mean_ci([])


def test_non_finite_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        bootstrap_mean_ci([1.0, float("nan"), 0.0])
