"""Null must be distinguishable from underpowered. This file enforces that.

It is the single most useful thing the tool does, and it is the thing most easily lost
to a refactor: collapse the two into "not significant" and every test about p-values and
intervals still passes. So the distinction is pinned here at three levels -- the pure
classification rule, the analysis pipeline over data built to produce each verdict, and
the rendered output a user actually reads.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from errorbars.ingest import IngestedRun
from errorbars.report import render_analysis_table
from errorbars.stats.effects import Arm, analyse_arms, classify, default_sesoi


def _arm(intervention_id: str, per_item: list[list[float]]) -> Arm:
    return Arm(
        intervention_id=intervention_id,
        scores={f"item-{index:04d}": scores for index, scores in enumerate(per_item)},
    )


def _simulate(
    rng: np.random.Generator,
    n_items: int,
    repeats: int,
    difficulty: np.ndarray,
    shift: float,
) -> list[list[float]]:
    probability = np.clip(difficulty + shift, 0.0, 1.0)
    draws = rng.random((n_items, repeats)) < probability[:, None]
    return [list(row.astype(float)) for row in draws]


# --------------------------------------------------------------------------------------
# The rule itself
# --------------------------------------------------------------------------------------


def test_significant_when_the_adjusted_p_clears_alpha() -> None:
    verdict, reason = classify(
        ci_low=-0.24, ci_high=-0.13, p_adjusted=0.0004, alpha=0.05, sesoi=0.03
    )
    assert verdict == "significant"
    assert "0.0004" in reason


def test_null_when_the_interval_excludes_an_effect_worth_caring_about() -> None:
    verdict, reason = classify(
        ci_low=-0.019, ci_high=0.012, p_adjusted=1.0, alpha=0.05, sesoi=0.028
    )
    assert verdict == "null"
    assert "excludes" in reason


def test_underpowered_when_the_interval_still_admits_one() -> None:
    verdict, reason = classify(
        ci_low=-0.048, ci_high=0.021, p_adjusted=0.612, alpha=0.05, sesoi=0.028
    )
    assert verdict == "underpowered"
    assert "still admits" in reason


def test_null_and_underpowered_are_different_verdicts() -> None:
    """Same p-value, same non-significance -- opposite conclusions.

    This is the whole point. Both rows below would read as "not significant, p = 0.40"
    in any other tool, and they mean opposite things: one rules out a meaningful effect,
    the other cannot.
    """
    tight, _ = classify(ci_low=-0.010, ci_high=0.004, p_adjusted=0.40, alpha=0.05, sesoi=0.028)
    wide, _ = classify(ci_low=-0.090, ci_high=0.035, p_adjusted=0.40, alpha=0.05, sesoi=0.028)
    assert tight == "null"
    assert wide == "underpowered"
    assert tight != wide


def test_an_unusable_sesoi_forces_underpowered_not_null() -> None:
    """Without a threshold there is no basis for a null claim, so it must not be given."""
    for bad in (math.nan, 0.0, -0.01):
        verdict, reason = classify(
            ci_low=-0.01, ci_high=0.01, p_adjusted=0.9, alpha=0.05, sesoi=bad
        )
        assert verdict == "underpowered"
        assert "null cannot be established" in reason


def test_significance_wins_over_a_wide_interval() -> None:
    verdict, _ = classify(ci_low=-0.30, ci_high=-0.01, p_adjusted=0.04, alpha=0.05, sesoi=0.05)
    assert verdict == "significant"


def test_default_margin_is_the_smaller_of_the_two_mdes() -> None:
    """Null is claimed only when design and achieved precision both say it was visible."""
    assert default_sesoi(0.048, 0.031) == pytest.approx(0.031)
    assert default_sesoi(0.020, 0.045) == pytest.approx(0.020)
    assert default_sesoi(math.nan, 0.031) == pytest.approx(0.031)
    assert default_sesoi(0.048, math.nan) == pytest.approx(0.048)
    assert math.isnan(default_sesoi(math.nan, math.nan))


def test_default_margin_always_leaves_underpowered_reachable() -> None:
    """A margin above twice the interval half-width would collapse the three-way split.

    If SESOI exceeded ``2 * half-width``, every non-significant result would fall inside
    it and "underpowered" could never be reported at all. Because ``mde_observed`` is
    ``(z_alpha + z_beta) * SE`` and the half-width is ``z_alpha * SE``, the ratio is
    fixed at 2.80/1.96 = 1.43 < 2, and taking the minimum can only lower it further.
    """
    for se in (0.001, 0.01, 0.05):
        half_width = 1.959963984540054 * se
        mde_observed = (1.959963984540054 + 0.8416212335729143) * se
        for mde_design in (se, 10 * se, math.nan):
            margin = default_sesoi(mde_design, mde_observed)
            assert margin < 2 * half_width


# --------------------------------------------------------------------------------------
# The pipeline, on data constructed so the right answer is known
# --------------------------------------------------------------------------------------


def _constructed_arms() -> tuple[Arm, list[Arm]]:
    """Three arms whose effects and variances are fixed by construction, not sampled.

    The control gets 320 of 400 items right. Each treatment then flips a named set of
    items, so every paired difference is known exactly and the arithmetic below is
    checkable by hand:

    * ``big-effect``   -- 80 items lost, 0 gained  => -20.0pp, SE 2.0pp,  z ~ -10
    * ``no-effect``    -- 20 items lost, 20 gained =>   0.0pp, SE 1.6pp,  z = 0
    * ``small-effect`` -- 60 items lost, 40 gained =>  -5.0pp, SE 2.5pp,  z ~ -2.0

    With a margin of 5pp, that is one of each verdict, and no seed can change it.
    """
    n_items = 400
    control_scores = [1.0 if index < 320 else 0.0 for index in range(n_items)]

    def flipped(lose: range, gain: range) -> list[list[float]]:
        scores = list(control_scores)
        for index in lose:
            scores[index] = 0.0
        for index in gain:
            scores[index] = 1.0
        return [[value] for value in scores]

    control = _arm("control", [[value] for value in control_scores])
    return control, [
        _arm("big-effect", flipped(range(0, 80), range(0, 0))),
        _arm("no-effect", flipped(range(100, 120), range(320, 340))),
        _arm("small-effect", flipped(range(100, 160), range(320, 360))),
    ]


def test_pipeline_produces_all_three_verdicts_on_constructed_data() -> None:
    """One arm with a large effect, one with none, one too small to resolve."""
    control, treatments = _constructed_arms()
    result = analyse_arms(control, treatments, sesoi=0.05, seed=7)
    by_id = {effect.intervention_id: effect for effect in result.effects}

    assert by_id["big-effect"].effect == pytest.approx(-0.20)
    assert by_id["no-effect"].effect == pytest.approx(0.0)
    assert by_id["small-effect"].effect == pytest.approx(-0.05)

    assert by_id["big-effect"].verdict == "significant"
    assert by_id["no-effect"].verdict == "null"
    assert by_id["small-effect"].verdict == "underpowered"


def test_the_null_and_underpowered_arms_have_indistinguishable_p_values() -> None:
    """Why the verdict column exists at all.

    Both arms below fail to reach significance. Any tool that reported only "p > 0.05"
    would present them as the same result. They are not: one rules out a 5pp effect and
    the other cannot.
    """
    control, treatments = _constructed_arms()
    result = analyse_arms(control, treatments, sesoi=0.05, seed=7)
    by_id = {effect.intervention_id: effect for effect in result.effects}

    assert by_id["no-effect"].p_adjusted > 0.05
    assert by_id["small-effect"].p_adjusted > 0.05
    assert by_id["no-effect"].verdict != by_id["small-effect"].verdict


def test_a_real_null_is_reported_as_null_not_as_a_failure_to_reject() -> None:
    """Identical arms and plenty of items: the answer is 'nothing happened'."""
    rng = np.random.default_rng(99)
    n_items, repeats = 600, 5
    difficulty = rng.uniform(0.5, 0.95, size=n_items)

    control = _arm("control", _simulate(rng, n_items, repeats, difficulty, 0.0))
    treatment = _arm("inert", _simulate(rng, n_items, repeats, difficulty, 0.0))

    result = analyse_arms(control, [treatment], seed=1)
    (effect,) = result.effects
    assert effect.verdict == "null"
    assert abs(effect.effect) < 0.02
    assert effect.ci_low > -effect.sesoi
    assert effect.ci_high < effect.sesoi


def test_a_fixed_sesoi_turns_underpowered_into_null_as_items_are_added() -> None:
    """The property users actually care about, and the reason --sesoi matters.

    Against a margin chosen in advance -- "I care about 3 percentage points" -- a bigger
    run genuinely does convert underpowered into null, because the interval shrinks
    while the margin stays where you put it. Same generating process (no effect at all),
    same margin, different run sizes, different conclusions.
    """
    sesoi = 0.03
    repeats = 3

    def verdict_at(n_items: int) -> str:
        generator = np.random.default_rng(404)
        difficulty = generator.uniform(0.55, 0.9, size=n_items)
        control = _arm("control", _simulate(generator, n_items, repeats, difficulty, 0.0))
        treatment = _arm("inert", _simulate(generator, n_items, repeats, difficulty, 0.0))
        return analyse_arms(control, [treatment], sesoi=sesoi, seed=3).effects[0].verdict

    assert verdict_at(40) == "underpowered", "40 items cannot rule out a 3pp effect"
    assert verdict_at(3000) == "null", "3000 items can"


def test_a_real_effect_becomes_significant_as_items_are_added() -> None:
    """The mirror image, constructed rather than sampled so no seed can flatter it.

    Exactly 10% of items are lost in every case, so the effect is -10pp at both sizes.
    At 20 items only two differences are non-zero and the sign-flip test cannot get
    anywhere near alpha; at 2000 the same effect is unmissable.
    """

    def verdict_at(n_items: int) -> str:
        lost = n_items // 10
        control = _arm("control", [[1.0] for _ in range(n_items)])
        treatment = _arm(
            "shifted",
            [[0.0] if index < lost else [1.0] for index in range(n_items)],
        )
        return analyse_arms(control, [treatment], sesoi=0.02, seed=3).effects[0].verdict

    assert verdict_at(20) == "underpowered"
    assert verdict_at(2000) == "significant"


def test_explicit_sesoi_overrides_the_measured_mde() -> None:
    """Naming your own smallest-effect-of-interest turns null into an equivalence claim."""
    rng = np.random.default_rng(17)
    n_items, repeats = 400, 3
    difficulty = rng.uniform(0.6, 0.95, size=n_items)
    control = _arm("control", _simulate(rng, n_items, repeats, difficulty, 0.0))
    treatment = _arm("inert", _simulate(rng, n_items, repeats, difficulty, 0.0))

    generous = analyse_arms(control, [treatment], sesoi=0.20, seed=2).effects[0]
    strict = analyse_arms(control, [treatment], sesoi=0.0005, seed=2).effects[0]

    assert generous.verdict == "null"
    assert strict.verdict == "underpowered"
    assert generous.sesoi == 0.20


def test_holm_adjustment_is_applied_across_the_family() -> None:
    rng = np.random.default_rng(64)
    n_items, repeats = 250, 3
    difficulty = rng.uniform(0.5, 0.9, size=n_items)
    control = _arm("control", _simulate(rng, n_items, repeats, difficulty, 0.0))
    arms = [
        _arm(f"arm-{index}", _simulate(rng, n_items, repeats, difficulty, -0.05))
        for index in range(4)
    ]

    result = analyse_arms(control, arms, seed=11)
    for effect in result.effects:
        assert effect.p_adjusted >= effect.p_raw - 1e-12


def test_single_arm_records_that_holm_was_a_no_op() -> None:
    rng = np.random.default_rng(5)
    difficulty = rng.uniform(0.5, 0.9, size=60)
    control = _arm("control", _simulate(rng, 60, 2, difficulty, 0.0))
    treatment = _arm("only", _simulate(rng, 60, 2, difficulty, -0.05))

    result = analyse_arms(control, [treatment], seed=0)
    assert result.effects[0].p_adjusted == result.effects[0].p_raw
    assert any("no-op" in note for note in result.notes)


def test_measured_icc_and_pairing_are_reported() -> None:
    rng = np.random.default_rng(77)
    n_items, repeats = 300, 5
    difficulty = rng.uniform(0.3, 0.95, size=n_items)
    control = _arm("control", _simulate(rng, n_items, repeats, difficulty, 0.0))
    treatment = _arm("shift", _simulate(rng, n_items, repeats, difficulty, -0.05))

    (effect,) = analyse_arms(control, [treatment], seed=0).effects
    assert 0.0 < effect.icc < 1.0, "spread-out item difficulty should give a non-trivial ICC"
    assert effect.pair_corr > 0.3, "shared item difficulty should show up as pairing"


def test_single_repeat_falls_back_and_says_so() -> None:
    """With one repeat the ICC is unidentifiable; the note must not be silent."""
    rng = np.random.default_rng(3)
    difficulty = rng.uniform(0.4, 0.95, size=200)
    control = _arm("control", _simulate(rng, 200, 1, difficulty, 0.0))
    treatment = _arm("shift", _simulate(rng, 200, 1, difficulty, -0.05))

    result = analyse_arms(control, [treatment], seed=0)
    assert any("1 repeat per item" in note for note in result.notes)
    assert result.effects[0].icc == 1.0


def test_unpaired_items_are_dropped_and_counted() -> None:
    control = Arm("control", {"a": [1.0], "b": [1.0], "c": [0.0]})
    treatment = Arm("t", {"a": [0.0], "b": [1.0], "z": [1.0]})
    result = analyse_arms(control, [treatment], bootstrap_resamples=200, permutations=200)
    assert result.effects[0].n_items == 2
    assert any("unpaired" in note for note in result.notes)


def test_arms_sharing_no_items_are_an_error_not_an_empty_result() -> None:
    control = Arm("control", {"a": [1.0], "b": [1.0]})
    treatment = Arm("t", {"y": [0.0], "z": [1.0]})
    with pytest.raises(ValueError, match="shares no items"):
        analyse_arms(control, [treatment])


def test_wilson_is_used_for_single_repeat_arms_and_bootstrap_otherwise() -> None:
    rng = np.random.default_rng(8)
    difficulty = rng.uniform(0.4, 0.9, size=80)

    single = analyse_arms(
        _arm("control", _simulate(rng, 80, 1, difficulty, 0.0)),
        [_arm("t", _simulate(rng, 80, 1, difficulty, 0.0))],
        bootstrap_resamples=500,
        permutations=500,
        seed=0,
    )
    repeated = analyse_arms(
        _arm("control", _simulate(rng, 80, 4, difficulty, 0.0)),
        [_arm("t", _simulate(rng, 80, 4, difficulty, 0.0))],
        bootstrap_resamples=500,
        permutations=500,
        seed=0,
    )
    assert single.effects[0].rate_ci_method == "wilson"
    assert repeated.effects[0].rate_ci_method == "bootstrap(items)"


# --------------------------------------------------------------------------------------
# The rendered output
# --------------------------------------------------------------------------------------


def test_rendered_table_distinguishes_null_from_underpowered() -> None:
    """A user reading the terminal must be able to tell the two apart at a glance."""
    control, treatments = _constructed_arms()
    analysis = analyse_arms(control, treatments, sesoi=0.05, seed=7)
    run = IngestedRun(
        results_dir=Path("results/constructed"),
        manifest={},
        tool_name="errorbars",
        tool_version="test",
        cxs_version="0.1",
        model_key="mock:test",
        available_models=["mock:test"],
        control_id="control",
        arms={arm.intervention_id: arm for arm in [control, *treatments]},
        repeats=1.0,
        n_items=400,
        scorer="exact_match:v1",
    )
    rendered = render_analysis_table(run, analysis)

    assert "null" in rendered
    assert "UNDERPOWERED" in rendered
    assert "did nothing, and this run was big enough to have seen it" in rendered
    assert "taught you nothing" in rendered
    # Every effect carries its interval; a bare percentage is never printed alone.
    for effect in analysis.effects:
        assert f"{effect.effect * 100:+.1f}pp" in rendered
