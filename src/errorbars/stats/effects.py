"""Paired effect estimation, and the null-vs-underpowered verdict.

This module holds the one judgement ``errorbars`` makes that other tools do not:
telling apart *"this intervention did nothing"* from *"this run was too small to say."*

Both look identical in every eval harness -- a difference near zero and a p-value above
0.05 -- and they mean opposite things. The distinction is a precision question, not a
significance question, so it is answered with the confidence interval rather than the
p-value:

``significant``
    The adjusted p-value clears alpha. There is an effect.

``null``
    Not significant, **and** the whole confidence interval sits inside +/- SESOI. The run
    could have seen an effect worth caring about, and it rules one out. This is a real
    finding and is reportable as such.

``underpowered``
    Not significant, and the confidence interval still contains an effect of at least
    SESOI. The run cannot distinguish "no effect" from "an effect big enough to matter".
    Nothing has been learned; report the MDE and go get more items.

Which MDE the verdict uses
--------------------------
Two different minimum detectable effects appear in the output and they answer different
questions.

``mde_design``
    What the power model says a run of this *shape* should detect, given the measured
    ICC and pairing correlation. This is the number that connects to ``errorbars power``,
    and it is what you use to size the next run.

``mde_observed``
    What this run's *actual* precision would have detected:
    ``(z_{1-alpha/2} + z_{1-beta}) * SE``, with SE taken from the bootstrap distribution
    of the effect.

The default margin is the **smaller of the two**: a null verdict is only given when the
design model and the achieved precision *both* say an effect that size would have been
visible. Every way the two can disagree is then handled in the safe direction. If the
design model is pessimistic -- which it is whenever an intervention's damage is
concentrated on a slice of items, leaving most paired differences at exactly zero and
the bootstrap interval far tighter than a normal approximation predicts -- the margin
falls back to what the run actually achieved, instead of awarding a null the data has
not earned. If the data is noisier than the design assumed, the margin tightens instead,
and a null becomes harder to claim. Both are the right way to be wrong: "null" is a
positive claim, "underpowered" is only an admission, so the asymmetry belongs there.

Taking the larger of the two, or the design figure alone, produces demonstrably false
nulls. There is one in this repository's own demo: ``strip-articles`` has a real effect
of roughly -3.8 percentage points built into the fixture, and a design-model margin
labels it "null" while the smaller margin correctly calls it underpowered.

Read this before trusting a default null verdict
------------------------------------------------
A null against the run's own MDE is a **self-referential** claim, and it does not get
easier to earn by running more items. Both the interval and the MDE shrink like
1/sqrt(N), so their ratio is roughly fixed, and under a true null the verdict comes out
"null" with roughly fixed probability however big the run is. That is not a bug in the
arithmetic; it is what "did this run rule out an effect it was sized to detect?" means.

The claim that *does* improve with N is a null against a margin you chose in advance.
Name it with ``--sesoi`` -- "I care about 2 percentage points" -- and a bigger run
genuinely does convert underpowered into null, because the interval shrinks while the
margin stays put. Equivalence cannot be established without an equivalence margin, and
the default is a convenience, not a substitute for picking one.

The `min()` rule itself is a **pragmatic composite**, chosen because it fails in the
safe direction in every case, not because any particular paper prescribes it. Nothing in
the literature endorses that exact choice. Where you can name your own margin, do.

Per-comparison intervals next to a family-wise p-value
-------------------------------------------------------
By default the interval beside each effect is a **per-comparison** 95% interval, while
the Holm-adjusted p-value beside it controls error across the **whole family** of
interventions. Those are two different error rates in one row, and pretending otherwise
would be the exact kind of quiet inconsistency this package exists to complain about. So
the columns are labelled with which is which, always.

Per-comparison intervals are the near-universal default -- it is what every eval harness,
every statistics package and almost every paper prints -- and they are the right thing
when each interval is read on its own: "how big is this one effect?"

They are the wrong thing when the interval is read as a *screening* device across many
arms, because then the chance that at least one interval misses its true value grows with
the number of arms, just as the chance of at least one false positive does. Pass
``simultaneous_ci=True`` (``--simultaneous-ci``) for that case. Each interval is then
built at ``alpha / m`` by taking the ``alpha/2m`` and ``1 - alpha/2m`` percentiles of the
bootstrap distribution, so all m intervals hold together at ``1 - alpha``. They are
strictly wider, and both MDEs move to the same level, so the whole row is internally
consistent at one error rate.

Bonferroni is used rather than a Holm-style step-down because there is no accepted
step-down analogue for interval estimation: Holm's power advantage comes from
sequentially rejecting, which produces decisions, not intervals. The consequence is that
the simultaneous intervals are slightly conservative relative to the Holm p-values beside
them -- an interval may straddle zero while its adjusted p clears alpha. That is the
honest cost of the wider guarantee, and it is stated here rather than discovered.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from .bootstrap import DEFAULT_RESAMPLES, bootstrap_mean_ci, paired_bootstrap_ci
from .icc import intraclass_correlation, pair_correlation
from .multiplicity import holm_adjust
from .normal import z_critical, z_quantile
from .permutation import DEFAULT_PERMUTATIONS, paired_permutation_test
from .power import minimum_detectable_effect
from .proportions import Interval, wilson_interval

__all__ = [
    "AnalysisResult",
    "Arm",
    "EffectEstimate",
    "Verdict",
    "analyse_arms",
    "classify",
    "default_sesoi",
]

Verdict = Literal["significant", "null", "underpowered"]

# Sentinel used when a fallback is needed for an unidentifiable ICC. With no usable
# within-item information, treating all variance as between-item is the assumption that
# does not overstate what repeats bought.
_ICC_FALLBACK = 1.0


@dataclass(frozen=True)
class Arm:
    """One intervention's scored results: item id -> that item's scores across repeats."""

    intervention_id: str
    scores: Mapping[str, Sequence[float]]

    def item_mean(self, item_id: str) -> float:
        values = self.scores[item_id]
        return float(np.mean(np.asarray(values, dtype=np.float64)))


def arm_rate_interval(
    groups: Sequence[npt.NDArray[np.float64]],
    *,
    alpha: float,
    resamples: int,
    seed: int | None,
) -> tuple[Interval, str]:
    """Confidence interval for one arm's overall success rate.

    Picks the right instrument rather than always reaching for the same one. With
    exactly one observation per item the arm is a plain binomial sample and Wilson is
    the correct (and much cheaper) interval. With repeats, observations are clustered
    within items and a Wilson interval would be too narrow, so the item-level bootstrap
    is used instead -- the same resampling unit as every other interval here.

    Returns:
        The interval and the name of the method used, so output can say which it was.
    """
    per_item = np.array([float(group.mean()) for group in groups], dtype=np.float64)
    if all(group.size == 1 for group in groups):
        successes = float(sum(float(group.sum()) for group in groups))
        return wilson_interval(successes, len(groups), alpha=alpha), "wilson"
    return (
        bootstrap_mean_ci(per_item, alpha=alpha, resamples=resamples, seed=seed),
        "bootstrap(items)",
    )


@dataclass(frozen=True)
class EffectEstimate:
    """The full statistical picture for one intervention against the control arm."""

    intervention_id: str
    n_items: int
    repeats: float
    control_rate: float
    treatment_rate: float
    control_ci: Interval
    treatment_ci: Interval
    rate_ci_method: str
    effect: float
    ci_low: float
    ci_high: float
    p_raw: float
    p_adjusted: float
    icc: float
    pair_corr: float
    mde_design: float
    mde_observed: float
    sesoi: float
    verdict: Verdict
    reason: str

    @property
    def effect_pp(self) -> float:
        """Effect in percentage points, which is how the terminal table prints it."""
        return self.effect * 100.0


@dataclass(frozen=True)
class _Partial:
    """Per-arm results before Holm adjustment, which needs the whole family at once."""

    intervention_id: str
    n_items: int
    repeats: float
    control_rate: float
    treatment_rate: float
    control_ci: Interval
    treatment_ci: Interval
    rate_ci_method: str
    effect: float
    ci_low: float
    ci_high: float
    p_raw: float
    icc: float
    pair_corr: float
    mde_design: float
    mde_observed: float


@dataclass(frozen=True)
class AnalysisResult:
    control_id: str
    alpha: float
    power: float
    bootstrap_resamples: int
    permutations: int
    effects: list[EffectEstimate]
    notes: list[str] = field(default_factory=list)
    simultaneous_ci: bool = False
    """True when the intervals are family-wise rather than per-comparison."""
    ci_alpha: float = 0.05
    """The level each interval was actually built at: ``alpha``, or ``alpha / m``
    when ``simultaneous_ci`` is set. Reported so output can say which it is instead of
    leaving a reader to assume."""
    family_size: int = 1
    """Number of treatment arms, i.e. the m in both the Holm and Bonferroni
    adjustments."""

    @property
    def ci_label(self) -> str:
        """How the interval column should be headed, given what it actually contains."""
        scope = "simultaneous" if self.simultaneous_ci else "per-comparison"
        return f"{round((1 - self.alpha) * 100)}% CI ({scope})"


def default_sesoi(mde_design: float, mde_observed: float) -> float:
    """The equivalence margin used when the caller does not supply one.

    The smaller of the two MDEs, so a null is only claimed when the design model and the
    precision actually achieved both agree the effect would have been visible. ``nan``
    values are ignored; if both are ``nan`` the result is ``nan`` and no null can be
    established. See the module docstring for why the minimum rather than either alone.
    """
    usable = [value for value in (mde_design, mde_observed) if math.isfinite(value)]
    return min(usable) if usable else math.nan


def classify(
    *,
    ci_low: float,
    ci_high: float,
    p_adjusted: float,
    alpha: float,
    sesoi: float,
) -> tuple[Verdict, str]:
    """Apply the significant / null / underpowered rule. Pure, and directly tested."""
    if p_adjusted <= alpha:
        return "significant", f"adjusted p = {p_adjusted:.4g} <= alpha = {alpha:g}"

    if not math.isfinite(sesoi) or sesoi <= 0.0:
        return (
            "underpowered",
            "no usable smallest-effect-of-interest: the design's MDE could not be "
            "computed, so a null cannot be established",
        )

    if ci_low > -sesoi and ci_high < sesoi:
        return (
            "null",
            f"CI [{ci_low * 100:+.1f}, {ci_high * 100:+.1f}] pp excludes effects of "
            f"+/-{sesoi * 100:.1f} pp",
        )

    return (
        "underpowered",
        f"CI [{ci_low * 100:+.1f}, {ci_high * 100:+.1f}] pp still admits an effect of "
        f"+/-{sesoi * 100:.1f} pp",
    )


def _aligned_items(control: Arm, treatment: Arm) -> list[str]:
    shared = set(control.scores) & set(treatment.scores)
    return sorted(shared)


def analyse_arms(
    control: Arm,
    treatments: Sequence[Arm],
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    bootstrap_resamples: int = DEFAULT_RESAMPLES,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int | None = 0,
    sesoi: float | None = None,
    icc_override: float | None = None,
    pair_corr_override: float | None = None,
    simultaneous_ci: bool = False,
) -> AnalysisResult:
    """Estimate every treatment's paired effect against the control arm.

    Args:
        control: the control (noop) arm. Required -- an effect size without a control is
            not reportable, which is why there is no way to call this without one.
        treatments: the intervention arms to compare against it.
        alpha: significance level. Holm always applies it family-wise; whether the
            *intervals* are family-wise depends on ``simultaneous_ci``.
        power: target power used when computing each arm's MDE.
        bootstrap_resamples: resamples for the percentile CI.
        permutations: sign-flip draws for the paired permutation test.
        seed: RNG seed; fixed by default so two analyses of the same results agree.
        sesoi: smallest effect of interest, as a proportion. Defaults per-arm to that
            arm's MDE.
        icc_override / pair_corr_override: use these instead of the measured values.
        simultaneous_ci: build Bonferroni-adjusted simultaneous intervals at
            ``alpha / m`` rather than per-comparison intervals at ``alpha``, where ``m``
            is the number of treatment arms. See the module docstring for why the
            default is off and when to turn it on.

    Returns:
        AnalysisResult with one EffectEstimate per treatment and any caveats in
        ``notes``.
    """
    notes: list[str] = []
    partials: list[_Partial] = []

    # Family size for both the Holm adjustment and, when asked for, the interval
    # adjustment. With one comparison there is no multiplicity and alpha/m == alpha, so
    # `simultaneous_ci` correctly becomes a no-op rather than a special case.
    family_size = max(1, len(treatments))
    ci_alpha = alpha / family_size if simultaneous_ci else alpha
    if simultaneous_ci and family_size > 1:
        notes.append(
            f"intervals are simultaneous: each is built at alpha/{family_size} = "
            f"{ci_alpha:.4g} (Bonferroni), so all {family_size} of them hold together "
            f"with {1 - alpha:.0%} confidence"
        )

    for treatment in treatments:
        items = _aligned_items(control, treatment)
        if not items:
            raise ValueError(
                f"intervention {treatment.intervention_id!r} shares no items with the "
                f"control arm {control.intervention_id!r}; a paired comparison needs the "
                "same items in both arms"
            )
        dropped = (len(control.scores) - len(items)) + (len(treatment.scores) - len(items))
        if dropped:
            notes.append(
                f"{treatment.intervention_id}: dropped {dropped} unpaired item entries; "
                f"{len(items)} items appear in both arms"
            )

        control_means = np.array([control.item_mean(i) for i in items], dtype=np.float64)
        treatment_means = np.array([treatment.item_mean(i) for i in items], dtype=np.float64)

        control_groups = [np.asarray(control.scores[i], dtype=np.float64) for i in items]
        treatment_groups = [np.asarray(treatment.scores[i], dtype=np.float64) for i in items]
        repeats_per_item = [group.size for group in control_groups + treatment_groups]
        mean_repeats = float(np.mean(repeats_per_item))
        design_repeats = max(1, round(mean_repeats))

        measured_icc = intraclass_correlation(control_groups + treatment_groups)
        measured_pair = pair_correlation(control_means, treatment_means)

        icc = icc_override if icc_override is not None else measured_icc
        if not math.isfinite(icc):
            icc = _ICC_FALLBACK
            if design_repeats > 1:
                notes.append(
                    f"{treatment.intervention_id}: ICC not identifiable from the data "
                    f"(scores show no usable variance); assumed {_ICC_FALLBACK:g}"
                )
            else:
                notes.append(
                    f"{treatment.intervention_id}: 1 repeat per item, so repeat-level and "
                    f"item-level variance cannot be separated; ICC assumed "
                    f"{_ICC_FALLBACK:g} (the only assumption that does not credit repeats "
                    "with information they did not provide)"
                )

        pair_corr = pair_corr_override if pair_corr_override is not None else measured_pair
        if not math.isfinite(pair_corr):
            pair_corr = 0.0
            notes.append(
                f"{treatment.intervention_id}: across-arm correlation not estimable "
                "(an arm has no variance across items); assumed 0.0, which understates "
                "how much pairing helped"
            )
        pair_corr = min(1.0, max(0.0, pair_corr))

        ci = paired_bootstrap_ci(
            control_means,
            treatment_means,
            alpha=ci_alpha,
            resamples=bootstrap_resamples,
            seed=seed,
        )
        test = paired_permutation_test(
            control_means,
            treatment_means,
            permutations=permutations,
            seed=seed,
        )

        control_rate = float(control_means.mean())
        mde_design = math.nan
        if 0.0 < control_rate < 1.0:
            mde_design = minimum_detectable_effect(
                control_rate,
                len(items),
                alpha=ci_alpha,
                power=power,
                repeats=design_repeats,
                icc=icc,
                pair_corr=pair_corr,
            )
        else:
            notes.append(
                f"{treatment.intervention_id}: the control arm scored exactly "
                f"{control_rate:.0%}, so the normal-approximation design MDE is "
                "undefined; only the observed-precision MDE is reported"
            )

        # What this run's own precision would have caught. See the module docstring for
        # why this, and not the design model, is what the verdict is measured against.
        mde_observed = (
            (z_critical(ci_alpha) + z_quantile(power)) * ci.se if math.isfinite(ci.se) else math.nan
        )

        # Arm rates stay at the per-comparison level deliberately. They are
        # descriptive summaries of one arm, not members of the family of comparisons
        # that the multiplicity correction exists to protect.
        control_ci, ci_method = arm_rate_interval(
            control_groups, alpha=alpha, resamples=bootstrap_resamples, seed=seed
        )
        treatment_ci, _ = arm_rate_interval(
            treatment_groups, alpha=alpha, resamples=bootstrap_resamples, seed=seed
        )

        partials.append(
            _Partial(
                intervention_id=treatment.intervention_id,
                n_items=len(items),
                repeats=mean_repeats,
                control_rate=control_rate,
                treatment_rate=float(treatment_means.mean()),
                control_ci=control_ci,
                treatment_ci=treatment_ci,
                rate_ci_method=ci_method,
                effect=ci.point,
                ci_low=ci.low,
                ci_high=ci.high,
                p_raw=test.p_value,
                icc=icc,
                pair_corr=pair_corr,
                mde_design=mde_design,
                mde_observed=mde_observed,
            )
        )

    raw_p = [partial.p_raw for partial in partials]
    adjusted = holm_adjust(raw_p) if len(raw_p) > 1 else list(raw_p)
    if len(raw_p) == 1:
        notes.append(
            "only one intervention compared, so Holm adjustment is a no-op; the "
            "adjusted and raw p-values are identical"
        )

    effects: list[EffectEstimate] = []
    for partial, p_adj in zip(partials, adjusted, strict=True):
        arm_sesoi = (
            sesoi if sesoi is not None else default_sesoi(partial.mde_design, partial.mde_observed)
        )
        verdict, reason = classify(
            ci_low=partial.ci_low,
            ci_high=partial.ci_high,
            p_adjusted=p_adj,
            alpha=alpha,
            sesoi=arm_sesoi,
        )
        effects.append(
            EffectEstimate(
                intervention_id=partial.intervention_id,
                n_items=partial.n_items,
                repeats=partial.repeats,
                control_rate=partial.control_rate,
                treatment_rate=partial.treatment_rate,
                control_ci=partial.control_ci,
                treatment_ci=partial.treatment_ci,
                rate_ci_method=partial.rate_ci_method,
                effect=partial.effect,
                ci_low=partial.ci_low,
                ci_high=partial.ci_high,
                p_raw=partial.p_raw,
                p_adjusted=p_adj,
                icc=partial.icc,
                pair_corr=partial.pair_corr,
                mde_design=partial.mde_design,
                mde_observed=partial.mde_observed,
                sesoi=arm_sesoi,
                verdict=verdict,
                reason=reason,
            )
        )

    return AnalysisResult(
        control_id=control.intervention_id,
        alpha=alpha,
        power=power,
        bootstrap_resamples=bootstrap_resamples,
        permutations=permutations,
        effects=effects,
        notes=notes,
        simultaneous_ci=simultaneous_ci,
        ci_alpha=ci_alpha,
        family_size=family_size,
    )
