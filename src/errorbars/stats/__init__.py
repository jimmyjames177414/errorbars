"""Statistics for LLM experiments. No scipy: everything here is numpy plus stdlib.

The four things this package exists to do:

* :mod:`~errorbars.stats.bootstrap` -- paired percentile CIs, resampling items.
* :mod:`~errorbars.stats.permutation` -- paired sign-flip significance test.
* :mod:`~errorbars.stats.multiplicity` -- Holm-Bonferroni, because eight interventions
  at alpha=0.05 will hand you a false positive roughly a third of the time.
* :mod:`~errorbars.stats.power` -- MDE and required N, which is the only one you can run
  *before* spending money.

:mod:`~errorbars.stats.effects` combines them into the significant / null / underpowered
verdict.
"""

from __future__ import annotations

from .bootstrap import DEFAULT_RESAMPLES, bootstrap_mean_ci, paired_bootstrap_ci
from .effects import (
    AnalysisResult,
    Arm,
    EffectEstimate,
    Verdict,
    analyse_arms,
    classify,
    default_sesoi,
)
from .icc import intraclass_correlation, pair_correlation
from .multiplicity import bonferroni_adjust, holm_adjust, holm_reject
from .normal import normal_cdf, z_critical, z_quantile
from .permutation import (
    DEFAULT_PERMUTATIONS,
    PermutationResult,
    paired_permutation_test,
)
from .power import (
    PowerResult,
    design_effect,
    design_factor,
    effective_n,
    minimum_detectable_effect,
    power_analysis,
    required_items,
    required_n_two_proportion,
)
from .proportions import Interval, wilson_interval

__all__ = [
    "DEFAULT_PERMUTATIONS",
    "DEFAULT_RESAMPLES",
    "AnalysisResult",
    "Arm",
    "EffectEstimate",
    "Interval",
    "PermutationResult",
    "PowerResult",
    "Verdict",
    "analyse_arms",
    "bonferroni_adjust",
    "bootstrap_mean_ci",
    "classify",
    "default_sesoi",
    "design_effect",
    "design_factor",
    "effective_n",
    "holm_adjust",
    "holm_reject",
    "intraclass_correlation",
    "minimum_detectable_effect",
    "normal_cdf",
    "pair_correlation",
    "paired_bootstrap_ci",
    "paired_permutation_test",
    "power_analysis",
    "required_items",
    "required_n_two_proportion",
    "wilson_interval",
    "z_critical",
    "z_quantile",
]
