<div align="center">

<img src="https://raw.githubusercontent.com/jimmyjames177414/errorbars/main/docs/banner.jpg" alt="jimmyjames177414" width="100%">

# errorbars

**Your eval says prompt B beat prompt A by 4 points. This tells you whether that's real.**

[![CI](https://github.com/jimmyjames177414/errorbars/actions/workflows/ci.yml/badge.svg)](https://github.com/jimmyjames177414/errorbars/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/errorbars.svg)](https://pypi.org/project/errorbars/)
[![Python](https://img.shields.io/pypi/pyversions/errorbars.svg)](https://pypi.org/project/errorbars/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/jimmyjames177414/errorbars/blob/main/LICENSE)

</div>

A small statistics layer for LLM experiments: paired control-vs-treatment effects,
bootstrap confidence intervals, multiplicity correction, and — the part nothing else does —
**power analysis that tells you what your run can detect before you spend a cent.**

---

## Why it exists

Eval harnesses print percentages. Almost none print an interval, a paired comparison
against a control, or a power calculation. So people size eval sets by feel, ship on
4-point deltas that sit inside the noise, and cannot tell "this change did nothing" apart
from "our run was too small to notice."

That last distinction is the point of this tool. In every other harness both come out as
`p > 0.05`, and they mean opposite things.

## Show me it working

No install, no config, no API key, no network:

<img src="https://raw.githubusercontent.com/jimmyjames177414/errorbars/main/docs/demo.png" alt="errorbars power analysis output" width="100%">

```console
$ uvx errorbars power --items 200 --repeats 5 --baseline 0.80

Power analysis  (alpha=0.05, power=80%, two-sided, paired)

  design              200 items x 5 repeats = 1,000 observations per arm
  baseline            80.0%
  ICC (assumed)       0.50   correlation of repeats within an item
  pairing r (assumed) 0.00   correlation of item difficulty across arms
  design effect       3.00   inflation from repeated measurement
  effective n         333   independent-equivalent observations per arm

  minimum detectable effect   9.3 pp

  A 9.3 pp result at this N is at the edge of what the run can see.
  Anything under 4.7 pp is noise you cannot distinguish from zero.

  sensitivity to the pairing assumption
    pairing r = 0.0   MDE   9.3 pp
    pairing r = 0.3   MDE   8.0 pp
    pairing r = 0.6   MDE   6.5 pp
    pairing r = 0.9   MDE   4.5 pp
    Higher pairing correlation means a tighter MDE. The default of 0.00 assumes
    pairing buys nothing, so the headline number is never optimistic. Run `errorbars
    analyze` on a pilot to measure the real value.
```

Two hundred items and a thousand model calls cannot reliably resolve anything smaller
than **9 percentage points**, unless both arms run on the same items. That is the number
almost nobody computes, and it costs nothing to get.

Then, on results — yours or another tool's:

```console
$ errorbars run examples/negation-sensitivity.yaml --analyze

results/negation-sensitivity
  tool          errorbars 0.1.0 (CXS 0.1.1)
  model         mock:comparison-sim-v1
  scorer        first_word:v1
  control       control  (86.2% correct)
  design        200 items x 5 repeats, alpha=0.05, power=80%

intervention       effect  95% CI (per-comparison)  p (raw)  p (Holm, family-wise)    n  verdict
----------------  -------  -----------------------  -------  ---------------------  ---  ------------
strip-negation    -23.7pp           [-29.3, -18.3]   <0.001                 <0.001  200  significant
strip-politeness   -0.1pp             [-1.6, +1.4]    1.000                  1.000  200  null
strip-articles     -2.3pp             [-4.6, -0.1]    0.050                  0.099  200  UNDERPOWERED

  strip-negation: MDE 7.3pp by design, 7.8pp achieved. ICC 0.46, pairing r 0.22
      adjusted p = 0.0003 <= alpha = 0.05
  strip-politeness: MDE 4.5pp by design, 2.2pp achieved. ICC 0.16, pairing r 0.84
      CI [-1.6, +1.4] pp excludes effects of +/-2.2 pp
  strip-articles: MDE 4.8pp by design, 3.2pp achieved. ICC 0.18, pairing r 0.72
      CI [-4.6, -0.1] pp still admits an effect of +/-3.2 pp

  The p-values are family-wise (Holm); the intervals are per-comparison. Each
  interval holds on its own, so the chance at least one of them misses grows with
  the number of arms. That is the usual default and is fine when you read one
  interval at a time. Use --simultaneous-ci to put both columns on the same footing.

  null      = not significant AND the interval excludes an effect as large as the MDE.
              strip-politeness did nothing, and this run was big enough to have seen it.

  UNDERPOWERED = not significant, but the interval still admits an effect worth caring about.
              strip-articles taught you nothing. Use `errorbars power` to size a run that would.
```

Look at rows two and three. Both fail to reach significance. In any other tool they would
print identically. Here one is a finding and the other is an admission.

> **That demo output is synthetic.** `examples/negation-sensitivity.yaml` runs against a
> deterministic simulator in `src/errorbars/providers/mock.py`, not a language model, so
> the demo works offline in CI with no key. It is not a measurement of any real system.
> Every other number in this README is real and reproducible by the command shown above
> it. The simulator is documented in full in its own module docstring.

## Install

```bash
uvx errorbars power          # zero-install
pipx run errorbars power
pip install errorbars
```

Runtime dependencies: `numpy` and `PyYAML`. **No scipy** — Holm, the permutation test and
the Wilson interval are implemented here, about twenty lines each, because scipy's install
weight would ruin `uvx` startup for a tool whose flagship command makes no network calls at
all.

## The three commands

| Command | What it does |
|---|---|
| `errorbars power` | MDE and required N for a design. No config, no network, no key, writes nothing. |
| `errorbars analyze <dir>` | Paired effects, bootstrap intervals, Holm correction and a verdict over any CXS results directory — including one another tool wrote. |
| `errorbars run <spec.yaml>` | A small intervention-grid runner with a mandatory control arm. |

Full statistical background, written for an engineer with no stats: **[docs/statistics.md](docs/statistics.md)**.
That document is the best single reason to look at this repository.

## What "paired" means, and why it is the whole game

Run the control and the treatment on the same items, and compare each item to itself.

Most of the variance in an eval score comes from *which questions you happened to pick*.
Both arms answering the same questions makes that variance shared, and the difference
cancels it out. In the sensitivity block above it takes the MDE from 9.3 pp to 4.5 pp for
the same thousand calls — a bigger gain than quintupling your repeat count, and free.

An effect measured without a control arm is not an effect size. `errorbars run` injects a
control if your spec has none, and warns you loudly, because injecting one silently would
misrepresent your design.

## Reading results other tools produced

`errorbars analyze` reads the [CXS v0.1.1](schemas/README.md) interchange format. The
claim is about **files, not imports** — nothing here imports another package, and no other
package imports this one.

It also handles the case a boolean cannot: an outcome marked `inconclusive` or `error` is
**skipped, counted and reported above the table**, never scored and never folded into a
denominator.

That is not a formality. On the shipped fixture with a tenth of the trials unresolved,
squashing them into `passed: false` moves the headline rate by **7.3 percentage points**
— 76.9% becomes 69.6% — while the measured *effect* moves only 1.7pp, because the bias
lands on every arm at once. That partial cancellation is what lets the bug survive review:
the deltas look nearly right while every absolute rate you quote is wrong by more than
most of the effects you are chasing. Both readings still print "significant".

    python examples/unresolved_trials.py

Worked through in [docs/statistics.md §10](docs/statistics.md).

`tests/fixtures/foreign_tool_results/` is a results directory deliberately written in a
different tool's style: another `tool.name`, arm names this project would never generate,
`passed` booleans with no `score` field, no declared control arm, and extra fields it has
never heard of. `tests/test_ingest.py` reads it. That fixture is the proof of the claim;
without it the sentence above would just be marketing.

## Prior art — read this before believing the pitch

This is a small library standing next to some very large, very good projects. Naming them
properly:

| Project | Stars | What it already does |
|---|---|---|
| [promptfoo](https://github.com/promptfoo/promptfoo) | 24,765 | YAML-native eval spec since 2023, `transformVars` input transforms, `evaluateOptions.repeat`, nine export formats, hosted sharing. **Acquired by OpenAI on 2026-03-09.** |
| [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) (UK AISI) | 2,686 | `--epochs`, `--epochs-reducer` (`mean/median/mode/max/at_least_{n}/pass_at_{k}`), and **scorers that report `stderr`**. It does trials and variance well. |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 13,872 | The de-facto academic benchmark runner, YAML-native. |

Star counts verified via the GitHub API on 2026-09-02.

**We do not claim "no framework reports variance."** Inspect AI does, and does it properly.
That claim would be disprovable in one click, and `NOVELTY.md` lists it among the things
this project may never say.

Methodology this builds on rather than invents:

- Miller, *Adding Error Bars to Evals: A Statistical Approach to Language Model
  Evaluations*, [arXiv:2411.00640](https://arxiv.org/abs/2411.00640). **This project is
  named after that paper.**
- *Resolution Diagnostics for Paired LLM Evaluation*,
  [arXiv:2605.30315](https://arxiv.org/abs/2605.30315).

### What is actually new here

Three things, and only three:

1. **A per-intervention effect table.** Nothing surveyed labels results by intervention and
   emits "removing X moved the score by δ, N items, 95% CI". promptfoo has
   `redteam.strategies` and `transformVars` but has never joined them into an effect report.
2. **Null results as a first-class output.** Everything else optimises for "did it pass".
   Nothing reports "this provably did nothing, and here is the power calculation showing we
   could have seen it if it hadn't."
3. **Power and MDE up front.** No surveyed tool answers "how many items do I need to detect
   two points?" The statistics are published; the tooling is not.

## What this is not

**It is a feature, not a platform.** Roughly "the statistics your eval harness doesn't do."
promptfoo's architecture is arguably one design decision away from absorbing it — a risk
the OpenAI acquisition sharpens. Calling it a platform would be the exact dishonesty this
repository is trying to avoid.

**It is not an eval harness and will not replace promptfoo.** There is no test-case
management, no assertion library, no CI reporting, no web UI, no dataset tooling, no
red-teaming, no tracing.

> If you want a regression test suite, use promptfoo. If you want rigorous trials and
> variance in a Python-native harness, use Inspect AI. Use `errorbars` when you have
> results and need to know whether a difference is real, or how big a run you would need
> for it to be.

`errorbars run` exists so the control/intervention primitive has a first-class expression.
It is deliberately less capable than promptfoo and is not trying to catch up.

## Honest limitations

Every one of these is real, and none of them is going to be fixed by a flag.

- **Bootstrap intervals assume items are independent draws.** 200 paraphrases of 5
  questions is really 5 items, every interval here would be far too narrow, and errorbars
  cannot detect that or warn you.
- **The design-effect correction is an approximation**, not a mixed-effects model. Two
  scalars standing in for a full random-effects structure.
- **Frequentist only.** No priors, no posteriors, no Bayes factors.
- **A three-valued scorer is read, not modelled.** Outcomes marked `inconclusive` or
  `error` (CXS 0.1.1) are skipped, counted and reported rather than scored — they never
  enter a denominator. But errorbars models no *uncertainty* about why they were
  unresolved, so a run that is 40% inconclusive gets intervals computed over the 60% that
  resolved, which may not be a representative 60%.
- **A noisy scorer adds variance that is not modelled.** LLM-judge scores carry their own
  disagreement, and treating a judge as ground truth understates uncertainty by an amount
  this package does not estimate. That is why no LLM-judge scorer ships.
- **A default null verdict is self-referential.** With no `--sesoi`, the margin is the run's
  own MDE, and both it and the interval shrink at the same rate — so a bigger run does not
  make "null" easier to earn. The claim that *does* improve with N is a null against a
  margin you chose in advance. Pass `--sesoi 2pp` and mean it.
- **Association under intervention, on your corpus, with your model.** Broader causal
  claims are yours to make, not the tool's output.
- **The power model is a normal approximation.** At very small N, or baselines within a
  couple of standard errors of 0 or 1, treat the answers as indicative.

## Not built yet

Deliberately absent rather than stubbed, because a stub that looks implemented is worse
than an admitted gap. Each of these is a `help-wanted` issue:

- promptfoo and Inspect AI result importers
- Bayesian analysis
- Sequential / adaptive stopping
- Mixed-effects models
- HTML reports, leaderboards, hosted anything

## Is the statistics right?

Fair question for a repository whose entire value is that the arithmetic is correct. It is
checked against things outside this codebase, not against its own past output:

```console
$ uv run pytest -q -m "not live"
247 passed, 1 deselected

$ uv run pytest tests/test_bootstrap.py -q -s -m slow
single-arm mean coverage: 94.9% over 1000 simulations
paired effect coverage: 94.4% over 1000 simulations

$ uv run pytest tests/test_permutation.py -q -s -m slow
continuous-null false positive rate: 0.051 over 2000 simulations
binary-null false positive rate: 0.048 over 1000 simulations
```

- The **95% interval is verified by simulation**: a thousand complete experiments from a
  distribution whose true value we chose, counting how often the interval contains it.
- The **permutation test's false-positive rate** under a true null lands on alpha.
- **Wilson** reproduces all four reference intervals in Newcombe (1998).
- **Holm** reproduces R's `p.adjust(..., method = "holm")` and the Wikipedia worked example.
- **Required N** reproduces hand computation at three points: 80%→85% needs 906 per arm,
  25%→40% needs 152, 50%→60% needs 388.
- The clustered/paired power model **collapses exactly to the textbook two-proportion
  formula** at one repeat with no pairing, which is what makes those hand checks bind on
  the general case too.

`mypy --strict` and `ruff` are clean. The figures above were measured on Python 3.10.12,
Linux — run the commands yourself and you should get the same ones, since every seed is
fixed. CI is configured for Python 3.10–3.13 on Linux plus macOS and Windows,
**with no secrets configured at all**, and fails the build if scipy ever appears.

## Contributing

Good first issues: add a scorer, add an output format, improve the terminal table, add a
worked example to `docs/statistics.md`.

You need no API key to contribute anything. The whole suite runs offline against a
deterministic simulator. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

Apache-2.0. See [LICENSE](LICENSE).

---

<div align="center">
<img src="https://raw.githubusercontent.com/jimmyjames177414/errorbars/main/docs/avatar.png" width="80" alt="James Miller">

**James Miller** · [@jimmyjames177414](https://github.com/jimmyjames177414)

<sub>One of five projects measuring what context actually does to AI systems:<br>
<a href="https://github.com/jimmyjames177414/stopless">stopless</a> ·
<a href="https://github.com/jimmyjames177414/stopbench">stopbench</a> ·
<a href="https://github.com/jimmyjames177414/mincontext">mincontext</a> ·
<a href="https://github.com/jimmyjames177414/validwhile">validwhile</a> ·
<a href="https://github.com/jimmyjames177414/errorbars">errorbars</a></sub>
</div>
