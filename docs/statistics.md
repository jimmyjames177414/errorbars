# Why your eval score is probably meaningless

*For a strong engineer with no statistics background. No formulas you have to take on
faith: every number below is one you can reproduce with a command in this repository.*

---

## 1. The situation

You changed a prompt. You ran your eval. It went from 80% to 84%.

Ship it?

The honest answer is that you cannot tell from those two numbers, and neither can anyone
else. Not because 4 points is a small improvement, but because **you do not know how big
a difference this eval can distinguish from nothing at all.** Until you know that, 80%
and 84% are two draws from a process whose spread you have not measured.

Here is the thing that makes it worse: the number looks precise. `84.0%` has a decimal
point in it. It came out of a computer. Every instinct a working engineer has says that a
number like that means something, and in this case it mostly does not.

## 2. Where the uncertainty comes from

Run the same eval twice and you get two different numbers. Three separate reasons, and
they are worth separating because the fixes are different.

**The items are a sample.** You have 200 questions. There are more questions in the world
than those 200. If you had picked a different 200, you would have got a different score.
This is the big one, and it does not go away by running the same 200 questions again.

**The model is stochastic.** Even at temperature 0, most production endpoints do not
guarantee bit-identical output — batching, hardware, and kernel selection all leak in. Ask
the same question five times and you may get four rights and a wrong.

**The scorer is a judgement.** Exact-match is deterministic. An LLM judge is not, and adds
variance nobody in this repository models. (Said plainly in the limitations, below.)

The first two have a shape you can calculate with. The third does not, which is why
`errorbars` ships no LLM-judge scorer.

## 3. How much noise is there, really?

Try it:

```
$ errorbars power --items 200 --repeats 1 --baseline 0.80 --icc 0 --no-sensitivity

Power analysis  (alpha=0.05, power=80%, two-sided, paired)

  design              200 items x 1 repeats = 200 observations per arm
  baseline            80.0%
  ICC (assumed)       0.00   correlation of repeats within an item
  pairing r (assumed) 0.00   correlation of item difficulty across arms
  design effect       1.00   inflation from repeated measurement
  effective n         200   independent-equivalent observations per arm

  minimum detectable effect   12.3 pp

  A 12.3 pp result at this N is at the edge of what the run can see.
  Anything under 6.1 pp is noise you cannot distinguish from zero.
```

**12.3 percentage points.** On a 200-item eval, comparing two prompts the plain way, an
improvement has to be roughly *twelve points* before you can reliably tell it from
nothing.

Your 4-point win is not a win. It is not a loss either. It is a number you cannot
interpret, and the difference between those two statements is the entire subject of this
document.

That figure — the **minimum detectable effect**, or MDE — is the single most useful number
in this repository, and almost nobody computes it. It costs nothing: no model calls, no
API key, no data. You can run it before you have written the eval.

## 4. "But I ran each question five times"

Good instinct, and it helps less than you would expect.

```
$ errorbars power --items 200 --repeats 5 --baseline 0.80 --icc 0 --no-sensitivity
  minimum detectable effect   5.2 pp

$ errorbars power --items 200 --repeats 5 --baseline 0.80 --icc 0.5 --no-sensitivity
  minimum detectable effect   9.3 pp

$ errorbars power --items 200 --repeats 5 --baseline 0.80 --icc 0.9 --no-sensitivity
  minimum detectable effect   11.7 pp
```

Five repeats of 200 items is a thousand model calls. If every one of those calls were an
independent coin flip, that would buy you the full five-fold gain: 12.3 pp down to 5.2 pp.

It usually does not, because **repeats of the same item are not independent.** A question
your model reliably gets right is right all five times. A question it reliably fumbles is
wrong all five times. Those five calls told you roughly what one call would have.

The parameter controlling this is the **intraclass correlation** (ICC): the share of the
outcome variance that lives *between* items rather than *within* them.

| ICC | What it means | What repeats buy |
|---|---|---|
| 0.0 | every item equally hard, each call a fresh coin flip | full value: 5 repeats = 5x the data |
| 0.5 | a typical mixed eval | about a third of the nominal gain |
| 1.0 | every item deterministically right or wrong | **nothing at all** |

Real evals sit high, because most items are not marginal — they are firmly inside or
firmly outside what the model can do. At ICC 0.9, a thousand calls bought you almost
exactly what two hundred would have. You paid five times over for a rounding error.

**More items beats more repeats, nearly always.** Repeats fight noise you may not have;
items fight the sampling problem you definitely have.

The multiplier is called the **design effect**: `1 + (repeats - 1) x ICC`. At 5 repeats
and ICC 0.9 it is 4.6, so your 1000 observations are worth 1000/4.6 ≈ 217. Which is why
the tool prints it.

## 5. The single highest-leverage change: run both arms on the same items

Everything above assumed you compare two independent runs. Don't.

Run the control and the treatment on **the same items**, and compare each item to itself.
The reason this works is worth spelling out, because it is the largest free improvement
available to almost any eval:

> Most of the variance in an eval score comes from *which questions you happened to pick*.
> If both arms answer the same questions, that variance is shared, and subtracting one arm
> from the other cancels it out.

You stop asking "is 84% bigger than 80%" and start asking "on how many individual items
did the answer change, and in which direction?" The second question has far less noise in
it.

```
$ errorbars power --items 200 --repeats 5 --baseline 0.80 --icc 0.9 --pair-corr 0.9 --no-sensitivity
  minimum detectable effect   3.8 pp
```

Same thousand calls. 11.7 pp becomes 3.8 pp, purely from running both arms on the same
questions. That is a bigger improvement than quintupling your repeat count, and it costs
nothing.

`pair-corr` is how strongly item difficulty carries across the two arms — hard questions
staying hard. It is usually high, because difficulty is mostly a property of the question.
The default is `0.0`, which assumes pairing buys nothing, so the headline number is never
optimistic. Measure yours with `errorbars analyze` and put the real value in.

## 6. So how many items do you actually need?

You wanted to detect a 2-point improvement:

```
$ errorbars power --items 200 --repeats 5 --baseline 0.80 --icc 0.9 --pair-corr 0.9 --detect 2pp --no-sensitivity
  minimum detectable effect   3.8 pp
  to detect 2.0 pp you need    ~717 items (at 5 repeats)
```

717 items. If you have 200, you are not measuring a 2-point effect; you are generating a
number that will move around by more than 2 points on its own.

This is the calculation that should happen *before* the eval is built, and essentially
never does.

## 7. The three-way answer

Now the payoff. You ran the experiment, and the difference was not significant. There are
**two completely different reasons** that can happen, and every eval harness prints them
identically:

> `p = 0.38, not significant`

Reason one: **the intervention did nothing.** Your run was big enough to have caught a
meaningful effect, and there wasn't one. That is a real finding. It is publishable, it
settles an argument, and you should stop working on it.

Reason two: **your run was too small to tell.** The effect could be zero. It could also
be twice what you were hoping for. You learned nothing and spent money doing it.

These are opposite conclusions from an identical p-value, and telling them apart is the
main thing this tool does:

```
intervention       effect  95% CI (per-comparison)  p (raw)  p (Holm, family-wise)    n  verdict
----------------  -------  -----------------------  -------  ---------------------  ---  ------------
strip-negation    -23.7pp           [-29.3, -18.3]   <0.001                 <0.001  200  significant
strip-politeness   -0.1pp             [-1.6, +1.4]    1.000                  1.000  200  null
strip-articles     -2.3pp             [-4.6, -0.1]    0.050                  0.099  200  UNDERPOWERED
```

Look at the intervals rather than the p-values.

- `strip-politeness` lands in `[-1.6, +1.4]`. Every value in that range is tiny. Whatever
  is going on, it is not big enough to care about. **That is a null result, and it is
  information.**
- `strip-articles` lands in `[-4.6, -0.1]`. That range includes effects worth caring
  about. The run cannot separate "nothing" from "a real 4-point regression". **That is not
  a null result; it is a shrug.**

Same p-value neighbourhood. Opposite meanings. A tool that printed only `p > 0.05` for
both would be actively misleading, and most of them do.

(That table is real output from `errorbars run examples/negation-sensitivity.yaml
--analyze`, against a *simulated* model that ships with this repository. See §12.)

## 8. Confidence intervals, and why a bare number is not enough

`-18.4pp` invites you to believe in the `.4`.

`-18.4pp [-24.1, -12.9]` tells you the truth: somewhere around 18, could plausibly be 13,
could plausibly be 24.

The interval is not decoration and there is no flag to turn it off.

The one here is a **bootstrap percentile interval**, which works like this: you have 200
items; draw 200 items *from your own 200, with replacement*; compute the score; do that
ten thousand times; look at where the middle 95% of those scores fell. That range is your
interval.

It sounds like cheating and it is not — it is a well-understood way of asking "how much
would this number have moved if the sample had come out differently", and it needs no
assumption that your data is bell-shaped, which eval data emphatically is not.

The one thing that matters: **resample items, not individual trials.** Trials on the same
item are correlated, so resampling trials treats correlated observations as independent
and produces an interval that is too narrow. Too narrow means overconfident, and
overconfident is the failure mode this whole document exists to prevent.

Is it right? Don't take it on trust:

```
$ uv run pytest tests/test_bootstrap.py -q -s -m slow
single-arm mean coverage: 94.9% over 1000 simulations
paired effect coverage: 94.4% over 1000 simulations
```

That test simulates a thousand complete experiments from a distribution whose true value
is known because we chose it, builds an interval for each, and counts how many contain
the truth. A 95% interval should contain it about 95% of the time. It does.

## 9. Testing eight things at once

You tried eight prompt variants. One came back at p = 0.04. Exciting?

Eight independent coin flips at 5% each: the chance that *at least one* looks significant
when none of them is works out at about **34%**. A third of the time, testing eight
nothings hands you a winner.

**Holm–Bonferroni** fixes this. Sort your p-values, make the smallest clear a bar eight
times stricter, the next one seven times, and so on. Plain Bonferroni multiplies
everything by eight and throws away real findings; Holm is strictly better and never
worse, which is why there is no reason to use Bonferroni.

`errorbars` reports raw and adjusted p-values side by side and says which is which. In the
table above, `strip-articles` has a raw p of 0.050 and a Holm-adjusted p of 0.099 — it was
already sitting exactly on the line, and correcting for the fact that three interventions
were tested pushes it clearly over.

If you have ever tried several prompt variants and reported the best one, you have done
this, and the correction is the honest way to report it.

### The mismatch this creates, and what to do about it

Look carefully at that table and there is an inconsistency sitting in plain sight.

The **p-value** column is family-wise: Holm controls the chance of *any* false positive
across all three interventions. The **interval** column is per-comparison: each interval
is a 95% interval *on its own*.

Those are two different error rates, printed side by side in the same row. It is not a
rounding detail. With three arms, each interval independently has a 5% chance of missing
its true value, so the chance that **at least one of them misses is about 14%**, not 5%.
Read the intervals as a set and the "95%" on the header is not the number you are getting.

`errorbars` labels the columns so you can see which is which:

```
intervention       effect  95% CI (per-comparison)  p (raw)  p (Holm, family-wise)    n  verdict
```

**Why per-comparison is still the default.** It is what every eval harness, every
statistics package and almost every paper prints, and it is the right answer to the
question people usually ask, which is "how big is *this* effect?" Silently widening
everyone's intervals to answer a question they did not ask would be its own kind of
dishonesty, and it would make the tool disagree with every other tool for no stated
reason.

**When to reach for `--simultaneous-ci`.** When you are scanning the interval column
across arms to decide what to chase — screening rather than reading one result. Then you
want a guarantee that covers the whole set:

```console
$ errorbars analyze results/negation-sensitivity --simultaneous-ci

intervention       effect  95% CI (simultaneous)  p (raw)  p (Holm, family-wise)    n  verdict
----------------  -------  ---------------------  -------  ---------------------  ---  ------------
strip-negation    -23.7pp         [-30.5, -17.1]   <0.001                 <0.001  200  significant
strip-politeness   -0.1pp           [-2.0, +1.8]    1.000                  1.000  200  null
strip-articles     -2.3pp           [-5.1, +0.3]    0.051                  0.103  200  UNDERPOWERED
```

Each interval is now built at `α/m` — the `α/2m` and `1 − α/2m` percentiles of the
bootstrap distribution instead of `α/2` and `1 − α/2` — so all three hold together at 95%.
They are strictly wider, which is the price of the stronger guarantee. Both MDEs move to
the same level too, so the whole row now runs on one error rate.

Notice `strip-articles` crossing zero at `[-5.1, +0.3]` once the correction is applied.
That is the correction doing its job.

**One honest wrinkle.** The simultaneous intervals use Bonferroni, not Holm, because
there is no accepted step-down analogue for *intervals* — Holm's extra power comes from
rejecting sequentially, which produces decisions rather than ranges. So the intervals are
slightly conservative relative to the Holm p-values beside them, and an interval can
straddle zero while its adjusted p clears alpha. That is a real limitation of the method,
not a bug, and it is better stated here than discovered later.

## 10. Reading a report, in order

1. **N, and repeats.** Small N means everything below is wide. Check it first.
2. **The interval, not the point estimate.** Ask whether the *whole* range would change
   your decision. If part of it would and part of it would not, you do not have an answer.
3. **The MDE.** If it is larger than the effect you care about, the run could not have
   found what you were looking for. Nothing else on the page matters.
4. **The adjusted p-value**, if more than one thing was tested.
5. **The verdict**, which is the previous four steps done for you.

## 11. The formulas, for anyone who wants them

Skippable. Everything above works without this section.

**Wilson score interval** for a single proportion — the sane replacement for
`p ± 1.96·sqrt(p(1-p)/n)`, which produces intervals running past 100% at the proportions
evals actually live at:

```
centre     = (p̂ + z²/2n) / (1 + z²/n)
half-width = z/(1 + z²/n) · sqrt( p̂(1-p̂)/n + z²/4n² )
```

**Design effect** for `m` repeats per item at intraclass correlation `ρ_w`:

```
DEFF = 1 + (m - 1)·ρ_w          effective observations = n·m / DEFF
```

**Variance of the paired difference**, which is the model the whole `power` command runs
on. With `ρ_pair` the across-arm correlation of item difficulty:

```
Var(D) = (p_c·q_c + p_t·q_t) · k / n_items
k      = ρ_w·(1 - ρ_pair) + (1 - ρ_w)/m
```

`k` is the design factor, and the two boundary cases are what make it checkable. At `m=1,
ρ_pair=0` it equals 1 and the required-N expression below collapses to the textbook
unpaired two-proportion formula — which is exactly how the tests verify it, against hand
computation. At `ρ_pair=0` it equals `DEFF/m`, the classical cluster correction.

**Required items** (both arms use the same items, so this is the item count, not double
it):

```
n = k · ( z_{1-α/2}·sqrt(2·p̄·q̄) + z_{1-β}·sqrt(p_c·q_c + p_t·q_t) )² / δ²
```

The sign of `δ` matters. From an 80% baseline, a 5-point gain needs 906 items per arm and
a 5-point loss needs 1094, because 85% carries less binomial variance than 75%. A
calculator that takes the magnitude and discards the sign is wrong by about 20% on one of
those, and most of them do.

**Disattenuating the pairing correlation.** `errorbars analyze` reports the observed
correlation of per-item scores across arms. Per-item scores are noisy estimates of true
item difficulty, and noise drags a correlation toward zero, so the reported figure is a
*lower bound*. If you want the underlying value:

```
ρ_pair = r_observed · (ρ_w + (1 - ρ_w)/m) / ρ_w
```

Using the observed figure directly understates how much pairing helped, which is the safe
direction, so that is what the tool does by default.

## 12. What this does not fix

Being straight about the limits is the price of asking anyone to trust the rest.

**Your items may not be independent.** The bootstrap assumes each item is an independent
draw. Two hundred paraphrases of five underlying questions is really five items, and every
interval in this document would be far too narrow. `errorbars` cannot detect that and will
not warn you. Nothing can except knowing your data.

**The design effect is an approximation.** Two scalars standing in for what a proper
mixed-effects model would estimate. Good enough for planning; not a substitute for one.

**Everything here is frequentist.** No priors, no posteriors, no Bayes factors. If you want
"probability the effect is positive", this is the wrong tool.

**A noisy scorer adds variance nobody models.** LLM-judge scores carry their own
disagreement, and treating a judge's verdict as ground truth understates your uncertainty
by an amount this package does not estimate. That is why no LLM-judge scorer ships here.

**A default null verdict is self-referential.** When you do not name a margin, `errorbars`
uses the run's own minimum detectable effect, and both that and the interval shrink at the
same rate — so a bigger run does not make "null" easier to earn. The claim that *does*
improve with N is a null against a margin you chose in advance. Pass `--sesoi 2pp` and mean
it. Equivalence cannot be established without an equivalence margin.

**And that default margin is a pragmatic composite.** It is the *smaller* of the design
MDE and the precision actually achieved, chosen because it fails in the safe direction
whichever way the two disagree. No paper prescribes that particular rule and there is no
literature behind it — it is a judgement call, documented so you can disagree with it.
Naming your own `--sesoi` sidesteps it entirely.

**Intervals and p-values carry different error rates by default.** Per-comparison
intervals next to family-wise Holm p-values, as covered in §9. Use `--simultaneous-ci`
when you are reading the interval column as a screen across arms.

**Association, not causation, and only on your corpus.** "Deleting negation cost 18 points
on these 200 items with this model at this temperature" is what was measured. Anything
broader is your inference, not the tool's output.

**The demo numbers are synthetic.** The table in §7 comes from a deterministic simulator
in `src/errorbars/providers/mock.py`, not a language model. It exists so the output shape
can be shown without an API key, and it is labelled everywhere it appears. It is not a
measurement of anything.

---

## Reproducing every number in this document

```bash
uv venv && uv pip install -e ".[dev]"

# Sections 3-6
errorbars power --items 200 --repeats 1 --baseline 0.80 --icc 0   --no-sensitivity
errorbars power --items 200 --repeats 5 --baseline 0.80 --icc 0   --no-sensitivity
errorbars power --items 200 --repeats 5 --baseline 0.80 --icc 0.5 --no-sensitivity
errorbars power --items 200 --repeats 5 --baseline 0.80 --icc 0.9 --no-sensitivity
errorbars power --items 200 --repeats 5 --baseline 0.80 --icc 0.9 --pair-corr 0.9 --no-sensitivity
errorbars power --items 200 --repeats 5 --baseline 0.80 --icc 0.9 --pair-corr 0.9 --detect 2pp --no-sensitivity

# Section 7
errorbars run examples/negation-sensitivity.yaml --analyze

# Section 8
uv run pytest tests/test_bootstrap.py -q -s -m slow

# Section 11
uv run pytest tests/test_power.py tests/test_proportions.py -q
```

## Further reading

- Miller, **Adding Error Bars to Evals: A Statistical Approach to Language Model
  Evaluations**, [arXiv:2411.00640](https://arxiv.org/abs/2411.00640). This project is
  named after it. Start here.
- **Resolution Diagnostics for Paired LLM Evaluation**,
  [arXiv:2605.30315](https://arxiv.org/abs/2605.30315) — paired testing, MDE at current N,
  Holm, cluster correction.
- Newcombe (1998), *Two-sided confidence intervals for the single proportion*, Statistics
  in Medicine 17:857-872 — the source of the Wilson reference values the tests check
  against.
- Efron & Tibshirani, *An Introduction to the Bootstrap*, ch. 13.
- Phipson & Smyth (2010), *Permutation p-values should never be zero* — why the p-value
  here is `(b+1)/(B+1)` and not `b/B`.
