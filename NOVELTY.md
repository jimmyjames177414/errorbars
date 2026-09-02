# NOVELTY.md — errorbars

Novelty gate completed 2026-09-02, before any code was written. This project was **renamed and
substantially cut down** as a result: the original concept was a YAML-driven LLM experiment
runner, which the gate found to be commoditized. Every claim below is sourced.

---

## 1. Problem

An eval reports that prompt B beat prompt A by 4 points. Nobody knows whether that is real.
Eval harnesses run trials and print percentages; almost none report an interval, a paired
comparison against a control, or a power calculation. So practitioners size eval sets by feel,
ship on 4-point deltas that are inside the noise, and cannot distinguish "this intervention did
nothing" from "our run was too small to tell."

## 2. Existing closest projects

| Tool | Stars | YAML spec | Input transforms | N-trials | Variance reported | Cross-model |
|---|---|---|---|---|---|---|
| [promptfoo](https://github.com/promptfoo/promptfoo) | 24,765 | **yes, native** | `transformVars` | `evaluateOptions.repeat` | **no** | yes |
| [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) (UK AISI) | 2,686 | no (Python) | solver chains | `--epochs` | **yes — `stderr` + reducers** | yes |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 13,872 | yes | template variants | not first-class | no | yes |
| [HELM](https://github.com/stanford-crfm/helm) | 2,898 | partly | run expanders | no | no | yes |
| [DeepEval](https://github.com/confident-ai/deepeval) | 18,059 | no | no | parametrize | no | yes |
| [Opik](https://github.com/comet-ml/opik) | 21,751 | no | no | no | no | yes |
| [Langfuse](https://github.com/langfuse/langfuse) | 34,109 | no | no | dataset runs | no | yes |

Star counts and push dates verified via the GitHub API on 2026-09-02.

**Critically: promptfoo was acquired by OpenAI on 2026-03-09**
([announcement](https://openai.com/index/openai-to-acquire-promptfoo/)), remaining open source
under its current licence. The incumbent in this niche is funded by a frontier lab.

Methodology prior art:
- **[arXiv:2411.00640](https://arxiv.org/abs/2411.00640)** — *Adding Error Bars to Evals: A
  Statistical Approach to Language Model Evaluations*. **This project's name is a deliberate nod
  to it, and it is the primary methodology citation.**
- **[arXiv:2605.30315](https://arxiv.org/abs/2605.30315)** — *Resolution Diagnostics for Paired
  LLM Evaluation*: paired hypothesis testing, MDE at current N, Holm multiplicity, design-effect
  cluster correction.

## 3. Exact overlap

The original "ContextLab" concept — a common YAML experiment specification and runner — is
**already built, twice**:

- **YAML experiment spec** → promptfoo (native, since 2023), lm-evaluation-harness, HELM.
- **Input-side transformation** → promptfoo `options.transformVars`, *"runs on the vars before
  they are substituted into the prompt."*
- **Repeated trials** → promptfoo `evaluateOptions.repeat`, Inspect `--epochs`.
- **Variance reporting** → **Inspect AI already does this well**: `--epochs-reducer`
  (`mean, median, mode, max, at_least_{n}, pass_at_{k}`) and scorers reporting `stderr`.
- **Cross-model comparison** → nearly everyone.
- **Shareable machine-readable artifacts** → promptfoo exports nine formats plus hosted sharing.

**A YAML-driven LLM experiment runner is a solved, crowded problem.** Building a fifth would be
a me-too.

## 4. Exact differentiation

Three things, all verified absent from the tools above:

1. **Intervention as the swept dimension, with a per-intervention effect table.** Nothing in the
   surveyed field labels results by intervention or emits "removing X moved the score by δ, N
   trials, 95% CI". promptfoo has `redteam.strategies` (a named mutator library) and
   `transformVars` (an input transform) but has never joined them into an ablation report.
2. **Null results as a first-class output.** Every framework optimises for "did it pass". None
   reports "this intervention provably did nothing, and here is the power calculation showing we
   could have detected it if it had." Distinguishing **null** from **underpowered** is the single
   most useful thing this tool does.
3. **Power / MDE analysis up front.** No surveyed tool answers "how many items and repeats do I
   need to detect a 2-point effect?" The statistics are published (§2); the tooling is not.

## 5. Honest sizing — read this before believing the pitch

**This is a feature-sized contribution, not a category-sized one.** It is roughly "the
statistics your eval harness doesn't do," and promptfoo's architecture is arguably one design
decision away from absorbing it — a risk sharpened by the OpenAI acquisition.

It is shipped as a separate small library anyway because (a) it is useful today, (b) it works on
results produced by *any* tool that emits the interchange format, and (c) a 900-line focused
library that does one thing correctly is more valuable than a platform that does ten things
approximately. The README states this sizing openly rather than implying a platform.

## 6. Why someone would use this instead

- You already have eval results and need to know whether a difference is real.
- You need to justify a run size *before* spending money on it.
- You need to report a null result honestly and be believed.
- You want the statistics without adopting a whole harness — it reads results, it does not
  demand you switch.

## 7. Research questions this makes askable

1. How many published prompt-engineering "improvements" are inside the noise of their own evals?
2. What is the typical MDE of a real-world eval suite — i.e. how small an effect can anyone
   actually detect?
3. How much does repeat-within-item correlation inflate apparent precision?
4. Does effect size on an ablation transfer across models, or is the variance mostly model-specific?

## 8. Claims we MUST NOT make

| Forbidden claim | Why it is false | Source |
|---|---|---|
| "The first YAML-driven LLM experiment runner" | promptfoo, YAML-native since 2023, 24,765★ | github.com/promptfoo/promptfoo |
| "No tool transforms context before the model call" | `transformVars` does exactly this | promptfoo.dev/docs/configuration/guide/ |
| "No eval framework supports repeated trials" | promptfoo `repeat`; Inspect `--epochs` | promptfoo.dev docs; inspect.aisi.org.uk |
| **"No framework reports variance across trials"** | **Inspect AI ships `--epochs-reducer` and scorers report `stderr`** | inspect.aisi.org.uk/options.html |
| "Cross-model comparison is unsolved" | `eval_set(model=[...])`; promptfoo provider matrix | inspect.aisi.org.uk/eval-sets.html |
| "There is no shareable artifact standard" | promptfoo exports nine formats plus hosted sharing | promptfoo.dev/docs/usage/command-line/ |
| "We invented error bars for evals" | arXiv:2411.00640 | arXiv |
| "promptfoo is an independent startup" | OpenAI acquired it 2026-03-09 | openai.com |

Also banned: "first ever", "revolutionary", "the standard for LLM experiments".

## 9. Recommendation

**RESHAPE AND RENAME, then GO small** — both done before implementation:

- **Renamed** `ContextLab` → `errorbars`. The GitHub org
  [`ContextLab`](https://github.com/ContextLab) is the **Contextual Dynamics Laboratory at
  Dartmouth** — created 2016, 118 public repos, pushed 2026-09-02, flagship `hypertools`
  (1,887★) — and it is *now publishing LLM-context research*. `ContextLab/contextlab` is
  unavailable and the npm scope `@contextlab` is taken. Shipping "ContextLab" doing context
  science would read as coming from them, compete with them in search, and be an unforced
  discourtesy to an adjacent lab we are more likely to cite than collide with.
- **Cut** from "umbrella experiment platform" (dead on arrival) to a focused statistics library.
- **Repositioned** as complementary: *if you want a regression suite use promptfoo; if you want
  rigorous trials in Python use Inspect AI; use errorbars when you need to know whether a
  difference is real.*

Note on vocabulary: the word "ablation" is avoided in the tagline. In open-source it has been
largely captured by weight-level **abliteration** (removing refusal directions from activations
— e.g. `elder-plinius/OBLITERATUS`, 8,164★), and would surface the wrong audience.
