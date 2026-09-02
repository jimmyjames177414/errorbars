# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `--simultaneous-ci`: Bonferroni-adjusted simultaneous intervals at `alpha/m`, so the
  interval column controls the same family-wise error rate as the Holm p-values beside
  it. Default off. The columns are now labelled `95% CI (per-comparison)` and
  `p (Holm, family-wise)` either way, because presenting two different error rates in one
  row without saying so is the kind of quiet inconsistency this package exists to object
  to.
- **CXS 0.1.1 `verdict` support.** `passed` is a boolean and cannot represent "we do not
  know". Outcomes carrying `verdict: "inconclusive"` or `"error"` are now **skipped,
  counted and reported** rather than raising or being coerced to a failure — they are
  excluded from every rate, interval and denominator. `verdict: "true"` / `"false"` score
  as `passed` does, and a record where `passed` and `verdict` disagree, or where an
  unresolved verdict carries `passed`/`score` anyway, is rejected as malformed.
  errorbars also now *emits* `verdict` on every outcome it writes.
- `examples/unresolved_trials.py`, which measures what squashing unresolved trials
  into failures costs: on the shipped fixture with a tenth of trials unresolved the
  headline rate moves 7.3 percentage points while the effect moves only 1.7, because
  the bias lands on every arm at once. Worked through in `docs/statistics.md` sec 10.

### Changed

- `cxs_version` written by `errorbars run` is now `0.1.1`; the reader accepts `0.1` and
  `0.1.1` without complaint. The change is purely additive, so existing 0.1 directories
  read exactly as before.

## [0.1.0] - 2026-09-02

First release. A small statistics layer for LLM experiments.

### Added

- **`errorbars power`** — minimum detectable effect and required N for a paired,
  repeated eval design. No config, no network, no API key, writes nothing. Models the two
  correlation parameters that actually decide what a run can see: `--icc` (repeats within
  an item) and `--pair-corr` (item difficulty across arms). Prints a sensitivity block
  because both are assumptions until measured, and defaults `--pair-corr` to 0 so the
  headline number is never optimistic.
- **`errorbars analyze <results-dir>`** — paired effects, bootstrap confidence intervals,
  a paired permutation test, Holm–Bonferroni correction, and a
  significant / null / **underpowered** verdict. Reads any CXS v0.1 results directory,
  including one written by a different tool. Measures ICC and pairing correlation from the
  data and reports them, so they can be fed back into `power`. Writes `report.json` and
  `report.md`.
- **`errorbars run <experiment.yaml>`** — a small intervention-grid runner with a
  mandatory control arm, `--dry-run` budgeting, `--max-calls`, a content-addressed
  response cache, and resume from an append-only `trials.jsonl`.
- Statistics, all implemented here with no scipy: paired percentile bootstrap resampling
  items, paired sign-flip permutation test, Holm–Bonferroni, Wilson score intervals,
  one-way ANOVA intraclass correlation, and a clustered/paired power model that reduces
  exactly to the textbook two-proportion formula at one repeat with no pairing.
- Providers: OpenAI-compatible (Ollama, vLLM, LM Studio, llama.cpp, OpenRouter, Together,
  Groq, DeepSeek, OpenAI), a deterministic offline simulator, and a cassette replay
  provider.
- Interventions: `noop`, `remove` and `replace`, with lexical-class selectors for
  negation, articles, politeness and hedges. Word lists are in the source, visible.
- Scorers: `exact_match`, `contains`, `first_word`, `regex`. All deterministic and local.
- Vendored CXS v0.1 JSON Schemas under `schemas/`, with a conformance test that validates
  a real run of the shipped demo against them, line by line.
- **`docs/statistics.md`** — a plain-language explanation of why a bare eval percentage is
  usually meaningless, written for an engineer with no statistics background, with worked
  numbers that are all reproducible by a documented command.

### Verified, not asserted

- 95% bootstrap interval coverage measured by simulating 1000 complete experiments from a
  known distribution: **94.9%** single-arm, **94.4%** paired.
- Permutation test false-positive rate under a true null: **0.051** on continuous data
  (2000 simulations), **0.048** on binary data (1000 simulations), at alpha = 0.05.
- Wilson intervals reproduce all four reference values in Newcombe (1998).
- Holm reproduces R's `p.adjust(..., method = "holm")` and the Wikipedia worked example.
- Required N reproduces hand computation at three points: 80%→85% needs 906 per arm,
  25%→40% needs 152, 50%→60% needs 388.

### Deliberately not included

Listed in the README as gaps rather than shipped as stubs: promptfoo and Inspect AI result
importers, Bayesian analysis, sequential/adaptive stopping, mixed-effects models, HTML
reports, leaderboards, and any hosted component. No LLM-judge scorer, because it would add
variance the statistics here do not model.

[Unreleased]: https://github.com/jimmyjames177414/errorbars/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jimmyjames177414/errorbars/releases/tag/v0.1.0
