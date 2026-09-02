# Contributing

Thanks for looking. This is a small library whose entire value is that the arithmetic is
correct, so contributions are held to that standard rather than to a house style.

## You do not need an API key

Not for anything. The whole test suite runs offline against a deterministic simulator, and
CI runs with **no secrets configured at all**. If you find yourself needing a key to work
on something, that is a bug in this repository — please open an issue.

```bash
git clone https://github.com/jimmyjames177414/errorbars
cd errorbars
uv venv && uv pip install -e ".[dev]"

uv run pytest -m "not live"     # the suite
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src/
```

Tests marked `live` need a real endpoint and are deselected by default. Point
`OPENAI_BASE_URL` at Ollama, vLLM or LM Studio if you want to run them.

## The bar for a statistics change

Anything under `src/errorbars/stats/` has to be checked against something **outside this
repository**. A test that compares an implementation against its own previous output tests
that nothing changed, not that anything is right.

Acceptable evidence, roughly in order of preference:

1. **A published reference value.** Newcombe's 1998 table for Wilson intervals, R's
   `p.adjust` for Holm, a textbook sample-size table. This is the best kind.
2. **Hand computation**, written out in the test docstring so a reviewer can follow it.
3. **A simulation with a known answer.** Generate a thousand experiments from a
   distribution whose true value you chose, and count. This is how the bootstrap coverage
   and the permutation false-positive rate are verified, and it is the only honest way to
   check a claim that is itself about long-run frequencies.

Snapshot tests are not acceptable for statistical code. They are fine for output
formatting.

## The bar for a claim in the README

Every number in `README.md` and `docs/statistics.md` must be reproducible by a command
printed next to it, or explicitly labelled synthetic. There are no exceptions and no
"approximately". If you change something that moves a number, re-run the command and paste
the new output — do not adjust the digits by hand.

Both documents also have a list of things this project must never claim (see `NOVELTY.md`
§8). "No framework reports variance" is on it, because Inspect AI does, and a disprovable
claim in the first screenful would cost more credibility than the feature is worth.

## Good first issues

- **Add a scorer.** `src/errorbars/scorers.py`, one function plus a registry entry. Keep
  it deterministic and local — an LLM-judge scorer adds variance the statistics here do
  not model, which is why there isn't one.
- **Add an output format.** `src/errorbars/report.py` already writes a terminal table,
  JSON and Markdown. CSV would be useful.
- **Improve the terminal table.** It is plain text on purpose (no `rich` dependency), but
  it could be better plain text.
- **Add a worked example to `docs/statistics.md`.** That document does more recruiting
  than the code does. A real scenario with real numbers is a genuine contribution.
- **Add a lexical class** to `src/errorbars/interventions.py`. Word lists live in the
  source, visible, because an intervention nobody can read is one nobody can check.

## Help wanted

Bigger pieces, listed in the README as *not built yet* rather than stubbed:

- promptfoo and Inspect AI result importers (read their native output, emit CXS)
- Bayesian analysis
- Sequential / adaptive stopping
- Mixed-effects models, replacing the two-scalar design-effect approximation

## Style

- `ruff` for linting and formatting. No black, no isort.
- `mypy --strict` on `src/`. Full annotations, no bare `Any` where a type is knowable.
- Comments explain **why**, especially where a choice is statistically load-bearing.
  "Resample items, not trials" needs its reason next to it; `# increment counter` does not.
- Errors name the file, the line, and what to do about it. `IngestError` messages are read
  by people who did not write the code that produced the file.

## Reporting a statistical bug

Please include the inputs, what you got, and what you expected **with a source**. A
disagreement with R, a textbook, or a published table is the most useful bug report this
project can receive, and it will be treated as a priority.

## Code of conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
