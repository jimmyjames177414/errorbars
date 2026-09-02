# Using errorbars on results another tool produced

`errorbars analyze` does not care which tool ran the experiment. It reads a directory of
files. If your harness can write four JSON files, you get paired effects, bootstrap
intervals, Holm correction and the null-vs-underpowered verdict without changing anything
about how you run evals.

That is a claim about **files, not imports.** Nothing in this package imports another
project, and no other project imports this one. There is no shared runtime to keep in
step, and each tool stays independently cloneable.

## The shortest possible integration

Write these four files. Full field lists are in [`schemas/`](../schemas/README.md).

```
results/my-run/
  manifest.json          # one object
  interventions.json     # array of arms
  trials.jsonl           # one line per model call
  outcomes.jsonl         # one line per score
```

**`manifest.json`** — the reproducibility envelope. Four fields are required:

```json
{
  "cxs_version": "0.1",
  "experiment_id": "my-run-2026-09-02",
  "tool": { "name": "my-harness", "version": "1.4.2" },
  "created_at": "2026-09-02T11:00:00Z",
  "determinism": "stochastic",
  "control_intervention_id": "baseline"
}
```

`determinism` must be `"deterministic"` only if you *observed* identical outputs across
repeats. Temperature 0 is not evidence.

`control_intervention_id` is optional but strongly recommended. Without it, errorbars
looks for an intervention with `"kind": "noop"`, then for an arm named something like
`control` or `baseline`, and refuses rather than guessing if neither works. An effect
measured against an arbitrarily chosen arm would be worse than an error.

**`interventions.json`** — one entry per arm. Exactly one should be the control:

```json
[
  { "id": "baseline",   "kind": "noop" },
  { "id": "no-context", "kind": "remove",
    "selector": { "type": "unit_kind", "value": "document" },
    "description": "Drop all retrieved documents." }
]
```

**`trials.jsonl`** — one line per model call. Only four fields are required
(`trial_id`, `intervention_id`, `item_id`, `response_text`), though `repeat_index` and
`model` make the analysis better:

```json
{"trial_id": "t-0001", "intervention_id": "baseline", "item_id": "q-042", "repeat_index": 0, "response_text": "yes", "model": {"provider": "openai-compatible", "model": "gpt-oss-20b", "params": {"temperature": 0.0}}}
```

Note that the *prompt itself* is not in the schema — only `prompt_sha256`, if you want it.
That is deliberate: a results directory can then be shared for re-analysis without handing
over your dataset.

**`outcomes.jsonl`** — one line per score, joined to trials by `trial_id`:

```json
{"trial_id": "t-0001", "scorer": "exact_match:v1", "passed": true, "verdict": "true"}
```

`score` (a float), `passed` (a boolean), or `verdict` (CXS 0.1.1) — any of the three is
enough. Partial-credit scorers should write `score`; pass/fail harnesses can write only
`passed` and errorbars will read it as 1.0 / 0.0.

### If your scorer has a third outcome, say so

`passed` is a boolean and cannot represent "we do not know". CXS 0.1.1 added `verdict`,
an enum of `"true"` / `"false"` / `"inconclusive"` / `"error"`, for exactly that case: a
truncated trace, a provider failure part-way, a judge that declined to answer.

```json
{"trial_id": "t-0002", "scorer": "ltl3:v1", "verdict": "inconclusive"}
```

Two rules, and errorbars enforces both:

* **Where `passed` and `verdict` are both present they must agree.** A record saying
  `passed: false, verdict: "true"` is malformed and is rejected rather than guessed at.
* **Omit `passed` and `score` entirely for `inconclusive` and `error`.** There is no
  honest value for either — `inconclusive` is not a failure. A reader that understands
  only `passed` then skips the record instead of silently counting it as one, which is
  the intended failure mode.

errorbars **skips unresolved trials, counts them, and reports the count** above the
results table. It never scores them, never coerces them to zero, and never lets them into
a denominator. A pass rate computed over trials nobody resolved is a confident-looking
wrong number, and producing one quietly would defeat the purpose of the tool.

Then:

```bash
errorbars analyze results/my-run --json --md
```

## What errorbars tolerates, and what it refuses

Being explicit about this matters, because "compatible" is a word people use loosely.

**Tolerated**, so a well-formed directory is never rejected on a technicality:

- Unknown fields anywhere. Every schema sets `additionalProperties: true`. Your harness's
  `run_group`, `wall_clock` and `judge_notes` are ignored, not fatal.
- A missing `interventions.json` — arms are then derived from the trials themselves.
- A missing `repeat_index` — order of appearance is used.
- `passed` with no `score`, or `score` with no `passed`.
- An unfamiliar `cxs_version` — you get a note and a best-effort read, not a refusal.
- Unbalanced repeats per item, and items present in one arm but not another. Both are
  reported in the notes rather than silently absorbed.

**Refused**, because guessing would be worse than failing:

- No manifest, or no trials.
- An ambiguous control arm — two `noop` interventions, or none identifiable. Pass
  `--control <id>`.
- Only one arm. An effect needs something to be an effect *against*.
- More than one model with no `--model` chosen. Pooling two models into one effect
  estimate mixes different populations, and errorbars will not do it quietly.

## The fixture is the proof

`tests/fixtures/foreign_tool_results/` is a complete results directory written in the
style of a **different** tool: another `tool.name`, arm names this project would never
generate, `passed` booleans with no `score` field, no declared control arm, and four
fields errorbars has never heard of. `tests/test_ingest.py` reads it end to end, and
`tests/test_cxs_conformance.py` validates it against the vendored schemas.

Regenerate it with `python tests/fixtures/make_foreign_fixture.py`. It is synthetic and
seeded, and labelled as such in its own manifest `notes` field — nothing in it was
produced by a language model.

Without that fixture, "reads other tools' output" would be an untested sentence in a
README. With it, it is a claim something checks on every commit.

## Emitting CXS from errorbars

`errorbars run` writes the same format, so the interop goes both ways. The one property
worth knowing about: **`trials.jsonl` is append-only and never rewritten.** A killed run
resumes from it, and the raw responses survive any later re-scoring. If your harness can
manage the same, a crashed overnight run stops being a total loss.
