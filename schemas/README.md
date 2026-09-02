# CXS v0.1 schemas (vendored)

These are a **vendored copy** of the Context Experiment Interchange Spec, v0.1. They are
not a package. Nothing here is imported from another project, and no other project
imports it.

That is the whole point. Five tools that `pip install` each other rise and fall together
and cannot be cloned in isolation. A file format lets each stay independently useful
while still interoperating: one tool writes a results directory, another reads it, and a
conformance test on each side is what makes the claim checkable.

## The files

| File | Object |
|---|---|
| `cxs-manifest.schema.json` | `RunManifest` -- the reproducibility envelope, one per run |
| `cxs-trial.schema.json` | `Trial` -- one execution, raw, one per line of `trials.jsonl` |
| `cxs-outcome.schema.json` | `Outcome` -- the scored result of a trial |
| `cxs-intervention.schema.json` | `Intervention` -- a named transformation of the context |
| `cxs-model-descriptor.schema.json` | `ModelDescriptor` -- what was asked, precisely enough to re-run |

## On-disk layout

```
results/<experiment_id>/
  manifest.json          # RunManifest
  interventions.json     # [Intervention]
  trials.jsonl           # one Trial per line, append-only, raw
  outcomes.jsonl         # one Outcome per line
  report.json            # aggregated + statistics   (errorbars analyze --json)
  report.md              # human-readable            (errorbars analyze --md)
```

`trials.jsonl` is append-only and never rewritten, so a crashed run resumes and the raw
data survives any later re-analysis. It is the single most important reproducibility
property in the spec.

## Conformance

A project is CXS v0.1 conformant if it writes a `manifest.json` with
`cxs_version: "0.1"` and all required fields, its `trials.jsonl` and `outcomes.jsonl`
validate against these schemas, and it ships a test asserting both on a real run of its
own demo.

errorbars' copy of that test is `tests/test_cxs_conformance.py`. It runs the shipped
demo end to end and validates every line of the output. Conformance is a claim about
files; no project should claim it without that test.

## Two deliberate additions

`errorbars` writes two fields the base v0.1 text does not name:

* `control_intervention_id` on the manifest. Which arm is the control is the one thing a
  reader must never guess at, and inferring it from `kind: "noop"` fails as soon as a
  spec has two of them.
* `analysis` on the manifest, recording alpha, resample count and multiplicity method,
  so the statistics that produced a claim travel with the claim.

Both are additive and every schema here sets `additionalProperties: true`, so a reader
that has never heard of them is unaffected.
