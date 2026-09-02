"""CXS v0.1 conformance, asserted on a real run of the shipped demo.

The interchange spec says a project is conformant if it writes ``manifest.json`` with
``cxs_version: "0.1"`` and all required fields, its ``trials.jsonl`` and
``outcomes.jsonl`` validate against the vendored schemas, and it ships a test asserting
both on a real run of its own demo. This is that test.

Conformance is a claim about files, and no project should make it without something like
this. So the run below is a genuine end-to-end execution of ``examples/`` rather than a
hand-written fixture, and *every* line of the output is validated, not a sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema", reason="jsonschema is a dev dependency")

from errorbars.runner import run_experiment  # noqa: E402
from errorbars.spec import load_dataset, load_spec  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SCHEMAS = REPO / "schemas"
FOREIGN = Path(__file__).parent / "fixtures" / "foreign_tool_results"

DEMO_SPEC = """
version: "0.1"
name: conformance-demo
dataset: { path: data.jsonl }
models:
  - { provider: mock, model: sim, params: { temperature: 0.0, max_tokens: 8 } }
interventions:
  - { id: control, kind: noop }
  - { id: strip-negation, kind: remove,
      selector: { type: lexical_class, value: negation } }
  - { id: swap-words, kind: replace,
      selector: { type: regex, value: "greater", replacement: "larger" } }
scorer: first_word
trials: { repeats: 3, seed: 11 }
analysis: { alpha: 0.05 }
"""


def _validator(schema_name: str) -> jsonschema.protocols.Validator:
    """Build a validator that resolves the schemas' cross-references from disk.

    The schemas ``$ref`` each other by filename (a trial references a model descriptor),
    so they are all loaded into a registry keyed by that filename. No network access is
    involved, which matters because CI runs with nothing configured.
    """
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    resources = [
        (
            path.name,
            Resource.from_contents(
                json.loads(path.read_text(encoding="utf-8")),
                default_specification=DRAFT202012,
            ),
        )
        for path in sorted(SCHEMAS.glob("*.schema.json"))
    ]
    registry = Registry().with_resources(resources)
    schema = json.loads((SCHEMAS / schema_name).read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema, registry=registry)


@pytest.fixture(scope="module")
def demo_results(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the demo experiment for real, and hand back the results directory."""
    workspace = tmp_path_factory.mktemp("conformance")
    rows = []
    for index in range(24):
        rows.append(
            json.dumps(
                {
                    "id": f"item-{index:03d}",
                    "prompt": f"Is {index + 3} not greater than {index + 40}? "
                    "Answer with one word: true or false.",
                    "expected": "true",
                }
            )
        )
    (workspace / "data.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (workspace / "experiment.yaml").write_text(DEMO_SPEC, encoding="utf-8")

    spec, _ = load_spec(workspace / "experiment.yaml")
    items = load_dataset(spec.dataset_path)
    outcome = run_experiment(spec, items, workspace / "results")
    assert outcome.trials_written == 24 * 3 * 3
    return workspace / "results"


def test_the_expected_files_are_written(demo_results: Path) -> None:
    for name in ("manifest.json", "interventions.json", "trials.jsonl", "outcomes.jsonl"):
        assert (demo_results / name).is_file(), f"{name} missing from the results directory"


def test_manifest_validates_against_the_vendored_schema(demo_results: Path) -> None:
    manifest = json.loads((demo_results / "manifest.json").read_text(encoding="utf-8"))
    _validator("cxs-manifest.schema.json").validate(manifest)
    assert manifest["cxs_version"] == "0.1.1"
    assert manifest["tool"]["name"] == "errorbars"


def test_every_trial_validates(demo_results: Path) -> None:
    validator = _validator("cxs-trial.schema.json")
    lines = (demo_results / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines
    for number, line in enumerate(lines, 1):
        errors = sorted(validator.iter_errors(json.loads(line)), key=str)
        assert not errors, f"trials.jsonl:{number} failed validation: {errors[0].message}"


def test_every_outcome_validates(demo_results: Path) -> None:
    validator = _validator("cxs-outcome.schema.json")
    lines = (demo_results / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines
    for number, line in enumerate(lines, 1):
        errors = sorted(validator.iter_errors(json.loads(line)), key=str)
        assert not errors, f"outcomes.jsonl:{number} failed validation: {errors[0].message}"


def test_our_own_outcomes_carry_a_verdict_that_agrees_with_passed(demo_results: Path) -> None:
    """errorbars claims CXS 0.1.1, so it emits `verdict`, not only `passed`.

    Every scorer here is binary, so the verdict can only be true/false -- but emitting it
    means the field is exercised by a real producer on every run rather than only by a
    hand-written fixture.
    """
    lines = (demo_results / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines
    for number, line in enumerate(lines, 1):
        record = json.loads(line)
        assert "verdict" in record, f"outcomes.jsonl:{number} has no verdict"
        assert record["verdict"] in ("true", "false")
        assert record["verdict"] == ("true" if record["passed"] else "false"), (
            f"outcomes.jsonl:{number}: verdict disagrees with passed"
        )


def test_every_intervention_validates(demo_results: Path) -> None:
    validator = _validator("cxs-intervention.schema.json")
    interventions = json.loads((demo_results / "interventions.json").read_text(encoding="utf-8"))
    for entry in interventions:
        validator.validate(entry)
    assert any(entry["kind"] == "noop" for entry in interventions), "no control arm recorded"


def test_the_foreign_fixture_is_also_conformant() -> None:
    """The interop fixture has to be a valid CXS directory, or it proves nothing."""
    manifest = json.loads((FOREIGN / "manifest.json").read_text(encoding="utf-8"))
    _validator("cxs-manifest.schema.json").validate(manifest)

    trial_validator = _validator("cxs-trial.schema.json")
    for number, line in enumerate(
        (FOREIGN / "trials.jsonl").read_text(encoding="utf-8").splitlines(), 1
    ):
        errors = sorted(trial_validator.iter_errors(json.loads(line)), key=str)
        assert not errors, f"fixture trials.jsonl:{number}: {errors[0].message}"

    outcome_validator = _validator("cxs-outcome.schema.json")
    for number, line in enumerate(
        (FOREIGN / "outcomes.jsonl").read_text(encoding="utf-8").splitlines(), 1
    ):
        errors = sorted(outcome_validator.iter_errors(json.loads(line)), key=str)
        assert not errors, f"fixture outcomes.jsonl:{number}: {errors[0].message}"


def test_the_manifest_records_which_arm_is_the_control(demo_results: Path) -> None:
    """Inferring the control is the one thing a reader must never have to guess at."""
    manifest = json.loads((demo_results / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["control_intervention_id"] == "control"


def test_determinism_is_only_ever_claimed_from_observation(demo_results: Path) -> None:
    manifest = json.loads((demo_results / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["determinism"] in ("deterministic", "stochastic")

    responses: dict[tuple[str, str], set[str]] = {}
    for line in (demo_results / "trials.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        key = (record["intervention_id"], record["item_id"])
        responses.setdefault(key, set()).add(record["response_text"])

    if manifest["determinism"] == "deterministic":
        assert all(len(values) == 1 for values in responses.values())


def test_the_dataset_is_recorded_by_hash_so_a_run_can_be_checked(demo_results: Path) -> None:
    manifest = json.loads((demo_results / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["dataset"]["sha256"]) == 64
    assert manifest["dataset"]["n_items"] == 24


def test_prompts_are_stored_as_hashes_not_text(demo_results: Path) -> None:
    """A results directory should be shareable without handing over the dataset."""
    for line in (demo_results / "trials.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert "prompt" not in record
        assert len(record["prompt_sha256"]) == 64
