"""Parsing experiment.yaml, and the control arm that must always be there.

The control-arm rule is the one piece of policy in the parser: a spec without a ``noop``
intervention gets one, and the caller is told. Silence would be the wrong behaviour in
both directions -- refusing the spec is unhelpful, and injecting quietly would leave a
reader of the results believing the author designed a control they never wrote.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from errorbars.errors import SpecError
from errorbars.spec import load_dataset, load_spec

MINIMAL = """
version: "0.1"
name: demo
dataset: { path: data.jsonl }
models:
  - { provider: mock, model: sim }
interventions:
  - { id: strip-negation, kind: remove,
      selector: { type: lexical_class, value: negation } }
scorer: exact_match
"""


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "data.jsonl").write_text(
        '{"id": "a", "prompt": "Is 3 not greater than 1?", "expected": "false"}\n'
        '{"id": "b", "prompt": "Is 9 greater than 4?", "expected": "true"}\n',
        encoding="utf-8",
    )
    return tmp_path


def _write(workspace: Path, body: str) -> Path:
    path = workspace / "experiment.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_missing_control_arm_is_injected(workspace: Path) -> None:
    spec, _ = load_spec(_write(workspace, MINIMAL))
    assert [i.id for i in spec.interventions] == ["control", "strip-negation"]
    assert spec.control_id == "control"
    assert spec.interventions[0].kind == "noop"


def test_missing_control_arm_emits_a_warning(workspace: Path) -> None:
    """Injecting silently would misrepresent the author's design."""
    _, warnings = load_spec(_write(workspace, MINIMAL))
    assert len(warnings) == 1
    assert "no control arm" in warnings[0]
    assert "'control'" in warnings[0]
    assert "not reportable" in warnings[0]


def test_declared_control_arm_produces_no_warning(workspace: Path) -> None:
    body = MINIMAL.replace("interventions:\n", "interventions:\n  - { id: baseline, kind: noop }\n")
    spec, warnings = load_spec(_write(workspace, body))
    assert warnings == []
    assert spec.control_id == "baseline"


def test_injected_control_avoids_colliding_with_an_existing_id(workspace: Path) -> None:
    """An intervention already called 'control' that is not a noop must not be shadowed."""
    body = MINIMAL.replace(
        "  - { id: strip-negation",
        "  - { id: control, kind: remove,\n"
        "      selector: { type: lexical_class, value: articles } }\n"
        "  - { id: strip-negation",
    )
    spec, warnings = load_spec(_write(workspace, body))
    assert spec.control_id == "control-2"
    assert "control-2" in warnings[0]
    assert {i.id for i in spec.interventions} == {"control-2", "control", "strip-negation"}


def test_treatments_exclude_the_control(workspace: Path) -> None:
    spec, _ = load_spec(_write(workspace, MINIMAL))
    assert [i.id for i in spec.treatments] == ["strip-negation"]


def test_defaults_are_applied(workspace: Path) -> None:
    spec, _ = load_spec(_write(workspace, MINIMAL))
    assert spec.trials.repeats == 1
    assert spec.trials.seed is None
    assert spec.analysis.alpha == 0.05
    assert spec.analysis.power == 0.80
    assert spec.analysis.multiplicity == "holm"
    assert spec.budget.max_calls is None


def test_analysis_block_is_parsed(workspace: Path) -> None:
    body = (
        MINIMAL
        + """
trials: { repeats: 5, seed: 7 }
analysis: { alpha: 0.01, power: 0.9, bootstrap_resamples: 2000, sesoi: 0.02 }
budget: { max_calls: 100 }
"""
    )
    spec, _ = load_spec(_write(workspace, body))
    assert spec.trials.repeats == 5
    assert spec.trials.seed == 7
    assert spec.analysis.alpha == 0.01
    assert spec.analysis.power == 0.9
    assert spec.analysis.bootstrap_resamples == 2000
    assert spec.analysis.sesoi == 0.02
    assert spec.budget.max_calls == 100


def test_dataset_path_resolves_relative_to_the_spec(workspace: Path) -> None:
    nested = workspace / "specs"
    nested.mkdir()
    path = nested / "experiment.yaml"
    path.write_text(MINIMAL.replace("data.jsonl", "../data.jsonl"), encoding="utf-8")
    spec, _ = load_spec(path)
    assert spec.dataset_path == (workspace / "data.jsonl").resolve()


def test_unknown_scorer_fails_at_parse_time_not_after_the_bill(workspace: Path) -> None:
    body = MINIMAL.replace("scorer: exact_match", "scorer: vibes")
    with pytest.raises(SpecError, match="unknown scorer"):
        load_spec(_write(workspace, body))


def test_unknown_lexical_class_lists_the_known_ones(workspace: Path) -> None:
    body = MINIMAL.replace("value: negation", "value: adverbs")
    with pytest.raises(SpecError, match="known classes are"):
        load_spec(_write(workspace, body))


def test_duplicate_intervention_ids_are_rejected(workspace: Path) -> None:
    body = """
version: "0.1"
name: demo
dataset: { path: data.jsonl }
models:
  - { provider: mock, model: sim }
interventions:
  - { id: dupe, kind: noop }
  - { id: dupe, kind: remove, selector: { type: lexical_class, value: articles } }
scorer: exact_match
"""
    with pytest.raises(SpecError, match="duplicate intervention id"):
        load_spec(_write(workspace, body))


NO_NAME = """
version: "0.1"
dataset: { path: data.jsonl }
models:
  - { provider: mock, model: sim }
interventions:
  - { id: control, kind: noop }
"""

NO_DATASET = """
version: "0.1"
name: demo
models:
  - { provider: mock, model: sim }
interventions:
  - { id: control, kind: noop }
"""

NO_MODELS = """
version: "0.1"
name: demo
dataset: { path: data.jsonl }
interventions:
  - { id: control, kind: noop }
"""

NO_INTERVENTIONS = """
version: "0.1"
name: demo
dataset: { path: data.jsonl }
models:
  - { provider: mock, model: sim }
"""


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (NO_NAME, "spec needs a non-empty 'name'"),
        (NO_DATASET, "dataset.path is required"),
        (NO_MODELS, "at least one entry in 'models'"),
        (NO_INTERVENTIONS, "at least one entry in 'interventions'"),
    ],
)
def test_missing_required_sections_name_the_field(workspace: Path, body: str, message: str) -> None:
    with pytest.raises(SpecError, match=message):
        load_spec(_write(workspace, body))


def test_unsupported_version_is_rejected(workspace: Path) -> None:
    body = MINIMAL.replace('version: "0.1"', 'version: "0.9"')
    with pytest.raises(SpecError, match="not supported"):
        load_spec(_write(workspace, body))


def test_invalid_yaml_names_the_file(workspace: Path) -> None:
    path = workspace / "experiment.yaml"
    path.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(SpecError, match="not valid YAML"):
        load_spec(path)


def test_missing_spec_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="not found"):
        load_spec(tmp_path / "nope.yaml")


def test_bad_alpha_is_rejected(workspace: Path) -> None:
    body = MINIMAL + "\nanalysis: { alpha: 1.5 }\n"
    with pytest.raises(SpecError, match="alpha"):
        load_spec(_write(workspace, body))


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------


def test_dataset_loads(workspace: Path) -> None:
    items = load_dataset(workspace / "data.jsonl")
    assert [item.id for item in items] == ["a", "b"]
    assert items[0].expected == "false"


def test_duplicate_item_ids_are_rejected(tmp_path: Path) -> None:
    """Two items with one id would silently collapse into one during pairing."""
    path = tmp_path / "dupe.jsonl"
    path.write_text(
        '{"id": "a", "prompt": "x", "expected": "1"}\n'
        '{"id": "a", "prompt": "y", "expected": "2"}\n',
        encoding="utf-8",
    )
    with pytest.raises(SpecError, match="repeats item id"):
        load_dataset(path)


def test_missing_dataset_fields_name_the_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "a", "prompt": "x"}\n', encoding="utf-8")
    with pytest.raises(SpecError, match=r"bad\.jsonl:1 is missing required field\(s\): expected"):
        load_dataset(path)


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "gaps.jsonl"
    path.write_text(
        '{"id": "a", "prompt": "x", "expected": "1"}\n\n\n'
        '{"id": "b", "prompt": "y", "expected": "2"}\n',
        encoding="utf-8",
    )
    assert len(load_dataset(path)) == 2


def test_empty_dataset_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(SpecError, match="contains no items"):
        load_dataset(path)
