"""Reading a results directory written by a *different* tool.

The interop claim in the README is that errorbars analyses any CXS results directory,
whoever produced it. That claim is only worth making if something checks it, so
``tests/fixtures/foreign_tool_results/`` is a synthetic directory written in another
tool's style -- different ``tool.name``, different arm names, ``passed`` booleans with no
``score`` field, no ``control_intervention_id``, and extra fields errorbars has never
heard of. Everything below reads that fixture rather than errorbars' own output.

Regenerate it with ``python tests/fixtures/make_foreign_fixture.py``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from errorbars.errors import IngestError
from errorbars.ingest import load_run
from errorbars.stats.effects import analyse_arms

FOREIGN = Path(__file__).parent / "fixtures" / "foreign_tool_results"


@pytest.fixture()
def foreign_copy(tmp_path: Path) -> Path:
    destination = tmp_path / "run"
    shutil.copytree(FOREIGN, destination)
    return destination


def test_reads_a_foreign_results_directory() -> None:
    run = load_run(FOREIGN)
    assert run.tool_name == "contextprobe"
    assert run.tool_version == "2.4.0"
    assert run.cxs_version == "0.1"
    assert run.model_key == "openai-compatible:example-8b"
    assert set(run.arms) == {"baseline", "shuffle-context", "drop-tool-schemas"}
    assert run.n_items == 80
    assert run.repeats == pytest.approx(3.0)


def test_finds_the_control_arm_without_being_told() -> None:
    """The fixture has no ``control_intervention_id``; the noop kind is the only clue."""
    run = load_run(FOREIGN)
    assert run.control_id == "baseline"
    assert any("kind=noop" in note for note in run.notes)
    assert [arm.intervention_id for arm in run.treatments] == [
        "shuffle-context",
        "drop-tool-schemas",
    ]


def test_scores_come_from_passed_booleans_when_there_is_no_score_field() -> None:
    run = load_run(FOREIGN)
    for scores in run.arms["baseline"].scores.values():
        assert all(value in (0.0, 1.0) for value in scores)


def test_unknown_fields_are_ignored_rather_than_fatal() -> None:
    """``run_group``, ``harness_seed``, ``wall_clock``, ``judge_notes`` all pass through."""
    raw = json.loads((FOREIGN / "trials.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert {"run_group", "harness_seed", "wall_clock"} <= set(raw)
    assert load_run(FOREIGN).n_items == 80


def test_the_foreign_run_analyses_end_to_end() -> None:
    """The fixture was built with a real -20pp effect on one arm and none on the other."""
    run = load_run(FOREIGN)
    analysis = analyse_arms(
        run.control,
        run.treatments,
        bootstrap_resamples=2000,
        permutations=2000,
        seed=0,
    )
    by_id = {effect.intervention_id: effect for effect in analysis.effects}

    assert by_id["shuffle-context"].verdict == "significant"
    assert by_id["shuffle-context"].effect < -0.10
    assert by_id["shuffle-context"].ci_high < 0.0

    assert abs(by_id["drop-tool-schemas"].effect) < 0.05
    assert by_id["drop-tool-schemas"].verdict != "significant"


def test_explicit_control_overrides_inference(foreign_copy: Path) -> None:
    run = load_run(foreign_copy, control="drop-tool-schemas")
    assert run.control_id == "drop-tool-schemas"
    assert any("--control" in note for note in run.notes)


def test_unknown_control_lists_what_is_available(foreign_copy: Path) -> None:
    with pytest.raises(IngestError, match="not one of the interventions present"):
        load_run(foreign_copy, control="nope")


def test_control_can_be_found_by_name_when_interventions_json_is_missing(
    foreign_copy: Path,
) -> None:
    (foreign_copy / "interventions.json").unlink()
    run = load_run(foreign_copy)
    assert run.control_id == "baseline"
    assert any("matched by name" in note for note in run.notes)


def test_no_identifiable_control_refuses_rather_than_guessing(foreign_copy: Path) -> None:
    """Reporting an effect against an arbitrarily chosen arm would be worse than failing."""
    (foreign_copy / "interventions.json").unlink()
    trials = foreign_copy / "trials.jsonl"
    trials.write_text(
        trials.read_text(encoding="utf-8").replace('"baseline"', '"arm-one"'),
        encoding="utf-8",
    )
    with pytest.raises(IngestError, match="no control arm found"):
        load_run(foreign_copy)


def test_missing_manifest_is_a_clear_error(tmp_path: Path) -> None:
    (tmp_path / "trials.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(IngestError, match="not a CXS results directory"):
        load_run(tmp_path)


def test_missing_trials_file_names_it(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text('{"cxs_version": "0.1"}', encoding="utf-8")
    with pytest.raises(IngestError, match=r"trials\.jsonl"):
        load_run(tmp_path)


def test_malformed_jsonl_names_the_line(foreign_copy: Path) -> None:
    path = foreign_copy / "outcomes.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[4] = "{not json"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(IngestError, match=r"outcomes\.jsonl:5"):
        load_run(foreign_copy)


def test_multiple_models_require_an_explicit_choice(foreign_copy: Path) -> None:
    """Pooling two models into one effect would mix different populations."""
    path = foreign_copy / "trials.jsonl"
    original = path.read_text(encoding="utf-8")
    doubled = original + original.replace('"example-8b"', '"example-70b"')
    path.write_text(doubled, encoding="utf-8")

    with pytest.raises(IngestError, match="Analyse one at a time"):
        load_run(foreign_copy)

    run = load_run(foreign_copy, model="openai-compatible:example-70b")
    assert run.model_key == "openai-compatible:example-70b"
    assert len(run.available_models) == 2


def test_unknown_model_lists_what_is_present(foreign_copy: Path) -> None:
    with pytest.raises(IngestError, match="this directory has"):
        load_run(foreign_copy, model="openai-compatible:not-here")


def test_trials_recording_an_error_are_dropped_and_counted(foreign_copy: Path) -> None:
    path = foreign_copy / "trials.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    for index in range(3):
        record = json.loads(lines[index])
        record["error"] = "connection refused"
        lines[index] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    run = load_run(foreign_copy)
    assert any("provider error" in note for note in run.notes)


def test_outcomes_without_a_score_or_passed_field_are_rejected(foreign_copy: Path) -> None:
    path = foreign_copy / "outcomes.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record.pop("passed")
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(IngestError, match="neither a 'score' nor a 'passed'"):
        load_run(foreign_copy)


def test_a_numeric_score_is_preferred_over_passed(foreign_copy: Path) -> None:
    """A partial-credit scorer writes floats; those must not be flattened to 0/1."""
    path = foreign_copy / "outcomes.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["score"] = 0.375
    record["passed"] = False
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    run = load_run(foreign_copy)
    all_scores = [value for scores in run.arms["baseline"].scores.values() for value in scores]
    assert 0.375 in all_scores


def test_a_single_arm_directory_is_refused(foreign_copy: Path) -> None:
    path = foreign_copy / "trials.jsonl"
    kept = [line for line in path.read_text(encoding="utf-8").splitlines() if '"baseline"' in line]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    with pytest.raises(IngestError, match=r"needs a control arm and at least one intervention"):
        load_run(foreign_copy)


def test_an_unexpected_cxs_version_is_a_note_not_a_refusal(foreign_copy: Path) -> None:
    """Refusing a future minor version outright would make the format brittle."""
    manifest_path = foreign_copy / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cxs_version"] = "0.2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    run = load_run(foreign_copy)
    assert any("best-effort" in note for note in run.notes)
