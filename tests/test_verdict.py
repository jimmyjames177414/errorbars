"""CXS 0.1.1 `verdict`: three-valued outcomes, and never folding them into a rate.

`passed` is a boolean and cannot carry a third state. Real runs produce one anyway --
a truncated trace, a provider error part-way, a judge that declined. Squashing those into
`passed: false` turns "we do not know" into "it failed", and a pass rate computed over a
denominator that quietly includes unresolved trials is exactly the confident-looking
wrong number this package exists to argue against.

So the contract these tests pin is narrow and absolute: an unresolved trial is skipped,
counted, and reported, and it changes no rate, no interval and no denominator.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from errorbars.errors import IngestError
from errorbars.ingest import _resolve_outcome, load_run
from errorbars.report import render_analysis_table, write_report_json, write_report_md
from errorbars.stats.effects import analyse_arms

FOREIGN = Path(__file__).parent / "fixtures" / "foreign_tool_results"

WHERE = "outcomes.jsonl:1"


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    destination = tmp_path / "run"
    shutil.copytree(FOREIGN, destination)
    return destination


def _rewrite_outcomes(path: Path, mutate: object) -> None:
    """Apply ``mutate(index, record)`` to every outcome line, in place."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for index, line in enumerate(lines):
        record = json.loads(line)
        mutate(index, record)  # type: ignore[operator]
        out.append(json.dumps(record))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# Resolving one record
# --------------------------------------------------------------------------------------


def test_verdict_true_and_false_score_like_passed() -> None:
    assert _resolve_outcome({"verdict": "true"}, WHERE).score == 1.0
    assert _resolve_outcome({"verdict": "false"}, WHERE).score == 0.0


def test_unresolved_verdicts_produce_no_score_at_all() -> None:
    """Not 0.0. Not 0.5. No score -- the record leaves the analysis entirely."""
    for verdict in ("inconclusive", "error"):
        resolved = _resolve_outcome({"verdict": verdict}, WHERE)
        assert resolved.score is None
        assert resolved.verdict == verdict


def test_a_numeric_score_still_wins_for_the_value() -> None:
    """A partial-credit scorer may emit both; `verdict` gates, `score` supplies the number."""
    assert _resolve_outcome({"verdict": "true", "score": 0.75}, WHERE).score == 0.75


def test_passed_alone_still_works() -> None:
    """0.1 producers are untouched by any of this."""
    assert _resolve_outcome({"passed": True}, WHERE).score == 1.0
    assert _resolve_outcome({"passed": False}, WHERE).score == 0.0
    assert _resolve_outcome({"passed": False}, WHERE).verdict is None


def test_passed_and_verdict_may_agree() -> None:
    assert _resolve_outcome({"passed": True, "verdict": "true"}, WHERE).score == 1.0
    assert _resolve_outcome({"passed": False, "verdict": "false"}, WHERE).score == 0.0


@pytest.mark.parametrize(
    "record",
    [
        {"passed": True, "verdict": "false"},
        {"passed": False, "verdict": "true"},
    ],
)
def test_passed_contradicting_verdict_is_malformed(record: dict[str, object]) -> None:
    with pytest.raises(IngestError, match="contradicts"):
        _resolve_outcome(record, WHERE)


@pytest.mark.parametrize(
    "record",
    [
        {"verdict": "inconclusive", "passed": False},
        {"verdict": "error", "passed": True},
        {"verdict": "inconclusive", "score": 0.0},
    ],
)
def test_an_unresolved_verdict_may_not_carry_passed_or_score(
    record: dict[str, object],
) -> None:
    """The spec requires producers to omit both, precisely so a reader cannot misread them."""
    with pytest.raises(IngestError, match="omit it for an unresolved verdict"):
        _resolve_outcome(record, WHERE)


def test_a_verdict_outside_the_enum_is_rejected() -> None:
    with pytest.raises(IngestError, match="must be one of"):
        _resolve_outcome({"verdict": "maybe"}, WHERE)
    with pytest.raises(IngestError, match="must be one of"):
        _resolve_outcome({"verdict": True}, WHERE)


def test_an_outcome_with_nothing_scoreable_is_rejected() -> None:
    with pytest.raises(IngestError, match="nothing to score"):
        _resolve_outcome({"scorer": "x:v1"}, WHERE)


# --------------------------------------------------------------------------------------
# Reading a whole directory
# --------------------------------------------------------------------------------------


def test_unresolved_trials_are_skipped_counted_and_reported(run_dir: Path) -> None:
    def mutate(index: int, record: dict[str, object]) -> None:
        if index % 20 == 0:
            record.pop("passed")
            record["verdict"] = "inconclusive" if index % 40 == 0 else "error"

    _rewrite_outcomes(run_dir / "outcomes.jsonl", mutate)
    run = load_run(run_dir)

    assert run.skipped_inconclusive > 0
    assert run.skipped_error > 0
    assert run.skipped_unresolved == run.skipped_inconclusive + run.skipped_error
    assert any("inconclusive" in note and "not scored" in note for note in run.notes)


def test_unresolved_trials_never_reach_a_denominator(run_dir: Path) -> None:
    """The load-bearing assertion. Marking a trial unresolved must not move any rate.

    Two directories, identical except that in one a slice of trials is marked
    ``inconclusive`` rather than being scored. If those trials were folded in as
    failures, the control rate would fall. It must not move at all -- the only thing that
    changes is how many observations remain.
    """
    baseline = load_run(run_dir)
    baseline_analysis = analyse_arms(
        baseline.control, baseline.treatments, bootstrap_resamples=500, permutations=500
    )
    baseline_rate = baseline_analysis.effects[0].control_rate

    # Mark every trial of one item unresolved, in every arm, so no item is half-dropped.
    doomed = "task-000"
    trial_ids = {
        json.loads(line)["trial_id"]
        for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["item_id"] == doomed
    }

    def mutate(index: int, record: dict[str, object]) -> None:
        if record["trial_id"] in trial_ids:
            record.pop("passed")
            record["verdict"] = "inconclusive"

    _rewrite_outcomes(run_dir / "outcomes.jsonl", mutate)

    after = load_run(run_dir)
    assert after.skipped_inconclusive == len(trial_ids)
    assert doomed not in after.control.scores, "an unresolved item must vanish, not score 0"
    assert after.n_items == baseline.n_items - 1

    after_analysis = analyse_arms(
        after.control, after.treatments, bootstrap_resamples=500, permutations=500
    )
    after_rate = after_analysis.effects[0].control_rate

    # Dropping one item of eighty moves the mean a little; folding 3 unresolved trials in
    # as failures would move it far more, and always downward.
    assert after_rate == pytest.approx(baseline_rate, abs=0.02)


def test_a_partially_unresolved_item_keeps_only_its_resolved_repeats(run_dir: Path) -> None:
    """One bad repeat should cost that repeat, not the whole item and not a false zero."""
    target = json.loads((run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()[0])

    def mutate(index: int, record: dict[str, object]) -> None:
        if record["trial_id"] == target["trial_id"]:
            record.pop("passed")
            record["verdict"] = "error"

    _rewrite_outcomes(run_dir / "outcomes.jsonl", mutate)
    run = load_run(run_dir)

    assert run.skipped_error == 1
    assert len(run.arms[target["intervention_id"]].scores[target["item_id"]]) == 2


def test_the_tally_appears_in_the_terminal_output_above_the_table(run_dir: Path) -> None:
    def mutate(index: int, record: dict[str, object]) -> None:
        if index % 15 == 0:
            record.pop("passed")
            record["verdict"] = "error"

    _rewrite_outcomes(run_dir / "outcomes.jsonl", mutate)
    run = load_run(run_dir)
    analysis = analyse_arms(run.control, run.treatments, bootstrap_resamples=500, permutations=500)
    rendered = render_analysis_table(run, analysis)

    assert "UNRESOLVED" in rendered
    assert "excluded from every rate" in rendered
    # Above the table, so it cannot be missed by someone reading the effects first.
    assert rendered.index("UNRESOLVED") < rendered.index("intervention ")


def test_the_tally_reaches_both_report_formats(run_dir: Path) -> None:
    def mutate(index: int, record: dict[str, object]) -> None:
        if index % 15 == 0:
            record.pop("passed")
            record["verdict"] = "inconclusive"

    _rewrite_outcomes(run_dir / "outcomes.jsonl", mutate)
    run = load_run(run_dir)
    analysis = analyse_arms(run.control, run.treatments, bootstrap_resamples=500, permutations=500)

    payload = json.loads(
        write_report_json(run_dir / "report.json", run, analysis).read_text(encoding="utf-8")
    )
    assert payload["skipped_inconclusive"] == run.skipped_inconclusive
    assert payload["skipped_error"] == run.skipped_error

    markdown = write_report_md(run_dir / "report.md", run, analysis).read_text(encoding="utf-8")
    assert "unresolved" in markdown


def test_no_tally_and_no_noise_when_everything_resolved() -> None:
    run = load_run(FOREIGN)
    assert run.skipped_unresolved == 0
    analysis = analyse_arms(run.control, run.treatments, bootstrap_resamples=500, permutations=500)
    assert "UNRESOLVED" not in render_analysis_table(run, analysis)


def test_a_0_1_1_manifest_reads_without_a_best_effort_warning(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cxs_version"] = "0.1.1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    run = load_run(run_dir)
    assert run.cxs_version == "0.1.1"
    assert not any("best-effort" in note for note in run.notes)


def test_a_plain_0_1_manifest_still_reads_without_a_warning() -> None:
    assert not any("best-effort" in note for note in load_run(FOREIGN).notes)
