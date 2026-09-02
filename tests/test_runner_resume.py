"""Killing a run part-way and picking it back up.

``trials.jsonl`` is append-only and never rewritten, which is what makes a run
resumable and what preserves the raw responses for re-analysis. The property that has to
hold is narrow and easy to break: a resumed run writes the trials that are missing and
**none** of the ones already there. Double-counting a trial would silently inflate N and
narrow every interval that follows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from errorbars.cache import ResponseCache
from errorbars.errors import BudgetExceededError
from errorbars.runner import run_experiment
from errorbars.spec import load_dataset, load_spec

SPEC = """
version: "0.1"
name: resume-demo
dataset: { path: data.jsonl }
models:
  - { provider: mock, model: sim, params: { temperature: 0.0 } }
interventions:
  - { id: control, kind: noop }
  - { id: strip-negation, kind: remove,
      selector: { type: lexical_class, value: negation } }
scorer: first_word
trials: { repeats: 3, seed: 5 }
"""

N_ITEMS = 10
TOTAL_TRIALS = N_ITEMS * 2 * 3  # items x arms x repeats


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    rows = []
    for index in range(N_ITEMS):
        left, right = 10 + index, 40 + index
        rows.append(
            json.dumps(
                {
                    "id": f"item-{index}",
                    "prompt": f"Is {left} not greater than {right}? "
                    "Answer with one word: true or false.",
                    "expected": "true",
                }
            )
        )
    (tmp_path / "data.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (tmp_path / "experiment.yaml").write_text(SPEC, encoding="utf-8")
    return tmp_path


def _identities(results: Path) -> list[tuple[str, str, int]]:
    identities = []
    for line in (results / "trials.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        identities.append((record["intervention_id"], record["item_id"], record["repeat_index"]))
    return identities


def test_a_budget_stop_can_be_resumed_with_no_duplicates(workspace: Path) -> None:
    spec, _ = load_spec(workspace / "experiment.yaml")
    items = load_dataset(spec.dataset_path)
    results = workspace / "results"
    cache = ResponseCache(workspace / "cache", enabled=False)

    with pytest.raises(BudgetExceededError, match="resume"):
        run_experiment(spec, items, results, cache=cache, max_calls=17)

    partial = _identities(results)
    assert len(partial) == 17
    assert len(set(partial)) == 17

    outcome = run_experiment(spec, items, results, cache=cache)

    full = _identities(results)
    assert len(full) == TOTAL_TRIALS, "the resumed run should complete the grid"
    assert len(set(full)) == TOTAL_TRIALS, "no trial may be written twice"
    assert set(partial) <= set(full)
    assert outcome.trials_written == TOTAL_TRIALS - 17
    assert outcome.trials_skipped == 17


def test_the_experiment_id_survives_a_resume(workspace: Path) -> None:
    """A resumed run is the same experiment; a new id would fragment the results."""
    spec, _ = load_spec(workspace / "experiment.yaml")
    items = load_dataset(spec.dataset_path)
    results = workspace / "results"

    with pytest.raises(BudgetExceededError):
        run_experiment(spec, items, results, max_calls=10)
    first_id = json.loads((results / "manifest.json").read_text(encoding="utf-8"))["experiment_id"]

    outcome = run_experiment(spec, items, results)
    assert outcome.experiment_id == first_id


def test_a_truncated_final_line_is_re_run_not_fatal(workspace: Path) -> None:
    """A kill mid-write leaves half a line. That trial is redone and the file stays valid.

    The trap here is that appending straight onto a fragment glues the next record to it
    and loses that one too -- silently, only surfacing much later as a parse error during
    analysis. The fragment has to be discarded before anything is appended.
    """
    spec, _ = load_spec(workspace / "experiment.yaml")
    items = load_dataset(spec.dataset_path)
    results = workspace / "results"

    with pytest.raises(BudgetExceededError):
        run_experiment(spec, items, results, max_calls=12)

    trials_path = results / "trials.jsonl"
    text = trials_path.read_text(encoding="utf-8")
    trials_path.write_text(text[: len(text) - 40], encoding="utf-8")

    outcome = run_experiment(spec, items, results)
    assert outcome.torn_records_discarded == 1

    # Every line still parses, and the grid is complete with no duplicates.
    full = _identities(results)
    assert len(set(full)) == TOTAL_TRIALS
    assert len(full) == TOTAL_TRIALS


def test_an_intact_file_is_never_truncated(workspace: Path) -> None:
    """The repair must only ever fire on a fragment, never on a complete record."""
    spec, _ = load_spec(workspace / "experiment.yaml")
    items = load_dataset(spec.dataset_path)
    results = workspace / "results"

    with pytest.raises(BudgetExceededError):
        run_experiment(spec, items, results, max_calls=12)
    before = (results / "trials.jsonl").read_text(encoding="utf-8")

    outcome = run_experiment(spec, items, results)
    assert outcome.torn_records_discarded == 0
    assert (results / "trials.jsonl").read_text(encoding="utf-8").startswith(before)


def test_no_resume_re_runs_everything(workspace: Path) -> None:
    spec, _ = load_spec(workspace / "experiment.yaml")
    items = load_dataset(spec.dataset_path)
    results = workspace / "results"

    run_experiment(spec, items, results)
    outcome = run_experiment(spec, items, results, resume=False)
    assert outcome.trials_written == TOTAL_TRIALS
    assert outcome.trials_skipped == 0
    assert len(_identities(results)) == TOTAL_TRIALS * 2, "append-only means both runs are kept"


def test_a_completed_run_resumes_to_nothing(workspace: Path) -> None:
    spec, _ = load_spec(workspace / "experiment.yaml")
    items = load_dataset(spec.dataset_path)
    results = workspace / "results"

    run_experiment(spec, items, results)
    outcome = run_experiment(spec, items, results)
    assert outcome.trials_written == 0
    assert outcome.trials_skipped == TOTAL_TRIALS


def test_repeats_get_different_seeds_so_they_are_not_the_same_call(workspace: Path) -> None:
    """Identical seeds across repeats would measure the cache, not the model's variance."""
    spec, _ = load_spec(workspace / "experiment.yaml")
    items = load_dataset(spec.dataset_path)
    results = workspace / "results"
    run_experiment(spec, items, results)

    seeds = set()
    for line in (results / "trials.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["item_id"] == "item-0" and record["intervention_id"] == "control":
            seeds.add(record["model"]["params"]["seed"])
    assert seeds == {5, 6, 7}


def test_determinism_is_observed_not_assumed(workspace: Path) -> None:
    """temperature=0 is not evidence. Identical outputs across repeats are."""
    spec, _ = load_spec(workspace / "experiment.yaml")
    items = load_dataset(spec.dataset_path)
    results = workspace / "results"
    outcome = run_experiment(spec, items, results)

    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["determinism"] in ("deterministic", "stochastic")
    assert manifest["determinism"] == outcome.determinism

    responses = {}
    for line in (results / "trials.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        key = (record["intervention_id"], record["item_id"])
        responses.setdefault(key, set()).add(record["response_text"])
    identical_everywhere = all(len(values) == 1 for values in responses.values())
    assert (manifest["determinism"] == "deterministic") == identical_everywhere


def test_the_cache_prevents_a_second_run_from_calling_the_model(workspace: Path) -> None:
    spec, _ = load_spec(workspace / "experiment.yaml")
    items = load_dataset(spec.dataset_path)
    cache = ResponseCache(workspace / "cache", enabled=True)

    first = run_experiment(spec, items, workspace / "run-a", cache=cache)
    assert first.model_calls == TOTAL_TRIALS

    second = run_experiment(spec, items, workspace / "run-b", cache=cache)
    assert second.model_calls == 0
    assert second.cache_hits == TOTAL_TRIALS


def test_outcomes_line_up_with_trials(workspace: Path) -> None:
    spec, _ = load_spec(workspace / "experiment.yaml")
    items = load_dataset(spec.dataset_path)
    results = workspace / "results"
    run_experiment(spec, items, results)

    trial_ids = {
        json.loads(line)["trial_id"]
        for line in (results / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    }
    outcome_ids = {
        json.loads(line)["trial_id"]
        for line in (results / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert outcome_ids == trial_ids
