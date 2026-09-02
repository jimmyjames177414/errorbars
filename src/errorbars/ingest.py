"""Read any CXS v0.1 results directory into arrays ready for analysis.

This is where the interop claim lives, so it is worth being precise about what the claim
*is*. errorbars reads **files**. It does not import any sibling project, and no sibling
project imports it. A tool is compatible if it writes the layout in ``schemas/`` --
nothing else is required of it, and there is no shared runtime to keep in step.

``tests/fixtures/foreign_tool_results/`` is a results directory written in the style of
a different tool (different ``tool.name``, different intervention ids, extra fields this
package has never heard of, ``passed`` booleans instead of ``score`` floats). It is the
proof of the claim, and ``tests/test_ingest.py`` is what checks it.

Tolerances, chosen so a well-formed foreign directory is not rejected on a technicality:

* unknown top-level and per-record fields are ignored, never fatal;
* an outcome may carry ``score`` (float), ``passed`` (bool), or both;
* ``repeat_index`` may be absent, in which case order of appearance is used;
* ``interventions.json`` may be absent, in which case arms are derived from the trials.

Things that are still errors, because guessing would be worse than failing: no manifest,
no trials, no outcomes joinable to trials, an ambiguous control arm, or more than one
model with no choice made.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import IngestError
from .stats.effects import Arm

__all__ = ["IngestedRun", "load_run", "read_jsonl"]

CONTROL_NAME_HINTS = ("control", "baseline", "noop", "none", "unmodified", "identity")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file, naming the file and line on any failure."""
    if not path.is_file():
        raise IngestError(f"expected {path.name} in the results directory: {path} not found")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise IngestError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise IngestError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def _score_of(outcome: dict[str, Any], where: str) -> float:
    if outcome.get("score") is not None:
        try:
            return float(outcome["score"])
        except (TypeError, ValueError) as exc:
            raise IngestError(f"{where}: 'score' is not a number: {outcome['score']!r}") from exc
    if outcome.get("passed") is not None:
        return 1.0 if bool(outcome["passed"]) else 0.0
    raise IngestError(f"{where}: outcome has neither a 'score' nor a 'passed' field")


def _model_key(trial: dict[str, Any]) -> str:
    model = trial.get("model")
    if not isinstance(model, dict):
        return "unknown:unknown"
    return f"{model.get('provider', 'unknown')}:{model.get('model', 'unknown')}"


@dataclass
class IngestedRun:
    """One results directory, resolved down to the arms of a single model."""

    results_dir: Path
    manifest: dict[str, Any]
    tool_name: str
    tool_version: str
    cxs_version: str
    model_key: str
    available_models: list[str]
    control_id: str
    arms: dict[str, Arm]
    repeats: float
    n_items: int
    scorer: str
    notes: list[str] = field(default_factory=list)

    @property
    def control(self) -> Arm:
        return self.arms[self.control_id]

    @property
    def treatments(self) -> list[Arm]:
        return [arm for arm_id, arm in self.arms.items() if arm_id != self.control_id]


def _resolve_control(
    explicit: str | None,
    manifest: dict[str, Any],
    interventions: list[dict[str, Any]],
    arm_ids: list[str],
) -> tuple[str, str]:
    """Pick the control arm and say how it was picked."""
    if explicit is not None:
        if explicit not in arm_ids:
            raise IngestError(
                f"--control {explicit!r} is not one of the interventions present: "
                f"{', '.join(arm_ids)}"
            )
        return explicit, "chosen with --control"

    declared = manifest.get("control_intervention_id")
    if isinstance(declared, str) and declared in arm_ids:
        return declared, "declared in manifest.control_intervention_id"

    noops = [
        str(entry["id"])
        for entry in interventions
        if entry.get("kind") == "noop" and str(entry.get("id")) in arm_ids
    ]
    if len(noops) == 1:
        return noops[0], "the only intervention with kind=noop"
    if len(noops) > 1:
        raise IngestError(
            f"several interventions have kind=noop ({', '.join(noops)}); pick one with --control"
        )

    hinted = [arm_id for arm_id in arm_ids if arm_id.lower() in CONTROL_NAME_HINTS]
    if len(hinted) == 1:
        return hinted[0], f"matched by name ({hinted[0]!r})"
    if len(hinted) > 1:
        raise IngestError(
            f"several interventions look like a control ({', '.join(hinted)}); "
            "pick one with --control"
        )

    raise IngestError(
        "no control arm found. errorbars will not report an effect without one -- a "
        "difference measured against nothing is not an effect size. Interventions "
        f"present: {', '.join(arm_ids)}. Name one with --control, or add "
        "'control_intervention_id' to manifest.json."
    )


def load_run(
    results_dir: str | Path,
    *,
    model: str | None = None,
    control: str | None = None,
) -> IngestedRun:
    """Load a CXS results directory.

    Args:
        results_dir: directory containing ``manifest.json``, ``trials.jsonl`` and
            ``outcomes.jsonl``.
        model: which model's trials to analyse, as ``provider:model``. Required only
            when the directory holds more than one.
        control: which intervention is the control arm. Usually inferred.

    Raises:
        IngestError: with a message naming the file and the fix, for anything that
            cannot be resolved without guessing.
    """
    directory = Path(results_dir)
    if not directory.is_dir():
        raise IngestError(f"results directory not found: {directory}")

    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise IngestError(
            f"{directory} has no manifest.json, so it is not a CXS results directory. "
            "See schemas/cxs-manifest.schema.json for the required shape."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestError(f"{manifest_path} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise IngestError(f"{manifest_path} must contain a JSON object")

    notes: list[str] = []
    cxs_version = str(manifest.get("cxs_version", ""))
    if cxs_version != "0.1":
        notes.append(
            f"manifest declares cxs_version {cxs_version!r}; this build reads 0.1 and is "
            "proceeding on a best-effort basis"
        )

    tool = manifest.get("tool") or {}
    tool_name = str(tool.get("name", "unknown"))
    tool_version = str(tool.get("version", "unknown"))

    trials = read_jsonl(directory / "trials.jsonl")
    outcomes = read_jsonl(directory / "outcomes.jsonl")
    if not trials:
        raise IngestError(f"{directory / 'trials.jsonl'} is empty; there is nothing to analyse")

    interventions_path = directory / "interventions.json"
    interventions: list[dict[str, Any]] = []
    if interventions_path.is_file():
        try:
            loaded = json.loads(interventions_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise IngestError(f"{interventions_path} is not valid JSON: {exc}") from exc
        if isinstance(loaded, list):
            interventions = [entry for entry in loaded if isinstance(entry, dict)]
    manifest_interventions = manifest.get("interventions")
    if not interventions and isinstance(manifest_interventions, list):
        interventions = [entry for entry in manifest_interventions if isinstance(entry, dict)]

    scores_by_trial: dict[str, float] = {}
    for index, outcome in enumerate(outcomes, 1):
        trial_id = outcome.get("trial_id")
        if trial_id is None:
            raise IngestError(f"{directory / 'outcomes.jsonl'}:{index} has no 'trial_id'")
        scores_by_trial[str(trial_id)] = _score_of(
            outcome, f"{directory / 'outcomes.jsonl'}:{index}"
        )

    available_models = sorted({_model_key(trial) for trial in trials})
    if model is None:
        if len(available_models) > 1:
            raise IngestError(
                f"{directory} holds trials for {len(available_models)} models "
                f"({', '.join(available_models)}). Analyse one at a time with "
                "--model provider:model -- pooling models would mix different "
                "populations into one effect estimate."
            )
        model_key = available_models[0]
    else:
        if model not in available_models:
            raise IngestError(
                f"--model {model!r} not present; this directory has: {', '.join(available_models)}"
            )
        model_key = model

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    unscored = 0
    errored = 0
    for trial in trials:
        if _model_key(trial) != model_key:
            continue
        if trial.get("error"):
            errored += 1
            continue
        trial_id = trial.get("trial_id")
        if trial_id is None or str(trial_id) not in scores_by_trial:
            unscored += 1
            continue
        intervention_id = trial.get("intervention_id")
        item_id = trial.get("item_id")
        if intervention_id is None or item_id is None:
            raise IngestError(
                f"{directory / 'trials.jsonl'}: a trial is missing 'intervention_id' or "
                f"'item_id' (trial_id {trial_id!r})"
            )
        grouped[str(intervention_id)][str(item_id)].append(scores_by_trial[str(trial_id)])

    if not grouped:
        raise IngestError(
            f"no scored trials for model {model_key!r} in {directory}. "
            f"{unscored} trial(s) had no matching outcome and {errored} recorded an error."
        )
    if unscored:
        notes.append(f"{unscored} trial(s) had no matching outcome and were dropped")
    if errored:
        notes.append(f"{errored} trial(s) recorded a provider error and were dropped")

    arm_ids = sorted(grouped)
    control_id, how = _resolve_control(control, manifest, interventions, arm_ids)
    notes.append(f"control arm is {control_id!r} ({how})")

    if len(arm_ids) < 2:
        raise IngestError(
            f"{directory} has only the arm {arm_ids[0]!r}. An effect needs a control arm "
            "and at least one intervention to compare against it."
        )

    arms = {
        arm_id: Arm(intervention_id=arm_id, scores=dict(items)) for arm_id, items in grouped.items()
    }
    repeat_counts = [len(scores) for items in grouped.values() for scores in items.values()]
    mean_repeats = sum(repeat_counts) / len(repeat_counts)
    if len(set(repeat_counts)) > 1:
        notes.append(
            f"unbalanced repeats per item ({min(repeat_counts)}-{max(repeat_counts)}); "
            f"design calculations use the mean, {mean_repeats:.2f}"
        )

    scorer = str(manifest.get("scorer", "unknown"))

    return IngestedRun(
        results_dir=directory,
        manifest=manifest,
        tool_name=tool_name,
        tool_version=tool_version,
        cxs_version=cxs_version,
        model_key=model_key,
        available_models=available_models,
        control_id=control_id,
        arms=arms,
        repeats=mean_repeats,
        n_items=len(grouped[control_id]),
        scorer=scorer,
        notes=notes,
    )
