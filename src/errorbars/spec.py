"""Parse and validate ``experiment.yaml``.

The spec is deliberately smaller than promptfoo's. Two things in it are the point:

1. **The control arm is mandatory.** An effect size without a control is not an effect
   size, it is a number. If a spec has no ``noop`` intervention, one is injected and a
   warning is emitted -- loudly, not silently, because a reader of the results needs to
   know the control was the tool's idea rather than the author's.
2. **The ``analysis`` block is first class.** alpha, resample count and multiplicity
   method are recorded in the manifest alongside the results, so the statistics that
   produced a claim travel with it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import SpecError
from .interventions import Intervention, build_intervention
from .scorers import get_scorer

__all__ = [
    "AnalysisSpec",
    "BudgetSpec",
    "DatasetItem",
    "ExperimentSpec",
    "ModelSpec",
    "TrialsSpec",
    "load_dataset",
    "load_spec",
]

SPEC_VERSION = "0.1"

CONTROL_ID = "control"
_CONTROL_WARNING = (
    "no control arm found in `interventions`: injected {id!r} (kind=noop). "
    "Every reported effect is measured against it. An effect size without a control "
    "is not reportable -- declare the control explicitly to silence this."
)


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    fixtures: str | None = None
    """For provider=cassette: directory of recorded responses."""

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"

    def descriptor(self) -> dict[str, Any]:
        """CXS ModelDescriptor, minus ``observed_at`` which the runner stamps."""
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class TrialsSpec:
    repeats: int = 1
    seed: int | None = None


@dataclass(frozen=True)
class AnalysisSpec:
    alpha: float = 0.05
    power: float = 0.80
    bootstrap_resamples: int = 10_000
    permutations: int = 10_000
    multiplicity: str = "holm"
    sesoi: float | None = None


@dataclass(frozen=True)
class BudgetSpec:
    max_calls: int | None = None


@dataclass(frozen=True)
class DatasetItem:
    id: str
    prompt: str
    expected: str


@dataclass(frozen=True)
class ExperimentSpec:
    version: str
    name: str
    dataset_path: Path
    models: list[ModelSpec]
    interventions: list[Intervention]
    scorer: str
    trials: TrialsSpec
    analysis: AnalysisSpec
    budget: BudgetSpec
    source_path: Path | None = None

    @property
    def control_id(self) -> str:
        for intervention in self.interventions:
            if intervention.kind == "noop":
                return intervention.id
        raise SpecError("spec has no control arm, which load_spec should have made impossible")

    @property
    def treatments(self) -> list[Intervention]:
        control = self.control_id
        return [i for i in self.interventions if i.id != control]


def _require_mapping(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _parse_model(raw: object, index: int) -> ModelSpec:
    data = _require_mapping(raw, f"models[{index}]")
    provider = data.get("provider")
    model = data.get("model")
    if not isinstance(provider, str) or not provider:
        raise SpecError(f"models[{index}] needs a non-empty 'provider'")
    if not isinstance(model, str) or not model:
        raise SpecError(f"models[{index}] needs a non-empty 'model'")

    params = data.get("params", {})
    if not isinstance(params, dict):
        raise SpecError(f"models[{index}].params must be a mapping")

    return ModelSpec(
        provider=provider,
        model=model,
        base_url=data.get("base_url"),
        api_key_env=data.get("api_key_env"),
        params=dict(params),
        fixtures=data.get("fixtures"),
    )


def load_spec(path: str | Path) -> tuple[ExperimentSpec, list[str]]:
    """Load an experiment spec.

    Returns:
        The parsed spec and a list of warnings -- currently the control-arm injection
        notice. Warnings are returned rather than printed so the caller decides where
        they land (stderr for the CLI, an assertion for the tests).

    Raises:
        SpecError: for anything malformed, with the field named in the message.
    """
    spec_path = Path(path)
    if not spec_path.is_file():
        raise SpecError(f"experiment spec not found: {spec_path}")

    try:
        loaded = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SpecError(f"{spec_path} is not valid YAML: {exc}") from exc

    data = _require_mapping(loaded, str(spec_path))
    warnings: list[str] = []

    version = str(data.get("version", SPEC_VERSION))
    if version != SPEC_VERSION:
        raise SpecError(
            f"spec version {version!r} is not supported; this build understands {SPEC_VERSION!r}"
        )

    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SpecError("spec needs a non-empty 'name'")

    dataset = _require_mapping(data.get("dataset", {}), "dataset")
    dataset_path_value = dataset.get("path")
    if not isinstance(dataset_path_value, str) or not dataset_path_value:
        raise SpecError("dataset.path is required and must be a string")
    dataset_path = (spec_path.parent / dataset_path_value).resolve()

    raw_models = data.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise SpecError("spec needs at least one entry in 'models'")
    models = [_parse_model(raw, index) for index, raw in enumerate(raw_models)]

    raw_interventions = data.get("interventions")
    if not isinstance(raw_interventions, list) or not raw_interventions:
        raise SpecError("spec needs at least one entry in 'interventions'")

    interventions: list[Intervention] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_interventions):
        entry = _require_mapping(raw, f"interventions[{index}]")
        intervention_id = entry.get("id")
        if not isinstance(intervention_id, str) or not intervention_id:
            raise SpecError(f"interventions[{index}] needs a non-empty 'id'")
        if intervention_id in seen:
            raise SpecError(f"duplicate intervention id {intervention_id!r}")
        seen.add(intervention_id)
        interventions.append(
            build_intervention(
                intervention_id,
                str(entry.get("kind", "")),
                entry.get("selector"),
                entry.get("description"),
            )
        )

    if not any(intervention.kind == "noop" for intervention in interventions):
        control_id = CONTROL_ID
        suffix = 2
        while control_id in seen:
            control_id = f"{CONTROL_ID}-{suffix}"
            suffix += 1
        interventions.insert(0, build_intervention(control_id, "noop"))
        warnings.append(_CONTROL_WARNING.format(id=control_id))

    scorer_name = data.get("scorer", "exact_match")
    if not isinstance(scorer_name, str):
        raise SpecError("scorer must be a string")
    get_scorer(scorer_name)  # fail now, not two hundred model calls from now

    trials_raw = _require_mapping(data.get("trials", {}), "trials")
    repeats = int(trials_raw.get("repeats", 1))
    if repeats < 1:
        raise SpecError(f"trials.repeats must be >= 1, got {repeats}")
    seed_raw = trials_raw.get("seed")
    trials = TrialsSpec(repeats=repeats, seed=None if seed_raw is None else int(seed_raw))

    analysis_raw = _require_mapping(data.get("analysis", {}), "analysis")
    multiplicity = str(analysis_raw.get("multiplicity", "holm"))
    if multiplicity not in ("holm", "none"):
        raise SpecError(f"analysis.multiplicity must be 'holm' or 'none', got {multiplicity!r}")
    sesoi_raw = analysis_raw.get("sesoi")
    analysis = AnalysisSpec(
        alpha=float(analysis_raw.get("alpha", 0.05)),
        power=float(analysis_raw.get("power", 0.80)),
        bootstrap_resamples=int(analysis_raw.get("bootstrap_resamples", 10_000)),
        permutations=int(analysis_raw.get("permutations", 10_000)),
        multiplicity=multiplicity,
        sesoi=None if sesoi_raw is None else float(sesoi_raw),
    )
    if not 0.0 < analysis.alpha < 1.0:
        raise SpecError(f"analysis.alpha must be in (0, 1), got {analysis.alpha}")
    if not 0.0 < analysis.power < 1.0:
        raise SpecError(f"analysis.power must be in (0, 1), got {analysis.power}")

    budget_raw = _require_mapping(data.get("budget", {}), "budget")
    max_calls_raw = budget_raw.get("max_calls")
    budget = BudgetSpec(max_calls=None if max_calls_raw is None else int(max_calls_raw))

    return (
        ExperimentSpec(
            version=version,
            name=name.strip(),
            dataset_path=dataset_path,
            models=models,
            interventions=interventions,
            scorer=scorer_name,
            trials=trials,
            analysis=analysis,
            budget=budget,
            source_path=spec_path,
        ),
        warnings,
    )


def load_dataset(path: str | Path) -> list[DatasetItem]:
    """Read a JSONL dataset of ``{"id", "prompt", "expected"}`` records.

    Raises:
        SpecError: if the file is missing, a line is not JSON, a field is absent, or an
            id repeats. Duplicate ids would silently collapse two items into one during
            pairing, so they are rejected rather than deduplicated.
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise SpecError(f"dataset not found: {dataset_path}")

    items: list[DatasetItem] = []
    seen: set[str] = set()
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SpecError(f"{dataset_path}:{line_number} is not valid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise SpecError(f"{dataset_path}:{line_number} must be a JSON object")
        missing = [key for key in ("id", "prompt", "expected") if key not in record]
        if missing:
            raise SpecError(
                f"{dataset_path}:{line_number} is missing required field(s): {', '.join(missing)}"
            )
        item_id = str(record["id"])
        if item_id in seen:
            raise SpecError(f"{dataset_path}:{line_number} repeats item id {item_id!r}")
        seen.add(item_id)
        items.append(
            DatasetItem(
                id=item_id,
                prompt=str(record["prompt"]),
                expected=str(record["expected"]),
            )
        )

    if not items:
        raise SpecError(f"{dataset_path} contains no items")
    return items
