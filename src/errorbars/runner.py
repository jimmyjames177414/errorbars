"""Execute an experiment and write a CXS v0.1 results directory.

Deliberately less capable than promptfoo. It exists so the intervention/control
primitive has a first-class expression, not to compete on features.

Two properties are load-bearing:

* ``trials.jsonl`` is **append-only and never rewritten.** A crashed run resumes from
  it, and the raw responses survive any later re-analysis. This is the single most
  important reproducibility property in the interchange spec.
* Scoring is separate from generation. ``outcomes.jsonl`` can be regenerated from
  ``trials.jsonl`` with a different scorer without calling a model again.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .cache import ResponseCache, cache_key
from .errors import BudgetExceededError, ProviderError
from .grid import CHARS_PER_TOKEN, TrialKey, iter_trials
from .providers import build_provider
from .providers.base import Provider
from .scorers import get_scorer
from .spec import DatasetItem, ExperimentSpec

__all__ = ["CXS_VERSION", "RunOutcome", "new_experiment_id", "run_experiment"]

CXS_VERSION = "0.1"

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_experiment_id() -> str:
    """A ULID-shaped, lexicographically sortable id: 48-bit timestamp + 80 bits random."""
    milliseconds = int(time.time() * 1000)
    randomness = secrets.randbits(80)
    value = (milliseconds << 80) | randomness
    characters = []
    for _ in range(26):
        characters.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(characters))


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunOutcome:
    results_dir: Path
    experiment_id: str
    trials_written: int
    trials_skipped: int
    model_calls: int
    cache_hits: int
    errors: int
    determinism: str
    torn_records_discarded: int = 0
    """Partially-written final lines dropped before appending. Non-zero means the
    previous run was killed mid-write; the trials involved are simply re-run."""


def _discard_torn_final_line(path: Path) -> bool:
    """Drop a partially-written final record so the next append does not merge into it.

    A process killed mid-write leaves a line with no trailing newline. Appending after
    it would glue the next record onto the fragment and destroy *both* -- the fragment
    was already unusable, but the new trial would be lost too, silently, and only show
    up later as a JSON error during analysis.

    This does not violate the append-only rule. A record without its terminating newline
    was never a complete record; nothing that ever parsed is touched.

    Returns:
        True if a fragment was discarded.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return False

    with path.open("rb+") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return False

        # Walk back to the last newline; the fragment is everything after it.
        size = path.stat().st_size
        window = 65536
        position = size
        while position > 0:
            start = max(0, position - window)
            handle.seek(start)
            chunk = handle.read(position - start)
            index = chunk.rfind(b"\n")
            if index != -1:
                handle.truncate(start + index + 1)
                return True
            position = start
        handle.truncate(0)
        return True


def _existing_trial_identities(trials_path: Path) -> set[tuple[str, str, str, int]]:
    """Identities already present in ``trials.jsonl``, so a resume writes no duplicates.

    A truncated final line (the usual shape of a kill mid-write) is skipped rather than
    treated as fatal: the trial it represents will simply be re-run.
    """
    if not trials_path.is_file():
        return set()
    identities: set[tuple[str, str, str, int]] = set()
    with trials_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            model = record.get("model") or {}
            model_key = f"{model.get('provider')}:{model.get('model')}"
            identities.add(
                (
                    model_key,
                    str(record.get("intervention_id")),
                    str(record.get("item_id")),
                    int(record.get("repeat_index", 0)),
                )
            )
    return identities


def _build_providers(spec: ExperimentSpec) -> dict[str, Provider]:
    return {
        model.key: build_provider(
            model.provider,
            model.model,
            base_url=model.base_url,
            api_key_env=model.api_key_env,
            fixtures=(
                str((spec.source_path.parent / model.fixtures).resolve())
                if model.fixtures and spec.source_path
                else model.fixtures
            ),
        )
        for model in spec.models
    }


def _call_params(spec: ExperimentSpec, trial: TrialKey) -> dict[str, Any]:
    """Per-call params, including a per-repeat seed when the spec fixed one.

    Varying the seed across repeats is what makes repeats informative: identical seeds
    would return identical responses, which is a cache test, not a variance estimate.
    """
    params = dict(trial.model.params)
    if spec.trials.seed is not None:
        params["seed"] = spec.trials.seed + trial.repeat_index
    return params


def run_experiment(
    spec: ExperimentSpec,
    items: list[DatasetItem],
    results_dir: str | Path,
    *,
    experiment_id: str | None = None,
    cache: ResponseCache | None = None,
    max_calls: int | None = None,
    resume: bool = True,
    on_progress: Any = None,
) -> RunOutcome:
    """Run the grid, writing CXS files into ``results_dir``.

    Args:
        spec: parsed experiment. Its control arm is guaranteed present by ``load_spec``.
        items: dataset items.
        results_dir: destination; created if missing. Re-running against an existing
            directory resumes it.
        experiment_id: reuse an existing id when resuming; generated otherwise.
        cache: response cache; a disabled cache is fine.
        max_calls: hard stop. Raises BudgetExceeded on the call that would exceed it,
            after flushing everything already done.
        resume: skip trials already in ``trials.jsonl``.
        on_progress: optional ``callable(done, total)`` for a progress line.

    Returns:
        RunOutcome describing what happened, including whether the run was observed to
        be deterministic.

    Raises:
        BudgetExceededError: when ``max_calls`` is reached. Results written so far are valid
            and resumable.
    """
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    trials_path = directory / "trials.jsonl"
    outcomes_path = directory / "outcomes.jsonl"
    manifest_path = directory / "manifest.json"

    if experiment_id is None:
        if manifest_path.is_file():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                experiment_id = str(existing["experiment_id"])
            except (json.JSONDecodeError, KeyError, OSError):
                experiment_id = new_experiment_id()
        else:
            experiment_id = new_experiment_id()

    torn = [path for path in (trials_path, outcomes_path) if _discard_torn_final_line(path)]
    already = _existing_trial_identities(trials_path) if resume else set()
    cache = cache if cache is not None else ResponseCache(enabled=False)
    providers = _build_providers(spec)
    scorer = get_scorer(spec.scorer)

    planned = list(iter_trials(spec, items))
    total = len(planned)
    written = skipped = calls = errors = 0
    prompt_tokens = completion_tokens = 0
    responses_by_group: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    budget_hit = False

    (directory / "interventions.json").write_text(
        json.dumps(
            [
                {
                    "id": intervention.id,
                    "kind": intervention.kind,
                    "selector": intervention.selector,
                    "description": intervention.description,
                    "deterministic": True,
                    "implementation": f"errorbars.interventions:{intervention.kind}:v1",
                }
                for intervention in spec.interventions
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with (
        trials_path.open("a", encoding="utf-8") as trials_handle,
        outcomes_path.open("a", encoding="utf-8") as outcomes_handle,
    ):
        for index, trial in enumerate(planned, 1):
            if trial.identity in already:
                skipped += 1
                continue

            params = _call_params(spec, trial)
            key = cache_key(trial.model.provider, trial.model.model, params, trial.prompt)
            completion = cache.get(key)
            cached = completion is not None
            error_message: str | None = None

            if completion is None:
                if max_calls is not None and calls >= max_calls:
                    budget_hit = True
                    break
                try:
                    completion = providers[trial.model.key].complete(trial.prompt, params)
                    cache.put(key, completion)
                except ProviderError as exc:
                    errors += 1
                    error_message = str(exc)
                calls += 1

            trial_id = new_experiment_id()
            descriptor = trial.model.descriptor()
            descriptor["params"] = params
            descriptor["observed_at"] = _utc_now()

            response_text = completion.text if completion is not None else ""
            trials_handle.write(
                json.dumps(
                    {
                        "trial_id": trial_id,
                        "experiment_id": experiment_id,
                        "intervention_id": trial.intervention.id,
                        "item_id": trial.item.id,
                        "repeat_index": trial.repeat_index,
                        "model": descriptor,
                        "prompt_sha256": _sha256(trial.prompt),
                        "response_text": response_text,
                        "cached": cached,
                        "latency_ms": completion.latency_ms if completion else 0,
                        "usage": {
                            "prompt_tokens": completion.prompt_tokens if completion else 0,
                            "completion_tokens": completion.completion_tokens if completion else 0,
                        },
                        "error": error_message,
                        "observed_at": _utc_now(),
                    }
                )
                + "\n"
            )
            trials_handle.flush()

            if completion is not None:
                prompt_tokens += completion.prompt_tokens
                completion_tokens += completion.completion_tokens
                responses_by_group[(trial.model.key, trial.intervention.id, trial.item.id)].add(
                    response_text
                )
                score = scorer(response_text, trial.item.expected)
                outcomes_handle.write(
                    json.dumps(
                        {
                            "trial_id": trial_id,
                            "scorer": f"{spec.scorer}:v1",
                            "score": score.score,
                            "passed": score.passed,
                            "detail": score.detail,
                        }
                    )
                    + "\n"
                )
                outcomes_handle.flush()

            written += 1
            if on_progress is not None:
                on_progress(index, total)

    # "deterministic" is a claim about observed behaviour, never an inference from
    # temperature=0. With one repeat there is nothing to compare, so it stays stochastic.
    determinism = "stochastic"
    if (
        spec.trials.repeats > 1
        and responses_by_group
        and all(len(values) == 1 for values in responses_by_group.values())
    ):
        determinism = "deterministic"

    manifest = {
        "cxs_version": CXS_VERSION,
        "experiment_id": experiment_id,
        "tool": {"name": "errorbars", "version": __version__},
        "created_at": _utc_now(),
        "models": [model.descriptor() | {"observed_at": _utc_now()} for model in spec.models],
        "interventions": [
            {
                "id": intervention.id,
                "kind": intervention.kind,
                "selector": intervention.selector,
                "description": intervention.description,
                "deterministic": True,
                "implementation": f"errorbars.interventions:{intervention.kind}:v1",
            }
            for intervention in spec.interventions
        ],
        "control_intervention_id": spec.control_id,
        "dataset": {
            "name": spec.dataset_path.name,
            "sha256": _sha256(spec.dataset_path.read_text(encoding="utf-8")),
            "n_items": len(items),
            "source": str(spec.dataset_path),
        },
        "scorer": f"{spec.scorer}:v1",
        "repeats": spec.trials.repeats,
        "seed": spec.trials.seed,
        "determinism": determinism,
        "cache": {"hits": cache.hits, "misses": cache.misses},
        "totals": {
            "model_calls": calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "analysis": {
            "alpha": spec.analysis.alpha,
            "power": spec.analysis.power,
            "bootstrap_resamples": spec.analysis.bootstrap_resamples,
            "permutations": spec.analysis.permutations,
            "multiplicity": spec.analysis.multiplicity,
            "sesoi": spec.analysis.sesoi,
        },
        "environment": {
            "python": os.environ.get("ERRORBARS_PYTHON_OVERRIDE") or _python_version(),
            "platform": _platform(),
        },
        "notes": (
            f"Token counts from providers that do not report usage are 0. Prompt-token "
            f"estimates elsewhere in errorbars assume {CHARS_PER_TOKEN:g} chars/token."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    outcome = RunOutcome(
        results_dir=directory,
        experiment_id=experiment_id,
        trials_written=written,
        trials_skipped=skipped,
        model_calls=calls,
        cache_hits=cache.hits,
        errors=errors,
        determinism=determinism,
        torn_records_discarded=len(torn),
    )
    if budget_hit:
        raise BudgetExceededError(
            f"stopped at the --max-calls ceiling of {max_calls}: "
            f"{written} trial(s) written to {directory}. "
            "Re-run the same command to resume from where it stopped."
        )
    return outcome


def _python_version() -> str:
    import platform

    return platform.python_version()


def _platform() -> str:
    import sys

    return sys.platform
