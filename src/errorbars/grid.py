"""The trial grid: model x intervention x item x repeat, and what it will cost.

``--dry-run`` prints the exact number of model calls and an estimated token count before
anything is spent (SHARED-FOUNDATION D5). "Exact" and "estimated" are used precisely
here: the call count is exact, because the grid is a cartesian product and is known up
front. The token count is not, because tokenisation is model-specific, so it is labelled
as an estimate everywhere it appears.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .interventions import Intervention
from .spec import DatasetItem, ExperimentSpec, ModelSpec

__all__ = ["Budget", "TrialKey", "estimate_budget", "iter_trials", "trial_identity"]

# Rough characters-per-token for English prose. Real tokenisers vary by 20-30% around
# this, which is why the output says "estimated" every single time it is printed.
CHARS_PER_TOKEN = 4.0


@dataclass(frozen=True)
class TrialKey:
    """One planned model call, fully identified before it is made."""

    model: ModelSpec
    intervention: Intervention
    item: DatasetItem
    repeat_index: int
    prompt: str

    @property
    def identity(self) -> tuple[str, str, str, int]:
        """The tuple that makes this trial unique, used for resume de-duplication."""
        return (self.model.key, self.intervention.id, self.item.id, self.repeat_index)


def trial_identity(
    model_key: str, intervention_id: str, item_id: str, repeat_index: int
) -> tuple[str, str, str, int]:
    """Build an identity tuple from loose parts, for comparing against existing trials."""
    return (model_key, intervention_id, item_id, repeat_index)


def iter_trials(spec: ExperimentSpec, items: list[DatasetItem]) -> Iterator[TrialKey]:
    """Yield every planned trial.

    Order is model, then intervention, then item, then repeat. Items are the inner loop
    over interventions rather than the reverse so that a run killed part-way still has
    complete arms for the interventions it finished, which is what makes a partial run
    analysable.
    """
    for model in spec.models:
        for intervention in spec.interventions:
            for item in items:
                prompt = intervention.apply(item.prompt)
                for repeat_index in range(spec.trials.repeats):
                    yield TrialKey(
                        model=model,
                        intervention=intervention,
                        item=item,
                        repeat_index=repeat_index,
                        prompt=prompt,
                    )


@dataclass(frozen=True)
class Budget:
    model_calls: int
    estimated_prompt_tokens: int
    n_models: int
    n_interventions: int
    n_items: int
    repeats: int

    def render(self) -> str:
        return (
            f"{self.n_models} model(s) x {self.n_interventions} intervention(s) x "
            f"{self.n_items} item(s) x {self.repeats} repeat(s)\n"
            f"  model calls            {self.model_calls:,} (exact)\n"
            f"  prompt tokens          ~{self.estimated_prompt_tokens:,} "
            f"(estimated at {CHARS_PER_TOKEN:g} chars/token; real tokenisers vary)"
        )


def estimate_budget(spec: ExperimentSpec, items: list[DatasetItem]) -> Budget:
    """Count the calls and estimate the prompt tokens for a full run.

    Ignores the cache deliberately: this is the worst case, which is the number a user
    needs before agreeing to spend money.
    """
    calls = 0
    characters = 0
    for trial in iter_trials(spec, items):
        calls += 1
        characters += len(trial.prompt)

    return Budget(
        model_calls=calls,
        estimated_prompt_tokens=int(characters / CHARS_PER_TOKEN),
        n_models=len(spec.models),
        n_interventions=len(spec.interventions),
        n_items=len(items),
        repeats=spec.trials.repeats,
    )
