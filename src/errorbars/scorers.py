"""Scorers: turn a model response into a number.

Scoring is separated from generation on purpose. CXS keeps ``trials.jsonl`` (what the
model said) apart from ``outcomes.jsonl`` (what we decided about it), which means a run
can be re-scored with a different scorer without calling the model again. That property
is worth more than any individual scorer here.

Every scorer in v0.1 is deterministic and local. An LLM-judge scorer would add variance
that the statistics in this package do not model -- see "Honest limitations" in the
README -- so it is deliberately absent rather than quietly wrong.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .errors import SpecError

__all__ = ["ScoreResult", "Scorer", "get_scorer", "list_scorers"]


@dataclass(frozen=True)
class ScoreResult:
    score: float
    passed: bool
    detail: dict[str, Any]


Scorer = Callable[[str, str], ScoreResult]


_WORD_SPLIT = re.compile(r"[^\w'-]+")


def _normalise(text: str) -> str:
    """Lowercase, strip surrounding punctuation and whitespace, collapse inner spaces."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"^[\s\"'`*]+|[\s\"'`*.,;:!]+$", "", cleaned)
    return re.sub(r"\s+", " ", cleaned)


def exact_match(response: str, expected: str) -> ScoreResult:
    """1.0 when the normalised response equals the normalised expected answer."""
    got = _normalise(response)
    want = _normalise(expected)
    hit = got == want
    return ScoreResult(score=1.0 if hit else 0.0, passed=hit, detail={"expected": want, "got": got})


def contains(response: str, expected: str) -> ScoreResult:
    """1.0 when the expected string appears anywhere in the response, case-insensitively."""
    got = _normalise(response)
    want = _normalise(expected)
    hit = want in got
    return ScoreResult(score=1.0 if hit else 0.0, passed=hit, detail={"expected": want, "got": got})


def first_word(response: str, expected: str) -> ScoreResult:
    """1.0 when the response's first word matches.

    The pragmatic scorer for yes/no items, where a model answers correctly and then
    explains itself for a paragraph. Scoring that as a miss would measure verbosity
    rather than accuracy.
    """
    # Split on anything that is not a word character, so "true, because ..." yields
    # "true" rather than "true," -- otherwise the scorer marks a correct answer wrong
    # for being followed by a comma, which is the exact failure it exists to avoid.
    words = [word for word in _WORD_SPLIT.split(_normalise(response)) if word]
    got = words[0] if words else ""
    want = _normalise(expected)
    hit = got == want
    return ScoreResult(score=1.0 if hit else 0.0, passed=hit, detail={"expected": want, "got": got})


def regex_match(response: str, expected: str) -> ScoreResult:
    """1.0 when ``expected``, treated as a regex, matches the response."""
    try:
        pattern = re.compile(expected, re.IGNORECASE)
    except re.error as exc:
        raise SpecError(f"scorer 'regex' got an invalid pattern {expected!r}: {exc}") from exc
    hit = pattern.search(response) is not None
    return ScoreResult(
        score=1.0 if hit else 0.0,
        passed=hit,
        detail={"pattern": expected, "got": response.strip()[:200]},
    )


_SCORERS: dict[str, Scorer] = {
    "exact_match": exact_match,
    "contains": contains,
    "first_word": first_word,
    "regex": regex_match,
}


def list_scorers() -> list[str]:
    return sorted(_SCORERS)


def get_scorer(name: str) -> Scorer:
    """Look up a scorer by the name used in ``experiment.yaml``.

    Raises:
        SpecError: if no such scorer exists, listing the ones that do.
    """
    try:
        return _SCORERS[name]
    except KeyError:
        raise SpecError(
            f"unknown scorer {name!r}; available scorers are: {', '.join(list_scorers())}"
        ) from None
