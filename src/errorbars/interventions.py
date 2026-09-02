"""Context interventions: named, deterministic, reproducible transformations of a prompt.

An intervention is the swept dimension of an errorbars experiment, the way a
hyperparameter is the swept dimension of a training sweep. Every one of them is a pure
function of the prompt text, so a run is reproducible from the spec alone.

The set here is deliberately small. It exists so that the intervention/control primitive
has a first-class expression, not to compete with promptfoo's mutator library. Adding a
selector is a good first issue.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .errors import SpecError

__all__ = [
    "LEXICAL_CLASSES",
    "Intervention",
    "apply_intervention",
    "build_intervention",
]

InterventionKind = Literal["noop", "remove", "replace"]

# Word lists are intentionally visible rather than hidden behind a dependency: an
# intervention nobody can read is an intervention nobody can check. These are ordinary
# English function words, listed here in full so a reader can see exactly what a run
# deleted.
LEXICAL_CLASSES: dict[str, tuple[str, ...]] = {
    "negation": (
        "not",
        "n't",
        "no",
        "never",
        "none",
        "nobody",
        "nothing",
        "nowhere",
        "neither",
        "nor",
        "without",
        "cannot",
    ),
    "articles": ("a", "an", "the"),
    "politeness": (
        "please",
        "kindly",
        "thanks",
        "thank",
        "sorry",
        "excuse",
        "appreciate",
        "grateful",
    ),
    "hedges": (
        "maybe",
        "perhaps",
        "possibly",
        "probably",
        "arguably",
        "roughly",
        "somewhat",
        "fairly",
        "quite",
    ),
}


@dataclass(frozen=True)
class Intervention:
    """A parsed, ready-to-apply intervention.

    ``transform`` is the compiled implementation; ``description`` is what goes into the
    CXS ``interventions.json`` so a reader of the results knows what was done to the
    prompt without having the spec in front of them.
    """

    id: str
    kind: InterventionKind
    description: str
    selector: dict[str, str]
    transform: Callable[[str], str]

    def apply(self, text: str) -> str:
        return self.transform(text)


def _tidy_whitespace(text: str) -> str:
    """Collapse the double spaces and orphaned punctuation that deletion leaves behind.

    Without this, a removal intervention also silently becomes a "weird whitespace"
    intervention, and the measured effect would be confounded by formatting.
    """
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def _word_remover(words: tuple[str, ...]) -> Callable[[str], str]:
    # "n't" needs the apostrophe form matched as a suffix; the rest are whole words.
    plain = [word for word in words if not word.startswith("n't")]
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(word) for word in plain) + r")(?!\w)",
        re.IGNORECASE,
    )
    contraction = re.compile(r"n['’]t(?!\w)", re.IGNORECASE)
    has_contraction = any(word.startswith("n't") for word in words)

    def transform(text: str) -> str:
        if has_contraction:
            text = contraction.sub("", text)
        return _tidy_whitespace(pattern.sub("", text))

    return transform


def _regex_remover(pattern_source: str) -> Callable[[str], str]:
    try:
        pattern = re.compile(pattern_source, re.IGNORECASE)
    except re.error as exc:
        raise SpecError(f"selector regex {pattern_source!r} does not compile: {exc}") from exc

    def transform(text: str) -> str:
        return _tidy_whitespace(pattern.sub("", text))

    return transform


def _regex_replacer(pattern_source: str, replacement: str) -> Callable[[str], str]:
    try:
        pattern = re.compile(pattern_source, re.IGNORECASE)
    except re.error as exc:
        raise SpecError(f"selector regex {pattern_source!r} does not compile: {exc}") from exc

    def transform(text: str) -> str:
        return _tidy_whitespace(pattern.sub(replacement, text))

    return transform


def build_intervention(
    intervention_id: str,
    kind: str,
    selector: dict[str, str] | None = None,
    description: str | None = None,
) -> Intervention:
    """Compile one intervention from its spec fields.

    Supported shapes::

        {id: control,   kind: noop}
        {id: strip-neg, kind: remove,  selector: {type: lexical_class, value: negation}}
        {id: strip-x,   kind: remove,  selector: {type: regex, value: "\\bfoo\\b"}}
        {id: swap,      kind: replace, selector: {type: regex, value: "yes",
                                                  replacement: "no"}}

    Raises:
        SpecError: for an unknown kind, an unknown selector type, or a missing field.
    """
    selector = dict(selector or {})

    # Check the kind before the selector, so an unrecognised kind is reported as one
    # rather than as a complaint about the selector it was never going to use.
    if kind not in ("noop", "remove", "replace"):
        raise SpecError(
            f"unknown intervention kind {kind!r} for {intervention_id!r}; "
            "expected 'noop', 'remove' or 'replace'"
        )

    if kind == "noop":
        return Intervention(
            id=intervention_id,
            kind="noop",
            description=description or "Control arm: the prompt is passed through unchanged.",
            selector={},
            transform=lambda text: text,
        )

    selector_type = selector.get("type")
    value = selector.get("value")
    if not selector_type or value is None:
        raise SpecError(
            f"intervention {intervention_id!r} of kind {kind!r} needs a selector with "
            "'type' and 'value'"
        )

    if kind == "remove":
        if selector_type == "lexical_class":
            words = LEXICAL_CLASSES.get(value)
            if words is None:
                known = ", ".join(sorted(LEXICAL_CLASSES))
                raise SpecError(f"unknown lexical_class {value!r}; known classes are: {known}")
            return Intervention(
                id=intervention_id,
                kind="remove",
                description=description or f"Delete {value} markers: {', '.join(words)}",
                selector=selector,
                transform=_word_remover(words),
            )
        if selector_type == "regex":
            return Intervention(
                id=intervention_id,
                kind="remove",
                description=description or f"Delete every match of the regex {value!r}",
                selector=selector,
                transform=_regex_remover(value),
            )
        raise SpecError(
            f"unknown selector type {selector_type!r} for kind 'remove'; "
            "expected 'lexical_class' or 'regex'"
        )

    if selector_type != "regex":
        raise SpecError(
            f"unknown selector type {selector_type!r} for kind 'replace'; expected 'regex'"
        )
    replacement = selector.get("replacement")
    if replacement is None:
        raise SpecError(
            f"intervention {intervention_id!r} of kind 'replace' needs selector.replacement"
        )
    return Intervention(
        id=intervention_id,
        kind="replace",
        description=description or f"Replace matches of {value!r} with {replacement!r}",
        selector=selector,
        transform=_regex_replacer(value, replacement),
    )


def apply_intervention(intervention: Intervention, prompt: str) -> str:
    """Convenience wrapper, mostly so tests read the way the concept does."""
    return intervention.apply(prompt)
