"""Interventions have to be exactly what they say they are.

An intervention that also quietly changes the whitespace is not "remove negation", it is
"remove negation and reformat", and the effect it measures is confounded. So the tests
here check both halves: that the named thing is removed, and that nothing else is.
"""

from __future__ import annotations

import pytest

from errorbars.errors import SpecError
from errorbars.interventions import LEXICAL_CLASSES, build_intervention


def _apply(kind: str, selector: dict[str, str] | None, text: str) -> str:
    return build_intervention("x", kind, selector).apply(text)


def test_noop_returns_the_prompt_byte_for_byte() -> None:
    """The control arm must not touch the prompt at all, whitespace included."""
    text = "  Is 3   not greater than 1?\n\nAnswer.  "
    assert build_intervention("control", "noop").apply(text) == text


def test_negation_removal_takes_the_whole_class() -> None:
    selector = {"type": "lexical_class", "value": "negation"}
    assert _apply("remove", selector, "Is 3 not greater than 1?") == "Is 3 greater than 1?"
    assert _apply("remove", selector, "It is never true.") == "It is true."
    assert _apply("remove", selector, "Answer without hedging.") == "Answer hedging."


def test_contractions_are_handled() -> None:
    """`isn't` carries the negation in a suffix, not a separate word."""
    selector = {"type": "lexical_class", "value": "negation"}
    assert _apply("remove", selector, "It isn't true.") == "It is true."
    assert _apply("remove", selector, "It isn’t true.") == "It is true."


def test_removal_does_not_break_words_that_merely_contain_the_target() -> None:
    """ "no" must not be stripped out of "notation" or "nothing" out of "nothingness"."""
    selector = {"type": "lexical_class", "value": "negation"}
    assert _apply("remove", selector, "Use the notation below.") == "Use the notation below."
    assert _apply("remove", selector, "Nobody knows.") == "knows."
    assert _apply("remove", selector, "The nominal value.") == "The nominal value."


def test_removal_tidies_the_whitespace_it_creates() -> None:
    """Otherwise the intervention is also a formatting change, and the effect is confounded."""
    selector = {"type": "lexical_class", "value": "politeness"}
    assert _apply("remove", selector, "Please, answer the question.") == ", answer the question."
    assert "  " not in _apply("remove", selector, "Kindly please answer now.")


def test_removal_does_not_leave_a_space_before_punctuation() -> None:
    selector = {"type": "lexical_class", "value": "articles"}
    assert _apply("remove", selector, "Consider the value, the total.") == "Consider value, total."


def test_article_removal_leaves_other_words_alone() -> None:
    selector = {"type": "lexical_class", "value": "articles"}
    result = _apply("remove", selector, "Is the first number above the second number?")
    assert result == "Is first number above second number?"
    assert "another" in _apply("remove", selector, "Pick another one.")


def test_removal_is_case_insensitive_across_the_class() -> None:
    selector = {"type": "lexical_class", "value": "negation"}
    assert _apply("remove", selector, "NOT true, Never false.") == "true, false."


def test_regex_removal() -> None:
    selector = {"type": "regex", "value": r"\bfoo\b"}
    assert _apply("remove", selector, "a foo b foobar") == "a b foobar"


def test_regex_replacement() -> None:
    selector = {"type": "regex", "value": "greater", "replacement": "larger"}
    assert _apply("replace", selector, "Is 3 greater than 1?") == "Is 3 larger than 1?"


def test_every_declared_lexical_class_compiles_and_does_something() -> None:
    for name, words in LEXICAL_CLASSES.items():
        selector = {"type": "lexical_class", "value": name}
        sample = " ".join(word for word in words if not word.startswith("n't"))
        assert _apply("remove", selector, f"start {sample} end") == "start end"


def test_the_word_lists_are_visible_rather_than_hidden() -> None:
    """An intervention nobody can read is an intervention nobody can check."""
    intervention = build_intervention("x", "remove", {"type": "lexical_class", "value": "negation"})
    for word in ("not", "never", "without"):
        assert word in intervention.description


def test_interventions_are_deterministic() -> None:
    intervention = build_intervention("x", "remove", {"type": "lexical_class", "value": "negation"})
    text = "Is 12 not greater than 40? Never mind."
    assert intervention.apply(text) == intervention.apply(text)


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(SpecError, match="unknown intervention kind"):
        build_intervention("x", "shuffle")


def test_unknown_lexical_class_lists_the_known_ones() -> None:
    with pytest.raises(SpecError, match="known classes are"):
        build_intervention("x", "remove", {"type": "lexical_class", "value": "verbs"})


def test_unknown_selector_type_is_rejected() -> None:
    with pytest.raises(SpecError, match="unknown selector type"):
        build_intervention("x", "remove", {"type": "embedding", "value": "0.3"})


def test_missing_selector_is_rejected() -> None:
    with pytest.raises(SpecError, match="needs a selector"):
        build_intervention("x", "remove")


def test_replace_without_a_replacement_is_rejected() -> None:
    with pytest.raises(SpecError, match=r"selector\.replacement"):
        build_intervention("x", "replace", {"type": "regex", "value": "foo"})


def test_an_invalid_regex_fails_at_parse_time() -> None:
    with pytest.raises(SpecError, match="does not compile"):
        build_intervention("x", "remove", {"type": "regex", "value": "("})
