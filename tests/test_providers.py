"""Providers, scorers and the response cache.

Everything here runs offline. The OpenAI-compatible adapter is exercised only for its
error handling -- the paths that need a real endpoint are marked ``live`` and deselected
by default, so the suite passes on a machine with no keys and no network, which is how
CI runs it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from errorbars.cache import ResponseCache, cache_key, default_cache_dir
from errorbars.errors import ProviderError, SpecError
from errorbars.providers import build_provider, write_cassette
from errorbars.providers.base import Completion
from errorbars.providers.cassette import CassetteProvider, cassette_key
from errorbars.providers.mock import MockProvider
from errorbars.providers.openai_compat import OpenAICompatibleProvider
from errorbars.scorers import get_scorer, list_scorers

ANSWER = "Answer with one word: true or false."


# --------------------------------------------------------------------------------------
# The simulated model
# --------------------------------------------------------------------------------------


def test_mock_answers_a_plain_comparison() -> None:
    provider = MockProvider(base_error=0.0, difficulty_sd=0.0)
    assert provider.complete(f"Is 40 greater than 9? {ANSWER}", {}).text == "true"
    assert provider.complete(f"Is 9 greater than 40? {ANSWER}", {}).text == "false"


def test_mock_respects_negation() -> None:
    """The whole reason deleting 'not' has an effect: it changes the question."""
    provider = MockProvider(base_error=0.0, difficulty_sd=0.0)
    assert provider.complete(f"Is 40 not greater than 9? {ANSWER}", {}).text == "false"
    assert provider.complete(f"Is 9 not greater than 40? {ANSWER}", {}).text == "true"


def test_mock_uses_articles_to_bind_ordinal_references() -> None:
    provider = MockProvider(base_error=0.0, difficulty_sd=0.0)
    with_articles = (
        f"Consider the numbers 12 and 30. Is the second number greater than the first "
        f"number? {ANSWER}"
    )
    assert provider.complete(with_articles, {}).text == "true"


def test_mock_is_unaffected_by_politeness() -> None:
    """Which is what makes the demo's null verdict a genuine null."""
    provider = MockProvider(base_error=0.0, difficulty_sd=0.0)
    plain = provider.complete(f"Is 40 greater than 9? {ANSWER}", {})
    polite = provider.complete(f"Please, Is 40 greater than 9? {ANSWER}", {})
    assert plain.text == polite.text == "true"


def test_mock_is_deterministic_for_a_given_prompt_and_seed() -> None:
    provider = MockProvider()
    prompt = f"Is 17 not greater than 55? {ANSWER}"
    first = provider.complete(prompt, {"seed": 3})
    second = provider.complete(prompt, {"seed": 3})
    assert first.text == second.text


def test_mock_varies_across_repeat_seeds() -> None:
    """Identical answers for every repeat would make repeats measure nothing."""
    provider = MockProvider(base_error=0.5, difficulty_sd=0.0)
    prompt = f"Is 17 greater than 55? {ANSWER}"
    answers = {provider.complete(prompt, {"seed": seed}).text for seed in range(20)}
    assert len(answers) == 2


def test_mock_item_difficulty_survives_an_intervention() -> None:
    """Difficulty keys off the operands, so the same items are hard in every arm.

    That is what gives a simulated run a realistic ICC and pairing correlation instead
    of zero, which in turn is what makes the demo's design numbers meaningful.
    """
    provider = MockProvider()
    plain = provider.complete(f"Is 17 greater than 55? {ANSWER}", {"seed": 1})
    stripped = provider.complete(f"Is 17 greater than 55? {ANSWER} extra", {"seed": 1})
    assert plain.raw["error_probability"] == stripped.raw["error_probability"]


def test_mock_reports_when_it_could_not_parse_the_prompt() -> None:
    result = MockProvider().complete("What is the capital of anywhere?", {})
    assert result.text == "unknown"
    assert "no operands" in result.raw["reason"]


# --------------------------------------------------------------------------------------
# Cassettes
# --------------------------------------------------------------------------------------


def test_cassette_round_trip(tmp_path: Path) -> None:
    prompt = "Is 4 greater than 2?"
    write_cassette(tmp_path, prompt, Completion(text="true"), recorded=True, note="from a run")

    provider = CassetteProvider(tmp_path)
    result = provider.complete(prompt, {})
    assert result.text == "true"
    assert result.raw["recorded"] is True
    assert result.raw["note"] == "from a run"


def test_cassette_records_whether_the_response_was_real(tmp_path: Path) -> None:
    """A synthetic fixture must never be readable as a measured result."""
    write_cassette(tmp_path, "p", Completion(text="x"), recorded=False, note="hand-written")
    stored = json.loads((tmp_path / f"{cassette_key('p')}.json").read_text(encoding="utf-8"))
    assert stored["recorded"] is False
    assert CassetteProvider(tmp_path).complete("p", {}).raw["recorded"] is False


def test_a_missing_cassette_says_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(ProviderError, match="Record it first"):
        CassetteProvider(tmp_path).complete("never recorded", {})


def test_a_missing_cassette_directory_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ProviderError, match="cassette directory not found"):
        CassetteProvider(tmp_path / "nope")


# --------------------------------------------------------------------------------------
# Provider selection
# --------------------------------------------------------------------------------------


def test_provider_selection_is_configuration_not_code() -> None:
    assert isinstance(build_provider("mock", "sim"), MockProvider)
    assert isinstance(
        build_provider("openai-compatible", "qwen3:8b", base_url="http://localhost:11434/v1"),
        OpenAICompatibleProvider,
    )


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(SpecError, match="unknown provider"):
        build_provider("telepathy", "sim")


def test_cassette_provider_needs_a_fixtures_directory() -> None:
    with pytest.raises(SpecError, match="needs a 'fixtures' directory"):
        build_provider("cassette", "sim")


def test_openai_adapter_reports_an_unreachable_endpoint_clearly() -> None:
    """Port 9 discards everything sent to it, so this fails without touching a real host."""
    provider = OpenAICompatibleProvider(
        "any-model", base_url="http://127.0.0.1:9/v1", api_key_env=None
    )
    with pytest.raises(ProviderError, match="could not reach"):
        provider.complete("hello", {})


def test_openai_adapter_omits_the_auth_header_when_no_key_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local servers need no key; sending an empty Bearer would be worse than nothing."""
    monkeypatch.delenv("ERRORBARS_TEST_KEY", raising=False)
    provider = OpenAICompatibleProvider("m", api_key_env="ERRORBARS_TEST_KEY")
    assert "Authorization" not in provider._headers()

    monkeypatch.setenv("ERRORBARS_TEST_KEY", "secret-value")
    assert provider._headers()["Authorization"] == "Bearer secret-value"


@pytest.mark.live
def test_openai_adapter_against_a_real_endpoint() -> None:  # pragma: no cover
    """Deselected by default. Point OPENAI_BASE_URL at Ollama or similar to run it."""
    import os

    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        pytest.skip("set OPENAI_BASE_URL to run the live adapter test")
    provider = OpenAICompatibleProvider(
        os.environ.get("OPENAI_MODEL", "qwen3:8b"), base_url=base_url
    )
    assert provider.complete("Reply with the single word: ok", {"max_tokens": 8}).text


# --------------------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------------------


def test_exact_match_normalises_punctuation_and_case() -> None:
    scorer = get_scorer("exact_match")
    assert scorer("  True. ", "true").passed
    assert scorer('"TRUE"', "true").passed
    assert not scorer("false", "true").passed


def test_first_word_tolerates_a_model_that_explains_itself() -> None:
    """Otherwise the scorer measures verbosity rather than accuracy."""
    scorer = get_scorer("first_word")
    assert scorer("true, because 40 is larger than 9", "true").passed
    assert not scorer("false, because ...", "true").passed


def test_contains_and_regex_scorers() -> None:
    assert get_scorer("contains")("the answer is true", "true").passed
    assert get_scorer("regex")("call get_user now", r"\bget_user\b").passed
    assert not get_scorer("regex")("call delete_user", r"\bget_user\b").passed


def test_scores_are_numbers_and_the_detail_says_what_happened() -> None:
    result = get_scorer("exact_match")("false", "true")
    assert result.score == 0.0
    assert result.detail == {"expected": "true", "got": "false"}


def test_unknown_scorer_lists_the_available_ones() -> None:
    with pytest.raises(SpecError, match="available scorers are"):
        get_scorer("vibes")
    assert "exact_match" in list_scorers()


def test_an_invalid_regex_pattern_is_a_spec_error() -> None:
    with pytest.raises(SpecError, match="invalid pattern"):
        get_scorer("regex")("anything", "(")


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------


def test_cache_round_trip(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    key = cache_key("mock", "sim", {"temperature": 0.0}, "hello")
    assert cache.get(key) is None
    cache.put(key, Completion(text="true", prompt_tokens=7, completion_tokens=1))

    hit = cache.get(key)
    assert hit is not None
    assert hit.text == "true"
    assert hit.prompt_tokens == 7
    assert cache.hits == 1
    assert cache.misses == 1


def test_cache_key_ignores_dict_ordering() -> None:
    """Otherwise a hit would depend on how a YAML parser happened to order a mapping."""
    first = cache_key("mock", "sim", {"a": 1, "b": 2}, "p")
    second = cache_key("mock", "sim", {"b": 2, "a": 1}, "p")
    assert first == second


def test_cache_key_changes_with_every_input() -> None:
    base = cache_key("mock", "sim", {"t": 0}, "p")
    assert cache_key("other", "sim", {"t": 0}, "p") != base
    assert cache_key("mock", "other", {"t": 0}, "p") != base
    assert cache_key("mock", "sim", {"t": 1}, "p") != base
    assert cache_key("mock", "sim", {"t": 0}, "q") != base


def test_a_disabled_cache_stores_nothing(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path, enabled=False)
    key = cache_key("mock", "sim", {}, "p")
    cache.put(key, Completion(text="true"))
    assert cache.get(key) is None
    assert not any(tmp_path.iterdir())


def test_a_corrupt_entry_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    """Worst case is paying for one call again; crashing mid-run would cost far more."""
    cache = ResponseCache(tmp_path)
    key = cache_key("mock", "sim", {}, "p")
    cache.put(key, Completion(text="true"))
    corrupted = next(tmp_path.rglob("*.json"))
    corrupted.write_text("{ truncated", encoding="utf-8")
    assert cache.get(key) is None


def test_default_cache_dir_follows_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_cache_dir() == tmp_path / "errorbars"
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_cache_dir() == tmp_path / ".cache" / "errorbars"
