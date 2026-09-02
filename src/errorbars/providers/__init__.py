"""Model backends. Selecting one is configuration, never a code change."""

from __future__ import annotations

from ..errors import SpecError
from .base import Completion, Provider
from .cassette import CassetteProvider, cassette_key, write_cassette
from .mock import MockProvider
from .openai_compat import OpenAICompatibleProvider

__all__ = [
    "CassetteProvider",
    "Completion",
    "MockProvider",
    "OpenAICompatibleProvider",
    "Provider",
    "build_provider",
    "cassette_key",
    "write_cassette",
]


def build_provider(
    provider: str,
    model: str,
    *,
    base_url: str | None = None,
    api_key_env: str | None = None,
    fixtures: str | None = None,
) -> Provider:
    """Construct a provider from spec fields.

    Raises:
        SpecError: for an unknown provider name, or a cassette provider with no
            ``fixtures`` directory.
    """
    if provider in ("openai-compatible", "openai_compatible", "openai"):
        return OpenAICompatibleProvider(
            model,
            base_url=base_url or "http://localhost:11434/v1",
            api_key_env=api_key_env or "OPENAI_API_KEY",
        )
    if provider == "mock":
        return MockProvider()
    if provider == "cassette":
        if not fixtures:
            raise SpecError("provider 'cassette' needs a 'fixtures' directory in the model entry")
        return CassetteProvider(fixtures)
    raise SpecError(
        f"unknown provider {provider!r}; expected 'openai-compatible', 'mock' or 'cassette'"
    )
