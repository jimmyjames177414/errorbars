"""The provider interface: one method, one result type.

Provider selection is a URL and env-var concern, never a code change (SHARED-FOUNDATION
D2). A contributor with Ollama and no credit card can run everything in this repo, and
the whole test suite runs with no keys configured at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["Completion", "Provider"]


@dataclass(frozen=True)
class Completion:
    """One model response, plus what it cost to get it."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Provider(Protocol):
    """Anything that can turn a prompt into a :class:`Completion`."""

    name: str

    def complete(self, prompt: str, params: dict[str, Any]) -> Completion:
        """Return the model's response to ``prompt``.

        Raises:
            ProviderError: on any failure the caller could act on -- a refused
                connection, a non-2xx status, a response missing its content.
        """
        ...
