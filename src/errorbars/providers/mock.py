"""A deterministic simulated model, so the whole demo runs offline with no API key.

READ THIS BEFORE QUOTING ANY NUMBER PRODUCED WITH IT
----------------------------------------------------
``MockProvider`` is **synthetic**. It is not a language model, and no result obtained
through it is a measurement of any real system. It exists so that ``errorbars run`` has
something to run against in CI and in the README demo, and so that the shape of the
output can be shown without spending money or requiring a key. Every number in the
README that came from this provider is labelled as such.

What it simulates, exactly
--------------------------
A model answering yes/no numeric comparison questions, with three documented behaviours:

1. **Semantics.** It extracts the two integers and the comparison word from the prompt,
   evaluates the comparison, then flips the answer once for every negation marker it
   finds. Deleting "not" therefore changes the *question being asked*, and the simulated
   model answers the question it was actually given. That is a real mechanism, not a
   fudge factor: it is the same reason the effect is expected in a real model.

2. **Ordinal reference needs its articles.** For prompts phrased "is the second number
   greater than the first number", the parser uses the articles to bind the operands.
   With them deleted the binding is gone, and it guesses -- a coin flip, the way a model
   that genuinely cannot tell which operand is which would behave. Only the items
   phrased that way are affected, so deleting articles produces a small effect spread
   thinly, which is exactly the shape the underpowered verdict exists to catch.

3. **Item difficulty and noise.** Each item gets a stable error probability drawn from
   the integer pair in its prompt, which survives every intervention -- so the same items
   are hard in every arm, exactly the correlation structure ``errorbars power`` models as
   ``icc`` and ``pair_corr``. The per-response noise draw additionally depends on the
   prompt text and the request seed, so repeats differ and arms are not locked together.

Politeness markers ("please", "kindly", ...) are invisible to all of the above, so
deleting them has an effect of *exactly* zero. That is what makes the demo's null
verdict a genuine null rather than a small effect that happened to miss.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any

import numpy as np

from .base import Completion

__all__ = ["MockProvider"]

_INTEGER = re.compile(r"-?\d+")
_NEGATION = re.compile(
    r"(?<!\w)(?:not|never|no|none|neither|nor|without|cannot)(?!\w)|n['’]t(?!\w)",
    re.IGNORECASE,
)
_LESS = re.compile(r"(?<!\w)(?:less|smaller|fewer|below)(?!\w)", re.IGNORECASE)
_ORDINAL_WITH_ARTICLES = re.compile(
    r"\bthe\s+(first|second)\s+number\b.*?\bthe\s+(first|second)\s+number\b",
    re.IGNORECASE | re.DOTALL,
)
_ORDINAL_BARE = re.compile(
    r"\b(first|second)\s+number\b.*?\b(first|second)\s+number\b",
    re.IGNORECASE | re.DOTALL,
)


def _seed_from(*parts: object) -> int:
    """A stable 63-bit seed. Python's ``hash()`` is salted per process and unusable here."""
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8)
    return int.from_bytes(digest.digest(), "big") >> 1


class MockProvider:
    """Simulated model. Deterministic given (prompt, params, constructor arguments)."""

    name = "mock"

    def __init__(
        self,
        *,
        seed: int = 7,
        base_error: float = 0.12,
        difficulty_sd: float = 0.20,
        max_error: float = 0.85,
    ) -> None:
        """
        Args:
            seed: master seed; changing it changes which items are hard.
            base_error: mean probability of answering incorrectly.
            difficulty_sd: spread of per-item error probability. Larger means more of
                the outcome variance is between items, i.e. a higher ICC.
            max_error: clip on per-item error probability.
        """
        self.seed = seed
        self.base_error = base_error
        self.difficulty_sd = difficulty_sd
        self.max_error = max_error

    def _operands(self, prompt: str) -> tuple[int, int] | None:
        found = _INTEGER.findall(prompt)
        if len(found) < 2:
            return None
        return int(found[0]), int(found[1])

    def _binding(self, prompt: str) -> tuple[bool, bool]:
        """Return (operands are swapped, the binding was ambiguous).

        "the second number ... the first number" swaps the operands. Strip the articles
        and there is nothing left to bind them with, so the reading is ambiguous.
        """
        ordinal = _ORDINAL_WITH_ARTICLES.search(prompt)
        if ordinal is not None:
            return ordinal.group(1).lower() == "second", False

        bare = _ORDINAL_BARE.search(prompt)
        if bare is not None:
            # No articles to bind with, so the simulated model falls back to the order
            # the operands appear in. A "first ... second" question asks about them in
            # that same order and is unaffected. A "second ... first" question is not
            # answerable from word order alone, so it guesses.
            return False, bare.group(1).lower() == "second"

        return False, False

    def _truth(self, prompt: str, left: int, right: int, swapped: bool) -> bool:
        """Evaluate the comparison the prompt actually asks, negation included."""
        first, second = (right, left) if swapped else (left, right)
        truth = first < second if _LESS.search(prompt) else first > second
        if len(_NEGATION.findall(prompt)) % 2 == 1:
            truth = not truth
        return truth

    def complete(self, prompt: str, params: dict[str, Any]) -> Completion:
        started = time.perf_counter()
        operands = self._operands(prompt)
        if operands is None:
            return Completion(
                text="unknown",
                prompt_tokens=len(prompt) // 4,
                completion_tokens=1,
                latency_ms=int((time.perf_counter() - started) * 1000),
                raw={"provider": "mock", "reason": "no operands found in prompt"},
            )

        left, right = operands

        # Item difficulty keys off the operands only, so it is identical in every arm.
        # That is what gives the simulated run a realistic ICC and pairing correlation.
        difficulty_rng = np.random.default_rng(_seed_from("difficulty", self.seed, left, right))
        error_probability = float(
            np.clip(
                self.base_error + difficulty_rng.normal(0.0, self.difficulty_sd),
                0.0,
                self.max_error,
            )
        )

        # The noise draw keys off the prompt text and the request seed too, so repeats
        # differ from each other and the two arms of an item get independent draws.
        noise_rng = np.random.default_rng(
            _seed_from("noise", self.seed, prompt, params.get("seed", 0))
        )

        swapped, ambiguous = self._binding(prompt)
        if ambiguous:
            swapped = bool(noise_rng.random() < 0.5)
        truth = self._truth(prompt, left, right, swapped)

        if noise_rng.random() < error_probability:
            truth = not truth

        return Completion(
            text="true" if truth else "false",
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=1,
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw={
                "provider": "mock",
                "error_probability": error_probability,
                "ambiguous_binding": ambiguous,
            },
        )
