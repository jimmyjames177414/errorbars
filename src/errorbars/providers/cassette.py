"""Replay provider: serve recorded responses from disk, never touch the network.

Cassettes make a real-model run reproducible by anyone, including CI with no secrets
configured. Record once against a real endpoint, commit the JSON, and every later run of
the same experiment replays byte-identical responses.

Cassette files are keyed by ``sha256(prompt)`` and must carry a ``recorded`` flag, so a
reader can always tell a real recorded response from a hand-written fixture
(SHARED-FOUNDATION D3: never present a synthetic fixture as a measured result).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..errors import ProviderError
from .base import Completion

__all__ = ["CassetteProvider", "cassette_key", "write_cassette"]


def cassette_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def write_cassette(
    directory: str | Path,
    prompt: str,
    completion: Completion,
    *,
    recorded: bool,
    note: str = "",
) -> Path:
    """Write one cassette entry.

    Args:
        recorded: True if this came from a real model call, False if it was written by
            hand. This flag is mandatory and is surfaced by the reader.
        note: free text describing provenance -- which endpoint, when, why.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{cassette_key(prompt)}.json"
    path.write_text(
        json.dumps(
            {
                "prompt": prompt,
                "response": completion.text,
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "recorded": recorded,
                "note": note,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class CassetteProvider:
    """Serve responses recorded under ``directory``."""

    name = "cassette"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise ProviderError(f"cassette directory not found: {self.directory}")

    def complete(self, prompt: str, params: dict[str, Any]) -> Completion:
        path = self.directory / f"{cassette_key(prompt)}.json"
        if not path.is_file():
            raise ProviderError(
                f"no cassette for this prompt in {self.directory} (expected {path.name}). "
                "Record it first, or switch to provider 'mock' for an offline run."
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProviderError(f"cassette {path} is not valid JSON: {exc}") from exc

        if "response" not in data:
            raise ProviderError(f"cassette {path} has no 'response' field")

        return Completion(
            text=str(data["response"]),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            latency_ms=0,
            raw={
                "provider": "cassette",
                "recorded": bool(data.get("recorded", False)),
                "note": data.get("note", ""),
            },
        )
