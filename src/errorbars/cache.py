"""Content-addressed response cache.

Keyed by ``sha256(provider | model | params | prompt)``, so re-running an experiment
after changing only the analysis costs nothing, and adding a fourth intervention to a
three-intervention spec re-uses everything already paid for.

The cache is on by default because the alternative -- silently re-billing a user for
work already done -- is the worse default. ``--no-cache`` turns it off.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .providers.base import Completion

__all__ = ["ResponseCache", "cache_key", "default_cache_dir"]


def default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME/errorbars`` if set, else ``~/.cache/errorbars``."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "errorbars"


def cache_key(provider: str, model: str, params: dict[str, Any], prompt: str) -> str:
    """Stable key for one model call.

    ``params`` is serialised with sorted keys so that dict ordering never changes the
    key -- otherwise a cache hit would depend on how a YAML parser happened to order a
    mapping.
    """
    payload = json.dumps(
        {"provider": provider, "model": model, "params": params, "prompt": prompt},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """A directory of JSON files, one per cached response."""

    def __init__(self, directory: str | Path | None = None, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.directory = Path(directory) if directory is not None else default_cache_dir()
        self.hits = 0
        self.misses = 0
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Two-character shard, so a big run does not create one directory with 100k
        # entries in it (which some filesystems handle badly).
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> Completion | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt or half-written entry is a miss, not a crash: the worst case is
            # paying for one call again.
            self.misses += 1
            return None
        self.hits += 1
        return Completion(
            text=str(data.get("text", "")),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            latency_ms=int(data.get("latency_ms", 0)),
            raw=dict(data.get("raw", {})),
        )

    def put(self, key: str, completion: Completion) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "text": completion.text,
                    "prompt_tokens": completion.prompt_tokens,
                    "completion_tokens": completion.completion_tokens,
                    "latency_ms": completion.latency_ms,
                    "raw": completion.raw,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
