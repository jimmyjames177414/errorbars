"""Any OpenAI-compatible ``/v1/chat/completions`` endpoint.

One adapter covers Ollama, vLLM, LM Studio, llama.cpp's server, OpenRouter, Together,
Groq, DeepSeek and OpenAI itself. Choosing a backend is a ``base_url`` and an env var,
never a code change.

``urllib`` from the stdlib rather than ``requests``: this is a handful of POSTs, and a
transitive HTTP dependency is not worth the install weight for a tool whose main command
makes no network calls at all.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from ..errors import ProviderError
from .base import Completion

__all__ = ["OpenAICompatibleProvider"]

DEFAULT_TIMEOUT_SECONDS = 120


class OpenAICompatibleProvider:
    """Chat-completions client for any OpenAI-compatible server."""

    name = "openai-compatible"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434/v1",
        api_key_env: str | None = "OPENAI_API_KEY",
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """
        Args:
            api_key_env: name of the environment variable holding the key. A missing
                variable is not an error -- local servers such as Ollama need no key --
                but the header is then omitted rather than sent empty.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    def complete(self, prompt: str, params: dict[str, Any]) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        payload.update({k: v for k, v in params.items() if v is not None})

        request = urllib.request.Request(
            self._endpoint(),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ProviderError(f"{self._endpoint()} returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(
                f"could not reach {self._endpoint()}: {exc.reason}. "
                "Is the server running, and is base_url right?"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self._endpoint()} returned a non-JSON body: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"{self._endpoint()} response had no choices[0].message.content; "
                f"got keys {sorted(body) if isinstance(body, dict) else type(body).__name__}"
            ) from exc

        usage = body.get("usage") or {}
        return Completion(
            text="" if text is None else str(text),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
            raw={"provider": "openai-compatible", "model": self.model},
        )
