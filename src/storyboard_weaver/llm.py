"""Text-generation clients.

The pipeline only needs ``complete(prompt) -> str``, captured by the
``LLMClient`` protocol, so any object with that method can be plugged in
(the tests use a scripted fake). Two HTTP implementations are provided:

* ``GeminiClient`` for the Google Gemini API (``generateContent``).
* ``OpenAICompatibleClient`` for any server exposing ``/chat/completions``:
  OpenAI itself, OpenRouter, Ollama, vLLM, LM Studio and similar.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import httpx

from .config import Settings
from .errors import ProviderError
from .transport import post_json


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self, prompt: str, *, system: str | None = None, json_output: bool = False
    ) -> str: ...


class _HTTPClient:
    provider = "provider"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = 90.0,
        retries: int = 2,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._retries = retries
        self._sleep = sleep
        self._http = http_client or httpx.Client(timeout=timeout)

    def _post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        return post_json(
            self._http,
            url,
            payload=payload,
            headers=headers,
            provider=self.provider,
            retries=self._retries,
            sleep=self._sleep,
        )

    def close(self) -> None:
        self._http.close()


class GeminiClient(_HTTPClient):
    provider = "gemini"

    def complete(self, prompt: str, *, system: str | None = None, json_output: bool = False) -> str:
        payload: dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if json_output:
            payload["generationConfig"] = {"responseMimeType": "application/json"}
        data = self._post(
            f"{self._base_url}/models/{self.model}:generateContent",
            payload,
            {"x-goog-api-key": self._api_key},
        )
        return gemini_text(data)


def gemini_text(data: dict[str, Any]) -> str:
    """Join the visible text parts of the first candidate."""
    candidates = data.get("candidates") or []
    if not candidates:
        reason = (data.get("promptFeedback") or {}).get("blockReason")
        detail = f" (blocked: {reason})" if reason else ""
        raise ProviderError(f"gemini: no answer was returned{detail}")
    first = candidates[0]
    parts = (first.get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
    if not text.strip():
        reason = first.get("finishReason", "unknown")
        raise ProviderError(f"gemini: the answer was empty (finish reason: {reason})")
    return text


class OpenAICompatibleClient(_HTTPClient):
    provider = "openai"

    def complete(self, prompt: str, *, system: str | None = None, json_output: bool = False) -> str:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        data = self._post(
            f"{self._base_url}/chat/completions",
            payload,
            {"Authorization": f"Bearer {self._api_key}"},
        )
        return chat_completion_text(data)


def chat_completion_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise ProviderError("openai: no choices were returned")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):  # some servers return content parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not content or not str(content).strip():
        refusal = message.get("refusal")
        reason = f"refused: {refusal}" if refusal else choices[0].get("finish_reason", "unknown")
        raise ProviderError(f"openai: the answer was empty ({reason})")
    return str(content)


def build_llm(settings: Settings) -> LLMClient:
    """Create the text client selected by ``settings``."""
    provider = settings.llm_provider
    common = {
        "api_key": settings.api_key(provider),
        "model": settings.text_model,
        "timeout": settings.request_timeout,
    }
    if provider == "gemini":
        return GeminiClient(base_url=settings.gemini_base_url, **common)
    return OpenAICompatibleClient(base_url=settings.openai_base_url, **common)
