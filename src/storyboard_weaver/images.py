"""Scene frame generation.

``ImageGenerator`` is the only thing the pipeline knows about: give it a
prompt, get back encoded image bytes. Two implementations are included and
chosen by ``STORYBOARD_IMAGE_PROVIDER``; ``none`` (the default) means no
image generator at all, so a run makes text calls only.
"""

from __future__ import annotations

import base64
import binascii
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from .config import Settings
from .errors import ProviderError
from .transport import post_json

EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}

# Aspect ratios accepted by Gemini image models.
GEMINI_ASPECT_RATIOS = ("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes
    mime_type: str = "image/png"

    @property
    def extension(self) -> str:
        return EXTENSIONS.get(self.mime_type, ".png")


@runtime_checkable
class ImageGenerator(Protocol):
    name: str
    model: str

    def generate(self, prompt: str) -> GeneratedImage: ...


def sniff_mime_type(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ProviderError("image response was not a PNG, JPEG or WebP file")


def _decode_base64(encoded: str, provider: str) -> bytes:
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ProviderError(f"{provider}: image data was not valid base64") from None


def closest_aspect_ratio(size: str) -> str:
    """Map ``WIDTHxHEIGHT`` to the nearest ratio Gemini accepts."""
    width, height = (int(part) for part in size.lower().split("x"))
    target = width / height

    def distance(ratio: str) -> float:
        w, h = (int(part) for part in ratio.split(":"))
        return abs(w / h - target)

    return min(GEMINI_ASPECT_RATIOS, key=distance)


class _HTTPImageGenerator:
    name = "provider"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        size: str = "1536x1024",
        timeout: float = 120.0,
        retries: int = 2,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self.size = size
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
            provider=self.name,
            retries=self._retries,
            sleep=self._sleep,
        )

    def close(self) -> None:
        self._http.close()


class OpenAIImageGenerator(_HTTPImageGenerator):
    """``/images/generations`` on OpenAI or any compatible server."""

    name = "openai"

    def generate(self, prompt: str) -> GeneratedImage:
        payload: dict[str, Any] = {"model": self.model, "prompt": prompt, "n": 1, "size": self.size}
        if not self.model.startswith("gpt-image"):
            # DALL-E style models return a short-lived URL unless asked for base64.
            payload["response_format"] = "b64_json"
        data = self._post(
            f"{self._base_url}/images/generations",
            payload,
            {"Authorization": f"Bearer {self._api_key}"},
        )
        items = data.get("data") or []
        if not items:
            raise ProviderError("openai: no image was returned")
        item = items[0]
        if item.get("b64_json"):
            raw = _decode_base64(item["b64_json"], self.name)
        elif item.get("url"):
            raw = self._download(item["url"])
        else:
            raise ProviderError("openai: image response had neither b64_json nor url")
        return GeneratedImage(data=raw, mime_type=sniff_mime_type(raw))

    def _download(self, url: str) -> bytes:
        try:
            response = self._http.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"openai: could not download the image ({type(exc).__name__})"
            ) from None
        return response.content


class GeminiImageGenerator(_HTTPImageGenerator):
    """Gemini image models through ``generateContent`` with image output enabled."""

    name = "gemini"

    def generate(self, prompt: str) -> GeneratedImage:
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {"aspectRatio": closest_aspect_ratio(self.size)},
            },
        }
        data = self._post(
            f"{self._base_url}/models/{self.model}:generateContent",
            payload,
            {"x-goog-api-key": self._api_key},
        )
        for candidate in data.get("candidates") or []:
            for part in (candidate.get("content") or {}).get("parts") or []:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    raw = _decode_base64(inline["data"], self.name)
                    mime = inline.get("mimeType") or inline.get("mime_type") or ""
                    return GeneratedImage(data=raw, mime_type=mime or sniff_mime_type(raw))
        reason = (data.get("promptFeedback") or {}).get("blockReason")
        detail = f" (blocked: {reason})" if reason else ""
        raise ProviderError(f"gemini: no image was returned{detail}")


def build_image_generator(settings: Settings) -> ImageGenerator | None:
    """Create the generator selected by ``settings``, or ``None`` for text-only runs."""
    provider = settings.image_provider
    if provider == "none":
        return None
    common = {
        "api_key": settings.api_key(provider),
        "model": settings.picture_model or "",
        "size": settings.image_size,
        "timeout": max(settings.request_timeout, 120.0),
    }
    if provider == "gemini":
        return GeminiImageGenerator(base_url=settings.gemini_base_url, **common)
    return OpenAIImageGenerator(base_url=settings.openai_base_url, **common)
