"""Runtime settings, read from environment variables.

The CLI loads a ``.env`` file from the working directory first, so the same
names work in a shell, in CI and in a local ``.env``. See ``.env.example``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import ConfigError

TextProvider = Literal["gemini", "openai"]
ImageProvider = Literal["none", "gemini", "openai"]

DEFAULT_TEXT_MODELS: dict[str, str] = {
    "gemini": "gemini-3.8-flash",
    "openai": "gpt-5-mini",
}
DEFAULT_IMAGE_MODELS: dict[str, str] = {
    "gemini": "gemini-3.1-flash-image",
    "openai": "gpt-image-1",
}

# Settings field -> environment variable.
ENV_VARS: dict[str, str] = {
    "llm_provider": "STORYBOARD_LLM_PROVIDER",
    "llm_model": "STORYBOARD_LLM_MODEL",
    "image_provider": "STORYBOARD_IMAGE_PROVIDER",
    "image_model": "STORYBOARD_IMAGE_MODEL",
    "image_size": "STORYBOARD_IMAGE_SIZE",
    "output_dir": "STORYBOARD_OUTPUT_DIR",
    "knowledge_base_path": "STORYBOARD_KB_PATH",
    "max_attempts": "STORYBOARD_MAX_ATTEMPTS",
    "request_timeout": "STORYBOARD_TIMEOUT",
    "gemini_api_key": "GEMINI_API_KEY",
    "gemini_base_url": "GEMINI_BASE_URL",
    "openai_api_key": "OPENAI_API_KEY",
    "openai_base_url": "OPENAI_BASE_URL",
}

_LOWERCASE_FIELDS = {"llm_provider", "image_provider"}


class Settings(BaseModel):
    """All knobs in one immutable object. Use ``model_copy(update=...)`` to override."""

    model_config = ConfigDict(frozen=True)

    llm_provider: TextProvider = "gemini"
    llm_model: str | None = None
    image_provider: ImageProvider = "none"
    image_model: str | None = None
    image_size: str = Field(default="1536x1024", pattern=r"^\d{2,5}x\d{2,5}$")
    output_dir: Path = Path("outputs")
    knowledge_base_path: Path | None = None
    max_attempts: int = Field(default=3, ge=1, le=10)
    request_timeout: float = Field(default=90.0, gt=0)
    gemini_api_key: str | None = Field(default=None, repr=False)
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    openai_api_key: str | None = Field(default=None, repr=False)
    openai_base_url: str = "https://api.openai.com/v1"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``env`` (defaults to ``os.environ``); blank values are ignored."""
        source = os.environ if env is None else env
        values: dict[str, str] = {}
        for field, var in ENV_VARS.items():
            raw = source.get(var, "").strip()
            if raw:
                values[field] = raw.lower() if field in _LOWERCASE_FIELDS else raw
        try:
            return cls(**values)
        except ValidationError as exc:
            problems = "; ".join(
                f"{ENV_VARS.get(str(err['loc'][0]), err['loc'][0])}: {err['msg']}"
                for err in exc.errors()
            )
            raise ConfigError(f"Invalid configuration: {problems}") from None

    @property
    def text_model(self) -> str:
        return self.llm_model or DEFAULT_TEXT_MODELS[self.llm_provider]

    @property
    def picture_model(self) -> str | None:
        if self.image_provider == "none":
            return None
        return self.image_model or DEFAULT_IMAGE_MODELS[self.image_provider]

    @property
    def kb_path(self) -> Path:
        return self.knowledge_base_path or self.output_dir / "knowledge_base.json"

    def api_key(self, provider: str) -> str:
        """Return the key for ``provider`` or explain which variable to set."""
        key = self.gemini_api_key if provider == "gemini" else self.openai_api_key
        if not key:
            var = ENV_VARS["gemini_api_key" if provider == "gemini" else "openai_api_key"]
            raise ConfigError(
                f"{var} is not set. Add it to your environment or .env file "
                f"to use the {provider} provider."
            )
        return key
