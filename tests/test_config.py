from pathlib import Path

import pytest

from storyboard_weaver.config import Settings
from storyboard_weaver.errors import ConfigError


def test_defaults_when_environment_is_empty():
    settings = Settings.from_env({})
    assert settings.llm_provider == "gemini"
    assert settings.image_provider == "none"
    assert settings.picture_model is None
    assert settings.text_model  # falls back to the provider default
    assert settings.kb_path == Path("outputs") / "knowledge_base.json"


def test_reads_and_normalises_environment_values():
    settings = Settings.from_env(
        {
            "STORYBOARD_LLM_PROVIDER": " OpenAI ",
            "STORYBOARD_LLM_MODEL": "local-model",
            "STORYBOARD_IMAGE_PROVIDER": "GEMINI",
            "STORYBOARD_OUTPUT_DIR": "runs",
            "STORYBOARD_MAX_ATTEMPTS": "5",
            "OPENAI_BASE_URL": "http://localhost:11434/v1",
            "OPENAI_API_KEY": "placeholder",
            "GEMINI_API_KEY": "",
        }
    )
    assert settings.llm_provider == "openai"
    assert settings.text_model == "local-model"
    assert settings.image_provider == "gemini"
    assert settings.picture_model  # provider default
    assert settings.max_attempts == 5
    assert settings.kb_path == Path("runs") / "knowledge_base.json"
    assert settings.api_key("openai") == "placeholder"


def test_invalid_values_name_the_environment_variable():
    with pytest.raises(ConfigError, match="STORYBOARD_LLM_PROVIDER"):
        Settings.from_env({"STORYBOARD_LLM_PROVIDER": "carrier-pigeon"})
    with pytest.raises(ConfigError, match="STORYBOARD_IMAGE_SIZE"):
        Settings.from_env({"STORYBOARD_IMAGE_SIZE": "huge"})


def test_missing_key_explains_what_to_set():
    settings = Settings.from_env({})
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        settings.api_key("gemini")
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        settings.api_key("openai")


def test_keys_are_hidden_from_repr():
    settings = Settings.from_env({"GEMINI_API_KEY": "super-secret-value"})
    assert "super-secret-value" not in repr(settings)
