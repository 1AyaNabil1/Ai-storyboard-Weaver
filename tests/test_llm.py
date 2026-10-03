"""Provider clients, exercised against an in-process mock transport (no network)."""

import json

import httpx
import pytest

from storyboard_weaver.config import Settings
from storyboard_weaver.errors import ConfigError, ProviderError
from storyboard_weaver.llm import (
    GeminiClient,
    LLMClient,
    OpenAICompatibleClient,
    build_llm,
)
from storyboard_weaver.transport import post_json


def mock_client(*responses: httpx.Response | Exception):
    """An httpx.Client that answers each request with the next queued response."""
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def no_sleep(_seconds: float) -> None:
    pass


def test_gemini_request_shape_and_answer():
    reply = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "thinking...", "thought": True},
                        {"text": '{"ok": '},
                        {"text": "true}"},
                    ]
                }
            }
        ]
    }
    http, seen = mock_client(httpx.Response(200, json=reply))
    client = GeminiClient(
        api_key="test-key",
        model="gemini-test",
        base_url="https://gemini.test/v1beta/",
        http_client=http,
    )

    assert client.complete("Write it", system="Be brief", json_output=True) == '{"ok": true}'

    request = seen[0]
    assert str(request.url) == "https://gemini.test/v1beta/models/gemini-test:generateContent"
    assert request.headers["x-goog-api-key"] == "test-key"
    assert "key=" not in str(request.url)  # the key never goes in the URL
    body = json.loads(request.content)
    assert body["contents"][0]["parts"][0]["text"] == "Write it"
    assert body["systemInstruction"]["parts"][0]["text"] == "Be brief"
    assert body["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_blocked_prompt_is_a_provider_error():
    http, _ = mock_client(httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    client = GeminiClient(api_key="k", model="m", base_url="https://g.test", http_client=http)
    with pytest.raises(ProviderError, match="blocked: SAFETY"):
        client.complete("x")


def test_openai_compatible_request_shape_and_answer():
    reply = {"choices": [{"message": {"role": "assistant", "content": '{"title": "x"}'}}]}
    http, seen = mock_client(httpx.Response(200, json=reply))
    client = OpenAICompatibleClient(
        api_key="sk-test",
        model="local-model",
        base_url="http://localhost:11434/v1",
        http_client=http,
    )

    assert client.complete("Write it", system="Be brief", json_output=True) == '{"title": "x"}'

    request = seen[0]
    assert str(request.url) == "http://localhost:11434/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body["model"] == "local-model"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["response_format"] == {"type": "json_object"}


def test_openai_refusal_is_reported():
    reply = {"choices": [{"message": {"content": None, "refusal": "Not allowed"}}]}
    http, _ = mock_client(httpx.Response(200, json=reply))
    client = OpenAICompatibleClient(
        api_key="k", model="m", base_url="https://o.test", http_client=http
    )
    with pytest.raises(ProviderError, match="refused: Not allowed"):
        client.complete("x")


def test_transient_failures_are_retried_then_succeed():
    http, seen = mock_client(
        httpx.ConnectTimeout("slow"),
        httpx.Response(429, json={"error": {"message": "rate limited"}}),
        httpx.Response(200, json={"fine": True}),
    )
    delays: list[float] = []
    data = post_json(
        http, "https://x.test/api", payload={}, provider="test", retries=2, sleep=delays.append
    )
    assert data == {"fine": True}
    assert len(seen) == 3
    assert delays == [1.5, 3.0]


def test_retries_give_up_with_the_last_error():
    http, seen = mock_client(*[httpx.Response(503, text="overloaded")] * 3)
    with pytest.raises(ProviderError, match="HTTP 503: overloaded"):
        post_json(http, "https://x.test", payload={}, provider="test", retries=2, sleep=no_sleep)
    assert len(seen) == 3


def test_client_errors_are_not_retried_and_hide_the_key():
    body = {"error": {"message": "API key not valid. Please pass a valid API key."}}
    http, seen = mock_client(httpx.Response(400, json=body))
    client = GeminiClient(
        api_key="secret-key", model="m", base_url="https://g.test", http_client=http
    )
    with pytest.raises(ProviderError) as caught:
        client.complete("x")
    assert len(seen) == 1
    assert "API key not valid" in str(caught.value)
    assert "secret-key" not in str(caught.value)


def test_non_json_success_body_is_a_provider_error():
    http, _ = mock_client(httpx.Response(200, text="<html>proxy page</html>"))
    with pytest.raises(ProviderError, match="not valid JSON"):
        post_json(http, "https://x.test", payload={}, provider="test", sleep=no_sleep)


def test_build_llm_picks_the_configured_provider():
    gemini = build_llm(Settings(gemini_api_key="g"))
    openai = build_llm(Settings(llm_provider="openai", openai_api_key="o", llm_model="m"))
    assert isinstance(gemini, GeminiClient)
    assert isinstance(openai, OpenAICompatibleClient)
    assert openai.model == "m"
    assert isinstance(gemini, LLMClient)
    gemini.close()
    openai.close()


def test_build_llm_requires_a_key():
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        build_llm(Settings())
