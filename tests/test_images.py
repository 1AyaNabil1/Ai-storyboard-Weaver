"""Image providers, exercised against an in-process mock transport (no network)."""

import base64
import json

import httpx
import pytest
from helpers import TINY_PNG, FakeImageGenerator

from storyboard_weaver.config import Settings
from storyboard_weaver.errors import ConfigError, ProviderError
from storyboard_weaver.images import (
    GeminiImageGenerator,
    GeneratedImage,
    ImageGenerator,
    OpenAIImageGenerator,
    build_image_generator,
    closest_aspect_ratio,
    sniff_mime_type,
)

ENCODED_PNG = base64.b64encode(TINY_PNG).decode()


def recording_client(handler):
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(wrapped)), seen


def test_openai_gpt_image_returns_base64_png():
    http, seen = recording_client(
        lambda _: httpx.Response(200, json={"data": [{"b64_json": ENCODED_PNG}]})
    )
    generator = OpenAIImageGenerator(
        api_key="k", model="gpt-image-1", base_url="https://o.test/v1", http_client=http
    )

    image = generator.generate("a lighthouse in a storm")

    assert image == GeneratedImage(TINY_PNG, "image/png")
    assert image.extension == ".png"
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://o.test/v1/images/generations"
    assert body == {
        "model": "gpt-image-1",
        "prompt": "a lighthouse in a storm",
        "n": 1,
        "size": "1536x1024",
    }


def test_openai_dalle_style_model_asks_for_base64_and_can_fall_back_to_url():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"data": [{"url": "https://cdn.test/frame.png"}]})
        return httpx.Response(200, content=TINY_PNG)

    http, seen = recording_client(handler)
    generator = OpenAIImageGenerator(
        api_key="k",
        model="dall-e-3",
        base_url="https://o.test/v1",
        size="1792x1024",
        http_client=http,
    )

    assert generator.generate("frame").data == TINY_PNG
    assert json.loads(seen[0].content)["response_format"] == "b64_json"
    assert str(seen[1].url) == "https://cdn.test/frame.png"


def test_openai_content_policy_rejection_is_a_provider_error():
    error = {"error": {"message": "Your request was rejected by the safety system."}}
    http, seen = recording_client(lambda _: httpx.Response(400, json=error))
    generator = OpenAIImageGenerator(
        api_key="k", model="gpt-image-1", base_url="https://o.test", http_client=http
    )
    with pytest.raises(ProviderError, match="safety system"):
        generator.generate("frame")
    assert len(seen) == 1


def test_gemini_image_request_and_inline_data():
    reply = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Here is the frame."},
                        {"inlineData": {"mimeType": "image/png", "data": ENCODED_PNG}},
                    ]
                }
            }
        ]
    }
    http, seen = recording_client(lambda _: httpx.Response(200, json=reply))
    generator = GeminiImageGenerator(
        api_key="k",
        model="img-model",
        base_url="https://g.test/v1beta",
        size="1920x1080",
        http_client=http,
    )

    image = generator.generate("frame")

    assert image.data == TINY_PNG
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://g.test/v1beta/models/img-model:generateContent"
    assert body["generationConfig"]["responseModalities"] == ["TEXT", "IMAGE"]
    assert body["generationConfig"]["imageConfig"] == {"aspectRatio": "16:9"}


def test_gemini_text_only_answer_is_a_provider_error():
    reply = {"candidates": [{"content": {"parts": [{"text": "I can't draw that."}]}}]}
    http, _ = recording_client(lambda _: httpx.Response(200, json=reply))
    generator = GeminiImageGenerator(
        api_key="k", model="m", base_url="https://g.test", http_client=http
    )
    with pytest.raises(ProviderError, match="no image was returned"):
        generator.generate("frame")


def test_bad_base64_is_a_provider_error():
    http, _ = recording_client(lambda _: httpx.Response(200, json={"data": [{"b64_json": "%%%"}]}))
    generator = OpenAIImageGenerator(
        api_key="k", model="gpt-image-1", base_url="https://o.test", http_client=http
    )
    with pytest.raises(ProviderError, match="base64"):
        generator.generate("frame")


@pytest.mark.parametrize(
    ("size", "ratio"),
    [
        ("1024x1024", "1:1"),
        ("1536x1024", "3:2"),
        ("1024x1536", "2:3"),
        ("1792x1024", "16:9"),
        ("2560x1080", "21:9"),
    ],
)
def test_closest_aspect_ratio(size, ratio):
    assert closest_aspect_ratio(size) == ratio


def test_sniff_mime_type():
    assert sniff_mime_type(TINY_PNG) == "image/png"
    assert sniff_mime_type(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    assert sniff_mime_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    with pytest.raises(ProviderError):
        sniff_mime_type(b"GIF89a")


def test_build_image_generator():
    assert build_image_generator(Settings()) is None
    generator = build_image_generator(Settings(image_provider="openai", openai_api_key="o"))
    assert isinstance(generator, OpenAIImageGenerator)
    assert generator.model == "gpt-image-1"
    generator.close()
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        build_image_generator(Settings(image_provider="gemini"))


def test_fake_generator_satisfies_the_protocol():
    assert isinstance(FakeImageGenerator(), ImageGenerator)
