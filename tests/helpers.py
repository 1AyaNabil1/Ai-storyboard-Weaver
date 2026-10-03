"""Test doubles and sample data. Nothing in the test suite talks to a network."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from storyboard_weaver.errors import ProviderError
from storyboard_weaver.images import GeneratedImage

# A real 1x1 transparent PNG, so files written by the pipeline are valid images.
TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d4944415478da63606060600000000500017aa857500000000049454e44ae426082"
)

LIGHTHOUSE_STORY = (
    "A lighthouse keeper on a remote island finds a message in a bottle that "
    "describes tomorrow's storm in perfect detail, and signs it with her own name."
)


def storyboard_payload(scenes: int = 3) -> dict[str, Any]:
    plan = [
        ("Bottle at Low Tide", "Mara", "mysterious", "establishing", []),
        (
            "Reading by Lamplight",
            "Mara",
            "tense",
            "close-up",
            [{"speaker": "Mara", "line": "That's my handwriting."}],
        ),
        (
            "The Harbour Master",
            "Mara, Tomas",
            "suspenseful",
            "two-shot",
            [
                {"speaker": "Tomas", "line": "Nobody sails tomorrow."},
                {"speaker": "Mara", "line": "Then who wrote this?"},
            ],
        ),
        ("Storm Front", "Mara", "chaotic", "wide", []),
        (
            "Morning After",
            "Mara, Tomas",
            "hopeful",
            "medium",
            [{"speaker": "Tomas", "line": "You kept the light on."}],
        ),
    ]
    chosen = [plan[i % len(plan)] for i in range(scenes)]
    return {
        "title": "The Keeper's Letter",
        "logline": "A lighthouse keeper receives a warning written in her own hand.",
        "genre": "mystery",
        "scenes": [
            {
                "number": index,
                "title": title,
                "description": f"Scene {index}: the island under a heavy sky as {who} reacts.",
                "characters": [name.strip() for name in who.split(",")],
                "mood": mood,
                "shot_type": shot,
                "dialogue": dialogue,
            }
            for index, (title, who, mood, shot, dialogue) in enumerate(chosen, start=1)
        ],
    }


def storyboard_json(scenes: int = 3, **overrides: Any) -> str:
    payload = storyboard_payload(scenes)
    payload.update(overrides)
    return json.dumps(payload)


class FakeLLM:
    """Replays queued replies; an ``Exception`` instance in the queue is raised instead."""

    def __init__(self, replies: Iterable[str | Exception]):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    def complete(self, prompt: str, *, system: str | None = None, json_output: bool = False) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        if not self.replies:
            raise AssertionError("FakeLLM ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeImageGenerator:
    """Returns a tiny PNG for every prompt, or fails on chosen calls (1-based)."""

    name = "fake"
    model = "fake-image-model"

    def __init__(self, fail_on: Iterable[int] = ()):
        self.fail_on = set(fail_on)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> GeneratedImage:
        self.prompts.append(prompt)
        if len(self.prompts) in self.fail_on:
            raise ProviderError("fake: image request was refused by the safety system")
        return GeneratedImage(data=TINY_PNG, mime_type="image/png")
