import json

import pytest
from helpers import storyboard_json, storyboard_payload

from storyboard_weaver.parsing import InvalidModelOutput, extract_json_object, parse_storyboard


def test_plain_json():
    assert parse_storyboard(storyboard_json(2)).title == "The Keeper's Letter"


def test_json_inside_a_markdown_fence_with_chatter():
    text = f"Sure! Here is your storyboard:\n```json\n{storyboard_json(2)}\n```\nEnjoy."
    assert len(parse_storyboard(text).scenes) == 2


def test_json_after_a_preamble_with_stray_braces():
    text = "Notes {not json} then the answer: " + storyboard_json(1)
    assert parse_storyboard(text).genre == "mystery"


def test_wrapper_object_is_unwrapped():
    text = json.dumps({"storyboard": storyboard_payload(2)})
    assert len(parse_storyboard(text).scenes) == 2


def test_no_json_at_all():
    with pytest.raises(InvalidModelOutput, match="did not contain a JSON object"):
        parse_storyboard("I'm sorry, I can't help with that.")


def test_top_level_array_is_not_an_object():
    with pytest.raises(InvalidModelOutput):
        extract_json_object("[1, 2, 3]")


def test_schema_errors_are_listed_by_path():
    payload = storyboard_payload(2)
    payload["scenes"][1]["mood"] = "sparkly"
    del payload["logline"]
    with pytest.raises(InvalidModelOutput) as caught:
        parse_storyboard(json.dumps(payload))
    message = str(caught.value)
    assert "logline: Field required" in message
    assert "scenes.1.mood" in message
