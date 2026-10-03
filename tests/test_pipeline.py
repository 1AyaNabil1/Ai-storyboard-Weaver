import json
from datetime import UTC, datetime

import pytest
from helpers import LIGHTHOUSE_STORY, TINY_PNG, FakeImageGenerator, FakeLLM, storyboard_json

from storyboard_weaver.errors import ProviderError, StoryboardGenerationError
from storyboard_weaver.knowledge import KnowledgeBase
from storyboard_weaver.models import Mood
from storyboard_weaver.pipeline import StoryboardWeaver, slugify
from storyboard_weaver.prompts import SYSTEM_PROMPT

FIXED_TIME = datetime(2026, 10, 3, 9, 30, tzinfo=UTC)


def weaver(llm, **kwargs) -> StoryboardWeaver:
    return StoryboardWeaver(llm, now=lambda: FIXED_TIME, **kwargs)


def test_text_only_run_writes_every_artifact(tmp_path):
    llm = FakeLLM([storyboard_json(3)])
    result = weaver(llm).run(LIGHTHOUSE_STORY, scenes=3, style="noir", output_dir=tmp_path)

    assert result.run_dir == tmp_path / "20261003-093000-the-keeper-s-letter"
    assert result.chart_path.read_bytes().startswith(b"\x89PNG")
    assert not (result.run_dir / "frames").exists()

    saved = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert saved["style"] == "noir"
    assert saved["text_model"] == "FakeLLM:unknown"
    assert saved["image_model"] is None
    assert saved["attempts"] == 1
    assert len(saved["storyboard"]["scenes"]) == 3
    assert saved["analysis"]["mood_arc"] == ["mysterious", "tense", "suspenseful"]

    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith("# The Keeper's Letter")
    assert "## Scene 2: Reading by Lamplight" in markdown
    assert "> **Mara:** That's my handwriting." in markdown
    assert "![Emotional arc and mood distribution](mood_chart.png)" in markdown

    prompt = llm.prompts[0]
    assert LIGHTHOUSE_STORY in prompt
    assert "exactly 3 scenes" in prompt
    assert "VISUAL STYLE: noir" in prompt
    assert llm.systems == [SYSTEM_PROMPT]


def test_invalid_reply_is_sent_back_for_repair(tmp_path):
    bad = storyboard_json(3).replace('"tense"', '"sparkly"')
    llm = FakeLLM(["Sorry, here you go: not json", bad, storyboard_json(3)])

    result = weaver(llm).run(LIGHTHOUSE_STORY, scenes=3, output_dir=tmp_path)

    assert result.manifest.attempts == 3
    assert "did not contain a JSON object" in llm.prompts[1]
    assert "Sorry, here you go" in llm.prompts[1]
    assert "scenes.1.mood" in llm.prompts[2]
    assert llm.prompts[2].startswith(llm.prompts[0])  # the original request is kept


def test_wrong_scene_count_is_corrected_when_possible(tmp_path):
    llm = FakeLLM([storyboard_json(2), storyboard_json(4)])
    result = weaver(llm).run(LIGHTHOUSE_STORY, scenes=4, output_dir=tmp_path)
    assert len(result.manifest.storyboard.scenes) == 4
    assert "Expected exactly 4 scenes but got 2" in llm.prompts[1]


def test_valid_storyboard_with_wrong_count_is_kept_as_a_last_resort(tmp_path):
    llm = FakeLLM([storyboard_json(2), "{}"])
    result = weaver(llm, max_attempts=2).run(LIGHTHOUSE_STORY, scenes=4, output_dir=tmp_path)
    assert len(result.manifest.storyboard.scenes) == 2


def test_gives_up_cleanly_when_nothing_validates(tmp_path):
    llm = FakeLLM(["nope", "still nope"])
    with pytest.raises(StoryboardGenerationError, match="after 2 attempt"):
        weaver(llm, max_attempts=2).run(LIGHTHOUSE_STORY, output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []  # nothing half-written


def test_provider_failure_is_not_retried_by_the_pipeline(tmp_path):
    llm = FakeLLM([ProviderError("gemini: HTTP 401: API key not valid"), storyboard_json(3)])
    with pytest.raises(StoryboardGenerationError, match="API key not valid"):
        weaver(llm).run(LIGHTHOUSE_STORY, output_dir=tmp_path)
    assert len(llm.prompts) == 1


def test_frames_are_generated_per_scene(tmp_path):
    images = FakeImageGenerator()
    result = weaver(FakeLLM([storyboard_json(3)]), image_generator=images).run(
        LIGHTHOUSE_STORY, scenes=3, style="anime", output_dir=tmp_path
    )

    assert result.manifest.frames == {
        1: "frames/scene_01.png",
        2: "frames/scene_02.png",
        3: "frames/scene_03.png",
    }
    assert (result.run_dir / "frames" / "scene_02.png").read_bytes() == TINY_PNG
    assert result.manifest.image_model == "fake:fake-image-model"
    assert "anime key frame" in images.prompts[0]
    assert "close-up shot" in images.prompts[1]
    assert "Do not include any text" in images.prompts[2]
    assert "![Frame for scene 1](frames/scene_01.png)" in result.markdown_path.read_text("utf-8")


def test_a_failed_frame_does_not_stop_the_run(tmp_path):
    images = FakeImageGenerator(fail_on={2})
    result = weaver(FakeLLM([storyboard_json(3)]), image_generator=images).run(
        LIGHTHOUSE_STORY, scenes=3, output_dir=tmp_path
    )
    assert set(result.manifest.frames) == {1, 3}
    assert "safety system" in result.manifest.frame_errors[2]
    assert "> Frame not generated:" in result.markdown_path.read_text("utf-8")


def test_frames_stop_after_repeated_failures(tmp_path):
    images = FakeImageGenerator(fail_on={1, 2, 3, 4, 5})
    result = weaver(FakeLLM([storyboard_json(5)]), image_generator=images).run(
        LIGHTHOUSE_STORY, scenes=5, output_dir=tmp_path
    )
    assert len(images.prompts) == 3
    assert result.manifest.frames == {}
    assert result.manifest.frame_errors[5] == "skipped after repeated image failures"


def test_knowledge_base_is_updated_and_used_as_reference(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.json")
    first = weaver(FakeLLM([storyboard_json(3)]), knowledge_base=kb).run(
        LIGHTHOUSE_STORY, scenes=3, output_dir=tmp_path
    )
    stored = kb.entries()
    assert [e.id for e in stored] == [first.run_dir.name]
    assert stored[0].moods == [Mood.MYSTERIOUS, Mood.TENSE, Mood.SUSPENSEFUL]

    llm = FakeLLM([storyboard_json(3)])
    second = weaver(llm, knowledge_base=kb).run(
        "A lonely lighthouse keeper on a storm-battered island receives a mysterious letter.",
        scenes=3,
        output_dir=tmp_path,
    )
    assert second.run_dir != first.run_dir  # same title and time get a fresh folder
    assert second.manifest.references == [first.run_dir.name]
    assert "Earlier storyboards for similar stories" in llm.prompts[0]
    assert len(kb.entries()) == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"story": "   "}, "empty"),
        ({"story": "x" * 9000}, "longer than"),
        ({"story": "ok story", "scenes": 0}, "between 1 and"),
        ({"story": "ok story", "scenes": 13}, "between 1 and"),
        ({"story": "ok story", "style": "vaporwave"}, "Unknown style"),
    ],
)
def test_input_validation(tmp_path, kwargs, message):
    llm = FakeLLM([])
    story = kwargs.pop("story")
    with pytest.raises(ValueError, match=message):
        weaver(llm).run(story, output_dir=tmp_path, **kwargs)
    assert llm.prompts == []


@pytest.mark.parametrize(
    ("title", "slug"),
    [
        ("The Keeper's Letter", "the-keeper-s-letter"),
        ("Café Noir: Part II", "cafe-noir-part-ii"),
        ("!!!", "storyboard"),
        ("A" * 60, "a" * 40),
    ],
)
def test_slugify(title, slug):
    assert slugify(title) == slug
