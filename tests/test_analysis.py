from helpers import storyboard_payload

from storyboard_weaver.analysis import MOOD_VALENCE, analyze
from storyboard_weaver.models import Mood, Storyboard
from storyboard_weaver.viz import render_mood_chart


def board(scenes: int = 5) -> Storyboard:
    return Storyboard.model_validate(storyboard_payload(scenes))


def test_every_mood_has_a_valence_in_range():
    assert set(MOOD_VALENCE) == set(Mood)
    assert all(-1 <= value <= 1 for value in MOOD_VALENCE.values())


def test_analysis_of_the_sample_storyboard():
    result = analyze(board(5))

    assert result.scene_count == 5
    assert result.mood_arc == [
        Mood.MYSTERIOUS,
        Mood.TENSE,
        Mood.SUSPENSEFUL,
        Mood.CHAOTIC,
        Mood.HOPEFUL,
    ]
    assert result.valence_arc == [MOOD_VALENCE[m] for m in result.mood_arc]
    assert result.mood_shifts == 4
    assert result.dialogue_lines == 4
    assert result.silent_scenes == [1, 4]
    assert result.shot_counts["two-shot"] == 1

    people = {c.name: c for c in result.characters}
    assert list(people) == ["Mara", "Tomas"]
    assert people["Mara"].scenes == [1, 2, 3, 4, 5]
    assert people["Mara"].lines == 2
    assert people["Tomas"].scenes == [3, 5]
    assert people["Tomas"].lines == 2


def test_dominant_mood_ties_go_to_the_earliest():
    payload = storyboard_payload(4)
    for scene, mood in zip(payload["scenes"], ["dark", "calm", "calm", "dark"], strict=True):
        scene["mood"] = mood
    result = analyze(Storyboard.model_validate(payload))
    assert result.dominant_mood is Mood.DARK
    assert result.mood_counts == {Mood.DARK: 2, Mood.CALM: 2}


def test_speakers_missing_from_the_character_list_are_still_counted():
    payload = storyboard_payload(1)
    payload["scenes"][0]["dialogue"] = [{"speaker": "radio voice", "line": "Storm warning."}]
    result = analyze(Storyboard.model_validate(payload))
    names = [c.name for c in result.characters]
    assert names == ["Mara", "radio voice"]


def test_mood_chart_is_written_as_png(tmp_path):
    path = render_mood_chart(analyze(board(5)), tmp_path / "charts" / "mood.png", title="Test")
    assert path.read_bytes().startswith(b"\x89PNG")


def test_single_scene_chart(tmp_path):
    path = render_mood_chart(analyze(board(1)), tmp_path / "mood.png", title="One scene")
    assert path.stat().st_size > 0
