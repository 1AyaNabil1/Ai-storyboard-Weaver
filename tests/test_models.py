import pytest
from helpers import storyboard_payload
from pydantic import ValidationError

from storyboard_weaver.models import DialogueLine, Mood, Scene, ShotType, Storyboard
from storyboard_weaver.styles import STYLES, get_style


def scene(**changes):
    data = storyboard_payload(1)["scenes"][0]
    data.update(changes)
    return data


def test_valid_payload_round_trips():
    board = Storyboard.model_validate(storyboard_payload(5))
    assert len(board.scenes) == 5
    assert board.scenes[2].shot_type is ShotType.TWO_SHOT
    assert board.characters == ["Mara", "Tomas"]
    again = Storyboard.model_validate_json(board.model_dump_json())
    assert again == board


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Close Up", ShotType.CLOSE_UP),
        ("close_up", ShotType.CLOSE_UP),
        ("CU", ShotType.CLOSE_UP),
        ("Extreme close-up shot", ShotType.EXTREME_CLOSE_UP),
        ("POV", ShotType.POINT_OF_VIEW),
        ("over the shoulder", ShotType.OVER_THE_SHOULDER),
        ("Establishing shot", ShotType.ESTABLISHING),
        ("drone", ShotType.AERIAL),
    ],
)
def test_shot_type_spellings_are_normalised(raw, expected):
    assert Scene.model_validate(scene(shot_type=raw)).shot_type is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Tense", Mood.TENSE),
        ("happy", Mood.JOYFUL),
        ("SAD", Mood.MELANCHOLIC),
        ("eerie", Mood.MYSTERIOUS),
    ],
)
def test_mood_synonyms_are_normalised(raw, expected):
    assert Scene.model_validate(scene(mood=raw)).mood is expected


def test_unknown_labels_are_rejected():
    with pytest.raises(ValidationError, match="mood"):
        Scene.model_validate(scene(mood="sparkly"))
    with pytest.raises(ValidationError, match="shot_type"):
        Scene.model_validate(scene(shot_type="dutch tilt"))


def test_alternative_keys_and_loose_fields_are_accepted():
    data = scene(characters="Mara, mara ,  Tomas,", dialogue=None)
    data["scene_number"] = data.pop("number")
    data["shot"] = data.pop("shot_type")
    parsed = Scene.model_validate(data)
    assert parsed.characters == ["Mara", "Tomas"]
    assert parsed.dialogue == []


def test_dialogue_accepts_script_style_strings():
    assert DialogueLine.model_validate("TOMAS: Nobody sails tomorrow.") == DialogueLine(
        speaker="TOMAS", line="Nobody sails tomorrow."
    )
    assert DialogueLine.model_validate("The wind answers.").speaker == "Narrator"


def test_scenes_are_sorted_and_renumbered():
    payload = storyboard_payload(3)
    for item, number in zip(payload["scenes"], (7, 2, 4), strict=True):
        item["number"] = number
    board = Storyboard.model_validate(payload)
    assert [s.number for s in board.scenes] == [1, 2, 3]
    assert [s.title for s in board.scenes] == [
        "Reading by Lamplight",
        "The Harbour Master",
        "Bottle at Low Tide",
    ]


def test_empty_storyboard_is_rejected():
    payload = storyboard_payload(1)
    payload["scenes"] = []
    with pytest.raises(ValidationError):
        Storyboard.model_validate(payload)


def test_style_lookup():
    assert get_style(" Noir ") is STYLES["noir"]
    with pytest.raises(ValueError, match="cinematic"):
        get_style("vaporwave")
