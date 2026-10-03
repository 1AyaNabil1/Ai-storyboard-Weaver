"""Typed storyboard schema.

The language model is asked for JSON in exactly this shape. Validation is
strict about structure but forgiving about spelling: ``"Close up"``,
``"close_up"`` and ``"CU"`` all become ``ShotType.CLOSE_UP``, and moods accept
a few common synonyms. Anything that still does not fit is reported back to
the model so it can correct itself.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class Mood(StrEnum):
    JOYFUL = "joyful"
    HOPEFUL = "hopeful"
    ROMANTIC = "romantic"
    CALM = "calm"
    MYSTERIOUS = "mysterious"
    SUSPENSEFUL = "suspenseful"
    TENSE = "tense"
    MELANCHOLIC = "melancholic"
    CHAOTIC = "chaotic"
    DARK = "dark"


class ShotType(StrEnum):
    ESTABLISHING = "establishing"
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close-up"
    EXTREME_CLOSE_UP = "extreme close-up"
    OVER_THE_SHOULDER = "over-the-shoulder"
    POINT_OF_VIEW = "point-of-view"
    TWO_SHOT = "two-shot"
    AERIAL = "aerial"
    TRACKING = "tracking"


def _key(text: str) -> str:
    """Reduce a label to bare letters so spelling variants compare equal."""
    return re.sub(r"[^a-z]", "", text.lower()).removesuffix("shot")


_MOOD_SYNONYMS = {
    "happy": Mood.JOYFUL,
    "cheerful": Mood.JOYFUL,
    "uplifting": Mood.HOPEFUL,
    "optimistic": Mood.HOPEFUL,
    "tender": Mood.ROMANTIC,
    "peaceful": Mood.CALM,
    "serene": Mood.CALM,
    "eerie": Mood.MYSTERIOUS,
    "mystery": Mood.MYSTERIOUS,
    "suspense": Mood.SUSPENSEFUL,
    "anxious": Mood.TENSE,
    "sad": Mood.MELANCHOLIC,
    "somber": Mood.MELANCHOLIC,
    "sombre": Mood.MELANCHOLIC,
    "frantic": Mood.CHAOTIC,
    "ominous": Mood.DARK,
    "grim": Mood.DARK,
}

_SHOT_SYNONYMS = {
    "pov": ShotType.POINT_OF_VIEW,
    "ots": ShotType.OVER_THE_SHOULDER,
    "cu": ShotType.CLOSE_UP,
    "ecu": ShotType.EXTREME_CLOSE_UP,
    "xcu": ShotType.EXTREME_CLOSE_UP,
    "ms": ShotType.MEDIUM,
    "mediumcloseup": ShotType.CLOSE_UP,
    "mediumlong": ShotType.WIDE,
    "ws": ShotType.WIDE,
    "long": ShotType.WIDE,
    "full": ShotType.WIDE,
    "extremewide": ShotType.WIDE,
    "birdseye": ShotType.AERIAL,
    "birdseyeview": ShotType.AERIAL,
    "drone": ShotType.AERIAL,
    "dolly": ShotType.TRACKING,
    "steadicam": ShotType.TRACKING,
}


def _coerce_label(value: Any, members: type[StrEnum], synonyms: dict[str, StrEnum]) -> Any:
    if not isinstance(value, str) or isinstance(value, members):
        return value
    key = _key(value)
    for member in members:
        if _key(member.value) == key:
            return member
    return synonyms.get(key, value)  # unknown labels fall through to a validation error


class DialogueLine(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    speaker: str = Field(min_length=1, max_length=80)
    line: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _from_script_line(cls, data: Any) -> Any:
        """Accept ``"MARA: Who sent this?"`` as well as the object form."""
        if isinstance(data, str):
            speaker, sep, line = data.partition(":")
            if sep and speaker.strip() and len(speaker) <= 80:
                return {"speaker": speaker, "line": line}
            return {"speaker": "Narrator", "line": data}
        return data


class Scene(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    number: int = Field(ge=1, validation_alias=AliasChoices("number", "scene_number"))
    title: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=10)
    characters: list[str] = Field(default_factory=list)
    mood: Mood
    shot_type: ShotType = Field(validation_alias=AliasChoices("shot_type", "shot"))
    dialogue: list[DialogueLine] = Field(default_factory=list)

    @field_validator("mood", mode="before")
    @classmethod
    def _normalise_mood(cls, value: Any) -> Any:
        return _coerce_label(value, Mood, _MOOD_SYNONYMS)

    @field_validator("shot_type", mode="before")
    @classmethod
    def _normalise_shot(cls, value: Any) -> Any:
        return _coerce_label(value, ShotType, _SHOT_SYNONYMS)

    @field_validator("characters", mode="before")
    @classmethod
    def _clean_characters(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.split(",")
        if not isinstance(value, list):
            return value
        seen: set[str] = set()
        names: list[str] = []
        for item in value:
            name = str(item).strip()
            if name and name.casefold() not in seen:
                seen.add(name.casefold())
                names.append(name)
        return names

    @field_validator("dialogue", mode="before")
    @classmethod
    def _null_dialogue_is_silence(cls, value: Any) -> Any:
        return [] if value is None else value


class Storyboard(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=120)
    logline: str = Field(min_length=1)
    genre: str = Field(min_length=1, max_length=60)
    scenes: list[Scene] = Field(min_length=1)

    @model_validator(mode="after")
    def _renumber(self) -> Storyboard:
        """Order scenes by the model's numbering, then number them 1..n with no gaps."""
        ordered = sorted(self.scenes, key=lambda scene: scene.number)
        self.scenes = [
            scene if scene.number == index else scene.model_copy(update={"number": index})
            for index, scene in enumerate(ordered, start=1)
        ]
        return self

    @property
    def characters(self) -> list[str]:
        """Every character in order of first appearance."""
        seen: dict[str, str] = {}
        for scene in self.scenes:
            for name in scene.characters:
                seen.setdefault(name.casefold(), name)
        return list(seen.values())
