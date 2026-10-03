"""Structural summary of a storyboard: who appears where, and how the mood moves.

The output is the per-storyboard knowledge file written next to each run
(``storyboard.json`` -> ``analysis``) and the input for the mood chart.
"""

from __future__ import annotations

from collections import Counter
from itertools import pairwise

from pydantic import BaseModel

from .models import Mood, Storyboard

# Rough emotional valence of each mood, from -1 (darkest) to +1 (brightest).
# A storytelling heuristic used only to draw the arc, not a measured quantity.
MOOD_VALENCE: dict[Mood, float] = {
    Mood.JOYFUL: 1.0,
    Mood.HOPEFUL: 0.7,
    Mood.ROMANTIC: 0.6,
    Mood.CALM: 0.3,
    Mood.MYSTERIOUS: -0.1,
    Mood.SUSPENSEFUL: -0.4,
    Mood.TENSE: -0.6,
    Mood.MELANCHOLIC: -0.6,
    Mood.CHAOTIC: -0.7,
    Mood.DARK: -0.9,
}


class CharacterSummary(BaseModel):
    name: str
    scenes: list[int]
    lines: int


class StoryAnalysis(BaseModel):
    scene_count: int
    mood_arc: list[Mood]
    valence_arc: list[float]
    mood_counts: dict[Mood, int]
    dominant_mood: Mood
    mood_shifts: int
    shot_counts: dict[str, int]
    characters: list[CharacterSummary]
    dialogue_lines: int
    silent_scenes: list[int]


def analyze(storyboard: Storyboard) -> StoryAnalysis:
    scenes = storyboard.scenes
    moods = [scene.mood for scene in scenes]
    mood_counts = Counter(moods)

    # Characters are keyed case-insensitively; speakers who never appear in a
    # scene's character list still count (for example, a voice-over).
    display: dict[str, str] = {}
    appearances: dict[str, list[int]] = {}
    lines: Counter[str] = Counter()
    for scene in scenes:
        present = [*scene.characters, *(line.speaker for line in scene.dialogue)]
        for name in present:
            key = name.casefold()
            display.setdefault(key, name)
            numbers = appearances.setdefault(key, [])
            if scene.number not in numbers:
                numbers.append(scene.number)
        lines.update(line.speaker.casefold() for line in scene.dialogue)

    return StoryAnalysis(
        scene_count=len(scenes),
        mood_arc=moods,
        valence_arc=[MOOD_VALENCE[mood] for mood in moods],
        mood_counts=dict(mood_counts.most_common()),
        # Ties go to the mood that appears first in the story.
        dominant_mood=max(moods, key=lambda mood: (mood_counts[mood], -moods.index(mood))),
        mood_shifts=sum(1 for a, b in pairwise(moods) if a != b),
        shot_counts=dict(Counter(scene.shot_type.value for scene in scenes).most_common()),
        characters=[
            CharacterSummary(name=display[key], scenes=appearances[key], lines=lines[key])
            for key in display
        ],
        dialogue_lines=sum(len(scene.dialogue) for scene in scenes),
        silent_scenes=[scene.number for scene in scenes if not scene.dialogue],
    )
