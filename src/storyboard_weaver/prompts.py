"""Prompt text for the storyboard writer and the frame illustrator."""

from __future__ import annotations

from collections.abc import Sequence

from .knowledge import Match
from .models import Mood, Scene, ShotType, Storyboard
from .styles import Style

SYSTEM_PROMPT = (
    "You are a storyboard artist and script editor for film. You break story ideas into "
    "shot-by-shot storyboards that a director could shoot from. You always answer with a "
    "single JSON object and nothing else."
)

_MAX_ECHOED_REPLY = 4000


def _choices(values: Sequence[str]) -> str:
    return ", ".join(f'"{value}"' for value in values)


def build_storyboard_prompt(
    story: str, *, scenes: int, style: Style, references: Sequence[Match] = ()
) -> str:
    moods = _choices([mood.value for mood in Mood])
    shots = _choices([shot.value for shot in ShotType])
    prompt = f"""Turn the story below into a storyboard of exactly {scenes} scenes.

STORY
<<<
{story.strip()}
>>>

VISUAL STYLE: {style.name}. {style.direction}

Answer with one JSON object in this shape:
{{
  "title": "a short film title",
  "logline": "one sentence that sums up the film",
  "genre": "the main genre, for example thriller",
  "scenes": [
    {{
      "number": 1,
      "title": "a short scene heading",
      "description": "2-4 sentences on what the camera sees: place, action, light",
      "characters": ["each character visible in the scene"],
      "mood": "one of the moods listed below",
      "shot_type": "one of the shot types listed below",
      "dialogue": [{{"speaker": "NAME", "line": "what they say"}}]
    }}
  ]
}}

Allowed moods: {moods}
Allowed shot types: {shots}

Rules:
- Number the scenes 1 to {scenes} in story order.
- Use the same spelling for a character's name in every scene.
- Use an empty dialogue list for a silent scene.
- Vary the shot types so the sequence is visually interesting.
- Write the description as images and action, not as a summary of feelings."""
    if references:
        examples = "\n".join(
            f'- "{match.entry.title}" ({match.entry.genre}): {match.entry.logline}'
            for match in references
        )
        prompt += (
            "\n\nEarlier storyboards for similar stories, for tone and pacing only. "
            f"Do not reuse their titles, names or scenes:\n{examples}"
        )
    return prompt


def build_repair_prompt(original_prompt: str, reply: str, problem: str) -> str:
    """Ask the model to fix its own answer, quoting what went wrong."""
    echoed = reply.strip()
    if len(echoed) > _MAX_ECHOED_REPLY:
        echoed = echoed[:_MAX_ECHOED_REPLY] + "\n...[truncated]"
    return (
        f"{original_prompt}\n\n"
        f"Your previous answer was:\n<<<\n{echoed}\n>>>\n\n"
        f"It could not be used because:\n{problem}\n\n"
        "Send the corrected storyboard as a single JSON object, with no other text."
    )


def build_image_prompt(scene: Scene, storyboard: Storyboard, style: Style) -> str:
    """Describe one storyboard frame for a text-to-image model."""
    cast = ", ".join(scene.characters) if scene.characters else "no people in frame"
    return (
        f"{style.look}. {scene.shot_type.value} shot, {scene.mood.value} mood. "
        f"{scene.description} "
        f"On screen: {cast}. "
        f'Frame {scene.number} of the {storyboard.genre} film "{storyboard.title}". '
        "Do not include any text, captions, speech bubbles or watermarks."
    )
