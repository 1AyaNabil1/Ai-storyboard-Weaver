"""Visual style presets.

Each preset steers two things: how the language model writes the scenes
(``direction``) and how scene frames are described to the image model
(``look``). The five names come from the original prototype notebook.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Style:
    name: str
    direction: str
    look: str


STYLES: dict[str, Style] = {
    style.name: style
    for style in (
        Style(
            name="cinematic",
            direction=(
                "Feature-film storytelling: motivated camera moves, clear visual contrast "
                "between scenes, and a strong opening and closing image."
            ),
            look="cinematic film still, anamorphic lens, dramatic motivated lighting",
        ),
        Style(
            name="documentary",
            direction=(
                "Observational and grounded: real locations, available light, and people "
                "who speak as if interviewed or overheard rather than performing."
            ),
            look="documentary photograph, handheld camera, natural light, light film grain",
        ),
        Style(
            name="anime",
            direction=(
                "Japanese animation staging: expressive reactions, dynamic angles, and "
                "emotional close-ups that hold on a character's face."
            ),
            look="anime key frame, clean line art, cel shading, saturated colour",
        ),
        Style(
            name="noir",
            direction=(
                "Classic film noir: night-time city streets, moral ambiguity, clipped "
                "dialogue, and characters half hidden in shadow."
            ),
            look="black-and-white film noir frame, low-key lighting, hard shadows, high contrast",
        ),
        Style(
            name="experimental",
            direction=(
                "Dreamlike and unconventional: symbolic images, surprising transitions, "
                "and moments that may break strict chronology."
            ),
            look="surreal art-house film frame, unusual composition, double exposure",
        ),
    )
}

DEFAULT_STYLE = "cinematic"


def get_style(name: str) -> Style:
    """Look up a preset by name (case-insensitive)."""
    try:
        return STYLES[name.strip().lower()]
    except KeyError:
        choices = ", ".join(STYLES)
        raise ValueError(f"Unknown style {name!r}. Choose one of: {choices}.") from None
