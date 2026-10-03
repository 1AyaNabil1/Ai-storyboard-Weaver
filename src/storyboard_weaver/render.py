"""Readable Markdown version of a run, for GitHub or any Markdown viewer."""

from __future__ import annotations

from .analysis import StoryAnalysis
from .models import Storyboard


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(
    storyboard: Storyboard,
    analysis: StoryAnalysis,
    *,
    style: str,
    frames: dict[int, str],
    frame_errors: dict[int, str],
    chart: str | None,
) -> str:
    """``frames`` and ``chart`` are paths relative to the Markdown file."""
    out = [
        f"# {storyboard.title}",
        "",
        f"*{storyboard.logline}*",
        "",
        f"**Genre:** {storyboard.genre} | **Style:** {style} | "
        f"**Scenes:** {analysis.scene_count} | **Dominant mood:** {analysis.dominant_mood.value}",
        "",
    ]
    for scene in storyboard.scenes:
        out += [f"## Scene {scene.number}: {scene.title}", ""]
        if scene.number in frames:
            out += [f"![Frame for scene {scene.number}]({frames[scene.number]})", ""]
        elif scene.number in frame_errors:
            out += [f"> Frame not generated: {frame_errors[scene.number]}", ""]
        on_screen = ", ".join(scene.characters) or "nobody"
        out += [
            f"**Shot:** {scene.shot_type.value} | **Mood:** {scene.mood.value} | "
            f"**On screen:** {on_screen}",
            "",
            scene.description,
            "",
        ]
        if scene.dialogue:
            out += [f"> **{line.speaker}:** {line.line}  " for line in scene.dialogue]
        else:
            out.append("*No dialogue.*")
        out.append("")

    out += ["## Characters", "", "| Character | Scenes | Lines |", "| --- | --- | --- |"]
    out += [
        f"| {_cell(c.name)} | {', '.join(map(str, c.scenes))} | {c.lines} |"
        for c in analysis.characters
    ]
    out += ["", "## Mood", ""]
    if chart:
        out += [f"![Emotional arc and mood distribution]({chart})", ""]
    arc = " -> ".join(mood.value for mood in analysis.mood_arc)
    out += [f"Arc: {arc} ({analysis.mood_shifts} mood shifts).", ""]
    return "\n".join(out)
