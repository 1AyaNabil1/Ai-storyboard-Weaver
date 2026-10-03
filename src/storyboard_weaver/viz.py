"""Mood chart: the emotional arc across scenes next to the mood distribution.

Uses matplotlib's object API (``Figure``) rather than pyplot, so it needs no
display, keeps no global state and is safe to call from tests or a server.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from .analysis import StoryAnalysis

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, length=0, labelsize=9)


def render_mood_chart(analysis: StoryAnalysis, path: Path, *, title: str) -> Path:
    """Draw the chart to ``path`` (PNG) and return the path."""
    scenes = list(range(1, analysis.scene_count + 1))
    width = max(9.0, 6.0 + 0.45 * analysis.scene_count)
    fig = Figure(figsize=(width, 4.6), dpi=150, facecolor=SURFACE, layout="constrained")
    arc_ax, count_ax = fig.subplots(1, 2, width_ratios=[3, 2])

    # Emotional arc: one line, scene number and mood name on the x axis.
    _style_axes(arc_ax)
    arc_ax.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    arc_ax.plot(
        scenes,
        analysis.valence_arc,
        color=SERIES,
        linewidth=2,
        solid_joinstyle="round",
        solid_capstyle="round",
        marker="o",
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=3,
    )
    labels = [f"{n}  {mood.value}" for n, mood in zip(scenes, analysis.mood_arc, strict=True)]
    arc_ax.set_xticks(scenes, labels, rotation=35, ha="right", rotation_mode="anchor")
    arc_ax.set_xlim(0.5, analysis.scene_count + 0.5)
    arc_ax.set_ylim(-1.15, 1.15)
    arc_ax.set_yticks([-1, 0, 1], ["darker", "neutral", "brighter"])
    arc_ax.grid(axis="y", color=GRID, linewidth=1)
    arc_ax.set_axisbelow(True)
    arc_ax.set_title("Emotional arc by scene", loc="left", color=INK, fontsize=11, pad=10)

    # Mood distribution: horizontal bars, most frequent on top, value at the tip.
    _style_axes(count_ax)
    count_ax.spines["bottom"].set_visible(False)
    count_ax.spines["left"].set_visible(True)  # bars grow from the left baseline
    count_ax.spines["left"].set_color(BASELINE)
    moods = list(analysis.mood_counts)
    counts = [analysis.mood_counts[mood] for mood in moods]
    # Reserve at least six rows so a short list still gets thin bars, stacked from the top.
    rows = max(len(moods), 6)
    positions = [rows - 1 - index for index in range(len(moods))]
    count_ax.barh(positions, counts, height=0.45, color=SERIES)
    count_ax.set_ylim(-0.5, rows - 0.5)
    count_ax.set_yticks(positions, [mood.value for mood in moods])
    count_ax.set_xticks([])
    count_ax.set_xlim(0, max(counts) * 1.2)
    for y, value in zip(positions, counts, strict=True):
        count_ax.annotate(
            str(value),
            (value, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=INK_SECONDARY,
            fontsize=9,
        )
    count_ax.set_title("Scenes per mood", loc="left", color=INK, fontsize=11, pad=10)

    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    return path
