"""End-to-end run: story -> validated storyboard -> frames -> analysis -> files.

story --> knowledge base search --> prompt --> LLM --> parse + validate
                                                ^             | invalid
                                                +-- repair <--+
storyboard --> ImageGenerator (optional, one call per scene) --> frames/
           --> analyze() --> mood_chart.png
           --> storyboard.json + storyboard.md --> knowledge base entry
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from .analysis import StoryAnalysis, analyze
from .errors import ProviderError, StoryboardGenerationError
from .images import ImageGenerator
from .knowledge import KnowledgeBase, KnowledgeEntry, Match
from .llm import LLMClient
from .models import Storyboard
from .parsing import InvalidModelOutput, parse_storyboard
from .prompts import (
    SYSTEM_PROMPT,
    build_image_prompt,
    build_repair_prompt,
    build_storyboard_prompt,
)
from .render import render_markdown
from .styles import DEFAULT_STYLE, Style, get_style
from .viz import render_mood_chart

logger = logging.getLogger(__name__)

MAX_SCENES = 12
MAX_STORY_CHARS = 8000
# Stop asking for frames after this many failures in a row (bad key, quota, outage).
MAX_CONSECUTIVE_FRAME_FAILURES = 3


def slugify(text: str, max_length: int = 40) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "storyboard"


def _describe(component: object, kind_attr: str) -> str:
    kind = getattr(component, kind_attr, type(component).__name__)
    model = getattr(component, "model", "unknown")
    return f"{kind}:{model}"


class RunManifest(BaseModel):
    """Everything about one run; saved as ``storyboard.json``."""

    created_at: datetime
    story: str
    style: str
    text_model: str
    image_model: str | None
    attempts: int
    references: list[str]
    storyboard: Storyboard
    analysis: StoryAnalysis
    frames: dict[int, str]
    frame_errors: dict[int, str]


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    manifest: RunManifest
    json_path: Path
    markdown_path: Path
    chart_path: Path


class StoryboardWeaver:
    def __init__(
        self,
        llm: LLMClient,
        *,
        image_generator: ImageGenerator | None = None,
        knowledge_base: KnowledgeBase | None = None,
        max_attempts: int = 3,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.llm = llm
        self.image_generator = image_generator
        self.knowledge_base = knowledge_base
        self.max_attempts = max_attempts
        self._now = now

    def write_storyboard(
        self, story: str, *, scenes: int, style: Style, references: Sequence[Match] = ()
    ) -> tuple[Storyboard, int]:
        """Ask the model for a storyboard, feeding validation problems back to it.

        Returns the storyboard and the number of attempts used. Raises
        ``StoryboardGenerationError`` if the provider fails or no attempt validates.
        """
        prompt = build_storyboard_prompt(story, scenes=scenes, style=style, references=references)
        request = prompt
        near_miss: Storyboard | None = None
        problem = "no attempt was made"
        for attempt in range(1, self.max_attempts + 1):
            try:
                reply = self.llm.complete(request, system=SYSTEM_PROMPT, json_output=True)
            except ProviderError as exc:
                raise StoryboardGenerationError(
                    f"The language model request failed. {exc}"
                ) from exc
            try:
                storyboard = parse_storyboard(reply)
            except InvalidModelOutput as exc:
                problem = str(exc)
            else:
                if len(storyboard.scenes) == scenes:
                    return storyboard, attempt
                near_miss = storyboard
                problem = f"Expected exactly {scenes} scenes but got {len(storyboard.scenes)}."
            logger.warning(
                "Attempt %d of %d was rejected: %s",
                attempt,
                self.max_attempts,
                problem.splitlines()[0],
            )
            request = build_repair_prompt(prompt, reply, problem)
        if near_miss is not None:
            logger.warning(
                "Keeping a valid storyboard with %d scenes instead of %d",
                len(near_miss.scenes),
                scenes,
            )
            return near_miss, self.max_attempts
        raise StoryboardGenerationError(
            f"The model did not return a usable storyboard after {self.max_attempts} "
            f"attempt(s). Last problem: {problem}"
        )

    def run(
        self,
        story: str,
        *,
        scenes: int = 4,
        style: str = DEFAULT_STYLE,
        output_dir: Path = Path("outputs"),
    ) -> RunResult:
        story = story.strip()
        if not story:
            raise ValueError("The story is empty.")
        if len(story) > MAX_STORY_CHARS:
            raise ValueError(f"The story is longer than {MAX_STORY_CHARS} characters.")
        if not 1 <= scenes <= MAX_SCENES:
            raise ValueError(f"Scenes must be between 1 and {MAX_SCENES}.")
        preset = get_style(style)

        references = self.knowledge_base.search(story) if self.knowledge_base else []
        storyboard, attempts = self.write_storyboard(
            story, scenes=scenes, style=preset, references=references
        )

        created_at = self._now()
        run_dir = self._new_run_dir(Path(output_dir), created_at, storyboard.title)
        frames, frame_errors = self._draw_frames(storyboard, preset, run_dir)
        analysis = analyze(storyboard)
        chart_path = render_mood_chart(analysis, run_dir / "mood_chart.png", title=storyboard.title)

        manifest = RunManifest(
            created_at=created_at,
            story=story,
            style=preset.name,
            text_model=_describe(self.llm, "provider"),
            image_model=_describe(self.image_generator, "name") if self.image_generator else None,
            attempts=attempts,
            references=[match.entry.id for match in references],
            storyboard=storyboard,
            analysis=analysis,
            frames=frames,
            frame_errors=frame_errors,
        )
        json_path = run_dir / "storyboard.json"
        json_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        markdown_path = run_dir / "storyboard.md"
        markdown_path.write_text(
            render_markdown(
                storyboard,
                analysis,
                style=preset.name,
                frames=frames,
                frame_errors=frame_errors,
                chart=chart_path.name,
            ),
            encoding="utf-8",
        )
        self._remember(manifest, run_dir)
        return RunResult(run_dir, manifest, json_path, markdown_path, chart_path)

    @staticmethod
    def _new_run_dir(output_dir: Path, created_at: datetime, title: str) -> Path:
        base = output_dir / f"{created_at:%Y%m%d-%H%M%S}-{slugify(title)}"
        candidate, counter = base, 2
        while candidate.exists():
            candidate = base.with_name(f"{base.name}-{counter}")
            counter += 1
        candidate.mkdir(parents=True)
        return candidate

    def _draw_frames(
        self, storyboard: Storyboard, style: Style, run_dir: Path
    ) -> tuple[dict[int, str], dict[int, str]]:
        """Generate one frame per scene. A failed frame is recorded, not fatal."""
        frames: dict[int, str] = {}
        errors: dict[int, str] = {}
        if self.image_generator is None:
            return frames, errors
        frames_dir = run_dir / "frames"
        frames_dir.mkdir()
        failures_in_a_row = 0
        for scene in storyboard.scenes:
            if failures_in_a_row >= MAX_CONSECUTIVE_FRAME_FAILURES:
                errors[scene.number] = "skipped after repeated image failures"
                continue
            try:
                image = self.image_generator.generate(build_image_prompt(scene, storyboard, style))
            except ProviderError as exc:
                failures_in_a_row += 1
                errors[scene.number] = str(exc)
                logger.warning("Frame for scene %d failed: %s", scene.number, exc)
                continue
            failures_in_a_row = 0
            name = f"scene_{scene.number:02d}{image.extension}"
            (frames_dir / name).write_bytes(image.data)
            frames[scene.number] = f"frames/{name}"
        return frames, errors

    def _remember(self, manifest: RunManifest, run_dir: Path) -> None:
        if self.knowledge_base is None:
            return
        board = manifest.storyboard
        entry = KnowledgeEntry(
            id=run_dir.name,
            created_at=manifest.created_at,
            story=manifest.story,
            style=manifest.style,
            title=board.title,
            logline=board.logline,
            genre=board.genre,
            moods=manifest.analysis.mood_arc,
            output_dir=str(run_dir),
        )
        try:
            self.knowledge_base.add(entry)
        except OSError as exc:  # the run itself succeeded; do not lose it over this
            logger.warning("Could not update the knowledge base: %s", exc)
