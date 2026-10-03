"""Command-line interface: ``storyboard-weaver`` or ``python -m storyboard_weaver``.

Exit codes: 0 success, 1 the model or provider failed, 2 bad input or
configuration, 130 interrupted.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from . import __version__
from .config import Settings
from .errors import ConfigError, StoryboardError
from .images import build_image_generator
from .knowledge import KnowledgeBase
from .llm import build_llm
from .pipeline import MAX_SCENES, RunResult, StoryboardWeaver
from .styles import DEFAULT_STYLE, STYLES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="storyboard-weaver",
        description="Turn a short story idea into a scene-by-scene film storyboard.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="show retries and other progress details"
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    generate = commands.add_parser("generate", help="write a storyboard for a story idea")
    generate.add_argument(
        "story",
        nargs="?",
        help='the story idea in quotes; use "-" or omit it to read standard input',
    )
    generate.add_argument("-f", "--file", type=Path, help="read the story from a text file")
    generate.add_argument(
        "-n",
        "--scenes",
        type=int,
        default=4,
        help=f"number of scenes, 1 to {MAX_SCENES} (default: 4)",
    )
    generate.add_argument(
        "-s",
        "--style",
        default=DEFAULT_STYLE,
        choices=list(STYLES),
        help=f"visual style preset (default: {DEFAULT_STYLE})",
    )
    generate.add_argument(
        "--provider", choices=["gemini", "openai"], help="text provider for this run"
    )
    generate.add_argument("--model", help="text model for this run")
    generate.add_argument(
        "--images",
        choices=["none", "gemini", "openai"],
        help="image provider for this run; 'none' skips frames",
    )
    generate.add_argument("-o", "--output-dir", type=Path, help="where to write the run folder")
    generate.add_argument(
        "--no-memory",
        action="store_true",
        help="neither read nor update the knowledge base of past storyboards",
    )

    commands.add_parser("styles", help="list the visual style presets")

    history = commands.add_parser("history", help="list storyboards in the knowledge base")
    history.add_argument("--limit", type=int, default=10, help="how many to show (default: 10)")
    return parser


def _read_story(args: argparse.Namespace) -> str:
    if args.story and args.story != "-":
        if args.file:
            raise ValueError("Give the story either as an argument or with --file, not both.")
        return args.story
    if args.file:
        try:
            return args.file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Could not read {args.file}: {exc.strerror or exc}") from None
    if args.story == "-" or not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError("Give a story as an argument, with --file, or on standard input.")


def _settings_for_run(settings: Settings, args: argparse.Namespace) -> Settings:
    overrides: dict[str, object] = {}
    if args.provider:
        overrides["llm_provider"] = args.provider
        if not args.model and args.provider != settings.llm_provider:
            overrides["llm_model"] = None  # the configured model belongs to the other provider
    if args.model:
        overrides["llm_model"] = args.model
    if args.images:
        overrides["image_provider"] = args.images
        if args.images != settings.image_provider:
            overrides["image_model"] = None
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    return settings.model_copy(update=overrides)


def _print_summary(result: RunResult) -> None:
    manifest = result.manifest
    board = manifest.storyboard
    print(f"\n{board.title} ({board.genre})")
    print(f"{board.logline}\n")
    print(f"  {'#':>2}  {'shot':<18} {'mood':<12} title")
    for scene in board.scenes:
        print(
            f"  {scene.number:>2}  {scene.shot_type.value:<18} {scene.mood.value:<12} {scene.title}"
        )
    if manifest.image_model:
        made, failed = len(manifest.frames), len(manifest.frame_errors)
        note = f" ({failed} failed, see storyboard.md)" if failed else ""
        print(f"\nFrames: {made} of {len(board.scenes)} generated{note}")
    print(f"\nSaved to {result.run_dir}")
    for path in (result.markdown_path, result.json_path, result.chart_path):
        print(f"  {path.name}")


def _generate(settings: Settings, args: argparse.Namespace) -> int:
    story = _read_story(args)
    settings = _settings_for_run(settings, args)
    llm = build_llm(settings)
    images = build_image_generator(settings)
    knowledge_base = None if args.no_memory else KnowledgeBase(settings.kb_path)
    weaver = StoryboardWeaver(
        llm,
        image_generator=images,
        knowledge_base=knowledge_base,
        max_attempts=settings.max_attempts,
    )
    using = f"{settings.llm_provider}:{settings.text_model}"
    if images:
        using += f", frames from {settings.image_provider}:{settings.picture_model}"
    print(f"Writing a {args.scenes}-scene {args.style} storyboard ({using})...", file=sys.stderr)
    result = weaver.run(story, scenes=args.scenes, style=args.style, output_dir=settings.output_dir)
    _print_summary(result)
    return 0


def _styles() -> int:
    for style in STYLES.values():
        print(f"{style.name:<13} {style.direction}")
    return 0


def _history(settings: Settings, limit: int) -> int:
    entries = KnowledgeBase(settings.kb_path).entries()
    if not entries:
        print(f"No storyboards stored yet in {settings.kb_path}.")
        return 0
    for entry in reversed(entries[-max(limit, 1) :]):
        print(f"{entry.created_at:%Y-%m-%d %H:%M}  {entry.title}  [{entry.genre}, {entry.style}]")
        if entry.output_dir:
            print(f"                  {entry.output_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    dotenv = Path.cwd() / ".env"
    if dotenv.is_file():
        load_dotenv(dotenv, override=False)  # real environment variables win

    try:
        settings = Settings.from_env()
        if args.command == "styles":
            return _styles()
        if args.command == "history":
            return _history(settings, args.limit)
        return _generate(settings, args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except StoryboardError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
