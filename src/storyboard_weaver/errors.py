"""Exception hierarchy.

Everything the package raises on purpose derives from ``StoryboardError`` so
callers (and the CLI) can catch one type and print a clean message instead of
a traceback.
"""


class StoryboardError(Exception):
    """Base class for all expected failures."""


class ConfigError(StoryboardError):
    """Settings are missing or invalid (for example, no API key)."""


class ProviderError(StoryboardError):
    """A model provider could not be reached or returned something unusable."""


class StoryboardGenerationError(StoryboardError):
    """The language model never produced a storyboard that passed validation."""
