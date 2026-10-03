"""Turn raw model text into a validated ``Storyboard``.

Models asked for JSON still sometimes wrap it in a Markdown fence or add a
sentence before it. ``parse_storyboard`` tolerates that, and when the content
itself is wrong it raises ``InvalidModelOutput`` with a short, model-readable
explanation that the pipeline sends back as a correction request.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .models import Storyboard

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_MAX_REPORTED_ERRORS = 8


class InvalidModelOutput(ValueError):
    """The model answered, but the answer is not a usable storyboard."""


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON object found in ``text``."""
    candidates = [match.group(1) for match in _FENCE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        start = candidate.find("{")
        while start != -1:
            try:
                value, _ = decoder.raw_decode(candidate, start)
            except json.JSONDecodeError:
                start = candidate.find("{", start + 1)
                continue
            if isinstance(value, dict):
                return value
            start = candidate.find("{", start + 1)
    raise InvalidModelOutput("The reply did not contain a JSON object.")


def describe_validation_error(error: ValidationError) -> str:
    """Summarise pydantic errors as ``path: message`` lines."""
    lines = []
    for item in error.errors()[:_MAX_REPORTED_ERRORS]:
        path = ".".join(str(part) for part in item["loc"]) or "(root)"
        lines.append(f"- {path}: {item['msg']}")
    hidden = error.error_count() - _MAX_REPORTED_ERRORS
    if hidden > 0:
        lines.append(f"- ...and {hidden} more")
    return "\n".join(lines)


def parse_storyboard(text: str) -> Storyboard:
    data = extract_json_object(text)
    if "storyboard" in data and isinstance(data["storyboard"], dict) and "scenes" not in data:
        data = data["storyboard"]  # some models add an extra wrapper object
    try:
        return Storyboard.model_validate(data)
    except ValidationError as exc:
        raise InvalidModelOutput(
            "The JSON did not match the storyboard schema:\n" + describe_validation_error(exc)
        ) from exc
