"""One small HTTP helper shared by every provider client.

It retries timeouts, dropped connections, rate limits and 5xx answers with
exponential backoff, and turns every other failure into ``ProviderError``
with a message that is safe to print (no URLs with keys, no headers).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

from .errors import ProviderError

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def _error_detail(response: httpx.Response) -> str:
    """Pull the human-readable message out of an error body if there is one."""
    try:
        body = response.json()
    except ValueError:
        return response.text.strip()[:200] or response.reason_phrase
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:300]
        if isinstance(error, str):
            return error[:300]
    return response.reason_phrase


def post_json(
    client: httpx.Client,
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    provider: str,
    retries: int = 2,
    backoff: float = 1.5,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """POST ``payload`` and return the decoded JSON object, retrying transient failures."""
    last_error = ProviderError(f"{provider}: request was not attempted")
    for attempt in range(retries + 1):
        try:
            response = client.post(url, json=payload, headers=headers)
        except httpx.TransportError as exc:
            last_error = ProviderError(
                f"{provider}: could not complete the request ({type(exc).__name__})"
            )
        else:
            if response.is_success:
                try:
                    data = response.json()
                except ValueError:
                    raise ProviderError(f"{provider}: response was not valid JSON") from None
                if not isinstance(data, dict):
                    raise ProviderError(f"{provider}: expected a JSON object in the response")
                return data
            last_error = ProviderError(
                f"{provider}: HTTP {response.status_code}: {_error_detail(response)}"
            )
            if response.status_code not in RETRYABLE_STATUS:
                raise last_error
        if attempt < retries:
            delay = backoff * 2**attempt
            logger.info("%s; retrying in %.1fs", last_error, delay)
            sleep(delay)
    raise last_error
