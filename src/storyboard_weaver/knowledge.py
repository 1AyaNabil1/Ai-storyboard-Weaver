"""A small persistent knowledge base of past storyboards.

Every finished run is stored as a compact entry (story, title, logline,
genre, moods). Before a new storyboard is written, the most similar past
stories are retrieved and shown to the model as tone references, which is
the retrieval step of the original prototype's RAG design.

Similarity uses ``HashingEmbedder``: a bag-of-words vector built with feature
hashing and compared by cosine similarity. It needs no model download and is
deterministic, at the cost of matching shared vocabulary rather than meaning.
Any object with an ``embed(text) -> list[float]`` method can replace it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from .models import Mood

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")
# Words of one or two letters are dropped anyway, so they are not listed here.
_STOPWORDS = frozenset(
    (
        "about after again all and any are because been before but can could did does for from "
        "had has have her here hers him his how into its just more most not now off once only "
        "other our out over own she some such than that the their them then there these they "
        "this those through too under until very was were what when where which while who whom "
        "why will with would you your"
    ).split()
)
_SUFFIXES = ("ing", "ies", "ed", "es", "ly", "y", "s")


def _stem(token: str) -> str:
    """Crude suffix stripping so that "storms", "stormy" and "storm" share a bucket."""
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Unit-length term-frequency vectors in a fixed number of hashed buckets."""

    def __init__(self, dimensions: int = 1024):
        self.dimensions = dimensions

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        sign = 1.0 if value & 1 else -1.0  # signed hashing keeps collisions unbiased
        return (value >> 1) % self.dimensions, sign

    def embed(self, text: str) -> list[float]:
        tokens = [
            _stem(t) for t in _TOKEN.findall(text.lower()) if len(t) > 2 and t not in _STOPWORDS
        ]
        vector = [0.0] * self.dimensions
        for token, count in Counter(tokens).items():
            index, sign = self._bucket(token)
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class KnowledgeEntry(BaseModel):
    id: str
    created_at: datetime
    story: str
    style: str
    title: str
    logline: str
    genre: str
    moods: list[Mood] = Field(default_factory=list)
    output_dir: str | None = None

    def search_text(self) -> str:
        return f"{self.story} {self.logline} {self.genre}"


class Match(BaseModel):
    entry: KnowledgeEntry
    score: float


class _KnowledgeFile(BaseModel):
    version: int = 1
    entries: list[KnowledgeEntry] = Field(default_factory=list)


class KnowledgeBase:
    def __init__(self, path: Path, embedder: Embedder | None = None):
        self.path = Path(path)
        self.embedder = embedder or HashingEmbedder()

    def entries(self) -> list[KnowledgeEntry]:
        """All stored entries, oldest first. A damaged file is reported and treated as empty."""
        if not self.path.exists():
            return []
        try:
            return _KnowledgeFile.model_validate_json(self.path.read_text("utf-8")).entries
        except (OSError, ValueError, ValidationError) as exc:
            logger.warning("Ignoring unreadable knowledge base %s (%s)", self.path, exc)
            return []

    def add(self, entry: KnowledgeEntry) -> None:
        current = self.entries()
        if not current and self.path.exists() and self.path.stat().st_size:
            backup = self.path.with_suffix(self.path.suffix + ".corrupt")
            self.path.replace(backup)
            logger.warning("Moved unreadable knowledge base aside to %s", backup)
        data = _KnowledgeFile(entries=[*current, entry]).model_dump(mode="json")
        self._write_atomically(json.dumps(data, indent=2, ensure_ascii=False))

    def search(self, story: str, *, limit: int = 2, min_score: float = 0.15) -> list[Match]:
        """Past entries most similar to ``story``, best first."""
        entries = self.entries()
        if not entries or limit <= 0:
            return []
        query = self.embedder.embed(story)
        scored = [
            Match(
                entry=entry, score=round(cosine(query, self.embedder.embed(entry.search_text())), 4)
            )
            for entry in entries
        ]
        ranked = sorted(scored, key=lambda match: match.score, reverse=True)
        return [match for match in ranked if match.score >= min_score][:limit]

    def _write_atomically(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".kb-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
