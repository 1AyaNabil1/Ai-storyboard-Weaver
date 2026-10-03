from datetime import UTC, datetime

import pytest

from storyboard_weaver.knowledge import (
    HashingEmbedder,
    KnowledgeBase,
    KnowledgeEntry,
    _stem,
    cosine,
)
from storyboard_weaver.models import Mood


def entry(entry_id: str, story: str, genre: str = "drama") -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        story=story,
        style="cinematic",
        title=entry_id.title(),
        logline=story[:40],
        genre=genre,
        moods=[Mood.TENSE, Mood.HOPEFUL],
    )


def test_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder(dimensions=64)
    first = embedder.embed("The storm hits the lighthouse at midnight")
    assert first == HashingEmbedder(dimensions=64).embed(
        "the STORM hits the lighthouse, at midnight!"
    )
    assert sum(v * v for v in first) == pytest.approx(1.0)
    assert embedder.embed("the and of") == [0.0] * 64  # only stopwords
    assert cosine(first, [0.0] * 64) == 0.0


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("storms", "storm"),
        ("stormy", "storm"),
        ("sailing", "sail"),
        ("city", "city"),
        ("sea", "sea"),
    ],
)
def test_stemming_is_light(word, stem):
    assert _stem(word) == stem


def test_empty_or_missing_file_means_no_entries(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.json")
    assert kb.entries() == []
    assert kb.search("anything") == []


def test_add_persists_and_search_ranks_by_similarity(tmp_path):
    path = tmp_path / "nested" / "kb.json"
    kb = KnowledgeBase(path)
    kb.add(
        entry("heist", "A crew of thieves plans a bank vault heist during a city blackout", "heist")
    )
    kb.add(entry("lighthouse", "A lighthouse keeper on a stormy island receives a strange letter"))
    kb.add(
        entry(
            "bakery", "Two rival bakers fall in love while competing at a village fair", "romance"
        )
    )

    reopened = KnowledgeBase(path)
    assert [e.id for e in reopened.entries()] == ["heist", "lighthouse", "bakery"]

    matches = reopened.search(
        "An old keeper guards a lighthouse while a storm closes on the island"
    )
    assert matches[0].entry.id == "lighthouse"
    assert all(m.entry.id != "bakery" for m in matches)
    assert matches == sorted(matches, key=lambda m: m.score, reverse=True)
    assert len(reopened.search("lighthouse storm island keeper", limit=1)) == 1


def test_corrupt_file_is_ignored_then_moved_aside(tmp_path, caplog):
    path = tmp_path / "kb.json"
    path.write_text("{not json", encoding="utf-8")
    kb = KnowledgeBase(path)

    assert kb.entries() == []
    assert "unreadable knowledge base" in caplog.text

    kb.add(entry("fresh", "A brand new story"))
    assert [e.id for e in kb.entries()] == ["fresh"]
    assert (tmp_path / "kb.json.corrupt").read_text(encoding="utf-8") == "{not json"
