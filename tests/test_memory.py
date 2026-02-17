"""Tests for Memory (episodic + user profile)."""

from __future__ import annotations

from enton.core.memory import Episode, Memory, UserProfile


def test_episode_creation():
    ep = Episode(kind="conversation", summary="User said hello")
    assert ep.kind == "conversation"
    assert ep.summary == "User said hello"
    assert ep.timestamp > 0
    assert ep.tags == []


def test_memory_remember_and_recall():
    mem = Memory()
    ep = Episode(kind="test", summary="Test episode", tags=["unit"])
    mem.remember(ep)

    recent = mem.recall_recent(5)
    assert len(recent) >= 1
    assert recent[-1].summary == "Test episode"


def test_memory_recall_by_kind():
    mem = Memory()
    mem.remember(Episode(kind="chat", summary="Chat 1"))
    mem.remember(Episode(kind="detection", summary="Det 1"))
    mem.remember(Episode(kind="chat", summary="Chat 2"))

    chats = mem.recall_by_kind("chat")
    assert all(e.kind == "chat" for e in chats)
    assert len(chats) == 2


def test_memory_recall_by_tag():
    mem = Memory()
    mem.remember(Episode(kind="test", summary="Tagged", tags=["important"]))
    mem.remember(Episode(kind="test", summary="Untagged"))

    tagged = mem.recall_by_tag("important")
    assert len(tagged) == 1
    assert tagged[0].summary == "Tagged"


def test_memory_recent_alias():
    mem = Memory()
    mem.remember(Episode(kind="test", summary="Alias test"))
    assert mem.recent(1) == mem.recall_recent(1)


def test_memory_max_recent_trimming():
    mem = Memory(max_recent=5)
    for i in range(15):
        mem.remember(Episode(kind="test", summary=f"Episode {i}"))
    # Should trim to max_recent after 2x threshold
    assert len(mem._episodes) <= 10


def test_memory_semantic_search_keyword_fallback():
    mem = Memory()
    mem.remember(Episode(kind="chat", summary="Gabriel likes Python programming"))
    mem.remember(Episode(kind="chat", summary="The weather is nice today"))

    results = mem.semantic_search("Python")
    assert len(results) >= 1
    assert "Python" in results[0]


def test_user_profile_defaults():
    p = UserProfile()
    assert p.name == "Gabriel"
    assert p.relationship_score == 0.5
    assert p.known_facts == []


def test_memory_learn_about_user():
    mem = Memory()
    mem.learn_about_user("Likes cats")
    assert "Likes cats" in mem.profile.known_facts
    # Duplicate should not be added
    mem.learn_about_user("Likes cats")
    assert mem.profile.known_facts.count("Likes cats") == 1


def test_memory_strengthen_relationship():
    mem = Memory()
    initial = mem.profile.relationship_score
    mem.strengthen_relationship(0.1)
    assert mem.profile.relationship_score == initial + 0.1


def test_memory_relationship_clamped():
    mem = Memory()
    mem.profile.relationship_score = 0.95
    mem.strengthen_relationship(0.1)
    assert mem.profile.relationship_score == 1.0


def test_memory_context_string():
    mem = Memory()
    mem.remember(Episode(kind="chat", summary="Hello there"))
    mem.learn_about_user("Likes gaming")
    ctx = mem.context_string()
    assert "Hello there" in ctx
    assert "gaming" in ctx


def test_memory_context_string_empty():
    mem = Memory()
    ctx = mem.context_string()
    # Default relationship_score=0.5 produces "good friend" even with no episodes
    assert "Gabriel" in ctx


def test_memory_persistence(tmp_path, monkeypatch):
    """Episodes persist to JSONL and reload."""
    monkeypatch.setattr("enton.core.memory.MEMORY_DIR", tmp_path)
    monkeypatch.setattr("enton.core.memory.EPISODES_FILE", tmp_path / "episodes.jsonl")
    monkeypatch.setattr("enton.core.memory.PROFILE_FILE", tmp_path / "profile.json")

    mem1 = Memory()
    mem1.remember(Episode(kind="test", summary="Persistent episode"))
    mem1.learn_about_user("Fact A")

    # Create new Memory — should load from files
    mem2 = Memory()
    assert len(mem2.recall_recent(10)) >= 1
    assert mem2.recall_recent(10)[-1].summary == "Persistent episode"
    assert "Fact A" in mem2.profile.known_facts
