"""Tests for ContextEngine — smart context management."""
from __future__ import annotations

import json
import time
from datetime import timedelta

import pytest

from enton.core.context_engine import (
    Checkpoint,
    ContextEngine,
    ContextEntry,
    _CHARS_PER_TOKEN,
)


# ------------------------------------------------------------------ #
# ContextEntry unit tests
# ------------------------------------------------------------------ #


def test_context_entry_token_estimation():
    """Token estimate is derived from content length when not explicit."""
    content = "a" * 35  # 35 chars / 3.5 = 10 tokens
    entry = ContextEntry(key="k", content=content, category="system")
    assert entry.token_estimate == 10


def test_context_entry_explicit_token_estimate():
    """Explicit token_estimate overrides auto-calculation."""
    entry = ContextEntry(
        key="k", content="short", category="system", token_estimate=999
    )
    assert entry.token_estimate == 999


def test_context_entry_zero_content_tokens():
    """Empty content yields zero tokens."""
    entry = ContextEntry(key="k", content="", category="system")
    assert entry.token_estimate == 0


def test_context_entry_not_stale_no_ttl():
    """Entries with ttl=0 never go stale."""
    entry = ContextEntry(key="k", content="x", category="system", ttl=0.0)
    assert entry.is_stale is False


def test_context_entry_not_stale_within_ttl(freezer):
    """Entries within their TTL are fresh."""
    now = time.time()
    entry = ContextEntry(
        key="k", content="x", category="system", ttl=60.0,
        timestamp=now,
    )
    assert entry.is_stale is False


def test_context_entry_stale_after_ttl(freezer):
    """Entries past their TTL are stale."""
    now = time.time()
    entry = ContextEntry(
        key="k", content="x", category="system", ttl=1.0,
        timestamp=now - 5.0,
    )
    assert entry.is_stale is True


def test_context_entry_age_seconds(freezer):
    """age_seconds reflects wall-clock time since creation."""
    now = time.time()
    past = now - 120.0
    entry = ContextEntry(key="k", content="x", category="system", timestamp=past)
    assert entry.age_seconds >= 119.0


def test_context_entry_relevance_score_fresh_high_priority(freezer):
    """Fresh, high-priority entries have high relevance."""
    now = time.time()
    entry = ContextEntry(
        key="k", content="x", category="system",
        priority=1.0, timestamp=now,
    )
    score = entry.relevance_score()
    # priority * 0.7 + recency * 0.3
    # recency ~ 1.0 for fresh entry => score ~ 0.7 + 0.3 = 1.0
    assert score > 0.9


def test_context_entry_relevance_score_old_low_priority(freezer):
    """Old, low-priority entries have low relevance."""
    now = time.time()
    entry = ContextEntry(
        key="k", content="x", category="system",
        priority=0.0, timestamp=now - 3600,
    )
    score = entry.relevance_score()
    assert score < 0.1


def test_context_entry_relevance_decay_over_time():
    """Relevance decays as an entry ages."""
    now = time.time()
    fresh = ContextEntry(
        key="a", content="x", category="system",
        priority=0.5, timestamp=now,
    )
    old = ContextEntry(
        key="b", content="x", category="system",
        priority=0.5, timestamp=now - 600,
    )
    assert fresh.relevance_score() > old.relevance_score()


# ------------------------------------------------------------------ #
# ContextEngine — basic CRUD
# ------------------------------------------------------------------ #


def test_engine_set_and_get():
    """set() stores content, get() retrieves it."""
    engine = ContextEngine(max_tokens=1000)
    engine.set("greeting", "hello world", category="system")
    assert engine.get("greeting") == "hello world"


def test_engine_get_missing_key():
    """get() returns None for missing keys."""
    engine = ContextEngine()
    assert engine.get("nonexistent") is None


def test_engine_set_overwrites():
    """set() with the same key overwrites the previous entry."""
    engine = ContextEngine()
    engine.set("k", "first")
    engine.set("k", "second")
    assert engine.get("k") == "second"


def test_engine_remove():
    """remove() deletes entry and returns True."""
    engine = ContextEngine()
    engine.set("k", "val")
    assert engine.remove("k") is True
    assert engine.get("k") is None


def test_engine_remove_missing():
    """remove() returns False for missing keys."""
    engine = ContextEngine()
    assert engine.remove("ghost") is False


def test_engine_get_stale_returns_none():
    """get() returns None for stale entries."""
    engine = ContextEngine()
    engine.set("k", "val", ttl=0.01)
    time.sleep(0.05)
    assert engine.get("k") is None


# ------------------------------------------------------------------ #
# Budget tracking
# ------------------------------------------------------------------ #


def test_current_tokens_empty():
    """Empty engine has zero tokens."""
    engine = ContextEngine()
    assert engine.current_tokens == 0


def test_current_tokens_sum():
    """current_tokens sums all entry estimates."""
    engine = ContextEngine()
    engine.set("a", "a" * 35)  # 10 tokens
    engine.set("b", "b" * 70)  # 20 tokens
    assert engine.current_tokens == 30


def test_budget_used_pct():
    """budget_used_pct calculates correctly."""
    engine = ContextEngine(max_tokens=100)
    engine.set("a", "a" * 35)  # 10 tokens => 10%
    assert abs(engine.budget_used_pct - 10.0) < 0.5


def test_budget_used_pct_zero_max():
    """budget_used_pct returns 0 when max_tokens is zero."""
    engine = ContextEngine(max_tokens=0)
    engine.set("a", "some content")
    assert engine.budget_used_pct == 0.0


def test_budget_used_pct_capped_at_100():
    """budget_used_pct never exceeds 100."""
    engine = ContextEngine(max_tokens=1)
    engine.set("a", "a very long content string that exceeds budget")
    assert engine.budget_used_pct == 100.0


def test_is_over_budget():
    """is_over_budget detects when tokens exceed max."""
    engine = ContextEngine(max_tokens=5)
    engine.set("big", "a" * 100)  # ~28 tokens >> 5
    assert engine.is_over_budget is True


def test_is_not_over_budget():
    """is_over_budget is False when within limits."""
    engine = ContextEngine(max_tokens=10000)
    engine.set("small", "hi")
    assert engine.is_over_budget is False


# ------------------------------------------------------------------ #
# assemble()
# ------------------------------------------------------------------ #


def test_assemble_empty():
    """Assembling empty engine returns empty string."""
    engine = ContextEngine()
    assert engine.assemble() == ""


def test_assemble_single_entry():
    """Assembled output includes category:key prefix."""
    engine = ContextEngine(max_tokens=10000)
    engine.set("hw", "hello world", category="system")
    result = engine.assemble()
    assert "[system:hw]" in result
    assert "hello world" in result


def test_assemble_respects_budget():
    """assemble() drops low-relevance entries when over budget."""
    engine = ContextEngine(max_tokens=15)
    # High priority => kept
    engine.set("important", "a" * 35, priority=1.0)  # 10 tokens
    # Low priority => dropped
    engine.set("filler", "b" * 35, priority=0.0)  # 10 tokens
    result = engine.assemble()
    assert "important" in result
    assert "filler" not in result


def test_assemble_extra_budget():
    """extra_budget extends the effective token limit."""
    engine = ContextEngine(max_tokens=10)
    engine.set("a", "a" * 35, priority=1.0)  # 10 tokens
    engine.set("b", "b" * 35, priority=0.9)  # 10 tokens
    result = engine.assemble(extra_budget=15)
    assert "a" in result
    assert "b" in result


def test_assemble_cleans_stale():
    """assemble() removes stale entries before building output."""
    engine = ContextEngine(max_tokens=10000)
    engine.set("fresh", "alive", ttl=60.0)
    engine.set("old", "dead", ttl=0.01)
    time.sleep(0.05)
    result = engine.assemble()
    assert "alive" in result
    assert "dead" not in result
    # Entry should be physically removed
    assert "old" not in engine._entries


def test_assemble_relevance_ordering():
    """Higher-relevance entries appear first in assembled output."""
    engine = ContextEngine(max_tokens=100000)
    engine.set("low", "LOW", priority=0.0)
    engine.set("high", "HIGH", priority=1.0)
    result = engine.assemble()
    high_pos = result.index("HIGH")
    low_pos = result.index("LOW")
    assert high_pos < low_pos


# ------------------------------------------------------------------ #
# assemble_by_category()
# ------------------------------------------------------------------ #


def test_assemble_by_category_groups():
    """Entries are grouped by their category."""
    engine = ContextEngine()
    engine.set("s1", "sensor data 1", category="sensor")
    engine.set("s2", "sensor data 2", category="sensor")
    engine.set("m1", "memory data", category="memory")
    grouped = engine.assemble_by_category()
    assert "sensor" in grouped
    assert "memory" in grouped
    assert "sensor data 1" in grouped["sensor"]
    assert "sensor data 2" in grouped["sensor"]
    assert "memory data" in grouped["memory"]


def test_assemble_by_category_filter():
    """Category filter limits which categories appear."""
    engine = ContextEngine()
    engine.set("s", "sensor", category="sensor")
    engine.set("m", "memory", category="memory")
    grouped = engine.assemble_by_category(categories=["sensor"])
    assert "sensor" in grouped
    assert "memory" not in grouped


def test_assemble_by_category_empty():
    """Empty engine returns empty dict."""
    engine = ContextEngine()
    assert engine.assemble_by_category() == {}


def test_assemble_by_category_cleans_stale():
    """Stale entries are cleaned before grouping."""
    engine = ContextEngine()
    engine.set("stale", "old data", category="sensor", ttl=0.01)
    engine.set("fresh", "new data", category="sensor", ttl=60.0)
    time.sleep(0.05)
    grouped = engine.assemble_by_category()
    assert "old data" not in grouped.get("sensor", "")
    assert "new data" in grouped["sensor"]


# ------------------------------------------------------------------ #
# Checkpointing — in-memory
# ------------------------------------------------------------------ #


def test_checkpoint_and_restore_in_memory():
    """Checkpoint saves state; restore brings it back."""
    engine = ContextEngine()
    engine.set("a", "alpha", category="system", priority=0.8)
    engine.set("b", "beta", category="memory", priority=0.3)
    cp_id = engine.checkpoint("test-cp")

    # Wipe and restore
    engine.set("c", "gamma")
    engine.remove("a")
    engine.remove("b")
    assert engine.get("a") is None

    ok = engine.restore(cp_id)
    assert ok is True
    assert engine.get("a") == "alpha"
    assert engine.get("b") == "beta"
    # Entry added after checkpoint should be gone
    assert engine.get("c") is None


def test_checkpoint_id_is_string():
    """Checkpoint returns a hex string ID."""
    engine = ContextEngine()
    cp_id = engine.checkpoint("cp1")
    assert isinstance(cp_id, str)
    assert len(cp_id) == 12


def test_restore_nonexistent_checkpoint():
    """Restoring a missing checkpoint returns False."""
    engine = ContextEngine()
    assert engine.restore("nonexistent") is False


def test_multiple_checkpoints():
    """Multiple checkpoints coexist independently."""
    engine = ContextEngine()
    engine.set("x", "version1")
    cp1 = engine.checkpoint("v1")
    engine.set("x", "version2")
    cp2 = engine.checkpoint("v2")

    engine.restore(cp1)
    assert engine.get("x") == "version1"

    engine.restore(cp2)
    assert engine.get("x") == "version2"


def test_list_checkpoints():
    """list_checkpoints returns info dicts sorted by time (newest first)."""
    engine = ContextEngine()
    engine.set("k", "v")
    cp1 = engine.checkpoint("first")
    cp2 = engine.checkpoint("second")
    listing = engine.list_checkpoints()
    assert len(listing) == 2
    # Newest first
    assert listing[0]["id"] == cp2
    assert listing[1]["id"] == cp1
    assert listing[0]["name"] == "second"
    assert "entries" in listing[0]


def test_list_checkpoints_empty():
    """Empty engine has no checkpoints."""
    engine = ContextEngine()
    assert engine.list_checkpoints() == []


# ------------------------------------------------------------------ #
# Checkpointing — disk persistence
# ------------------------------------------------------------------ #


def test_checkpoint_persists_to_disk(tmp_path):
    """Checkpoints are written as JSON files to disk."""
    cp_dir = tmp_path / "checkpoints"
    engine = ContextEngine(checkpoint_dir=cp_dir)
    engine.set("k", "value", category="tool_result")
    cp_id = engine.checkpoint("disk-test")

    path = cp_dir / f"{cp_id}.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["name"] == "disk-test"
    assert len(data["entries"]) == 1
    assert data["entries"][0]["key"] == "k"


def test_restore_from_disk_only(tmp_path):
    """Restore works from disk even if in-memory checkpoint is missing."""
    cp_dir = tmp_path / "checkpoints"
    engine1 = ContextEngine(checkpoint_dir=cp_dir)
    engine1.set("k", "disk-value", category="memory", priority=0.9)
    cp_id = engine1.checkpoint("to-disk")

    # New engine instance (no in-memory checkpoints)
    engine2 = ContextEngine(checkpoint_dir=cp_dir)
    ok = engine2.restore(cp_id)
    assert ok is True
    assert engine2.get("k") == "disk-value"


def test_list_checkpoints_includes_disk(tmp_path):
    """list_checkpoints discovers on-disk checkpoints not yet in memory."""
    cp_dir = tmp_path / "checkpoints"
    engine1 = ContextEngine(checkpoint_dir=cp_dir)
    engine1.set("k", "v")
    cp_id = engine1.checkpoint("on-disk")

    # Fresh engine with same dir
    engine2 = ContextEngine(checkpoint_dir=cp_dir)
    listing = engine2.list_checkpoints()
    ids = [cp["id"] for cp in listing]
    assert cp_id in ids


def test_checkpoint_metadata(tmp_path):
    """Metadata is preserved through checkpoint/restore cycle."""
    cp_dir = tmp_path / "checkpoints"
    engine = ContextEngine(checkpoint_dir=cp_dir)
    engine.set("k", "v")
    meta = {"reason": "before big change", "version": 42}
    cp_id = engine.checkpoint("with-meta", metadata=meta)

    listing = engine.list_checkpoints()
    cp_info = [c for c in listing if c["id"] == cp_id][0]
    assert cp_info["metadata"]["reason"] == "before big change"
    assert cp_info["metadata"]["version"] == 42


def test_checkpoint_dir_created_automatically(tmp_path):
    """Checkpoint dir is created if it does not exist."""
    cp_dir = tmp_path / "deep" / "nested" / "checkpoints"
    assert not cp_dir.exists()
    ContextEngine(checkpoint_dir=cp_dir)
    assert cp_dir.exists()


# ------------------------------------------------------------------ #
# _cleanup_stale()
# ------------------------------------------------------------------ #


def test_cleanup_stale_removes_expired():
    """_cleanup_stale removes entries past their TTL."""
    engine = ContextEngine()
    engine.set("ephemeral", "gone", ttl=0.01)
    engine.set("permanent", "stays")
    time.sleep(0.05)
    removed = engine._cleanup_stale()
    assert removed == 1
    assert engine.get("permanent") == "stays"
    assert "ephemeral" not in engine._entries


def test_cleanup_stale_nothing_to_clean():
    """_cleanup_stale returns 0 when nothing is stale."""
    engine = ContextEngine()
    engine.set("a", "alive", ttl=60.0)
    engine.set("b", "forever")
    removed = engine._cleanup_stale()
    assert removed == 0


def test_cleanup_stale_all_expired():
    """All entries can be stale at once."""
    engine = ContextEngine()
    engine.set("a", "x", ttl=0.01)
    engine.set("b", "y", ttl=0.01)
    time.sleep(0.05)
    removed = engine._cleanup_stale()
    assert removed == 2
    assert len(engine._entries) == 0


# ------------------------------------------------------------------ #
# rot_score() and needs_compression()
# ------------------------------------------------------------------ #


def test_rot_score_empty_engine():
    """Empty engine has zero rot."""
    engine = ContextEngine()
    assert engine.rot_score() == 0.0


def test_rot_score_fresh_under_budget():
    """Fresh entries under budget have low rot."""
    engine = ContextEngine(max_tokens=10000)
    engine.set("fresh", "hi", priority=1.0)
    score = engine.rot_score()
    assert score < 0.5


def test_rot_score_increases_with_budget_pressure():
    """Rot score increases as token budget fills up."""
    engine_low = ContextEngine(max_tokens=10000)
    engine_low.set("small", "x" * 35, priority=0.8)  # 10 tokens / 10000

    engine_high = ContextEngine(max_tokens=15)
    engine_high.set("big", "x" * 35, priority=0.8)  # 10 tokens / 15

    assert engine_high.rot_score() > engine_low.rot_score()


def test_rot_score_increases_with_low_priority():
    """More low-priority (noisy) entries increase rot."""
    engine = ContextEngine(max_tokens=10000)
    for i in range(10):
        engine.set(f"noise{i}", "filler", priority=0.0)
    score = engine.rot_score()
    # noise factor should be high
    assert score > 0.2


def test_rot_score_capped_at_one():
    """rot_score never exceeds 1.0."""
    engine = ContextEngine(max_tokens=1)
    for i in range(50):
        engine.set(f"e{i}", "x" * 100, priority=0.0, ttl=0.01)
    time.sleep(0.05)
    assert engine.rot_score() <= 1.0


def test_needs_compression_false_when_fresh():
    """needs_compression is False for a healthy engine."""
    engine = ContextEngine(max_tokens=100000)
    engine.set("ok", "data", priority=0.8)
    assert engine.needs_compression() is False


def test_needs_compression_true_when_over_budget():
    """needs_compression is True when over budget regardless of rot."""
    engine = ContextEngine(max_tokens=1)
    engine.set("huge", "a" * 1000)
    assert engine.needs_compression() is True


def test_needs_compression_custom_threshold():
    """Custom threshold adjusts sensitivity."""
    engine = ContextEngine(max_tokens=10000)
    engine.set("x", "data", priority=0.5)
    # With a very low threshold, even mild rot triggers compression
    assert engine.needs_compression(threshold=0.01) is True


# ------------------------------------------------------------------ #
# stats() and summary()
# ------------------------------------------------------------------ #


def test_stats_keys():
    """stats() returns all expected keys."""
    engine = ContextEngine()
    s = engine.stats()
    expected_keys = {
        "entries", "tokens_used", "tokens_max", "budget_pct",
        "rot_score", "needs_compression", "categories",
        "checkpoints", "total_compressions",
    }
    assert expected_keys == set(s.keys())


def test_stats_values():
    """stats() values reflect engine state."""
    engine = ContextEngine(max_tokens=5000)
    engine.set("a", "hello", category="sensor")
    engine.set("b", "world", category="memory")
    engine.checkpoint("cp")
    s = engine.stats()
    assert s["entries"] == 2
    assert s["tokens_max"] == 5000
    assert s["checkpoints"] == 1
    assert "sensor" in s["categories"]
    assert "memory" in s["categories"]
    assert s["categories"]["sensor"] == 1
    assert s["categories"]["memory"] == 1


def test_stats_empty_engine():
    """stats() works on empty engine."""
    engine = ContextEngine()
    s = engine.stats()
    assert s["entries"] == 0
    assert s["tokens_used"] == 0
    assert s["rot_score"] == 0.0
    assert s["categories"] == {}


def test_summary_format():
    """summary() returns a human-readable one-liner."""
    engine = ContextEngine(max_tokens=4000)
    engine.set("k", "something")
    s = engine.summary()
    assert "Context:" in s
    assert "4000" in s
    assert "tokens" in s
    assert "rot=" in s
    assert "entries" in s


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #


def test_empty_engine_assemble():
    """No crash on assembling empty engine."""
    engine = ContextEngine()
    assert engine.assemble() == ""
    assert engine.assemble_by_category() == {}


def test_over_budget_all_entries_dropped():
    """When every entry exceeds budget individually, assemble returns empty."""
    engine = ContextEngine(max_tokens=1)
    engine.set("a", "a" * 100, priority=1.0)
    engine.set("b", "b" * 100, priority=0.5)
    result = engine.assemble()
    assert result == ""


def test_all_stale_entries(freezer):
    """Engine with all stale entries behaves like empty after cleanup."""
    engine = ContextEngine(max_tokens=10000)
    # Set them now (time is frozen at start of test)
    engine.set("x", "data1", ttl=0.01)
    engine.set("y", "data2", ttl=0.01)
    
    # Tick 10s (well past the 0.01s TTL)
    freezer.tick(timedelta(seconds=10.0))
    
    engine._cleanup_stale()
    result = engine.assemble()
    assert result == ""
    assert engine.current_tokens == 0


def test_restore_clears_existing_entries():
    """Restoring a checkpoint replaces all current entries."""
    engine = ContextEngine()
    engine.set("before", "old")
    cp_id = engine.checkpoint("snap")

    engine.set("after", "new")
    engine.set("extra", "more")
    engine.restore(cp_id)

    assert engine.get("before") == "old"
    assert engine.get("after") is None
    assert engine.get("extra") is None


def test_checkpoint_empty_engine():
    """Checkpointing an empty engine works and restores to empty."""
    engine = ContextEngine()
    cp_id = engine.checkpoint("empty-snap")
    engine.set("temp", "data")
    engine.restore(cp_id)
    assert engine.get("temp") is None
    assert len(engine._entries) == 0


def test_large_number_of_entries():
    """Engine handles hundreds of entries without error."""
    engine = ContextEngine(max_tokens=1_000_000)
    for i in range(500):
        engine.set(f"entry_{i}", f"content {i}", priority=i / 500.0)
    assert engine.current_tokens > 0
    result = engine.assemble()
    assert len(result) > 0
    s = engine.stats()
    assert s["entries"] == 500


def test_set_various_categories():
    """All standard categories work correctly."""
    engine = ContextEngine()
    cats = ["sensor", "memory", "tool_result", "conversation", "system"]
    for cat in cats:
        engine.set(f"k_{cat}", f"data for {cat}", category=cat)
    grouped = engine.assemble_by_category()
    for cat in cats:
        assert cat in grouped


def test_checkpoint_preserves_priority_and_ttl():
    """Priority and TTL survive the checkpoint/restore cycle."""
    engine = ContextEngine()
    engine.set("k", "val", priority=0.99, ttl=120.0, category="sensor")
    cp_id = engine.checkpoint("prio-check")

    engine._entries.clear()
    engine.restore(cp_id)

    entry = engine._entries["k"]
    assert entry.priority == 0.99
    assert entry.ttl == 120.0
    assert entry.category == "sensor"
