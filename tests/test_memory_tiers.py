"""Tests for MemoryTiers."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from enton.core.memory_tiers import (
    MemoryTiers,
    ObjectLocation,
    TemporalPattern,
    TierResult,
)


def _make_memory():
    mem = MagicMock()
    mem.semantic_search.return_value = []
    mem.profile.name = "Gabriel"
    return mem


def test_object_location_creation():
    loc = ObjectLocation(
        label="cup",
        camera_id="main",
        bbox=(10, 20, 50, 60),
        confidence=0.9,
    )
    assert loc.label == "cup"
    assert loc.confidence == 0.9


def test_temporal_pattern_creation():
    p = TemporalPattern(
        description="person appears often",
        hour=14,
        tag="person",
        count=20,
    )
    assert p.hour == 14
    assert p.count == 20


def test_tier_result_creation():
    r = TierResult(tier="spatial", content="cup seen on camera main")
    assert r.score == 1.0
    assert r.tier == "spatial"


def test_update_object_location():
    mt = MemoryTiers(memory=_make_memory())
    mt.update_object_location("cup", "main", (10, 20, 50, 60), 0.9)
    assert "cup" in mt._spatial
    assert mt._spatial["cup"].camera_id == "main"


def test_where_is():
    mt = MemoryTiers(memory=_make_memory())
    mt.update_object_location("cup", "hack", (10, 20, 50, 60), 0.85)
    loc = mt.where_is("cup")
    assert loc is not None
    assert loc.camera_id == "hack"


def test_where_is_unknown():
    mt = MemoryTiers(memory=_make_memory())
    assert mt.where_is("unicorn") is None


def test_all_objects():
    mt = MemoryTiers(memory=_make_memory())
    mt.update_object_location("cup", "main", (0, 0, 0, 0))
    mt.update_object_location("laptop", "hack", (0, 0, 0, 0))
    objs = mt.all_objects()
    assert len(objs) == 2


def test_update_evicts_oldest():
    mt = MemoryTiers(memory=_make_memory())
    mt.MAX_OBJECTS = 3
    mt.update_object_location("a", "m", (0, 0, 0, 0))
    mt.update_object_location("b", "m", (0, 0, 0, 0))
    mt.update_object_location("c", "m", (0, 0, 0, 0))
    mt.update_object_location("d", "m", (0, 0, 0, 0))  # should evict 'a'
    assert len(mt._spatial) == 3
    assert "a" not in mt._spatial


def test_add_pattern():
    mt = MemoryTiers(memory=_make_memory())
    p = TemporalPattern(description="test", hour=10, tag="person", count=5)
    mt.add_pattern(p)
    assert len(mt._patterns) == 1


def test_patterns_for_hour():
    mt = MemoryTiers(memory=_make_memory())
    mt.add_pattern(TemporalPattern("morning", 8, "person", 10))
    mt.add_pattern(TemporalPattern("noon", 12, "car", 5))
    mt.add_pattern(TemporalPattern("morning2", 8, "dog", 3))
    assert len(mt.patterns_for_hour(8)) == 2
    assert len(mt.patterns_for_hour(12)) == 1
    assert len(mt.patterns_for_hour(20)) == 0


@pytest.mark.asyncio()
async def test_search_spatial_match():
    mt = MemoryTiers(memory=_make_memory())
    mt.update_object_location("cup", "main", (10, 20, 50, 60), 0.9)
    results = await mt.search("cup")
    spatial = [r for r in results if r.tier == "spatial"]
    assert len(spatial) >= 1
    assert "cup" in spatial[0].content


@pytest.mark.asyncio()
async def test_search_temporal_match():
    mt = MemoryTiers(memory=_make_memory())
    mt.add_pattern(TemporalPattern("person appears often", 14, "person", 20))
    results = await mt.search("person")
    temporal = [r for r in results if r.tier == "temporal"]
    assert len(temporal) >= 1


@pytest.mark.asyncio()
async def test_search_episodic():
    mem = _make_memory()
    mem.semantic_search.return_value = ["Conversa sobre Python"]
    mt = MemoryTiers(memory=mem)
    results = await mt.search("Python")
    episodic = [r for r in results if r.tier == "episodic"]
    assert len(episodic) >= 1
    assert "Python" in episodic[0].content


@pytest.mark.asyncio()
async def test_search_with_visual():
    vm = MagicMock()
    vm.search = AsyncMock(
        return_value=[
            {
                "timestamp": time.time(),
                "camera_id": "main",
                "detections": ["cup"],
                "score": 0.8,
            }
        ]
    )
    mt = MemoryTiers(memory=_make_memory(), visual_memory=vm)
    results = await mt.search("cup")
    visual = [r for r in results if "Visual" in r.content]
    assert len(visual) >= 1


@pytest.mark.asyncio()
async def test_search_with_knowledge():
    kc = MagicMock()
    kc.search = AsyncMock(
        return_value=[
            {
                "subject": "Python",
                "predicate": "is",
                "obj": "fast",
                "score": 0.7,
            }
        ]
    )
    mt = MemoryTiers(memory=_make_memory(), knowledge=kc)
    results = await mt.search("Python speed")
    semantic = [r for r in results if r.tier == "semantic"]
    assert len(semantic) >= 1


@pytest.mark.asyncio()
async def test_search_no_results():
    mt = MemoryTiers(memory=_make_memory())
    results = await mt.search("xyznonexistent123")
    assert isinstance(results, list)


def test_context_string_empty():
    mt = MemoryTiers(memory=_make_memory())
    assert mt.context_string() == ""


def test_context_string_with_objects():
    mt = MemoryTiers(memory=_make_memory())
    mt.update_object_location("cup", "main", (0, 0, 0, 0))
    ctx = mt.context_string()
    assert "cup" in ctx
    assert "Objects:" in ctx


def test_to_dict():
    mt = MemoryTiers(memory=_make_memory())
    d = mt.to_dict()
    assert d["spatial_objects"] == 0
    assert d["visual_memory"] is False
    assert d["knowledge"] is False
