"""Tests for DreamMode."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from enton.cognition.dream import DreamMode


def _make_memory():
    mem = MagicMock()
    mem.profile.name = "Gabriel"
    mem.recall_recent.return_value = []
    mem.recall_by_kind.return_value = []
    return mem


def _make_brain():
    brain = MagicMock()
    brain.think = AsyncMock(return_value="Test insight from dream")
    return brain


def test_initial_state():
    dm = DreamMode(memory=_make_memory(), brain=_make_brain())
    assert dm.dreaming is False
    assert dm.dream_count == 0


def test_on_interaction_resets_idle():
    dm = DreamMode(memory=_make_memory(), brain=_make_brain())
    dm._last_interaction = 0  # long time ago
    dm.on_interaction()
    assert dm.idle_seconds < 1.0


def test_should_dream_after_idle():
    dm = DreamMode(memory=_make_memory(), brain=_make_brain())
    dm._last_interaction = time.time() - 200  # idle for 200s
    dm._last_dream = 0  # no recent dream
    assert dm.should_dream is True


def test_should_not_dream_recent_activity():
    dm = DreamMode(memory=_make_memory(), brain=_make_brain())
    dm._last_interaction = time.time()  # just interacted
    assert dm.should_dream is False


def test_should_not_dream_already_dreaming():
    dm = DreamMode(memory=_make_memory(), brain=_make_brain())
    dm._dreaming = True
    dm._last_interaction = time.time() - 200
    assert dm.should_dream is False


def test_should_not_dream_cooldown():
    dm = DreamMode(memory=_make_memory(), brain=_make_brain())
    dm._last_interaction = time.time() - 200
    dm._last_dream = time.time()  # just dreamed
    assert dm.should_dream is False


@pytest.mark.asyncio()
async def test_dream_cycle_consolidates():
    mem = _make_memory()
    brain = _make_brain()

    # Provide enough episodes for consolidation
    from enton.core.memory import Episode

    episodes = [
        Episode(kind="conversation", summary=f"Chat #{i}", tags=["chat"]) for i in range(10)
    ]
    mem.recall_recent.return_value = episodes

    dm = DreamMode(memory=mem, brain=brain)
    await dm._dream_cycle()

    assert dm.dream_count == 1
    assert dm.dreaming is False
    # brain.think was called at least once for consolidation
    assert brain.think.call_count >= 1
    # memory.remember was called with dream episode
    assert mem.remember.called


@pytest.mark.asyncio()
async def test_dream_cycle_extracts_patterns():
    mem = _make_memory()
    brain = _make_brain()

    from enton.core.memory import Episode

    # Create episodes with repeated tags at same hour
    now = time.time()
    episodes = [
        Episode(kind="detection", summary=f"Det {i}", tags=["person"], timestamp=now)
        for i in range(15)
    ]
    mem.recall_recent.return_value = episodes

    dm = DreamMode(memory=mem, brain=brain)
    await dm._dream_cycle()

    # Should find pattern: "person" appears frequently
    assert dm.dream_count == 1


@pytest.mark.asyncio()
async def test_dream_handles_brain_failure():
    mem = _make_memory()
    brain = _make_brain()
    brain.think = AsyncMock(side_effect=Exception("LLM down"))

    from enton.core.memory import Episode

    episodes = [Episode(kind="conversation", summary=f"Chat {i}", tags=["chat"]) for i in range(10)]
    mem.recall_recent.return_value = episodes

    dm = DreamMode(memory=mem, brain=brain)
    # Should not raise
    await dm._dream_cycle()
    assert dm.dream_count == 1
    assert dm.dreaming is False


def test_summary():
    dm = DreamMode(memory=_make_memory(), brain=_make_brain())
    dm._dream_count = 3
    s = dm.summary()
    assert "Dreams: 3" in s


def test_to_dict():
    dm = DreamMode(memory=_make_memory(), brain=_make_brain())
    dm._dream_count = 2
    d = dm.to_dict()
    assert d["dream_count"] == 2
    assert "idle_seconds" in d


def test_interaction_interrupts_dream():
    dm = DreamMode(memory=_make_memory(), brain=_make_brain())
    dm._dreaming = True
    dm.on_interaction()
    assert dm.dreaming is False
