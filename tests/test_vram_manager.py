"""Tests for VRAMManager."""
from __future__ import annotations

import pytest

from enton.core.vram_manager import ModelPriority, ModelSlot, VRAMManager


class FakeModel:
    """Fake model that tracks device location."""
    def __init__(self, name: str = "fake"):
        self.name = name
        self._on_cuda = False

    def cuda(self):
        self._on_cuda = True
        return self

    def cpu(self):
        self._on_cuda = False
        return self


def _make_slot(
    name: str, vram: int = 100, priority: ModelPriority = ModelPriority.NORMAL,
) -> ModelSlot:
    return ModelSlot(
        name=name,
        loader=lambda n=name: FakeModel(n),
        vram_mb=vram,
        priority=priority,
    )


def test_slot_load():
    slot = _make_slot("test")
    assert slot.model is None
    model = slot.load()
    assert model is not None
    assert model.name == "test"


def test_slot_to_cuda():
    slot = _make_slot("test")
    slot.load()
    slot.to_cuda()
    assert slot.on_device is True
    assert slot.last_used > 0


def test_slot_to_cpu():
    slot = _make_slot("test")
    slot.load()
    slot.to_cuda()
    slot.to_cpu()
    assert slot.on_device is False


def test_slot_unload():
    slot = _make_slot("test")
    slot.load()
    slot.unload()
    assert slot.model is None
    assert slot.on_device is False


def test_manager_register():
    mgr = VRAMManager(budget_mb=1000)
    mgr.register(_make_slot("a", 100))
    assert "a" in mgr._slots
    assert mgr.used_mb == 0


@pytest.mark.asyncio()
async def test_acquire_basic():
    mgr = VRAMManager(budget_mb=1000)
    mgr.register(_make_slot("a", 100))
    model = await mgr.acquire("a")
    assert model is not None
    assert mgr.used_mb == 100


@pytest.mark.asyncio()
async def test_acquire_cached():
    mgr = VRAMManager(budget_mb=1000)
    mgr.register(_make_slot("a", 100))
    m1 = await mgr.acquire("a")
    m2 = await mgr.acquire("a")
    assert m1 is m2
    assert mgr._slots["a"].use_count == 2


@pytest.mark.asyncio()
async def test_eviction_lru():
    mgr = VRAMManager(budget_mb=250)
    mgr.register(_make_slot("a", 100))
    mgr.register(_make_slot("b", 100))
    mgr.register(_make_slot("c", 100))

    await mgr.acquire("a")
    await mgr.acquire("b")
    assert mgr.used_mb == 200

    # acquiring c should evict a (oldest)
    await mgr.acquire("c")
    assert mgr.used_mb == 200  # b + c
    assert mgr._slots["a"].on_device is False
    assert mgr._slots["c"].on_device is True


@pytest.mark.asyncio()
async def test_eviction_respects_priority():
    mgr = VRAMManager(budget_mb=250)
    mgr.register(_make_slot("critical", 100, ModelPriority.CRITICAL))
    mgr.register(_make_slot("normal", 100, ModelPriority.NORMAL))
    mgr.register(_make_slot("new", 100, ModelPriority.NORMAL))

    await mgr.acquire("critical")
    await mgr.acquire("normal")

    # should evict normal, not critical
    await mgr.acquire("new")
    assert mgr._slots["critical"].on_device is True
    assert mgr._slots["normal"].on_device is False
    assert mgr._slots["new"].on_device is True


@pytest.mark.asyncio()
async def test_acquire_unknown_raises():
    mgr = VRAMManager(budget_mb=1000)
    with pytest.raises(KeyError):
        await mgr.acquire("nonexistent")


@pytest.mark.asyncio()
async def test_acquire_no_room_raises():
    mgr = VRAMManager(budget_mb=50)
    mgr.register(_make_slot("big", 100, ModelPriority.CRITICAL))
    with pytest.raises(RuntimeError, match="Cannot fit"):
        await mgr.acquire("big")


def test_set_priority():
    mgr = VRAMManager()
    mgr.register(_make_slot("a"))
    assert mgr._slots["a"].priority == ModelPriority.NORMAL
    mgr.set_priority("a", ModelPriority.CRITICAL)
    assert mgr._slots["a"].priority == ModelPriority.CRITICAL


def test_evict_all():
    mgr = VRAMManager(budget_mb=1000)
    s1 = _make_slot("a", 100, ModelPriority.CRITICAL)
    s2 = _make_slot("b", 100, ModelPriority.NORMAL)
    s1.load()
    s1.to_cuda()
    s2.load()
    s2.to_cuda()
    mgr.register(s1)
    mgr.register(s2)

    mgr.evict_all(keep_critical=True)
    assert mgr._slots["a"].on_device is True
    assert mgr._slots["b"].on_device is False


def test_status():
    mgr = VRAMManager(budget_mb=1000)
    mgr.register(_make_slot("a", 100))
    s = mgr.status()
    assert "VRAM:" in s
    assert "a:" in s


def test_to_dict():
    mgr = VRAMManager(budget_mb=1000)
    mgr.register(_make_slot("a", 100))
    d = mgr.to_dict()
    assert d["budget_mb"] == 1000
    assert "a" in d["models"]
