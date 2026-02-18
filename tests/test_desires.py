"""Tests for DesireEngine and Desire."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from enton.cognition.desires import Desire, DesireEngine


def test_desire_tick():
    d = Desire(name="test", description="Test desire", growth_rate=0.1)
    d.tick(dt=1.0)
    assert d.urgency == 0.1


def test_desire_tick_clamped():
    d = Desire(name="test", description="Test", urgency=0.95, growth_rate=0.1)
    d.tick(dt=1.0)
    assert d.urgency == 1.0


def test_desire_should_activate():
    d = Desire(
        name="test",
        description="Test",
        urgency=0.8,
        threshold=0.7,
        cooldown=0,
        last_activated=0,
    )
    assert d.should_activate() is True


def test_desire_cooldown_blocks_activation():
    d = Desire(
        name="test",
        description="Test",
        urgency=0.8,
        threshold=0.7,
        cooldown=9999,
        last_activated=time.time(),
    )
    assert d.should_activate() is False


def test_desire_disabled():
    d = Desire(name="test", description="Test", urgency=0.9, threshold=0.5, enabled=False)
    assert d.should_activate() is False
    d.tick()
    assert d.urgency == 0.9  # disabled = no growth


def test_desire_activate_resets():
    d = Desire(name="test", description="Test", urgency=0.9)
    d.activate()
    assert d.urgency == 0.0
    assert d.last_activated > 0


def test_desire_suppress():
    d = Desire(name="test", description="Test", urgency=0.5)
    d.suppress(0.3)
    assert d.urgency == pytest.approx(0.2)


def test_engine_init():
    engine = DesireEngine()
    assert len(engine._desires) == 9


def test_engine_tick():
    engine = DesireEngine()
    sm = MagicMock()
    sm.mood.social = 0.5
    sm.mood.engagement = 0.5
    engine.tick(sm, dt=1.0)
    # All desires should have increased urgency
    for d in engine._desires.values():
        assert d.urgency > 0


def test_engine_lonely_boost():
    engine = DesireEngine()
    sm = MagicMock()
    sm.mood.social = 0.1  # lonely
    sm.mood.engagement = 0.5

    engine.tick(sm, dt=10.0)
    assert engine._desires["socialize"].urgency > engine._desires["optimize"].urgency


def test_engine_on_interaction():
    engine = DesireEngine()
    engine._desires["socialize"].urgency = 0.8
    engine.on_interaction()
    assert engine._desires["socialize"].urgency < 0.8


def test_engine_get_active_desire():
    engine = DesireEngine()
    # Nothing should be active initially
    assert engine.get_active_desire() is None

    # Force one desire to activate
    engine._desires["socialize"].urgency = 1.0
    engine._desires["socialize"].cooldown = 0
    engine._desires["socialize"].last_activated = 0
    d = engine.get_active_desire()
    assert d is not None
    assert d.name == "socialize"


def test_engine_get_prompt():
    engine = DesireEngine()
    d = engine._desires["socialize"]
    prompt = engine.get_prompt(d)
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_engine_serialization():
    engine = DesireEngine()
    engine._desires["learn"].urgency = 0.42
    data = engine.to_dict()
    assert data["learn"]["urgency"] == pytest.approx(0.42)

    engine2 = DesireEngine()
    engine2.from_dict(data)
    assert engine2._desires["learn"].urgency == pytest.approx(0.42)


def test_engine_summary():
    engine = DesireEngine()
    s = engine.summary()
    assert "Desires:" in s


import pytest
