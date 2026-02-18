"""Tests for AwarenessStateMachine."""

from __future__ import annotations

from unittest.mock import MagicMock

from enton.core.awareness import LEVEL_CONFIGS, AwarenessLevel, AwarenessStateMachine


def _make_sm(engagement: float = 0.5, social: float = 0.3):
    sm = MagicMock()
    sm.mood.engagement = engagement
    sm.mood.social = social
    return sm


def test_initial_state():
    asm = AwarenessStateMachine()
    assert asm.state == AwarenessLevel.SENTINEL


def test_config_per_level():
    for _level, cfg in LEVEL_CONFIGS.items():
        assert cfg.vision_fps >= 0
        assert isinstance(cfg.audio, bool)


def test_transition_basic():
    asm = AwarenessStateMachine()
    asm._last_transition = 0  # bypass debounce
    ok = asm.transition(AwarenessLevel.ATTENTIVE, "test")
    assert ok is True
    assert asm.state == AwarenessLevel.ATTENTIVE


def test_transition_same_state_noop():
    asm = AwarenessStateMachine()
    ok = asm.transition(AwarenessLevel.SENTINEL, "same")
    assert ok is False


def test_transition_debounce():
    asm = AwarenessStateMachine()
    # first transition OK
    asm._last_transition = 0
    asm.transition(AwarenessLevel.ATTENTIVE, "first")
    # immediate second blocked
    ok = asm.transition(AwarenessLevel.FOCUSED, "too fast")
    assert ok is False
    assert asm.state == AwarenessLevel.ATTENTIVE


def test_evaluate_sentinel_to_attentive():
    asm = AwarenessStateMachine()
    asm._last_transition = 0
    sm = _make_sm(social=0.5)
    asm.evaluate(sm)
    assert asm.state == AwarenessLevel.ATTENTIVE


def test_evaluate_sentinel_to_creative():
    asm = AwarenessStateMachine()
    asm._last_transition = 0
    asm._state_enter_time = 0  # been in SENTINEL for ages
    sm = _make_sm(social=0.0, engagement=0.0)
    asm.evaluate(sm)
    assert asm.state == AwarenessLevel.CREATIVE


def test_evaluate_attentive_to_focused():
    asm = AwarenessStateMachine()
    asm._state = AwarenessLevel.ATTENTIVE
    asm._last_transition = 0
    sm = _make_sm(engagement=0.8)
    asm.evaluate(sm)
    assert asm.state == AwarenessLevel.FOCUSED


def test_evaluate_focused_to_attentive():
    asm = AwarenessStateMachine()
    asm._state = AwarenessLevel.FOCUSED
    asm._last_transition = 0
    asm._state_enter_time = 0  # been focused for ages
    sm = _make_sm(engagement=0.1)
    asm.evaluate(sm)
    assert asm.state == AwarenessLevel.ATTENTIVE


def test_trigger_alert():
    asm = AwarenessStateMachine()
    asm._last_transition = 0
    asm.trigger_alert("loud sound")
    assert asm.state == AwarenessLevel.ALERT


def test_on_interaction_wakes():
    asm = AwarenessStateMachine()
    asm._last_transition = 0
    asm._state = AwarenessLevel.CREATIVE
    asm.on_interaction()
    assert asm.state == AwarenessLevel.ATTENTIVE


def test_is_dreaming():
    asm = AwarenessStateMachine()
    assert asm.is_dreaming is False
    asm._state = AwarenessLevel.CREATIVE
    assert asm.is_dreaming is True


def test_is_active():
    asm = AwarenessStateMachine()
    assert asm.is_active is False  # SENTINEL
    asm._state = AwarenessLevel.FOCUSED
    assert asm.is_active is True


def test_serialization():
    asm = AwarenessStateMachine()
    asm._state = AwarenessLevel.FOCUSED
    asm._transition_count = 5
    data = asm.to_dict()
    assert data["state"] == "FOCUSED"
    assert data["transitions"] == 5

    asm2 = AwarenessStateMachine()
    asm2.from_dict(data)
    assert asm2.state == AwarenessLevel.FOCUSED


def test_summary():
    asm = AwarenessStateMachine()
    s = asm.summary()
    assert "SENTINEL" in s
    assert "fps" in s
