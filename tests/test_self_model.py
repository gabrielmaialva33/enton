"""Tests for SelfModel, Mood, and SensoryState."""

from __future__ import annotations

from enton.core.self_model import Mood, SelfModel, SensoryState


def test_mood_initial_state():
    m = Mood()
    assert m.engagement == 0.5
    assert m.social == 0.3
    assert m.label == "tranquilo"


def test_mood_on_interaction():
    m = Mood()
    m.on_interaction()
    assert m.engagement == 0.65
    assert m.social == 0.5


def test_mood_on_detection_cat():
    m = Mood()
    initial = m.engagement
    m.on_detection("cat")
    assert m.engagement == initial + 0.3


def test_mood_on_detection_person():
    m = Mood()
    initial = m.social
    m.on_detection("person")
    assert m.social == initial + 0.15


def test_mood_clamping():
    m = Mood()
    for _ in range(20):
        m.on_interaction()
    assert m.engagement <= 1.0
    assert m.social <= 1.0


def test_mood_on_idle():
    m = Mood()
    m.engagement = 0.5
    m.on_idle()
    assert m.engagement == 0.45


def test_mood_on_error():
    m = Mood()
    m.engagement = 0.5
    m.on_error()
    assert m.engagement == 0.4


def test_mood_labels():
    m = Mood()
    m.engagement = 0.9
    m.social = 0.9
    assert m.label == "empolgado"

    m.engagement = 0.5
    m.social = 0.5
    assert m.label == "tranquilo"

    m.engagement = 0.2
    m.social = 0.2
    assert m.label == "entediado"

    m.engagement = 0.0
    m.social = 0.0
    assert m.label == "largado"


def test_sensory_state_summary():
    s = SensoryState()
    assert "camera OFF" in s.summary()
    assert "mic OFF" in s.summary()

    s.camera_online = True
    s.llm_ready = True
    s.active_providers["llm"] = "qwen2.5:14b"
    summary = s.summary()
    assert "camera ON" in summary
    assert "brain via qwen2.5:14b" in summary


def test_self_model_record_interaction(mock_settings):
    sm = SelfModel(mock_settings)
    initial_eng = sm.mood.engagement
    sm.record_interaction()
    assert sm._interactions_count == 1
    assert sm.mood.engagement > initial_eng


def test_self_model_record_detection(mock_settings):
    sm = SelfModel(mock_settings)
    sm.record_detection("person")
    assert sm._detections_count == 1


def test_self_model_record_activity(mock_settings):
    sm = SelfModel(mock_settings)
    sm.record_activity("Acenando")
    assert sm.last_activity == "Acenando"


def test_self_model_record_emotion(mock_settings):
    sm = SelfModel(mock_settings)
    sm.record_emotion("feliz")
    assert sm.last_emotion == "feliz"
    # Happy emotion boosts engagement
    assert sm.mood.engagement > 0.5


def test_self_model_uptime(mock_settings):
    sm = SelfModel(mock_settings)
    assert sm.uptime_seconds >= 0
    assert isinstance(sm.uptime_human, str)


def test_self_model_introspect(mock_settings):
    sm = SelfModel(mock_settings)
    text = sm.introspect()
    assert "Enton" in text
    assert "Mood" in text
