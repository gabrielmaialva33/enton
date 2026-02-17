"""Tests for Fuser (perception context fusion)."""

from __future__ import annotations

from enton.cognition.fuser import Fuser
from enton.core.events import ActivityEvent, DetectionEvent, EmotionEvent


def test_fuser_empty():
    f = Fuser()
    result = f.fuse([], [], [])
    assert "Nenhum objeto" in result


def test_fuser_single_detection():
    f = Fuser()
    detections = [DetectionEvent(label="person", confidence=0.9)]
    result = f.fuse(detections, [], [])
    assert "person" in result


def test_fuser_multiple_same():
    f = Fuser()
    detections = [
        DetectionEvent(label="person"),
        DetectionEvent(label="person"),
        DetectionEvent(label="cat"),
    ]
    result = f.fuse(detections, [], [])
    assert "2x person" in result
    assert "cat" in result


def test_fuser_with_activity():
    f = Fuser()
    detections = [DetectionEvent(label="person")]
    activities = [ActivityEvent(person_index=0, activity="Acenando")]
    result = f.fuse(detections, activities, [])
    assert "acenando" in result.lower()


def test_fuser_with_emotion():
    f = Fuser()
    detections = [DetectionEvent(label="person")]
    emotions = [EmotionEvent(person_index=0, emotion="Feliz", score=0.85)]
    result = f.fuse(detections, [], emotions)
    assert "feliz" in result.lower()
    assert "85%" in result


def test_fuser_full_scene():
    f = Fuser()
    detections = [
        DetectionEvent(label="person"),
        DetectionEvent(label="laptop"),
    ]
    activities = [ActivityEvent(person_index=0, activity="No celular")]
    emotions = [EmotionEvent(person_index=0, emotion="Neutro", score=0.6)]
    result = f.fuse(detections, activities, emotions)
    assert "person" in result
    assert "laptop" in result
    assert "celular" in result.lower()
    assert "neutro" in result.lower()
