"""Tests for Persona (system prompt builder)."""

from __future__ import annotations

from unittest.mock import MagicMock

from enton.cognition.persona import (
    REACTION_TEMPLATES,
    _build_env_context,
    _get_empathy_instruction,
    build_system_prompt,
)


def test_build_system_prompt_basic():
    self_model = MagicMock()
    self_model.introspect.return_value = "I am Enton."
    self_model.last_emotion = "neutral"
    memory = MagicMock()
    memory.context_string.return_value = "No memories yet"

    prompt = build_system_prompt(self_model, memory)
    assert "Enton" in prompt
    assert "I am Enton." in prompt
    assert "No memories yet" in prompt


def test_build_system_prompt_with_detections():
    self_model = MagicMock()
    self_model.introspect.return_value = "Running."
    self_model.last_emotion = "neutral"
    memory = MagicMock()
    memory.context_string.return_value = ""

    prompt = build_system_prompt(
        self_model,
        memory,
        detections=[{"label": "person"}, {"label": "cat"}],
    )
    assert "person" in prompt
    assert "cat" in prompt


def test_build_system_prompt_empathy():
    self_model = MagicMock()
    self_model.introspect.return_value = "Running."
    self_model.last_emotion = "sad"
    memory = MagicMock()
    memory.context_string.return_value = ""

    prompt = build_system_prompt(self_model, memory)
    assert "EMOTIONAL CONTEXT" in prompt
    assert "sad" in prompt.lower()


def test_empathy_instruction():
    assert "happy" in _get_empathy_instruction("happy").lower()
    assert "sad" in _get_empathy_instruction("triste").lower()
    assert _get_empathy_instruction("unknown") == ""


def test_env_context_with_detections():
    ctx = _build_env_context(
        [{"label": "person"}, {"label": "laptop"}],
        hour=14,
    )
    assert "person" in ctx
    assert "laptop" in ctx
    assert "afternoon" in ctx


def test_env_context_empty():
    ctx = _build_env_context([], hour=22)
    assert "night" in ctx


def test_env_context_morning():
    ctx = _build_env_context([], hour=9)
    assert "morning" in ctx


def test_reaction_templates_exist():
    expected = [
        "person_appeared",
        "person_left",
        "cat_detected",
        "idle",
        "startup",
        "face_recognized",
        "doorbell",
        "alarm",
        "tool_executed",
    ]
    for key in expected:
        assert key in REACTION_TEMPLATES
        assert len(REACTION_TEMPLATES[key]) >= 1
