"""Tests for EntonBrain (model init, no actual LLM calls)."""

from __future__ import annotations

from enton.cognition.brain import EntonBrain


def test_brain_init_models_local_only(mock_settings):
    """With no API keys, only Ollama model should be created."""
    models = EntonBrain._init_models(mock_settings)
    assert len(models) >= 1
    assert getattr(models[0], "id", "") == "qwen2.5:14b"


def test_brain_init_vision_models(mock_settings):
    """Vision models should include at least Ollama VLM."""
    models = EntonBrain._init_vision_models(mock_settings)
    assert len(models) >= 1


def test_brain_clean():
    assert EntonBrain._clean("<think>reasoning</think>Hello") == "Hello"
    assert EntonBrain._clean("No tags here") == "No tags here"
    assert EntonBrain._clean("<think>foo</think> bar <think>baz</think> qux") == "bar  qux"
    assert EntonBrain._clean("  spaces  ") == "spaces"
