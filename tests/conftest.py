"""Shared fixtures for Enton test suite."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Prevent tests from touching real ~/.enton or real API keys."""
    monkeypatch.setenv("ENTON_HOME", str(tmp_path))
    monkeypatch.setattr("enton.core.memory.MEMORY_DIR", tmp_path / "memory")
    monkeypatch.setattr("enton.core.memory.EPISODES_FILE", tmp_path / "memory" / "episodes.jsonl")
    monkeypatch.setattr("enton.core.memory.PROFILE_FILE", tmp_path / "memory" / "profile.json")
    monkeypatch.setattr("enton.core.visual_memory.FRAMES_DIR", tmp_path / "frames")
    monkeypatch.setenv("ENTON_SKILLS_DIR", str(tmp_path / "skills"))


@pytest.fixture()
def tmp_planner_file(tmp_path, monkeypatch):
    """Redirect planner persistence to tmp."""
    f = tmp_path / "planner.json"
    monkeypatch.setattr("enton.cognition.planner._PLANNER_FILE", f)
    return f


@pytest.fixture()
def mock_settings():
    """Minimal Settings mock — no real API keys needed."""
    s = MagicMock()
    s.ollama_model = "qwen2.5:14b"
    s.ollama_vlm_model = "qwen2.5-vl:7b"
    s.nvidia_api_keys = ""
    s.nvidia_api_key = ""
    s.nvidia_nim_model = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    s.nvidia_nim_vision_model = "nvidia/llama-3.2-neva-72b-v1"
    s.huggingface_token = ""
    s.huggingface_model = ""
    s.huggingface_vision_model = ""
    s.groq_api_key = ""
    s.groq_model = "llama-3.3-70b-versatile"
    s.openrouter_api_key = ""
    s.openrouter_model = "qwen/qwen3-235b-a22b:free"
    s.openrouter_vision_model = ""
    s.aimlapi_api_key = ""
    s.aimlapi_model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    s.google_project = ""
    s.google_brain_model = "gemini-2.0-flash"
    s.google_vision_model = "gemini-2.0-flash"
    s.qdrant_url = "http://localhost:6333"
    s.memory_size = 10
    s.brain_max_turns = 5
    s.vlm_transformers_model = "Qwen/Qwen2.5-VL-3B-Instruct"
    s.yolo_device = "cpu"
    # TTS settings
    s.kokoro_lang = "p"
    s.kokoro_voice = "pm_alex"
    s.qwen3_tts_model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    s.qwen3_tts_voice_instruct = "A deep male robotic voice."
    s.qwen3_tts_device = "cpu"
    s.edge_tts_voice = "pt-BR-AntonioNeural"
    s.nvidia_tts_voice = "English-US.Male-1"
    s.sample_rate = 16000
    s.tts_provider = "qwen3"
    return s
