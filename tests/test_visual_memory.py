"""Tests for VisualMemory."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from enton.core.visual_memory import VisualEpisode, VisualMemory


def _fake_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_visual_episode_creation():
    ep = VisualEpisode(
        timestamp=1.0, camera_id="main",
        detections=["person", "cup"], thumbnail_path="/tmp/t.jpg",
    )
    assert ep.camera_id == "main"
    assert "person" in ep.detections


def test_visual_memory_init():
    vm = VisualMemory(qdrant_url="http://fake:6333")
    assert vm._episode_count == 0
    assert vm._model is None


def test_should_embed_first_time():
    vm = VisualMemory()
    assert vm._should_embed(["person"], "main") is True


def test_should_not_embed_same_scene():
    vm = VisualMemory()
    vm._last_embed["main"] = time.time()
    vm._last_labels["main"] = {"person"}
    assert vm._should_embed(["person"], "main") is False


def test_should_embed_new_objects():
    vm = VisualMemory()
    vm._last_embed["main"] = time.time() - 60
    vm._last_labels["main"] = {"person"}
    assert vm._should_embed(["person", "cup"], "main") is True


def test_should_not_embed_cooldown():
    vm = VisualMemory()
    vm._last_embed["main"] = time.time() - 5  # only 5s ago
    vm._last_labels["main"] = {"person"}
    assert vm._should_embed(["person", "cup"], "main") is False


def test_save_thumbnail(tmp_path, monkeypatch):
    monkeypatch.setattr("enton.core.visual_memory.FRAMES_DIR", tmp_path)
    vm = VisualMemory(frames_dir=tmp_path)
    frame = _fake_frame()
    path = vm._save_thumbnail(frame, time.time())
    assert path.endswith(".jpg")


@pytest.mark.asyncio()
async def test_embed_frame_no_model():
    vm = VisualMemory()
    # Model won't load (no open_clip installed in test env)
    with patch.object(vm, "_load_model", return_value=False):
        result = await vm.embed_frame(_fake_frame())
    assert result == []


@pytest.mark.asyncio()
async def test_embed_text_no_model():
    vm = VisualMemory()
    with patch.object(vm, "_load_model", return_value=False):
        result = await vm.embed_text("where is my cup?")
    assert result == []


@pytest.mark.asyncio()
async def test_remember_scene_skips_non_novel():
    vm = VisualMemory()
    vm._last_embed["main"] = time.time()
    vm._last_labels["main"] = {"person"}
    result = await vm.remember_scene(_fake_frame(), ["person"], "main")
    assert result is None


@pytest.mark.asyncio()
async def test_remember_scene_stores(tmp_path, monkeypatch):
    monkeypatch.setattr("enton.core.visual_memory.FRAMES_DIR", tmp_path)
    vm = VisualMemory(frames_dir=tmp_path)

    fake_embedding = [0.1] * 512
    with (
        patch.object(vm, "embed_frame", return_value=fake_embedding),
        patch.object(vm, "_init_qdrant", return_value=False),
    ):
        result = await vm.remember_scene(_fake_frame(), ["person", "cup"], "main")

    assert result is not None
    assert result.camera_id == "main"
    assert "person" in result.detections
    assert vm._episode_count == 1


@pytest.mark.asyncio()
async def test_search_no_qdrant():
    vm = VisualMemory()
    with (
        patch.object(vm, "embed_text", return_value=[0.1] * 512),
        patch.object(vm, "_init_qdrant", return_value=False),
    ):
        results = await vm.search("cup")
    assert results == []


@pytest.mark.asyncio()
async def test_search_with_results():
    vm = VisualMemory()
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.payload = {
        "timestamp": time.time(), "camera_id": "main",
        "detections": ["cup"], "thumbnail_path": "/tmp/t.jpg",
    }
    mock_result.score = 0.9
    mock_client.search.return_value = [mock_result]
    vm._qdrant = mock_client

    with patch.object(vm, "embed_text", return_value=[0.1] * 512):
        results = await vm.search("cup")

    assert len(results) == 1
    assert results[0]["detections"] == ["cup"]
    assert results[0]["score"] == 0.9


@pytest.mark.asyncio()
async def test_recent_scenes_no_qdrant():
    vm = VisualMemory()
    results = await vm.recent_scenes()
    assert results == []
