"""Tests for ActionRecognizer (VideoMAE temporal action recognition)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from enton.core.events import ActionEvent
from enton.perception.actions import ACTION_LABELS_PT, ActionRecognizer

# ---------------------------------------------------------------------------
# ActionEvent
# ---------------------------------------------------------------------------


def test_action_event_defaults():
    e = ActionEvent()
    assert e.action == ""
    assert e.action_en == ""
    assert e.confidence == 0.0
    assert e.person_index == 0
    assert e.camera_id == "main"
    assert e.timestamp > 0


def test_action_event_fields():
    e = ActionEvent(
        action="bebendo",
        action_en="drinking",
        confidence=0.85,
        person_index=1,
        camera_id="cam2",
    )
    assert e.action == "bebendo"
    assert e.action_en == "drinking"
    assert e.confidence == 0.85
    assert e.person_index == 1
    assert e.camera_id == "cam2"


def test_action_event_is_frozen():
    e = ActionEvent(action="lendo")
    with pytest.raises(AttributeError):
        e.action = "escrevendo"


# ---------------------------------------------------------------------------
# ActionRecognizer — init / state
# ---------------------------------------------------------------------------


def test_init_model_not_loaded():
    """Modelo nao deve ser carregado no __init__."""
    rec = ActionRecognizer(device="cpu")
    assert rec._loaded is False
    assert rec._model is None
    assert rec._processor is None


def test_init_empty_buffer():
    rec = ActionRecognizer(device="cpu")
    assert len(rec._frame_buffer) == 0
    assert rec._frame_count == 0
    assert rec.last_actions == []


# ---------------------------------------------------------------------------
# feed_frame
# ---------------------------------------------------------------------------


def _make_bgr_frame(h: int = 224, w: int = 224) -> np.ndarray:
    """Cria um frame BGR sintetico."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def test_feed_frame_adds_to_buffer():
    rec = ActionRecognizer(device="cpu")
    frame = _make_bgr_frame()
    rec.feed_frame(frame)
    assert len(rec._frame_buffer) == 1
    assert rec._frame_count == 1


def test_feed_frame_converts_bgr_to_rgb():
    """O frame salvo no buffer deve ser RGB (nao BGR)."""
    rec = ActionRecognizer(device="cpu")
    # Cria frame com canal B=255, G=0, R=0 (azul puro em BGR)
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255  # B channel
    rec.feed_frame(bgr)
    stored = rec._frame_buffer[0]
    # Em RGB, o canal R (index 0) deve ser 0 e canal B (index 2) deve ser 255
    assert stored[0, 0, 0] == 0  # R
    assert stored[0, 0, 2] == 255  # B


def test_feed_frame_buffer_maxlen():
    """Buffer circular deve respeitar maxlen=64."""
    rec = ActionRecognizer(device="cpu")
    for _ in range(70):
        rec.feed_frame(_make_bgr_frame())
    assert len(rec._frame_buffer) == 64
    assert rec._frame_count == 70


# ---------------------------------------------------------------------------
# _sample_frames
# ---------------------------------------------------------------------------


def test_sample_frames_not_enough():
    rec = ActionRecognizer(device="cpu")
    for _ in range(10):
        rec.feed_frame(_make_bgr_frame())
    assert rec._sample_frames() is None


def test_sample_frames_exact():
    rec = ActionRecognizer(device="cpu")
    for _ in range(ActionRecognizer.NUM_FRAMES):
        rec.feed_frame(_make_bgr_frame())
    frames = rec._sample_frames()
    assert frames is not None
    assert len(frames) == ActionRecognizer.NUM_FRAMES


def test_sample_frames_from_larger_buffer():
    rec = ActionRecognizer(device="cpu")
    for _ in range(50):
        rec.feed_frame(_make_bgr_frame())
    frames = rec._sample_frames()
    assert frames is not None
    assert len(frames) == ActionRecognizer.NUM_FRAMES


# ---------------------------------------------------------------------------
# should_classify
# ---------------------------------------------------------------------------


def test_should_classify_false_initially():
    rec = ActionRecognizer(device="cpu")
    assert rec.should_classify() is False


def test_should_classify_false_not_enough_frames():
    rec = ActionRecognizer(device="cpu")
    for _ in range(ActionRecognizer.CLASSIFY_EVERY):
        rec.feed_frame(_make_bgr_frame())
    # frame_count == 32, buffer has 32 frames (>= 16) — should be True
    assert rec.should_classify() is True


def test_should_classify_true_at_interval():
    rec = ActionRecognizer(device="cpu")
    for _ in range(ActionRecognizer.CLASSIFY_EVERY):
        rec.feed_frame(_make_bgr_frame())
    assert rec._frame_count == ActionRecognizer.CLASSIFY_EVERY
    assert len(rec._frame_buffer) >= ActionRecognizer.NUM_FRAMES
    assert rec.should_classify() is True


def test_should_classify_false_between_intervals():
    rec = ActionRecognizer(device="cpu")
    for _ in range(ActionRecognizer.CLASSIFY_EVERY + 1):
        rec.feed_frame(_make_bgr_frame())
    # frame_count == 33, 33 % 32 != 0
    assert rec.should_classify() is False


# ---------------------------------------------------------------------------
# classify (mocked model)
# ---------------------------------------------------------------------------


def _build_mock_model_and_processor():
    """Cria mocks do VideoMAE model e processor."""
    import torch

    processor = MagicMock()
    processor.return_value = {"pixel_values": torch.randn(1, 16, 3, 224, 224)}

    # Simula logits com 400 classes (Kinetics-400)
    # Logit 8.0 -> softmax ~0.75 (bem acima de MIN_CONFIDENCE=0.3)
    logits = torch.zeros(1, 400)
    logits[0, 7] = 8.0  # classe dominante
    logits[0, 42] = 6.0  # segunda classe

    model = MagicMock()
    model.return_value = SimpleNamespace(logits=logits)
    id2label = {i: f"action_{i}" for i in range(400)}
    id2label[7] = "drinking"
    id2label[42] = "reading"
    model.config = SimpleNamespace(id2label=id2label)
    model.eval.return_value = model
    model.to.return_value = model

    return model, processor


def test_classify_empty_buffer():
    rec = ActionRecognizer(device="cpu")
    result = rec.classify()
    assert result == []


@patch("enton.perception.actions.ActionRecognizer._ensure_loaded")
def test_classify_model_load_fails(mock_load):
    mock_load.return_value = False
    rec = ActionRecognizer(device="cpu")
    for _ in range(20):
        rec.feed_frame(_make_bgr_frame())
    result = rec.classify()
    assert result == []


def test_classify_with_mocked_model():
    """Testa classificacao com modelo mockado."""
    model, processor = _build_mock_model_and_processor()
    rec = ActionRecognizer(device="cpu")
    rec._model = model
    rec._processor = processor
    rec._loaded = True

    for _ in range(20):
        rec.feed_frame(_make_bgr_frame())

    results = rec.classify()
    assert len(results) > 0

    # Primeira acao deve ser "drinking" (logit mais alto)
    action_en, action_pt, conf = results[0]
    assert action_en == "drinking"
    assert action_pt == "bebendo"
    assert conf > 0.0


def test_classify_updates_last_actions():
    model, processor = _build_mock_model_and_processor()
    rec = ActionRecognizer(device="cpu")
    rec._model = model
    rec._processor = processor
    rec._loaded = True

    for _ in range(20):
        rec.feed_frame(_make_bgr_frame())

    rec.classify()
    assert rec.last_actions == rec._last_actions
    assert len(rec.last_actions) > 0


def test_classify_filters_low_confidence():
    """Acoes com confianca abaixo do limiar devem ser filtradas."""
    import torch

    processor = MagicMock()
    processor.return_value = {"pixel_values": torch.randn(1, 16, 3, 224, 224)}

    # Todas as classes com logits iguais -> probabilidades ~0.0025 (muito abaixo de 0.3)
    logits = torch.zeros(1, 400)
    model = MagicMock()
    model.return_value = SimpleNamespace(logits=logits)
    model.config = SimpleNamespace(id2label={i: f"action_{i}" for i in range(400)})
    model.eval.return_value = model
    model.to.return_value = model

    rec = ActionRecognizer(device="cpu")
    rec._model = model
    rec._processor = processor
    rec._loaded = True

    for _ in range(20):
        rec.feed_frame(_make_bgr_frame())

    results = rec.classify()
    assert results == []


def test_classify_handles_inference_error():
    """Erros de inferencia devem retornar lista vazia, nao crashar."""
    rec = ActionRecognizer(device="cpu")
    rec._loaded = True
    rec._processor = MagicMock(side_effect=RuntimeError("CUDA OOM"))
    rec._model = MagicMock()

    for _ in range(20):
        rec.feed_frame(_make_bgr_frame())

    results = rec.classify()
    assert results == []


# ---------------------------------------------------------------------------
# unload
# ---------------------------------------------------------------------------


def test_unload_clears_model():
    rec = ActionRecognizer(device="cpu")
    rec._model = MagicMock()
    rec._processor = MagicMock()
    rec._loaded = True

    with patch("torch.cuda.empty_cache"):
        rec.unload()

    assert rec._model is None
    assert rec._processor is None
    assert rec._loaded is False


def test_unload_noop_when_not_loaded():
    """Unload sem modelo carregado nao deve dar erro."""
    rec = ActionRecognizer(device="cpu")
    rec.unload()  # nao deve levantar excecao
    assert rec._loaded is False


# ---------------------------------------------------------------------------
# ACTION_LABELS_PT
# ---------------------------------------------------------------------------


def test_action_labels_has_entries():
    assert len(ACTION_LABELS_PT) >= 20


def test_action_labels_all_strings():
    for en, pt in ACTION_LABELS_PT.items():
        assert isinstance(en, str)
        assert isinstance(pt, str)
        assert len(en) > 0
        assert len(pt) > 0


def test_action_labels_known_translations():
    assert ACTION_LABELS_PT["drinking"] == "bebendo"
    assert ACTION_LABELS_PT["typing"] == "digitando"
    assert ACTION_LABELS_PT["walking"] == "andando"
    assert ACTION_LABELS_PT["dancing"] == "dancando"
