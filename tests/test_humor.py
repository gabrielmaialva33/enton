"""Tests for HumorDetector — multimodal sarcasm/humor detection."""

from __future__ import annotations

from enton.cognition.humor import NEGATIVE, NEUTRAL, POSITIVE, HumorDetector
from enton.core.events import EmotionEvent, HumorEvent, TranscriptionEvent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_detector() -> HumorDetector:
    return HumorDetector()


# ---------------------------------------------------------------------------
# Text Sentiment Analysis
# ---------------------------------------------------------------------------


def test_text_positive_sentiment():
    d = _make_detector()
    valence, conf = d.analyze_text("Que legal, adorei isso!")
    assert valence == POSITIVE
    assert conf > 0.5


def test_text_negative_sentiment():
    d = _make_detector()
    valence, conf = d.analyze_text("Que lixo, horrível demais")
    assert valence == NEGATIVE
    assert conf > 0.5


def test_text_neutral_sentiment():
    d = _make_detector()
    valence, conf = d.analyze_text("Hoje eu fui ao mercado")
    assert valence == NEUTRAL


def test_text_empty_string():
    d = _make_detector()
    valence, conf = d.analyze_text("")
    assert valence == NEUTRAL
    assert conf == 0.0


def test_text_whitespace_only():
    d = _make_detector()
    valence, conf = d.analyze_text("   ")
    assert valence == NEUTRAL
    assert conf == 0.0


def test_text_mixed_sentiment_positive_wins():
    d = _make_detector()
    valence, _conf = d.analyze_text("Bom, excelente, mas ruim")
    # 2 positive vs 1 negative
    assert valence == POSITIVE


def test_text_mixed_sentiment_negative_wins():
    d = _make_detector()
    valence, _conf = d.analyze_text("Bom, mas horrível e lixo")
    # 1 positive vs 2 negative
    assert valence == NEGATIVE


def test_text_sarcasm_amplifier_detected():
    d = _make_detector()
    assert d._has_sarcasm_amplifier("Que maravilha, adorei")
    assert d._has_sarcasm_amplifier("Muito bom hein")
    assert d._has_sarcasm_amplifier("que legal")
    assert d._has_sarcasm_amplifier("To ótimo")


def test_text_no_sarcasm_amplifier():
    d = _make_detector()
    assert not d._has_sarcasm_amplifier("Eu fui ao mercado")
    assert not d._has_sarcasm_amplifier("Preciso de ajuda")


# ---------------------------------------------------------------------------
# Face Emotion Valence Mapping
# ---------------------------------------------------------------------------


def test_face_positive_emotion():
    d = _make_detector()
    valence, conf = d.analyze_face("Feliz", 0.9)
    assert valence == POSITIVE
    assert conf == 0.9


def test_face_negative_emotion_raiva():
    d = _make_detector()
    valence, conf = d.analyze_face("Raiva", 0.85)
    assert valence == NEGATIVE
    assert conf == 0.85


def test_face_negative_emotion_triste():
    d = _make_detector()
    valence, conf = d.analyze_face("Triste", 0.7)
    assert valence == NEGATIVE
    assert conf == 0.7


def test_face_negative_emotion_nojo():
    d = _make_detector()
    valence, conf = d.analyze_face("Nojo", 0.8)
    assert valence == NEGATIVE
    assert conf == 0.8


def test_face_neutral_emotion():
    d = _make_detector()
    valence, conf = d.analyze_face("Neutro", 0.6)
    assert valence == NEUTRAL
    assert conf == 0.6


def test_face_surprise_is_positive():
    d = _make_detector()
    valence, _conf = d.analyze_face("Surpreso", 0.75)
    assert valence == POSITIVE


def test_face_empty_string():
    d = _make_detector()
    valence, conf = d.analyze_face("", 0.5)
    assert valence == NEUTRAL
    assert conf == 0.0


# ---------------------------------------------------------------------------
# Incongruity Detection — Sarcasm
# ---------------------------------------------------------------------------


def test_sarcasm_positive_text_negative_face():
    """Texto positivo + face com raiva = sarcasmo."""
    d = _make_detector()
    result = d.detect("Que ótimo, adorei isso", face_emotion="Raiva", face_score=0.9)
    assert isinstance(result, HumorEvent)
    assert result.is_sarcastic is True
    assert result.confidence > 0.5
    assert result.text_sentiment == POSITIVE
    assert result.face_emotion == "Raiva"


def test_sarcasm_amplifier_with_negative_face():
    """Amplificador de sarcasmo + face negativa = sarcasmo forte."""
    d = _make_detector()
    result = d.detect("Que maravilha", face_emotion="Raiva", face_score=0.9)
    assert result.is_sarcastic is True
    assert result.confidence >= 0.8


def test_sarcasm_amplifier_with_sad_face():
    """Amplificador + face triste = sarcasmo."""
    d = _make_detector()
    result = d.detect("Muito bom mesmo", face_emotion="Triste", face_score=0.85)
    assert result.is_sarcastic is True
    assert result.confidence > 0.5


def test_joke_negative_text_positive_face():
    """Texto negativo + face feliz = possivel piada."""
    d = _make_detector()
    result = d.detect("Que merda horrivel", face_emotion="Feliz", face_score=0.9)
    assert result.is_sarcastic is True
    assert result.confidence > 0.3
    assert "piada" in result.reason.lower()


def test_congruent_positive_no_sarcasm():
    """Texto positivo + face feliz = sem sarcasmo."""
    d = _make_detector()
    result = d.detect("Adorei, que legal!", face_emotion="Feliz", face_score=0.9)
    assert result.is_sarcastic is False
    assert result.confidence == 0.0
    assert "congruente" in result.reason.lower()


def test_congruent_negative_no_sarcasm():
    """Texto negativo + face triste = sem sarcasmo."""
    d = _make_detector()
    result = d.detect("Que droga, péssimo", face_emotion="Triste", face_score=0.8)
    assert result.is_sarcastic is False
    assert result.confidence == 0.0


def test_congruent_neutral_no_sarcasm():
    """Texto neutro + face neutra = sem sarcasmo."""
    d = _make_detector()
    result = d.detect("Hoje eu fui ao mercado", face_emotion="Neutro", face_score=0.7)
    assert result.is_sarcastic is False


# ---------------------------------------------------------------------------
# Missing Modalities
# ---------------------------------------------------------------------------


def test_no_face_emotion_inconclusive():
    """Sem dados faciais = resultado inconclusivo."""
    d = _make_detector()
    result = d.detect("Que maravilha", face_emotion="", face_score=0.0)
    assert result.is_sarcastic is False
    assert result.confidence == 0.0
    assert "sem dados faciais" in result.reason.lower()


def test_low_face_confidence_inconclusive():
    """Face com confianca muito baixa = inconclusivo."""
    d = _make_detector()
    result = d.detect("Adorei!", face_emotion="Raiva", face_score=0.05)
    assert result.is_sarcastic is False
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# PT-BR Specific Patterns
# ---------------------------------------------------------------------------


def test_ptbr_que_maravilha_angry():
    """'que maravilha' + cara de raiva = sarcasmo classico BR."""
    d = _make_detector()
    result = d.detect("Que maravilha hein", face_emotion="Raiva", face_score=0.92)
    assert result.is_sarcastic is True
    assert result.confidence > 0.8


def test_ptbr_to_otimo_angry():
    """'to ótimo' + cara de raiva = sarcasmo."""
    d = _make_detector()
    result = d.detect("To ótimo, obrigado", face_emotion="Raiva", face_score=0.88)
    assert result.is_sarcastic is True


def test_ptbr_que_legal_nojo():
    """'que legal' + nojo = sarcasmo."""
    d = _make_detector()
    result = d.detect("Que legal isso", face_emotion="Nojo", face_score=0.8)
    assert result.is_sarcastic is True


def test_ptbr_brabo_foda_positive():
    """'brabo' e 'foda' sao positivos no contexto BR."""
    d = _make_detector()
    valence, _conf = d.analyze_text("Brabo demais, foda!")
    assert valence == POSITIVE


# ---------------------------------------------------------------------------
# on_transcription Entry Point
# ---------------------------------------------------------------------------


def test_on_transcription_with_emotion():
    """on_transcription integra TranscriptionEvent + EmotionEvent."""
    d = _make_detector()
    trans = TranscriptionEvent(text="Que maravilha")
    emotion = EmotionEvent(emotion="Raiva", score=0.9)
    result = d.on_transcription(trans, emotion)
    assert isinstance(result, HumorEvent)
    assert result.is_sarcastic is True


def test_on_transcription_without_emotion():
    """on_transcription sem emocao facial = inconclusivo."""
    d = _make_detector()
    trans = TranscriptionEvent(text="Adorei!")
    result = d.on_transcription(trans, None)
    assert result.is_sarcastic is False
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Stats Tracking
# ---------------------------------------------------------------------------


def test_detection_count_increments():
    d = _make_detector()
    assert d.detection_count == 0
    d.detect("oi", face_emotion="Neutro", face_score=0.8)
    d.detect("oi", face_emotion="Neutro", face_score=0.8)
    assert d.detection_count == 2


def test_sarcasm_count_increments():
    d = _make_detector()
    assert d.sarcasm_count == 0
    d.detect("Adorei!", face_emotion="Raiva", face_score=0.9)
    assert d.sarcasm_count == 1
    d.detect("Legal", face_emotion="Feliz", face_score=0.9)
    assert d.sarcasm_count == 1  # no sarcasm, count unchanged


def test_to_dict():
    d = _make_detector()
    d.detect("Que maravilha", face_emotion="Raiva", face_score=0.9)
    info = d.to_dict()
    assert info["detection_count"] == 1
    assert info["sarcasm_count"] == 1


# ---------------------------------------------------------------------------
# HumorEvent Fields
# ---------------------------------------------------------------------------


def test_humor_event_fields():
    """HumorEvent retains all cross-modal analysis fields."""
    d = _make_detector()
    result = d.detect(
        "Que ótimo, adorei",
        face_emotion="Raiva",
        face_score=0.9,
    )
    assert result.text == "Que ótimo, adorei"
    assert result.face_emotion == "Raiva"
    assert result.text_sentiment in {POSITIVE, NEGATIVE, NEUTRAL}
    assert result.timestamp > 0


def test_humor_event_is_frozen():
    """HumorEvent is immutable (frozen dataclass)."""
    d = _make_detector()
    result = d.detect("Oi", face_emotion="Neutro", face_score=0.8)
    try:
        result.is_sarcastic = True  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except AttributeError:
        pass  # expected — frozen dataclass
