"""HumorDetector — deteccao multimodal de sarcasmo e humor.

Cruza emocao facial + sentimento textual + tom de voz para detectar
incongruencia entre modalidades. Ex: "To otimo" + cara de raiva + voz
monotona = sarcasmo.

Abordagem leve (keyword-based) para PT-BR, sem dependencias pesadas.
"""

from __future__ import annotations

import logging
import re

from enton.core.events import EmotionEvent, HumorEvent, TranscriptionEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulario PT-BR para analise de sentimento
# ---------------------------------------------------------------------------

_POSITIVE_WORDS: set[str] = {
    "otimo",
    "ótimo",
    "maravilha",
    "maravilhoso",
    "adorei",
    "legal",
    "bom",
    "boa",
    "excelente",
    "incrível",
    "incrivel",
    "massa",
    "top",
    "show",
    "brabo",
    "foda",
    "dahora",
    "demais",
    "perfeito",
    "perfeita",
    "lindo",
    "linda",
    "sensacional",
    "animal",
    "genial",
    "maneiro",
    "bacana",
    "espetacular",
    "fantástico",
    "fantastico",
    "gostei",
    "amei",
    "feliz",
    "alegria",
    "curtindo",
    "curti",
    "melhor",
}

_NEGATIVE_WORDS: set[str] = {
    "ruim",
    "merda",
    "horrível",
    "horrivel",
    "péssimo",
    "pessimo",
    "lixo",
    "droga",
    "porcaria",
    "bosta",
    "nojo",
    "odiei",
    "odeio",
    "terrível",
    "terrivel",
    "triste",
    "raiva",
    "irritado",
    "irritada",
    "chato",
    "chata",
    "desgraça",
    "desgraca",
    "inferno",
    "pior",
    "detesto",
    "detestei",
    "ridículo",
    "ridiculo",
    "absurdo",
    "patético",
    "patetico",
}

# Expressoes que, quando combinadas com face negativa, amplificam sarcasmo.
_SARCASM_AMPLIFIERS: list[re.Pattern[str]] = [
    re.compile(r"\bque\s+maravilha\b", re.IGNORECASE),
    re.compile(r"\badorei\b", re.IGNORECASE),
    re.compile(r"\bmuito\s+bom\b", re.IGNORECASE),
    re.compile(r"\bque\s+legal\b", re.IGNORECASE),
    re.compile(r"\bque\s+lindo\b", re.IGNORECASE),
    re.compile(r"\bto\s+ótimo\b", re.IGNORECASE),
    re.compile(r"\bto\s+otimo\b", re.IGNORECASE),
    re.compile(r"\bque\s+ótimo\b", re.IGNORECASE),
    re.compile(r"\bque\s+otimo\b", re.IGNORECASE),
    re.compile(r"\bparabéns\b", re.IGNORECASE),
    re.compile(r"\bparabens\b", re.IGNORECASE),
    re.compile(r"\bque\s+show\b", re.IGNORECASE),
    re.compile(r"\bgrande\s+dia\b", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Mapeamento de emocoes faciais para valencia
# ---------------------------------------------------------------------------

_FACE_POSITIVE: set[str] = {"Feliz", "Surpreso"}
_FACE_NEGATIVE: set[str] = {"Triste", "Raiva", "Nojo", "Medo"}
_FACE_NEUTRAL: set[str] = {"Neutro"}

# Sentimento string constants
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"


class HumorDetector:
    """Detector multimodal de sarcasmo e humor.

    Cruza emocao facial + sentimento textual + tom de voz para detectar
    incongruencia. Ex: "To otimo" + cara de raiva + voz monotona = sarcasmo.
    """

    def __init__(self) -> None:
        self._detection_count: int = 0
        self._sarcasm_count: int = 0

    # -- analise textual --

    def analyze_text(self, text: str) -> tuple[str, float]:
        """Classifica sentimento do texto por keywords PT-BR.

        Returns:
            (valence, confidence) onde valence eh POSITIVE/NEGATIVE/NEUTRAL.
        """
        if not text or not text.strip():
            return NEUTRAL, 0.0

        lower = text.lower().strip()
        words = set(re.findall(r"\w+", lower))

        pos_count = len(words & _POSITIVE_WORDS)
        neg_count = len(words & _NEGATIVE_WORDS)

        total = pos_count + neg_count
        if total == 0:
            return NEUTRAL, 0.3

        if pos_count > neg_count:
            confidence = min(0.95, 0.5 + (pos_count - neg_count) * 0.15)
            return POSITIVE, confidence
        elif neg_count > pos_count:
            confidence = min(0.95, 0.5 + (neg_count - pos_count) * 0.15)
            return NEGATIVE, confidence
        else:
            # empate
            return NEUTRAL, 0.4

    def _has_sarcasm_amplifier(self, text: str) -> bool:
        """Verifica se o texto contem expressoes amplificadoras de sarcasmo."""
        return any(pat.search(text) for pat in _SARCASM_AMPLIFIERS)

    # -- analise facial --

    def analyze_face(self, emotion: str, score: float) -> tuple[str, float]:
        """Mapeia emocao facial para valencia.

        Args:
            emotion: String de emocao (ex: "Feliz", "Raiva").
            score: Confianca do detector de emocao (0-1).

        Returns:
            (valence, confidence) onde valence eh POSITIVE/NEGATIVE/NEUTRAL.
        """
        if not emotion:
            return NEUTRAL, 0.0

        if emotion in _FACE_POSITIVE:
            return POSITIVE, score
        elif emotion in _FACE_NEGATIVE:
            return NEGATIVE, score
        else:
            return NEUTRAL, score

    # -- deteccao de incongruencia --

    def detect(
        self,
        text: str,
        face_emotion: str = "",
        face_score: float = 0.0,
    ) -> HumorEvent:
        """Detecta sarcasmo cruzando texto e emocao facial.

        Regras de incongruencia (ordenadas por forca):
        1. Amplificador de sarcasmo + face nao-positiva -> sarcasmo (0.90)
        2. Texto POSITIVE + face NEGATIVE -> sarcasmo (0.85)
        3. Texto NEGATIVE + face POSITIVE -> possivel piada (0.65)
        4. Modalidades concordam -> sem sarcasmo

        Args:
            text: Texto transcrito do usuario.
            face_emotion: Emocao facial detectada (ex: "Raiva").
            face_score: Confianca da deteccao facial (0-1).

        Returns:
            HumorEvent com resultado da analise.
        """
        self._detection_count += 1

        text_valence, text_conf = self.analyze_text(text)
        face_valence, face_conf = self.analyze_face(face_emotion, face_score)

        has_amplifier = self._has_sarcasm_amplifier(text)

        # Sem dados faciais — inconclusivo
        if not face_emotion or face_conf < 0.1:
            return HumorEvent(
                is_sarcastic=False,
                confidence=0.0,
                reason="Sem dados faciais para cruzamento",
                text=text,
                face_emotion=face_emotion,
                text_sentiment=text_valence,
            )

        # Regra 1: amplificador de sarcasmo + face nao-positiva
        if has_amplifier and face_valence != POSITIVE:
            self._sarcasm_count += 1
            conf = min(0.95, 0.90 * face_conf)
            reason = f"Amplificador de sarcasmo no texto + face {face_emotion} (nao-positiva)"
            logger.info(
                "Sarcasmo detectado (amplificador): text='%s' face=%s conf=%.2f",
                text[:50],
                face_emotion,
                conf,
            )
            return HumorEvent(
                is_sarcastic=True,
                confidence=conf,
                reason=reason,
                text=text,
                face_emotion=face_emotion,
                text_sentiment=text_valence,
            )

        # Regra 2: texto positivo + face negativa
        if text_valence == POSITIVE and face_valence == NEGATIVE:
            self._sarcasm_count += 1
            conf = min(0.95, 0.85 * min(text_conf, face_conf) / 0.5)
            conf = max(0.0, min(1.0, conf))
            reason = f"Texto positivo ({text_valence}) + face negativa ({face_emotion})"
            logger.info(
                "Sarcasmo detectado (incongruencia): text='%s' face=%s conf=%.2f",
                text[:50],
                face_emotion,
                conf,
            )
            return HumorEvent(
                is_sarcastic=True,
                confidence=conf,
                reason=reason,
                text=text,
                face_emotion=face_emotion,
                text_sentiment=text_valence,
            )

        # Regra 3: texto negativo + face positiva (possivel piada)
        if text_valence == NEGATIVE and face_valence == POSITIVE:
            self._sarcasm_count += 1
            conf = min(0.95, 0.65 * min(text_conf, face_conf) / 0.5)
            conf = max(0.0, min(1.0, conf))
            reason = (
                f"Texto negativo ({text_valence}) + face positiva ({face_emotion}) — possivel piada"
            )
            logger.info(
                "Possivel humor detectado: text='%s' face=%s conf=%.2f",
                text[:50],
                face_emotion,
                conf,
            )
            return HumorEvent(
                is_sarcastic=True,
                confidence=conf,
                reason=reason,
                text=text,
                face_emotion=face_emotion,
                text_sentiment=text_valence,
            )

        # Regra 4: modalidades concordam — sem sarcasmo
        return HumorEvent(
            is_sarcastic=False,
            confidence=0.0,
            reason="Modalidades congruentes",
            text=text,
            face_emotion=face_emotion,
            text_sentiment=text_valence,
        )

    # -- entry point para EventBus --

    def on_transcription(
        self,
        event: TranscriptionEvent,
        latest_emotion: EmotionEvent | None,
    ) -> HumorEvent:
        """Entry point principal — chamado quando chega transcricao.

        Cruza transcricao com a emocao facial mais recente.

        Args:
            event: Evento de transcricao com texto do usuario.
            latest_emotion: Ultimo EmotionEvent detectado (ou None).

        Returns:
            HumorEvent com resultado da analise.
        """
        face_emotion = ""
        face_score = 0.0
        if latest_emotion is not None:
            face_emotion = latest_emotion.emotion
            face_score = latest_emotion.score

        return self.detect(
            text=event.text,
            face_emotion=face_emotion,
            face_score=face_score,
        )

    # -- stats --

    @property
    def detection_count(self) -> int:
        return self._detection_count

    @property
    def sarcasm_count(self) -> int:
        return self._sarcasm_count

    def to_dict(self) -> dict:
        return {
            "detection_count": self._detection_count,
            "sarcasm_count": self._sarcasm_count,
        }
