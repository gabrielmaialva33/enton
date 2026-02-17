from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from enton.core.tools import tool

if TYPE_CHECKING:
    from enton.cognition.brain import Brain
    from enton.perception.vision import Vision

logger = logging.getLogger(__name__)

# Module-level references set during App.__init__
_vision: Vision | None = None
_brain: Brain | None = None


def init(vision: Vision, brain: Brain) -> None:
    """Initialize with app components. Called from App.__init__."""
    global _vision, _brain
    _vision = vision
    _brain = brain


_DEFAULT_PROMPT = "Descreva o que você está vendo de forma breve e interessante."


@tool
async def describe_scene(prompt: str = _DEFAULT_PROMPT) -> str:
    """Descreve a cena atual da câmera usando visão computacional + VLM.

    Captura o frame atual e pede ao modelo de visão para descrever.

    Args:
        prompt: Instrução para o modelo de visão sobre o que descrever.
    """
    if _vision is None or _brain is None:
        return "Erro: Sistema de visão não inicializado."

    jpeg = _vision.get_frame_jpeg()
    if jpeg is None:
        return "Sem imagem disponível — câmera pode estar desconectada."

    try:
        description = await _brain.describe_scene(
            jpeg,
            system=(
                "Você é o Enton, um robô assistente brasileiro. "
                "Descreva a cena em português de forma natural."
            ),
        )
        return description or "Não consegui descrever a cena."
    except Exception as e:
        logger.exception("describe_scene failed")
        return f"Erro ao descrever cena: {e}"


@tool
async def what_do_you_see() -> str:
    """Retorna uma lista das detecções atuais da câmera (objetos, atividades, emoções)."""
    if _vision is None:
        return "Sistema de visão não inicializado."

    detections = _vision.last_detections
    activities = _vision.last_activities
    emotions = _vision.last_emotions

    if not detections and not activities and not emotions:
        return "Não estou vendo nada no momento."

    parts = []
    if detections:
        obj_counts: dict[str, int] = {}
        for d in detections:
            obj_counts[d.label] = obj_counts.get(d.label, 0) + 1
        obj_str = ", ".join(f"{v}x {k}" for k, v in obj_counts.items())
        parts.append(f"Objetos: {obj_str}")

    if activities:
        act_str = ", ".join(f"pessoa {a.person_index}: {a.activity}" for a in activities)
        parts.append(f"Atividades: {act_str}")

    if emotions:
        emo_str = ", ".join(
            f"pessoa {e.person_index}: {e.emotion} ({e.score:.0%})"
            for e in emotions
        )
        parts.append(f"Emoções: {emo_str}")

    return " | ".join(parts)
