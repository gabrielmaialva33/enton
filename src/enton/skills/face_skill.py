from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from enton.core.tools import tool

if TYPE_CHECKING:
    from enton.perception.faces import FaceRecognizer
    from enton.perception.vision import Vision

logger = logging.getLogger(__name__)

_vision: Vision | None = None
_face_recognizer: FaceRecognizer | None = None


def init(vision: Vision, face_recognizer: FaceRecognizer) -> None:
    """Initialize with app components. Called from App.__init__."""
    global _vision, _face_recognizer
    _vision = vision
    _face_recognizer = face_recognizer


@tool
def enroll_face(name: str) -> str:
    """Cadastra o rosto da pessoa visível na câmera com o nome dado.

    Deve haver exatamente 1 pessoa no frame para funcionar.

    Args:
        name: Nome da pessoa a ser cadastrada.
    """
    if _vision is None or _face_recognizer is None:
        return "Sistema de reconhecimento facial não inicializado."

    frame = _vision.last_frame
    if frame is None:
        return "Sem frame disponível — câmera pode estar desconectada."

    return _face_recognizer.enroll(name, frame)


@tool
def list_faces() -> str:
    """Lista todas as pessoas cadastradas no banco de rostos."""
    if _face_recognizer is None:
        return "Sistema de reconhecimento facial não inicializado."

    names = _face_recognizer.list_enrolled()
    if not names:
        return "Nenhum rosto cadastrado ainda."
    return f"Rostos cadastrados ({len(names)}): {', '.join(names)}"


@tool
def remove_face(name: str) -> str:
    """Remove uma pessoa do banco de rostos.

    Args:
        name: Nome da pessoa a remover.
    """
    if _face_recognizer is None:
        return "Sistema de reconhecimento facial não inicializado."

    if _face_recognizer.remove(name):
        return f"Rosto de '{name}' removido."
    return f"'{name}' não encontrado no banco de rostos."
