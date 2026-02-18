"""Face recognition toolkit -- Agno-compatible Enton face enrollment tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agno.tools import Toolkit

if TYPE_CHECKING:
    from enton.perception.faces import FaceRecognizer
    from enton.perception.vision import Vision

logger = logging.getLogger(__name__)


class FaceTools(Toolkit):
    """Enrolls, lists, and removes faces from Enton's recognition database."""

    def __init__(self, vision: Vision, face_recognizer: FaceRecognizer | None) -> None:
        super().__init__(name="face_tools")
        self._vision = vision
        self._face_recognizer = face_recognizer
        self.register(self.enroll_face)
        self.register(self.list_faces)
        self.register(self.remove_face)

    def enroll_face(self, name: str) -> str:
        """Cadastra o rosto da pessoa visivel na camera com o nome dado.

        Deve haver exatamente 1 pessoa no frame para funcionar.

        Args:
            name: Nome da pessoa a ser cadastrada.
        """
        if self._face_recognizer is None:
            return "Face recognition nao disponivel."

        frame = self._vision.last_frame
        if frame is None:
            return "Sem frame disponivel -- camera pode estar desconectada."

        return self._face_recognizer.enroll(name, frame)

    def list_faces(self) -> str:
        """Lista todas as pessoas cadastradas no banco de rostos."""
        if self._face_recognizer is None:
            return "Face recognition nao disponivel."

        names = self._face_recognizer.list_enrolled()
        if not names:
            return "Nenhum rosto cadastrado ainda."
        return f"Rostos cadastrados ({len(names)}): {', '.join(names)}"

    def remove_face(self, name: str) -> str:
        """Remove uma pessoa do banco de rostos.

        Args:
            name: Nome da pessoa a remover.
        """
        if self._face_recognizer is None:
            return "Face recognition nao disponivel."

        if self._face_recognizer.remove(name):
            return f"Rosto de '{name}' removido."
        return f"'{name}' nao encontrado no banco de rostos."
