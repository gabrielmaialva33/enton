"""Vision description toolkit — scene description and detection summary."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agno.tools import Toolkit

if TYPE_CHECKING:
    from enton.cognition.brain import EntonBrain
    from enton.perception.vision import Vision

from enton.cognition.prompts import DESCRIBE_TOOL_DEFAULT, DESCRIBE_TOOL_SYSTEM

logger = logging.getLogger(__name__)


class DescribeTools(Toolkit):
    """Tools that let the agent describe what the camera sees."""

    def __init__(self, vision: Vision, brain: EntonBrain | None = None) -> None:
        super().__init__(name="describe_tools")
        self._vision = vision
        self._brain = brain
        self.register(self.describe_scene)
        self.register(self.what_do_you_see)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    async def describe_scene(self, prompt: str = DESCRIBE_TOOL_DEFAULT) -> str:
        """Descreve a cena atual da camera usando visao computacional + VLM.

        Captura o frame atual e pede ao modelo de visao para descrever.

        Args:
            prompt: Instrucao para o modelo de visao sobre o que descrever.
        """
        jpeg = self._vision.get_frame_jpeg()
        if jpeg is None:
            return "Camera nao disponivel."

        try:
            description = await self._brain.describe_scene(
                jpeg,
                system=DESCRIBE_TOOL_SYSTEM,
            )
            return description or "Nao consegui descrever a cena."
        except Exception as exc:
            logger.exception("describe_scene failed")
            return f"Erro ao descrever cena: {exc}"

    async def what_do_you_see(self) -> str:
        """Retorna uma lista das deteccoes atuais da camera (objetos, atividades, emocoes).

        Args:
            (nenhum)
        """
        detections = self._vision.last_detections
        activities = self._vision.last_activities
        emotions = self._vision.last_emotions

        if not detections and not activities and not emotions:
            return "Nao estou vendo nada no momento."

        parts: list[str] = []

        if detections:
            counts: dict[str, int] = {}
            for d in detections:
                counts[d.label] = counts.get(d.label, 0) + 1
            obj_str = ", ".join(f"{v}x {k}" for k, v in counts.items())
            parts.append(f"Objetos: {obj_str}")

        if activities:
            act_str = ", ".join(f"pessoa {a.person_index}: {a.activity}" for a in activities)
            parts.append(f"Atividades: {act_str}")

        if emotions:
            emo_str = ", ".join(
                f"pessoa {e.person_index}: {e.emotion} ({e.score:.0%})" for e in emotions
            )
            parts.append(f"Emocoes: {emo_str}")

        return " | ".join(parts)
