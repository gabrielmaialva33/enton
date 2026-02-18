"""Visual Memory toolkit — search and recall visual memories."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from agno.tools import Toolkit

if TYPE_CHECKING:
    from enton.core.visual_memory import VisualMemory

logger = logging.getLogger(__name__)


class VisualMemoryTools(Toolkit):
    """Tools for searching visual episodic memories."""

    def __init__(self, visual_memory: VisualMemory) -> None:
        super().__init__(name="visual_memory_tools")
        self._vm = visual_memory
        self.register(self.search_visual_memory)
        self.register(self.recall_recent_scenes)

    async def search_visual_memory(self, query: str) -> str:
        """Busca na memoria visual do Enton. Use para perguntas como 'onde esta minha caneca?'

        Args:
            query: O que buscar nas cenas visuais memorizadas.
        """
        results = await self._vm.search(query, n=5)
        if not results:
            return "Nenhuma memoria visual encontrada."

        lines: list[str] = []
        for i, r in enumerate(results, 1):
            ts = datetime.fromtimestamp(r["timestamp"]).strftime("%d/%m %H:%M")
            det = ", ".join(r["detections"]) if r["detections"] else "cena vazia"
            lines.append(
                f"{i}. [{ts}] {det} (camera: {r['camera_id']}, "
                f"relevancia: {r.get('score', 0):.0%})",
            )
        return f"Encontrei {len(results)} memorias visuais:\n" + "\n".join(lines)

    async def recall_recent_scenes(self, n: int = 3) -> str:
        """Relembra as cenas visuais mais recentes.

        Args:
            n: Numero de cenas a relembrar (default: 3).
        """
        results = await self._vm.recent_scenes(n=n)
        if not results:
            return "Sem memorias visuais recentes."

        lines: list[str] = []
        for r in results:
            ts = datetime.fromtimestamp(r["timestamp"]).strftime("%d/%m %H:%M")
            det = ", ".join(r["detections"]) if r["detections"] else "cena vazia"
            lines.append(f"- [{ts}] {det} (camera: {r['camera_id']})")
        return "\n".join(lines)
