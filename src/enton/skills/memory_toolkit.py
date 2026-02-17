"""Memory management toolkit — search, recall, and user profile."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agno.tools import Toolkit

if TYPE_CHECKING:
    from enton.core.memory import Memory

logger = logging.getLogger(__name__)


class MemoryTools(Toolkit):
    """Tools for searching, recalling, and inspecting stored memories."""

    def __init__(self, memory: Memory) -> None:
        super().__init__(name="memory_tools")
        self._memory = memory
        self.register(self.search_memory)
        self.register(self.recall_recent)
        self.register(self.what_do_you_know_about_user)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def search_memory(self, query: str) -> str:
        """Busca nas memorias do Enton por algo relevante via busca semantica.

        Args:
            query: O que buscar nas memorias.
        """
        results = self._memory.semantic_search(query, n=5)
        if not results:
            return "Nenhuma memoria encontrada."

        lines = [f"{i + 1}. {r}" for i, r in enumerate(results)]
        return f"Encontrei {len(results)} memorias relevantes:\n" + "\n".join(lines)

    def recall_recent(self, n: int = 5) -> str:
        """Relembra os episodios mais recentes da memoria.

        Args:
            n: Numero de episodios a relembrar (default: 5).
        """
        episodes = self._memory.recall_recent(n)
        if not episodes:
            return "Sem memorias recentes."

        lines: list[str] = []
        for ep in episodes:
            lines.append(f"- [{ep.kind}] {ep.summary}")
        return "\n".join(lines)

    def what_do_you_know_about_user(self) -> str:
        """Retorna tudo que o Enton sabe sobre o usuario.

        Args:
            (nenhum)
        """
        profile = self._memory.profile

        parts = [
            f"Nome: {profile.name}",
            f"Relacionamento: {profile.relationship_score:.0%}",
        ]

        if profile.known_facts:
            facts = "\n".join(f"  - {f}" for f in profile.known_facts)
            parts.append(f"Fatos conhecidos:\n{facts}")

        if profile.preferences:
            prefs = "\n".join(
                f"  - {k}: {v}" for k, v in profile.preferences.items()
            )
            parts.append(f"Preferencias:\n{prefs}")

        return "\n".join(parts)
