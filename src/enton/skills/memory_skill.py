from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from enton.core.tools import tool

if TYPE_CHECKING:
    from enton.core.memory import Memory

logger = logging.getLogger(__name__)

_memory: Memory | None = None


def init(memory: Memory) -> None:
    """Initialize with app memory. Called from App.__init__."""
    global _memory
    _memory = memory


@tool
def search_memory(query: str) -> str:
    """Busca nas memórias do Enton por algo relevante.

    Usa busca semântica (Qdrant) se disponível, senão keyword.

    Args:
        query: O que buscar nas memórias.
    """
    if _memory is None:
        return "Sistema de memória não inicializado."

    results = _memory.semantic_search(query, n=5)
    if not results:
        return "Não encontrei nada nas minhas memórias sobre isso."

    formatted = "\n".join(f"- {r}" for r in results)
    return f"Encontrei {len(results)} memórias relevantes:\n{formatted}"


@tool
def recall_recent(n: int = 5) -> str:
    """Relembra os episódios mais recentes da memória.

    Args:
        n: Número de episódios a relembrar (default 5).
    """
    if _memory is None:
        return "Sistema de memória não inicializado."

    episodes = _memory.recall_recent(n)
    if not episodes:
        return "Sem memórias recentes."

    parts = []
    for ep in episodes:
        parts.append(f"- [{ep.kind}] {ep.summary}")
    return "\n".join(parts)


@tool
def what_do_you_know_about_user() -> str:
    """Retorna tudo que o Enton sabe sobre o usuário."""
    if _memory is None:
        return "Sistema de memória não inicializado."

    profile = _memory.profile
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
        parts.append(f"Preferências:\n{prefs}")

    return "\n".join(parts)
