"""Knowledge toolkit — web learning and knowledge search."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agno.tools import Toolkit

if TYPE_CHECKING:
    from enton.core.knowledge_crawler import KnowledgeCrawler

logger = logging.getLogger(__name__)


class KnowledgeTools(Toolkit):
    """Tools for web-based knowledge acquisition and search."""

    def __init__(self, crawler: KnowledgeCrawler) -> None:
        super().__init__(name="knowledge_tools")
        self._crawler = crawler
        self.register(self.learn_from_url)
        self.register(self.search_knowledge)
        self.register(self.learn_about_topic)
        self.register(self.deep_research)

    async def learn_from_url(self, url: str) -> str:
        """Aprende conhecimento de uma URL da web. Extrai fatos e armazena.

        Args:
            url: URL completa para aprender (ex: https://docs.python.org/3/library/asyncio.html).
        """
        triples = await self._crawler.learn_from_url(url)
        if not triples:
            return "Nao consegui extrair conhecimento desta URL."

        lines = [f"- {t.subject} {t.predicate} {t.obj}" for t in triples[:5]]
        return (
            f"Aprendi {len(triples)} fatos:\n"
            + "\n".join(lines)
            + (f"\n... e mais {len(triples) - 5}" if len(triples) > 5 else "")
        )

    async def search_knowledge(self, query: str) -> str:
        """Busca no banco de conhecimento do Enton (fatos aprendidos da web).

        Args:
            query: O que buscar no conhecimento armazenado.
        """
        results = await self._crawler.search(query, n=5)
        if not results:
            return "Nenhum conhecimento encontrado sobre isso."

        lines: list[str] = []
        for i, r in enumerate(results, 1):
            src = f" (fonte: {r['source_url']})" if r.get("source_url") else ""
            lines.append(
                f"{i}. {r['subject']} {r['predicate']} {r['obj']}{src}",
            )
        return "\n".join(lines)

    async def learn_about_topic(self, topic: str) -> str:
        """Pesquisa e aprende sobre um topico. Busca na web e extrai conhecimento.

        Args:
            topic: Topico para aprender (ex: 'Python asyncio', 'fotossintese').
        """
        triples = await self._crawler.learn_topic(topic, max_pages=3)
        if not triples:
            return f"Nao consegui aprender sobre '{topic}' agora."

        lines = [f"- {t.subject} {t.predicate} {t.obj}" for t in triples[:8]]
        return f"Aprendi {len(triples)} fatos sobre '{topic}':\n" + "\n".join(lines)

    async def deep_research(self, topic: str) -> str:
        """Realiza uma pesquisa profunda sobre um tema (Deep Research).

        Busca em multiplas fontes, le o conteudo completo e sintetiza um conhecimento denso.
        Use isso quando precisar de informacoes detalhadas ou complexas.

        Args:
            topic: Topico para pesquisar profundamente.
        """
        triples = await self._crawler.learn_topic(topic, max_pages=10)
        if not triples:
            return f"Falha na pesquisa profunda sobre '{topic}'."

        # Return more details for deep research
        lines = [f"- {t.subject} {t.predicate} {t.obj}" for t in triples[:20]]
        return (
            f"**Deep Research concluido**: {len(triples)} fatos extraidos sobre '{topic}'.\n"
            "Principais descobertas:\n" + "\n".join(lines)
        )
