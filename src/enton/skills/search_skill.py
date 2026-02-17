import logging

from duckduckgo_search import DDGS

from enton.core.tools import tool

logger = logging.getLogger(__name__)

@tool
def search_web(query: str) -> str:
    """Pesquisa na web usando DuckDuckGo e retorna os principais resultados.
    
    Args:
        query: O termo de busca ou pergunta.
    """
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "Nenhum resultado encontrado."
        
        summary = []
        for r in results:
            summary.append(f"- {r['title']}: {r['body']} ({r['href']})")
        
        return "\n".join(summary)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Erro na busca: {e}"
