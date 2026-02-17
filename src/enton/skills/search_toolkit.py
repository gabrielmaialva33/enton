"""Agno Toolkit for web search via DuckDuckGo (using httpx)."""

from __future__ import annotations

import logging
import re
from html import unescape

from agno.tools import Toolkit

logger = logging.getLogger(__name__)

_DDG_URL = "https://html.duckduckgo.com/html/"
_MAX_RESULTS = 3
_TIMEOUT = 10.0

# Regex patterns to extract results from DuckDuckGo HTML response
_RESULT_LINK_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_RESULT_SNIPPET_RE = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(html: str) -> str:
    """Remove HTML tags and decode entities."""
    return unescape(_TAG_RE.sub("", html)).strip()


def _extract_url(raw_url: str) -> str:
    """Extract actual URL from DuckDuckGo redirect URL."""
    # DuckDuckGo wraps URLs like //duckduckgo.com/l/?uddg=<encoded_url>&...
    if "uddg=" in raw_url:
        from urllib.parse import parse_qs, unquote, urlparse

        parsed = urlparse(raw_url)
        qs = parse_qs(parsed.query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    return raw_url


class SearchTools(Toolkit):
    """Web search using DuckDuckGo HTML endpoint (no external search lib needed)."""

    def __init__(self, max_results: int = _MAX_RESULTS):
        super().__init__(name="search_tools")
        self.max_results = max_results
        self.register(self.search_web)

    def search_web(self, query: str) -> str:
        """Pesquisa na web usando DuckDuckGo e retorna os principais resultados.

        Faz uma busca web e retorna titulo, resumo e URL de cada resultado.
        Util para responder perguntas sobre fatos atuais, noticias, ou qualquer
        informacao que o modelo nao tenha.

        Args:
            query: O termo de busca ou pergunta.
        """
        try:
            import httpx
        except ImportError:
            return "Erro: httpx nao esta instalado. Instale com: pip install httpx"

        try:
            resp = httpx.post(
                _DDG_URL,
                data={"q": query, "b": ""},
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
                timeout=_TIMEOUT,
                follow_redirects=True,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.error("Search request failed: %s", e)
            return f"Erro na busca: {e}"

        html = resp.text

        # Extract links
        links = _RESULT_LINK_RE.findall(html)
        snippets = _RESULT_SNIPPET_RE.findall(html)

        if not links:
            return "Nenhum resultado encontrado."

        results: list[str] = []
        for i, (raw_url, raw_title) in enumerate(links[: self.max_results]):
            title = _strip_tags(raw_title)
            url = _extract_url(raw_url)
            body = _strip_tags(snippets[i]) if i < len(snippets) else ""
            results.append(f"- {title}: {body} ({url})")

        return "\n".join(results)
