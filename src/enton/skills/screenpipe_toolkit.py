"""Screenpipe Toolkit — Digital Eyes for Enton."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from agno.tools import Toolkit

from enton.core.config import settings

logger = logging.getLogger(__name__)


class ScreenpipeTools(Toolkit):
    """Tools for searching and retrieving screen context via Screenpipe."""

    def __init__(self) -> None:
        super().__init__(name="screenpipe_tools")
        self.register(self.search_screen)
        self.register(self.get_recent_activity)

    def search_screen(
        self,
        query: str,
        limit: int = 5,
        content_type: str = "ocr",
        minutes_back: int | None = None,
    ) -> str:
        """Pesquisa por texto ou audio no historico de tela do usuario (Screenpipe).

        Use isso para saber o que o usuario viu, leu ou conversou recentemente.
        Ex: "O que eu estava vendo sobre Porsche?", "Qual foi o erro que apareceu no terminal?"

        Args:
            query: O termo de busca (texto que apareceu na tela ou foi falado).
            limit: Numero maximo de resultados (default: 5).
            content_type: Tipo de conteudo: "ocr" (texto na tela), "audio" (fala), ou "all".
            minutes_back: Quantos minutos para tras pesquisar. Se None, busca em tudo.
        """
        try:
            import httpx
        except ImportError:
            return "Erro: httpx nao instalado."

        try:
            params: dict[str, Any] = {
                "q": query,
                "limit": limit,
                "content_type": content_type if content_type != "all" else None,
            }
            
            if minutes_back:
                # Screenpipe expects ISO formatted start_time if filtering by time
                start_time = (datetime.now() - timedelta(minutes=minutes_back)).isoformat()
                params["start_time"] = start_time

            # Screenpipe search endpoint: /search?q=...
            # Remove trailing slash from base if present to avoid double slash
            base_url = settings.screenpipe_url.rstrip("/")
            url = f"{base_url}/search"
            
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                payload = resp.json()

            # The structure of response depends on Screenpipe version.
            # Assuming standard response format: { "data": [ { "content": ..., "timestamp": ... } ] }
            results = payload.get("data", [])
            if not results:
                return f"Nenhum resultado encontrado para '{query}'."

            formatted = []
            for item in results:
                content_obj = item.get("content", {})
                # Try to get text or transcription
                text = content_obj.get("text") or content_obj.get("transcription") or ""
                
                meta = item.get("meta", {})
                app_name = meta.get("app_name", "unknown")
                window_name = meta.get("window_name", "unknown")
                timestamp = item.get("timestamp", "unknown")
                
                formatted.append(
                    f"- [{timestamp}] [{app_name}] {window_name}: {text[:200].replace('\n', ' ')}..."
                )

            return "\n".join(formatted)

        except Exception as e:
            logger.error("Screenpipe search failed: %s", e)
            return f"Erro ao buscar no Screenpipe: {e}"

    def get_recent_activity(self, minutes: int = 5) -> str:
        """Recupera o contexto do que o usuario fez nos ultimos minutos.

        Args:
            minutes: Quantos minutos atras pesquisar (default: 5).
        """
        # Empty query usually returns latest items
        return self.search_screen(query="", limit=20, minutes_back=minutes)
