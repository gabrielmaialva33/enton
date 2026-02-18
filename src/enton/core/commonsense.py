"""Commonsense KB — ASCENT++ query interface.

Lazy initialization: only activates if 'enton_commonsense' Qdrant collection
exists (populated by scripts/load_commonsense.py).
"""

from __future__ import annotations

import logging
from typing import Any

from agno.knowledge.embedder.ollama import OllamaEmbedder
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

COMMONSENSE_COLLECTION = "enton_commonsense"
EMBED_DIM = 768


class CommonsenseKB:
    """Runtime query interface for ASCENT++ commonsense knowledge."""

    def __init__(self, qdrant_url: str = "http://localhost:6333") -> None:
        self._qdrant_url = qdrant_url
        self._qdrant: Any = None
        self._embedder: Any = None
        self._available: bool | None = None

    def _init_qdrant(self) -> bool:
        """Check if commonsense collection exists."""
        if self._qdrant is not None:
            return self._available or False
        try:
            client = QdrantClient(url=self._qdrant_url, timeout=5)
            collections = [c.name for c in client.get_collections().collections]
            if COMMONSENSE_COLLECTION in collections:
                self._qdrant = client
                self._available = True
                logger.info("Commonsense KB available (%s)", COMMONSENSE_COLLECTION)
                return True
            self._available = False
            return False
        except Exception:
            self._available = False
            return False

    @property
    def available(self) -> bool:
        """Whether the commonsense collection is populated and ready."""
        if self._available is None:
            self._init_qdrant()
        return self._available or False

    def _get_embedder(self) -> Any:
        """Return OllamaEmbedder for nomic-embed-text."""
        if self._embedder is not None:
            return self._embedder
        try:
            self._embedder = OllamaEmbedder(
                id="nomic-embed-text",
                dimensions=EMBED_DIM,
            )
            return self._embedder
        except Exception:
            return None

    async def search(self, query: str, n: int = 5) -> list[dict]:
        """Search commonsense assertions semantically."""
        if not self._init_qdrant():
            return []

        embedder = self._get_embedder()
        if embedder is None:
            return []

        try:
            embedding = embedder.get_embedding(query)
            if embedding is None:
                return []

            response = self._qdrant.query_points(
                collection_name=COMMONSENSE_COLLECTION,
                query=embedding,
                limit=n,
            )
            return [
                {
                    "subject": r.payload.get("subject", ""),
                    "predicate": r.payload.get("predicate", ""),
                    "obj": r.payload.get("obj", ""),
                    "score": r.score,
                }
                for r in response.points
            ]
        except Exception:
            logger.warning("Commonsense search failed")
            return []

    async def what_is(self, concept: str, n: int = 3) -> list[str]:
        """Get commonsense facts about a concept as natural language."""
        results = await self.search(concept, n=n)
        return [f"{r['subject']} {r['predicate']} {r['obj']}" for r in results]
