"""Knowledge Crawler — httpx + BeautifulSoup for web learning.

Crawls URLs, extracts text, uses LLM to extract knowledge triples.
Stores triples in Qdrant collection 'enton_knowledge' with embeddings.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from enton.cognition.brain import EntonBrain

logger = logging.getLogger(__name__)

KNOWLEDGE_COLLECTION = "enton_knowledge"
EMBED_DIM = 768  # nomic-embed-text dimension
MAX_TEXT_LEN = 4000


@dataclass(frozen=True, slots=True)
class KnowledgeTriple:
    """A single knowledge fact: (subject, predicate, object)."""

    subject: str
    predicate: str
    obj: str
    source_url: str = ""
    confidence: float = 1.0


class KnowledgeCrawler:
    """Crawls web pages and extracts knowledge triples via LLM."""

    def __init__(
        self,
        brain: EntonBrain | None = None,
        qdrant_url: str = "http://localhost:6333",
    ) -> None:
        self._brain = brain
        self._qdrant_url = qdrant_url
        self._qdrant: Any = None
        self._embedder: Any = None
        self._triple_count = 0

    # -- initialization --

    def _init_qdrant(self) -> bool:
        """Initialize Qdrant collection for knowledge triples (lazy)."""
        if self._qdrant is not None:
            return True
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            client = QdrantClient(url=self._qdrant_url, timeout=5)
            collections = [c.name for c in client.get_collections().collections]
            if KNOWLEDGE_COLLECTION not in collections:
                client.create_collection(
                    collection_name=KNOWLEDGE_COLLECTION,
                    vectors_config=VectorParams(
                        size=EMBED_DIM, distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection '%s'", KNOWLEDGE_COLLECTION)
            self._qdrant = client
            return True
        except Exception:
            logger.warning("Qdrant unavailable for knowledge crawler")
            return False

    def _get_embedder(self) -> Any:
        """Return OllamaEmbedder for nomic-embed-text."""
        if self._embedder is not None:
            return self._embedder
        try:
            from agno.embedder.ollama import OllamaEmbedder

            self._embedder = OllamaEmbedder(
                id="nomic-embed-text", dimensions=EMBED_DIM,
            )
            return self._embedder
        except Exception:
            logger.warning("Ollama embedder unavailable")
            return None

    # -- crawling --

    async def crawl_url(self, url: str) -> str:
        """Fetch URL content, extract text via BeautifulSoup."""
        try:
            import httpx
            from bs4 import BeautifulSoup

            async with httpx.AsyncClient(
                timeout=15, follow_redirects=True,
            ) as client:
                resp = await client.get(url, headers={
                    "User-Agent": "Enton/0.3 (AI Assistant)",
                })
                resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove scripts, styles, nav
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            # Collapse whitespace
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text[:MAX_TEXT_LEN]
        except Exception:
            logger.warning("Failed to crawl %s", url)
            return ""

    # -- extraction --

    async def extract_triples(
        self, text: str, source_url: str = "",
    ) -> list[KnowledgeTriple]:
        """Use LLM to extract knowledge triples from text."""
        if not self._brain or not text:
            return []

        prompt = (
            "Extraia fatos do texto abaixo como JSON array de objetos com "
            '"subject", "predicate", "obj". Maximo 10 fatos. '
            "Retorne APENAS o JSON array, sem markdown.\n\n"
            f"Texto:\n{text[:3000]}"
        )

        try:
            response = await self._brain.think(prompt)
            if not response:
                return []

            # Strip markdown fences if present
            clean = response.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```\w*\n?", "", clean)
                clean = re.sub(r"\n?```$", "", clean)

            data = json.loads(clean)
            if not isinstance(data, list):
                return []

            triples = []
            for item in data[:10]:
                if all(k in item for k in ("subject", "predicate", "obj")):
                    triples.append(KnowledgeTriple(
                        subject=str(item["subject"]),
                        predicate=str(item["predicate"]),
                        obj=str(item["obj"]),
                        source_url=source_url,
                    ))
            return triples
        except (json.JSONDecodeError, Exception):
            logger.warning("Failed to extract triples from LLM response")
            return []

    # -- learning --

    async def learn_from_url(self, url: str) -> list[KnowledgeTriple]:
        """Crawl URL -> extract text -> extract triples -> store in Qdrant."""
        text = await self.crawl_url(url)
        if not text:
            return []

        triples = await self.extract_triples(text, source_url=url)
        if not triples:
            return []

        await self._store_triples(triples)
        logger.info("Learned %d triples from %s", len(triples), url)
        return triples

    async def learn_topic(self, topic: str) -> list[KnowledgeTriple]:
        """Search web for topic, crawl top results, extract triples."""
        urls = await self._search_web(topic)
        all_triples: list[KnowledgeTriple] = []
        for url in urls[:2]:
            triples = await self.learn_from_url(url)
            all_triples.extend(triples)
        return all_triples

    async def _search_web(self, query: str) -> list[str]:
        """DuckDuckGo HTML search — returns list of URLs."""
        try:
            import httpx
            from bs4 import BeautifulSoup

            async with httpx.AsyncClient(
                timeout=10, follow_redirects=True,
            ) as client:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "Enton/0.3 (AI Assistant)"},
                )

            soup = BeautifulSoup(resp.text, "html.parser")
            urls = []
            for link in soup.select("a.result__a"):
                href = link.get("href", "")
                if href.startswith("http"):
                    urls.append(href)
            return urls[:5]
        except Exception:
            logger.warning("Web search failed for '%s'", query)
            return []

    # -- storage --

    async def _store_triples(self, triples: list[KnowledgeTriple]) -> None:
        """Embed and store triples in Qdrant."""
        if not triples or not self._init_qdrant():
            return

        embedder = self._get_embedder()
        if embedder is None:
            return

        try:
            from qdrant_client.models import PointStruct

            points = []
            for triple in triples:
                text = f"{triple.subject} {triple.predicate} {triple.obj}"
                resp = embedder.get_embedding(text)
                if resp is None:
                    continue
                embedding = resp

                self._triple_count += 1
                points.append(PointStruct(
                    id=self._triple_count,
                    vector=embedding,
                    payload={
                        "subject": triple.subject,
                        "predicate": triple.predicate,
                        "obj": triple.obj,
                        "source_url": triple.source_url,
                    },
                ))

            if points:
                self._qdrant.upsert(
                    collection_name=KNOWLEDGE_COLLECTION, points=points,
                )
        except Exception:
            logger.warning("Failed to store triples in Qdrant")

    # -- search --

    async def search(self, query: str, n: int = 5) -> list[dict]:
        """Semantic search over knowledge triples."""
        if not self._init_qdrant():
            return []

        embedder = self._get_embedder()
        if embedder is None:
            return []

        try:
            embedding = embedder.get_embedding(query)
            if embedding is None:
                return []

            results = self._qdrant.search(
                collection_name=KNOWLEDGE_COLLECTION,
                query_vector=embedding,
                limit=n,
            )
            return [
                {
                    "subject": r.payload.get("subject", ""),
                    "predicate": r.payload.get("predicate", ""),
                    "obj": r.payload.get("obj", ""),
                    "source_url": r.payload.get("source_url", ""),
                    "score": r.score,
                }
                for r in results
            ]
        except Exception:
            logger.warning("Knowledge search failed")
            return []
