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

import httpx
from agno.knowledge.embedder.ollama import OllamaEmbedder
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from enton.core.crawler_engine import Crawl4AIEngine

if TYPE_CHECKING:
    from enton.cognition.brain import EntonBrain

logger = logging.getLogger(__name__)

KNOWLEDGE_COLLECTION = "enton_knowledge"
EMBED_DIM = 768  # nomic-embed-text dimension
MAX_TEXT_LEN = 10000  # Increased for better context


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
        self._engine = Crawl4AIEngine()

    # -- initialization --

    def _init_qdrant(self) -> bool:
        """Initialize Qdrant collection for knowledge triples (lazy)."""
        if self._qdrant is not None:
            return True
        try:
            client = QdrantClient(url=self._qdrant_url, timeout=5)
            collections = [c.name for c in client.get_collections().collections]
            if KNOWLEDGE_COLLECTION not in collections:
                client.create_collection(
                    collection_name=KNOWLEDGE_COLLECTION,
                    vectors_config=VectorParams(
                        size=EMBED_DIM,
                        distance=Distance.COSINE,
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
            self._embedder = OllamaEmbedder(
                id="nomic-embed-text",
                dimensions=EMBED_DIM,
            )
            return self._embedder
        except Exception:
            logger.warning("Ollama embedder unavailable")
            return None

    # -- crawling --

    async def crawl_url(self, url: str) -> str:
        """Fetch URL content, extract markdown via Crawl4AI."""
        result = await self._engine.crawl(url)
        if result.get("error"):
            logger.warning("Failed to crawl %s: %s", url, result["error"])
            return ""

        markdown = result.get("markdown", "")
        if not markdown:
            # Fallback to HTML -> Text if markdown failed
            return ""

        return markdown[:MAX_TEXT_LEN]

    # -- extraction --

    async def extract_triples(
        self,
        text: str,
        source_url: str = "",
    ) -> list[KnowledgeTriple]:
        """Use LLM to extract knowledge triples from text."""
        if not self._brain or not text:
            return []

        from enton.cognition.prompts import KNOWLEDGE_EXTRACT_PROMPT

        prompt = KNOWLEDGE_EXTRACT_PROMPT.format(text=text[:3000])

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
                    triples.append(
                        KnowledgeTriple(
                            subject=str(item["subject"]),
                            predicate=str(item["predicate"]),
                            obj=str(item["obj"]),
                            source_url=source_url,
                        )
                    )
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

    async def learn_text(
        self, text: str, source: str = "internal_thought"
    ) -> list[KnowledgeTriple]:
        """Learn directly from provided text (e.g. from GitHubModule)."""
        if not text:
            return []

        triples = await self.extract_triples(text, source_url=source)
        if not triples:
            return []

        await self._store_triples(triples)
        logger.info("Learned %d triples from %s", len(triples), source)
        return triples

    async def learn_topic(self, topic: str, max_pages: int = 5) -> list[KnowledgeTriple]:
        """Deep research: Search web, parallel crawl, extract knowledge."""
        urls = await self._search_web(topic)
        target_urls = urls[:max_pages]

        if not target_urls:
            return []

        logger.info(f"Deep researching '{topic}' across {len(target_urls)} pages...")

        # Parallel crawl with Crawl4AI
        results = await self._engine.crawl_many(target_urls)

        all_triples: list[KnowledgeTriple] = []

        for res in results:
            url = res.get("url", "")
            markdown = res.get("markdown", "")

            if res.get("error") or not markdown:
                logger.warning(f"Failed to learn from {url}: {res.get('error')}")
                continue

            # Extract knowledge from content
            triples = await self.extract_triples(markdown, source_url=url)
            if triples:
                await self._store_triples(triples)
                all_triples.extend(triples)
                logger.info(f"Learned {len(triples)} facts from {url}")

        return all_triples

    async def _search_web(self, query: str) -> list[str]:
        """DuckDuckGo HTML search — returns list of URLs."""
        try:
            async with httpx.AsyncClient(
                timeout=10,
                follow_redirects=True,
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
            points = []
            for triple in triples:
                text = f"{triple.subject} {triple.predicate} {triple.obj}"
                resp = embedder.get_embedding(text)
                if resp is None:
                    continue
                embedding = resp

                self._triple_count += 1
                points.append(
                    PointStruct(
                        id=self._triple_count,
                        vector=embedding,
                        payload={
                            "subject": triple.subject,
                            "predicate": triple.predicate,
                            "obj": triple.obj,
                            "source_url": triple.source_url,
                        },
                    )
                )

            if points:
                self._qdrant.upsert(
                    collection_name=KNOWLEDGE_COLLECTION,
                    points=points,
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

            response = self._qdrant.query_points(
                collection_name=KNOWLEDGE_COLLECTION,
                query=embedding,
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
                for r in response.points
            ]
        except Exception:
            logger.warning("Knowledge search failed")
            return []
