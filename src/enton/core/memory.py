"""Memory — Enton's episodic memory and user profile.

Uses Agno Knowledge (Qdrant + Ollama embeddings) for semantic search.
JSONL file as durable backup. User profile persisted as JSON.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from enton.core.config import settings

logger = logging.getLogger(__name__)

MEMORY_DIR = Path.home() / ".enton"
EPISODES_FILE = MEMORY_DIR / "episodes.jsonl"
PROFILE_FILE = MEMORY_DIR / "profile.json"


@dataclass(frozen=True, slots=True)
class Episode:
    kind: str  # conversation, detection, system, observation
    summary: str
    timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class UserProfile:
    name: str = "Gabriel"
    known_facts: list[str] = field(default_factory=list)
    preferences: dict[str, str] = field(default_factory=dict)
    relationship_score: float = 0.5  # 0=stranger, 1=best friend


def create_knowledge():
    """Create Agno Knowledge backed by Qdrant + Ollama embeddings."""
    try:
        from agno.embedder.ollama import OllamaEmbedder
        from agno.knowledge import AgentKnowledge
        from agno.vectordb.qdrant import Qdrant

        vector_db = Qdrant(
            collection="enton_episodes",
            embedder=OllamaEmbedder(id="nomic-embed-text", dimensions=768),
            url=settings.qdrant_url,
        )
        knowledge = AgentKnowledge(vector_db=vector_db)
        logger.info("Agno Knowledge + Qdrant initialized")
        return knowledge
    except Exception:
        logger.warning("Agno Knowledge unavailable (Qdrant/Ollama embedder)")
        return None


class Memory:
    def __init__(self, max_recent: int = 50) -> None:
        self._max_recent = max_recent
        self._episodes: list[Episode] = []
        self.profile = UserProfile()
        self._knowledge = None
        self._ensure_dir()
        self._load()
        self._init_knowledge()

    def _ensure_dir(self) -> None:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    def _init_knowledge(self) -> None:
        """Initialize Agno Knowledge for semantic search."""
        self._knowledge = create_knowledge()

    @property
    def knowledge(self):
        """Agno AgentKnowledge instance (or None)."""
        return self._knowledge

    def _load(self) -> None:
        if EPISODES_FILE.exists():
            try:
                lines = EPISODES_FILE.read_text().strip().splitlines()
                for line in lines[-self._max_recent :]:
                    data = json.loads(line)
                    self._episodes.append(Episode(**data))
                logger.info("Loaded %d episodes from memory", len(self._episodes))
            except Exception:
                logger.warning("Failed to load episodes, starting fresh")

        if PROFILE_FILE.exists():
            try:
                data = json.loads(PROFILE_FILE.read_text())
                self.profile = UserProfile(**data)
                logger.info("Loaded user profile: %s", self.profile.name)
            except Exception:
                logger.warning("Failed to load profile, using defaults")

    def remember(self, episode: Episode) -> None:
        self._episodes.append(episode)
        if len(self._episodes) > self._max_recent * 2:
            self._episodes = self._episodes[-self._max_recent :]

        # JSONL persistence (always — durable backup)
        try:
            with EPISODES_FILE.open("a") as f:
                f.write(json.dumps(asdict(episode), ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("Failed to persist episode")

        # Agno Knowledge indexing (if available)
        if self._knowledge is not None:
            try:
                self._knowledge.load_text(
                    episode.summary,
                    upsert=True,
                )
            except Exception:
                logger.debug("Knowledge indexing failed for episode")

    def recall_recent(self, n: int = 5) -> list[Episode]:
        return self._episodes[-n:]

    def recent(self, n: int = 5) -> list[Episode]:
        """Alias for recall_recent."""
        return self.recall_recent(n)

    def recall_by_kind(self, kind: str, n: int = 5) -> list[Episode]:
        matching = [e for e in self._episodes if e.kind == kind]
        return matching[-n:]

    def recall_by_tag(self, tag: str, n: int = 5) -> list[Episode]:
        matching = [e for e in self._episodes if tag in e.tags]
        return matching[-n:]

    def semantic_search(self, query: str, n: int = 5) -> list[str]:
        """Search memories semantically via Qdrant. Falls back to keyword."""
        if self._knowledge is not None:
            try:
                results = self._knowledge.search(query=query, num_documents=n)
                return [doc.content for doc in results if doc.content]
            except Exception:
                logger.debug("Knowledge search failed, falling back to keyword")

        # Keyword fallback
        query_lower = query.lower()
        matches = []
        for ep in reversed(self._episodes):
            if query_lower in ep.summary.lower():
                matches.append(ep.summary)
                if len(matches) >= n:
                    break
        return matches

    def learn_about_user(self, fact: str) -> None:
        if fact not in self.profile.known_facts:
            self.profile.known_facts.append(fact)
            self._save_profile()
        # Also index in Knowledge
        if self._knowledge is not None:
            try:
                self._knowledge.load_text(f"Fato sobre {self.profile.name}: {fact}")
            except Exception:
                logger.debug("Knowledge fact indexing failed")

    def strengthen_relationship(self, amount: float = 0.02) -> None:
        self.profile.relationship_score = min(
            1.0, self.profile.relationship_score + amount,
        )
        self._save_profile()

    def _save_profile(self) -> None:
        try:
            data = {
                "name": self.profile.name,
                "known_facts": self.profile.known_facts,
                "preferences": self.profile.preferences,
                "relationship_score": self.profile.relationship_score,
            }
            PROFILE_FILE.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
            )
        except Exception:
            logger.warning("Failed to save user profile")

    def context_string(self) -> str:
        parts = []
        recent = self.recall_recent(3)
        if recent:
            summaries = [e.summary for e in recent]
            parts.append(f"Recent memories: {'; '.join(summaries)}")
        if self.profile.known_facts:
            facts = self.profile.known_facts[-5:]
            parts.append(
                f"Known about {self.profile.name}: {', '.join(facts)}",
            )
        rel = self.profile.relationship_score
        if rel >= 0.8:
            parts.append(f"{self.profile.name} is my best friend")
        elif rel >= 0.5:
            parts.append(f"{self.profile.name} is a good friend")
        elif rel >= 0.3:
            parts.append(f"Getting to know {self.profile.name}")
        return " | ".join(parts) if parts else "No memories yet — fresh start"
