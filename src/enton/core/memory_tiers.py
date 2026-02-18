"""4-Tier Memory Architecture (RoboMemory / ICLR 2026 inspired).

Tiers:
  1. Spatial  — object locations from DetectionEvents
  2. Temporal — time-based patterns (from DreamMode)
  3. Episodic — text episodes (Memory) + visual episodes (VisualMemory)
  4. Semantic — knowledge triples + commonsense + user profile

Unified search queries all tiers and ranks results.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enton.core.commonsense import CommonsenseKB
    from enton.core.knowledge_crawler import KnowledgeCrawler
    from enton.core.memory import Memory
    from enton.core.visual_memory import VisualMemory

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ObjectLocation:
    """Tracks where an object was last seen."""

    label: str
    camera_id: str
    bbox: tuple[int, int, int, int]
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.0


@dataclass(slots=True)
class TemporalPattern:
    """A detected time-based pattern."""

    description: str
    hour: int
    tag: str
    count: int
    last_seen: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class TierResult:
    """A single result from any tier."""

    tier: str  # "spatial", "temporal", "episodic", "semantic"
    content: str
    score: float = 1.0
    metadata: dict = field(default_factory=dict)


class MemoryTiers:
    """4-tier memory coordinator. Unifies spatial, temporal, episodic, semantic."""

    MAX_OBJECTS = 200
    MAX_PATTERNS = 100

    def __init__(
        self,
        memory: Memory,
        visual_memory: VisualMemory | None = None,
        knowledge: KnowledgeCrawler | None = None,
        commonsense: CommonsenseKB | None = None,
    ) -> None:
        self._memory = memory
        self._visual = visual_memory
        self._knowledge = knowledge
        self._commonsense = commonsense
        self._spatial: dict[str, ObjectLocation] = {}
        self._patterns: list[TemporalPattern] = []

    # -- Spatial Tier --

    def update_object_location(
        self,
        label: str,
        camera_id: str,
        bbox: tuple[int, int, int, int],
        confidence: float = 0.0,
    ) -> None:
        """Update spatial map from a DetectionEvent."""
        self._spatial[label] = ObjectLocation(
            label=label,
            camera_id=camera_id,
            bbox=bbox,
            timestamp=time.time(),
            confidence=confidence,
        )
        # Evict oldest if too many
        if len(self._spatial) > self.MAX_OBJECTS:
            oldest_key = min(
                self._spatial,
                key=lambda k: self._spatial[k].timestamp,
            )
            del self._spatial[oldest_key]

    def where_is(self, label: str) -> ObjectLocation | None:
        """Query spatial tier: where was this object last seen?"""
        return self._spatial.get(label)

    def all_objects(self) -> list[ObjectLocation]:
        """Return all tracked object locations."""
        return list(self._spatial.values())

    # -- Temporal Tier --

    def add_pattern(self, pattern: TemporalPattern) -> None:
        """Add a temporal pattern (called from DreamMode)."""
        self._patterns.append(pattern)
        if len(self._patterns) > self.MAX_PATTERNS:
            self._patterns = self._patterns[-self.MAX_PATTERNS :]

    def patterns_for_hour(self, hour: int) -> list[TemporalPattern]:
        """Get temporal patterns active around a given hour."""
        return [p for p in self._patterns if p.hour == hour]

    # -- Unified Search --

    async def search(self, query: str, n: int = 5) -> list[TierResult]:
        """Search all tiers, merge and rank results."""
        results: list[TierResult] = []

        # Spatial: keyword match
        q_lower = query.lower()
        for label, loc in self._spatial.items():
            if label.lower() in q_lower or q_lower in label.lower():
                age = time.time() - loc.timestamp
                age_str = f"{int(age)}s ago" if age < 60 else f"{int(age / 60)}min ago"
                results.append(
                    TierResult(
                        tier="spatial",
                        content=f"{label} seen on camera '{loc.camera_id}' ({age_str})",
                        score=loc.confidence,
                        metadata={"camera_id": loc.camera_id, "bbox": loc.bbox},
                    )
                )

        # Temporal: keyword match
        for pattern in self._patterns:
            if pattern.tag.lower() in q_lower or q_lower in pattern.description.lower():
                results.append(
                    TierResult(
                        tier="temporal",
                        content=pattern.description,
                        score=pattern.count / 100.0,
                    )
                )

        # Episodic + Visual + Semantic: run in parallel
        tasks = []

        # Episodic text search
        async def _episodic() -> list[TierResult]:
            hits = self._memory.semantic_search(query, n=n)
            return [TierResult(tier="episodic", content=h, score=0.7) for h in hits]

        tasks.append(_episodic())

        # Visual search
        if self._visual is not None:

            async def _visual() -> list[TierResult]:
                hits = await self._visual.search(query, n=n)
                return [
                    TierResult(
                        tier="episodic",
                        content=(
                            f"Visual: {h['detections']} "
                            f"({h['camera_id']}, "
                            f"{datetime.fromtimestamp(h['timestamp']).strftime('%H:%M')})"
                        ),
                        score=h.get("score", 0.5),
                        metadata=h,
                    )
                    for h in hits
                ]

            tasks.append(_visual())

        # Knowledge search
        if self._knowledge is not None:

            async def _knowledge() -> list[TierResult]:
                hits = await self._knowledge.search(query, n=n)
                return [
                    TierResult(
                        tier="semantic",
                        content=f"{h['subject']} {h['predicate']} {h['obj']}",
                        score=h.get("score", 0.5),
                        metadata=h,
                    )
                    for h in hits
                ]

            tasks.append(_knowledge())

        # Commonsense search
        if self._commonsense is not None and self._commonsense.available:

            async def _commonsense() -> list[TierResult]:
                hits = await self._commonsense.search(query, n=n)
                return [
                    TierResult(
                        tier="semantic",
                        content=f"{h['subject']} {h['predicate']} {h['obj']}",
                        score=h.get("score", 0.3),
                        metadata=h,
                    )
                    for h in hits
                ]

            tasks.append(_commonsense())

        # Gather all async searches
        if tasks:
            tier_results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in tier_results:
                if isinstance(res, list):
                    results.extend(res)

        # Sort by score descending, limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: n * 2]

    # -- context --

    def context_string(self, include_spatial: bool = True) -> str:
        """Build a context string from all tiers for brain system prompt."""
        parts: list[str] = []

        if include_spatial and self._spatial:
            recent = sorted(
                self._spatial.values(),
                key=lambda o: o.timestamp,
                reverse=True,
            )[:5]
            objs = ", ".join(f"{o.label}@{o.camera_id}" for o in recent)
            parts.append(f"Objects: {objs}")

        hour = datetime.now().hour
        hour_patterns = self.patterns_for_hour(hour)
        if hour_patterns:
            pats = "; ".join(p.description for p in hour_patterns[:3])
            parts.append(f"Patterns: {pats}")

        return " | ".join(parts) if parts else ""

    # -- serialization --

    def to_dict(self) -> dict:
        return {
            "spatial_objects": len(self._spatial),
            "temporal_patterns": len(self._patterns),
            "visual_memory": self._visual is not None,
            "knowledge": self._knowledge is not None,
            "commonsense": (self._commonsense.available if self._commonsense else False),
        }
