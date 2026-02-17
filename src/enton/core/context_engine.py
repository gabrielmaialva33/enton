"""Context Engine — Enton's smart context management.

Inspired by goose's context efficiency strategies, wcgw's context
checkpointing, and LangChain's context engineering patterns.

Key features:
- Context checkpointing: save/restore conversation state
- Relevance scoring: prioritize important context
- Summarization triggers: compress when approaching limits
- Token-aware budget management
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Rough token estimation: 1 token ≈ 4 chars for English, ~3 for Portuguese
_CHARS_PER_TOKEN = 3.5


@dataclass
class ContextEntry:
    """A single piece of context with metadata."""

    key: str  # unique identifier
    content: str
    category: str  # "sensor", "memory", "tool_result", "conversation", "system"
    priority: float = 0.5  # 0.0 = low, 1.0 = critical
    timestamp: float = field(default_factory=time.time)
    token_estimate: int = 0
    ttl: float = 0.0  # 0 = no expiry, else seconds until stale

    def __post_init__(self) -> None:
        if self.token_estimate == 0:
            self.token_estimate = int(len(self.content) / _CHARS_PER_TOKEN)

    @property
    def is_stale(self) -> bool:
        if self.ttl <= 0:
            return False
        return (time.time() - self.timestamp) > self.ttl

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def relevance_score(self) -> float:
        """Combined relevance: priority + recency decay."""
        age = self.age_seconds
        # Recency decay: halves every 5 minutes
        recency = 1.0 / (1.0 + age / 300.0)
        return self.priority * 0.7 + recency * 0.3


@dataclass
class Checkpoint:
    """Saved context state for restore."""

    id: str
    name: str
    entries: list[dict]
    metadata: dict
    created_at: float = field(default_factory=time.time)


class ContextEngine:
    """Manages context budget for LLM interactions.

    Keeps track of all context pieces (sensor data, memories, tool results),
    scores them by relevance, and trims to fit the token budget.
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self._entries: dict[str, ContextEntry] = {}
        self._max_tokens = max_tokens
        self._checkpoint_dir = checkpoint_dir
        self._checkpoints: dict[str, Checkpoint] = {}
        self._total_compressions = 0

        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Context Management
    # ------------------------------------------------------------------ #

    def set(
        self,
        key: str,
        content: str,
        category: str = "system",
        priority: float = 0.5,
        ttl: float = 0.0,
    ) -> None:
        """Set/update a context entry."""
        self._entries[key] = ContextEntry(
            key=key,
            content=content,
            category=category,
            priority=priority,
            ttl=ttl,
        )

    def remove(self, key: str) -> bool:
        """Remove a context entry."""
        return self._entries.pop(key, None) is not None

    def get(self, key: str) -> str | None:
        """Get context content by key."""
        entry = self._entries.get(key)
        if entry and not entry.is_stale:
            return entry.content
        return None

    def _cleanup_stale(self) -> int:
        """Remove expired entries. Returns count removed."""
        stale = [k for k, v in self._entries.items() if v.is_stale]
        for k in stale:
            del self._entries[k]
        return len(stale)

    # ------------------------------------------------------------------ #
    # Budget & Assembly
    # ------------------------------------------------------------------ #

    @property
    def current_tokens(self) -> int:
        """Estimated total tokens in all entries."""
        return sum(e.token_estimate for e in self._entries.values())

    @property
    def budget_used_pct(self) -> float:
        """Percentage of token budget used."""
        if self._max_tokens <= 0:
            return 0.0
        return min(100.0, self.current_tokens / self._max_tokens * 100)

    @property
    def is_over_budget(self) -> bool:
        return self.current_tokens > self._max_tokens

    def assemble(self, extra_budget: int = 0) -> str:
        """Assemble context string within token budget.

        Entries are sorted by relevance score. Lower-priority entries
        are dropped if we exceed the budget.
        """
        self._cleanup_stale()
        budget = self._max_tokens + extra_budget

        # Sort by relevance (highest first)
        entries = sorted(
            self._entries.values(),
            key=lambda e: e.relevance_score(),
            reverse=True,
        )

        parts: list[str] = []
        used = 0
        for entry in entries:
            if used + entry.token_estimate > budget:
                continue  # skip low-relevance entries
            parts.append(f"[{entry.category}:{entry.key}] {entry.content}")
            used += entry.token_estimate

        return "\n".join(parts)

    def assemble_by_category(
        self, categories: list[str] | None = None,
    ) -> dict[str, str]:
        """Assemble context grouped by category."""
        self._cleanup_stale()
        groups: dict[str, list[str]] = {}
        for entry in self._entries.values():
            if categories and entry.category not in categories:
                continue
            groups.setdefault(entry.category, []).append(entry.content)
        return {cat: "\n".join(items) for cat, items in groups.items()}

    # ------------------------------------------------------------------ #
    # Checkpointing (wcgw-inspired)
    # ------------------------------------------------------------------ #

    def checkpoint(self, name: str, metadata: dict | None = None) -> str:
        """Save current context state as a checkpoint.

        Returns checkpoint ID.
        """
        cp_id = uuid.uuid4().hex[:12]
        entries_data = [
            {
                "key": e.key,
                "content": e.content,
                "category": e.category,
                "priority": e.priority,
                "ttl": e.ttl,
            }
            for e in self._entries.values()
        ]
        cp = Checkpoint(
            id=cp_id,
            name=name,
            entries=entries_data,
            metadata=metadata or {},
        )
        self._checkpoints[cp_id] = cp

        # Persist to disk if checkpoint dir configured
        if self._checkpoint_dir:
            path = self._checkpoint_dir / f"{cp_id}.json"
            path.write_text(
                json.dumps({
                    "id": cp.id,
                    "name": cp.name,
                    "entries": cp.entries,
                    "metadata": cp.metadata,
                    "created_at": cp.created_at,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Checkpoint saved: %s → %s", name, path)

        return cp_id

    def restore(self, checkpoint_id: str) -> bool:
        """Restore context from a checkpoint."""
        cp = self._checkpoints.get(checkpoint_id)

        # Try loading from disk
        if not cp and self._checkpoint_dir:
            path = self._checkpoint_dir / f"{checkpoint_id}.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                cp = Checkpoint(
                    id=data["id"],
                    name=data["name"],
                    entries=data["entries"],
                    metadata=data.get("metadata", {}),
                    created_at=data.get("created_at", 0),
                )
                self._checkpoints[checkpoint_id] = cp

        if not cp:
            return False

        self._entries.clear()
        for entry_data in cp.entries:
            self.set(
                key=entry_data["key"],
                content=entry_data["content"],
                category=entry_data.get("category", "system"),
                priority=entry_data.get("priority", 0.5),
                ttl=entry_data.get("ttl", 0.0),
            )
        logger.info("Checkpoint restored: %s (%d entries)", cp.name, len(cp.entries))
        return True

    def list_checkpoints(self) -> list[dict]:
        """List available checkpoints."""
        # Include on-disk checkpoints
        if self._checkpoint_dir:
            for path in self._checkpoint_dir.glob("*.json"):
                cp_id = path.stem
                if cp_id not in self._checkpoints:
                    try:
                        data = json.loads(path.read_text(encoding="utf-8"))
                        self._checkpoints[cp_id] = Checkpoint(
                            id=cp_id,
                            name=data.get("name", "?"),
                            entries=data.get("entries", []),
                            metadata=data.get("metadata", {}),
                            created_at=data.get("created_at", 0),
                        )
                    except (json.JSONDecodeError, KeyError):
                        continue

        return [
            {
                "id": cp.id,
                "name": cp.name,
                "entries": len(cp.entries),
                "created_at": cp.created_at,
                "metadata": cp.metadata,
            }
            for cp in sorted(
                self._checkpoints.values(),
                key=lambda c: c.created_at,
                reverse=True,
            )
        ]

    # ------------------------------------------------------------------ #
    # Context Rot Detection (goose-inspired)
    # ------------------------------------------------------------------ #

    def rot_score(self) -> float:
        """Estimate context rot: 0.0 = fresh, 1.0 = severely degraded.

        Context rot increases when:
        - Many entries are stale
        - Token budget is heavily utilized
        - Average priority is low (lots of noise)
        """
        if not self._entries:
            return 0.0

        entries = list(self._entries.values())
        total = len(entries)

        # Factor 1: staleness ratio
        stale_count = sum(1 for e in entries if e.is_stale)
        stale_ratio = stale_count / total

        # Factor 2: budget pressure
        budget_pressure = min(1.0, self.current_tokens / self._max_tokens)

        # Factor 3: noise (low average relevance)
        avg_relevance = sum(e.relevance_score() for e in entries) / total
        noise = 1.0 - avg_relevance

        # Weighted combination
        rot = stale_ratio * 0.3 + budget_pressure * 0.4 + noise * 0.3
        return min(1.0, rot)

    def needs_compression(self, threshold: float = 0.7) -> bool:
        """Check if context needs compression/summarization."""
        return self.rot_score() > threshold or self.is_over_budget

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        """Context engine statistics."""
        entries = list(self._entries.values())
        categories = {}
        for e in entries:
            categories.setdefault(e.category, 0)
            categories[e.category] += 1

        return {
            "entries": len(entries),
            "tokens_used": self.current_tokens,
            "tokens_max": self._max_tokens,
            "budget_pct": round(self.budget_used_pct, 1),
            "rot_score": round(self.rot_score(), 3),
            "needs_compression": self.needs_compression(),
            "categories": categories,
            "checkpoints": len(self._checkpoints),
            "total_compressions": self._total_compressions,
        }

    def summary(self) -> str:
        """One-line summary for logging."""
        s = self.stats()
        return (
            f"Context: {s['tokens_used']}/{s['tokens_max']} tokens "
            f"({s['budget_pct']}%), rot={s['rot_score']:.2f}, "
            f"{s['entries']} entries"
        )
