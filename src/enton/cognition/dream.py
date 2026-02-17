"""DreamMode — background memory consolidation during idle.

When Enton is idle (no interactions for a threshold period), DreamMode
activates and runs consolidation cycles:

1. Summarize recent episodes into higher-level memories
2. Find temporal patterns in detections / events
3. Extract user preferences from conversation history
4. Emit dream insights as new episodes tagged [dream]

Inspired by: Letta/MemGPT "Sleep-time Compute" (arXiv 2504.13171),
SimNap (Dreaming AI Whitepaper).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enton.cognition.brain import EntonBrain
    from enton.core.memory import Memory

logger = logging.getLogger(__name__)


class DreamMode:
    """Background processing during idle time."""

    IDLE_THRESHOLD = 120.0   # seconds before entering dream mode
    DREAM_INTERVAL = 300.0   # seconds between dream cycles
    MAX_DREAM_DURATION = 300  # max seconds per cycle

    def __init__(
        self,
        memory: Memory,
        brain: EntonBrain,
    ) -> None:
        self._memory = memory
        self._brain = brain
        self._last_interaction = time.time()
        self._dreaming = False
        self._dream_count = 0
        self._last_dream = 0.0
        self._insights: list[str] = []

    # -- interaction tracking --

    def on_interaction(self) -> None:
        """Reset idle timer on any user interaction."""
        self._last_interaction = time.time()
        if self._dreaming:
            logger.info("Dream interrupted by interaction")
            self._dreaming = False

    @property
    def idle_seconds(self) -> float:
        return time.time() - self._last_interaction

    @property
    def dreaming(self) -> bool:
        return self._dreaming

    @property
    def dream_count(self) -> int:
        return self._dream_count

    @property
    def should_dream(self) -> bool:
        """Check if conditions are right for dreaming."""
        if self._dreaming:
            return False
        if self.idle_seconds < self.IDLE_THRESHOLD:
            return False
        return not (time.time() - self._last_dream < self.DREAM_INTERVAL)

    # -- main loop --

    async def run(self) -> None:
        """Main dream loop — run as asyncio task in TaskGroup."""
        await asyncio.sleep(60)  # let system initialize
        while True:
            await asyncio.sleep(10)
            if self.should_dream:
                await self._dream_cycle()

    async def _dream_cycle(self) -> None:
        """Execute one consolidation cycle."""
        self._dreaming = True
        self._dream_count += 1
        self._last_dream = time.time()
        logger.info(
            "Dream cycle #%d starting (idle %.0fs)",
            self._dream_count, self.idle_seconds,
        )

        try:
            # 1. consolidate recent episodes
            insight = await self._consolidate_episodes()
            if insight:
                self._insights.append(insight)

            # 2. find patterns
            patterns = self._analyze_patterns()
            if patterns:
                self._insights.extend(patterns)

            # 3. extract user preferences
            await self._update_profile()

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Dream cycle #%d error", self._dream_count)
        finally:
            self._dreaming = False
            logger.info("Dream cycle #%d complete", self._dream_count)

    # -- consolidation steps --

    async def _consolidate_episodes(self) -> str:
        """Summarize recent episodes into higher-level memories."""
        recent = self._memory.recall_recent(20)
        if len(recent) < 5:
            return ""

        summaries = [e.summary for e in recent]
        prompt = (
            "Voce e o sistema de consolidacao de memoria do Enton. "
            "Resuma estes episodios recentes em 2-3 insights chave. "
            "Foque em padroes, preferencias do usuario, e eventos importantes.\n\n"
            + "\n".join(f"- {s}" for s in summaries[-10:])
        )

        try:
            result = await self._brain.think(
                prompt,
                system=(
                    "Voce e um sistema de memoria. Seja conciso e factual. "
                    "Responda em 2-3 frases no maximo."
                ),
            )
        except Exception:
            logger.debug("Consolidation LLM call failed")
            return ""

        if result and len(result) > 10:
            from enton.core.memory import Episode
            self._memory.remember(Episode(
                kind="consolidation",
                summary=f"[Dream #{self._dream_count}] {result}",
                tags=["dream", "consolidation"],
            ))
            logger.info("Dream insight: %s", result[:100])
            return result

        return ""

    def _analyze_patterns(self) -> list[str]:
        """Find temporal patterns in recent episodes."""
        episodes = self._memory.recall_recent(50)
        if len(episodes) < 10:
            return []

        # group tags by hour
        hour_tags: Counter[tuple[int, str]] = Counter()
        for ep in episodes:
            h = datetime.fromtimestamp(ep.timestamp).hour
            for tag in ep.tags:
                hour_tags[(h, tag)] += 1

        patterns = []
        for (h, tag), count in hour_tags.most_common(5):
            if count >= 3:
                pattern = f"{tag} appears frequently around {h}:00 ({count}x)"
                patterns.append(pattern)

        if patterns:
            from enton.core.memory import Episode
            summary = f"[Dream] Patterns: {'; '.join(patterns[:3])}"
            self._memory.remember(Episode(
                kind="pattern",
                summary=summary,
                tags=["dream", "pattern"],
            ))

        return patterns

    async def _update_profile(self) -> None:
        """Extract user preferences from conversation history."""
        convos = self._memory.recall_by_kind("conversation", n=15)
        if len(convos) < 3:
            return

        texts = [e.summary for e in convos[-10:]]
        prompt = (
            f"Destes trechos de conversa com {self._memory.profile.name}, "
            "extraia preferencias, habitos ou fatos sobre a pessoa. "
            "Retorne como JSON: lista de strings com cada fato.\n\n"
            + "\n".join(f"- {t}" for t in texts)
        )

        try:
            result = await self._brain.think(
                prompt,
                system="Retorne APENAS um JSON array de strings. Nada mais.",
            )
        except Exception:
            return

        if not result:
            return

        # try to parse JSON from response
        try:
            # strip markdown fences if present
            clean = result.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            facts = json.loads(clean)
            if isinstance(facts, list):
                for fact in facts[:5]:
                    self._memory.learn_about_user(str(fact))
                logger.info("Dream extracted %d user facts", min(len(facts), 5))
        except (json.JSONDecodeError, TypeError, IndexError):
            # LLM didn't return valid JSON — skip
            pass

    # -- queries --

    @property
    def recent_insights(self) -> list[str]:
        return self._insights[-5:]

    def summary(self) -> str:
        parts = [
            f"Dreams: {self._dream_count}",
            f"idle: {self.idle_seconds:.0f}s",
        ]
        if self._dreaming:
            parts.append("STATUS: dreaming")
        if self._insights:
            parts.append(f"insights: {len(self._insights)}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        return {
            "dream_count": self._dream_count,
            "dreaming": self._dreaming,
            "idle_seconds": round(self.idle_seconds, 1),
            "insights_count": len(self._insights),
        }
