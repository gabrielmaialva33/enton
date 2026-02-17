"""Lifecycle — Enton's state persistence between boots.

Saves and restores:
- Mood state (engagement, social)
- Desire urgencies
- Boot count, total uptime
- Last shutdown reason
- Planner data is handled by planner.py itself
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enton.cognition.desires import DesireEngine
    from enton.core.self_model import SelfModel

logger = logging.getLogger(__name__)

_STATE_FILE = Path.home() / ".enton" / "state.json"


class Lifecycle:
    """Manages Enton's persistent living state."""

    def __init__(self) -> None:
        self._state: dict = {}
        self._boot_time = time.time()
        self._load()

    def _load(self) -> None:
        if _STATE_FILE.exists():
            try:
                self._state = json.loads(_STATE_FILE.read_text())
                logger.info("Lifecycle state loaded (boot #%d)", self.boot_count)
            except Exception:
                logger.warning("Failed to load lifecycle state")
                self._state = {}

    def _save(self) -> None:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(self._state, indent=2, ensure_ascii=False))

    @property
    def boot_count(self) -> int:
        return self._state.get("boot_count", 0)

    @property
    def total_uptime_hours(self) -> float:
        return self._state.get("total_uptime_seconds", 0) / 3600

    @property
    def last_shutdown(self) -> float:
        return self._state.get("last_shutdown", 0)

    @property
    def time_asleep(self) -> float:
        """How long Enton was 'asleep' (time between last shutdown and this boot)."""
        last = self.last_shutdown
        if last > 0:
            return self._boot_time - last
        return 0

    @property
    def time_asleep_human(self) -> str:
        s = int(self.time_asleep)
        if s < 60:
            return f"{s}s"
        if s < 3600:
            return f"{s // 60}min"
        h, m = divmod(s, 3600)
        if h < 24:
            return f"{h}h{m // 60}min"
        d, h = divmod(h, 24)
        return f"{d}d{h}h"

    def on_boot(self, self_model: SelfModel, desires: DesireEngine) -> str:
        """Called on startup. Restores state and returns wake-up context."""
        # Increment boot count
        self._state["boot_count"] = self.boot_count + 1

        # Restore mood
        mood_data = self._state.get("mood", {})
        if mood_data:
            self_model.mood.engagement = mood_data.get("engagement", 0.5)
            self_model.mood.social = mood_data.get("social", 0.3)

        # Restore desires
        desire_data = self._state.get("desires", {})
        if desire_data:
            desires.from_dict(desire_data)

        # Generate wake-up message based on how long we slept
        asleep = self.time_asleep
        if asleep > 86400:  # > 1 day
            return f"Caramba, dormi {self.time_asleep_human}! Saudade de mim?"
        if asleep > 3600:  # > 1 hour
            return f"Voltei! Fiquei {self.time_asleep_human} offline."
        if asleep > 60:
            return "Rapidinho, já voltei!"
        if self.boot_count <= 1:
            return ""  # First boot ever, use startup template
        return ""

    def on_shutdown(self, self_model: SelfModel, desires: DesireEngine) -> None:
        """Called on graceful shutdown. Persists state."""
        self._state["last_shutdown"] = time.time()
        session_uptime = time.time() - self._boot_time
        self._state["total_uptime_seconds"] = (
            self._state.get("total_uptime_seconds", 0) + session_uptime
        )

        # Save mood
        self._state["mood"] = {
            "engagement": self_model.mood.engagement,
            "social": self_model.mood.social,
        }

        # Save desires
        self._state["desires"] = desires.to_dict()

        self._save()
        logger.info(
            "Lifecycle saved: boot #%d, total uptime %.1fh",
            self.boot_count, self.total_uptime_hours,
        )

    def save_periodic(self, self_model: SelfModel, desires: DesireEngine) -> None:
        """Periodic save (crash safety)."""
        self.on_shutdown(self_model, desires)

    def summary(self) -> str:
        return (
            f"Boot #{self.boot_count}, "
            f"total uptime {self.total_uptime_hours:.1f}h, "
            f"slept {self.time_asleep_human}"
        )
