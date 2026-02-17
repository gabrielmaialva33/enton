"""React skill — responds to interesting detections (cats, unusual objects)."""
from __future__ import annotations

import random
import time

from enton.action.voice import Voice
from enton.cognition.persona import REACTION_TEMPLATES
from enton.core.events import DetectionEvent, Event, EventBus
from enton.core.memory import Episode, Memory


class ReactSkill:
    """Reacts vocally to interesting detections like cats."""

    def __init__(self, voice: Voice, memory: Memory, cooldown: float = 15.0) -> None:
        self._voice = voice
        self._memory = memory
        self._cooldown = cooldown
        self._last_react: float = 0

    @property
    def name(self) -> str:
        return "react"

    def attach(self, bus: EventBus) -> None:
        bus.on(DetectionEvent, self.handle)

    async def handle(self, event: Event) -> None:
        if not isinstance(event, DetectionEvent):
            return

        now = time.time()
        if now - self._last_react < self._cooldown:
            return

        if event.label == "cat":
            self._last_react = now
            text = random.choice(REACTION_TEMPLATES["cat_detected"])
            await self._voice.say(text)
            self._memory.remember(
                Episode(kind="detection", summary="Cat detected!", tags=["cat"]),
            )
