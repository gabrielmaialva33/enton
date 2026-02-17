"""Greet skill — reacts when a person appears in camera."""
from __future__ import annotations

import random
import time

from enton.action.voice import Voice
from enton.cognition.persona import REACTION_TEMPLATES
from enton.core.events import DetectionEvent, Event, EventBus
from enton.core.memory import Episode, Memory


class GreetSkill:
    """Greets people when they first appear on camera."""

    def __init__(self, voice: Voice, memory: Memory, cooldown: float = 10.0) -> None:
        self._voice = voice
        self._memory = memory
        self._cooldown = cooldown
        self._last_greet: float = 0
        self._person_present: bool = False

    @property
    def name(self) -> str:
        return "greet"

    def attach(self, bus: EventBus) -> None:
        bus.on(DetectionEvent, self.handle)

    async def handle(self, event: Event) -> None:
        if not isinstance(event, DetectionEvent):
            return
        if event.label != "person":
            return

        now = time.time()
        if self._person_present or now - self._last_greet < self._cooldown:
            return

        self._person_present = True
        self._last_greet = now
        text = random.choice(REACTION_TEMPLATES["person_appeared"])
        await self._voice.say(text)
        self._memory.remember(
            Episode(kind="detection", summary="Person appeared", tags=["person", "arrival"]),
        )

    def reset_presence(self) -> None:
        self._person_present = False
