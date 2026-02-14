from __future__ import annotations

import asyncio
import logging
import random
import time

from enton.brain import Brain
from enton.config import settings
from enton.ears import Ears
from enton.events import DetectionEvent, EventBus, SpeechRequest, SystemEvent, TranscriptionEvent
from enton.memory import Episode, Memory
from enton.persona import REACTION_TEMPLATES, build_system_prompt
from enton.self_model import SelfModel
from enton.vision import Vision
from enton.voice import Voice

logger = logging.getLogger(__name__)


class App:
    def __init__(self) -> None:
        self.bus = EventBus()
        self.self_model = SelfModel(settings)
        self.memory = Memory()
        self.vision = Vision(settings, self.bus)
        self.ears = Ears(settings, self.bus)
        self.brain = Brain(settings)
        self.voice = Voice(settings, ears=self.ears)
        self._last_reaction: float = 0
        self._person_present: bool = False
        self._register_handlers()
        self._probe_capabilities()

    def _probe_capabilities(self) -> None:
        sm = self.self_model.senses
        sm.llm_ready = bool(self.brain._providers)
        if self.brain._providers:
            sm.active_providers["llm"] = str(self.brain._primary)
        sm.tts_ready = bool(self.voice._providers)
        if self.voice._providers:
            sm.active_providers["tts"] = str(self.voice._primary)
        sm.stt_ready = bool(self.ears._providers)
        if self.ears._providers:
            sm.active_providers["stt"] = str(self.ears._primary)

    def _register_handlers(self) -> None:
        self.bus.on(DetectionEvent, self._on_detection)
        self.bus.on(TranscriptionEvent, self._on_transcription)
        self.bus.on(SpeechRequest, self._on_speech_request)
        self.bus.on(SystemEvent, self._on_system_event)

    async def _on_detection(self, event: DetectionEvent) -> None:
        now = time.time()
        self.self_model.record_detection(event.label)

        if now - self._last_reaction < settings.reaction_cooldown:
            return

        if event.label == "person" and not self._person_present:
            self._person_present = True
            self._last_reaction = now
            text = random.choice(REACTION_TEMPLATES["person_appeared"])
            await self.voice.say(text)
            self.memory.remember(
                Episode(
                    kind="detection",
                    summary="Person appeared in camera",
                    tags=["person", "arrival"],
                )
            )

        elif event.label == "cat":
            self._last_reaction = now
            text = random.choice(REACTION_TEMPLATES["cat_detected"])
            await self.voice.say(text)
            self.memory.remember(
                Episode(
                    kind="detection",
                    summary="Cat detected!",
                    tags=["cat"],
                )
            )

    async def _on_transcription(self, event: TranscriptionEvent) -> None:
        if not event.text.strip():
            return

        self.self_model.record_interaction()
        self.memory.strengthen_relationship()

        detections = [
            {"label": d.label, "confidence": d.confidence} for d in self.vision.last_detections
        ]
        system = build_system_prompt(
            self.self_model,
            self.memory,
            detections=detections,
        )

        response = await self.brain.think(event.text, system=system)
        if response:
            await self.voice.say(response)
            self.memory.remember(
                Episode(
                    kind="conversation",
                    summary=f"User said: '{event.text[:60]}' → I replied: '{response[:60]}'",
                    tags=["chat"],
                )
            )

    async def _on_speech_request(self, event: SpeechRequest) -> None:
        await self.voice.say(event.text)

    async def _on_system_event(self, event: SystemEvent) -> None:
        if event.kind == "startup":
            text = random.choice(REACTION_TEMPLATES["startup"])
            await self.voice.say(text)
            self.memory.remember(
                Episode(
                    kind="system",
                    summary="Enton booted up",
                    tags=["startup"],
                )
            )
        elif event.kind == "camera_lost":
            self.self_model.senses.camera_online = False
            logger.warning("Camera connection lost")
        elif event.kind == "camera_connected":
            self.self_model.senses.camera_online = True

    async def run(self) -> None:
        logger.info("Enton starting up...")
        logger.info("Self-state: %s", self.self_model.introspect())
        await self.bus.emit(SystemEvent(kind="startup"))

        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.bus.run(), name="event_bus")
            tg.create_task(self.vision.run(), name="vision")
            tg.create_task(self.voice.run(), name="voice")
            tg.create_task(self.ears.run(), name="ears")
            tg.create_task(self._idle_loop(), name="idle")
            tg.create_task(self._mood_decay_loop(), name="mood_decay")

    async def _idle_loop(self) -> None:
        while True:
            await asyncio.sleep(settings.idle_timeout)
            self.self_model.mood.on_idle()
            if not self._person_present and not self.voice.is_speaking:
                text = random.choice(REACTION_TEMPLATES["idle"])
                await self.voice.say(text)
            self._person_present = False

    async def _mood_decay_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            self.self_model.mood.tick()
