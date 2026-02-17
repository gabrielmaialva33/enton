from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from time import time
from typing import Any

logger = logging.getLogger(__name__)

type EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


@dataclass(frozen=True, slots=True)
class Event:
    timestamp: float = field(default_factory=time)


@dataclass(frozen=True, slots=True)
class DetectionEvent(Event):
    label: str = ""
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    frame_shape: tuple[int, int] = (0, 0)
    camera_id: str = "main"


@dataclass(frozen=True, slots=True)
class ActivityEvent(Event):
    person_index: int = 0
    activity: str = ""
    color: tuple[int, int, int] = (200, 200, 200)
    camera_id: str = "main"


@dataclass(frozen=True, slots=True)
class EmotionEvent(Event):
    person_index: int = 0
    emotion: str = ""
    emotion_en: str = ""
    score: float = 0.0
    color: tuple[int, int, int] = (180, 180, 180)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    camera_id: str = "main"


@dataclass(frozen=True, slots=True)
class TranscriptionEvent(Event):
    text: str = ""
    is_final: bool = True
    language: str = "pt-BR"


@dataclass(frozen=True, slots=True)
class SpeechRequest(Event):
    text: str = ""
    priority: int = 0  # higher = more important


@dataclass(frozen=True, slots=True)
class BrainResponse(Event):
    text: str = ""
    source: str = ""  # provider that generated it


@dataclass(frozen=True, slots=True)
class FaceEvent(Event):
    identity: str = "unknown"
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    camera_id: str = "main"


@dataclass(frozen=True, slots=True)
class SoundEvent(Event):
    label: str = ""
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class SceneChangeEvent(Event):
    """Emitted when visual scene changes significantly."""
    camera_id: str = "main"
    new_objects: list[str] = field(default_factory=list)
    removed_objects: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SkillEvent(Event):
    """Emitted when a dynamic skill is loaded, unloaded, or forged."""
    kind: str = ""  # "loaded", "unloaded", "forge_created", "forge_failed"
    name: str = ""
    detail: str = ""


@dataclass(frozen=True, slots=True)
class SystemEvent(Event):
    kind: str = ""  # startup, shutdown, error, camera_lost, etc.
    detail: str = ""


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[EventHandler]] = {}
        self._queue: asyncio.Queue[Event] = asyncio.Queue()

    def on(self, event_type: type[Event], handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    async def emit(self, event: Event) -> None:
        await self._queue.put(event)

    def emit_nowait(self, event: Event) -> None:
        self._queue.put_nowait(event)

    async def run(self) -> None:
        while True:
            event = await self._queue.get()
            handlers = self._handlers.get(type(event), [])
            for handler in handlers:
                try:
                    await handler(event)
                except Exception:
                    logger.exception("Handler error for %s", type(event).__name__)
