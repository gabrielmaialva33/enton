"""Tests for EventBus and Event types."""

from __future__ import annotations

import asyncio

from enton.core.events import (
    DetectionEvent,
    EventBus,
    SoundEvent,
    SystemEvent,
    TranscriptionEvent,
)


async def test_event_bus_emit_and_handle():
    bus = EventBus()
    received: list[DetectionEvent] = []

    async def handler(event: DetectionEvent):
        received.append(event)

    bus.on(DetectionEvent, handler)
    await bus.emit(DetectionEvent(label="person", confidence=0.95))

    # Process one event
    event = await asyncio.wait_for(bus._queue.get(), timeout=1.0)
    handlers = bus._handlers.get(type(event), [])
    for h in handlers:
        await h(event)

    assert len(received) == 1
    assert received[0].label == "person"
    assert received[0].confidence == 0.95


async def test_event_bus_multiple_handlers():
    bus = EventBus()
    calls: list[str] = []

    async def handler_a(event):
        calls.append("a")

    async def handler_b(event):
        calls.append("b")

    bus.on(SystemEvent, handler_a)
    bus.on(SystemEvent, handler_b)
    await bus.emit(SystemEvent(kind="startup"))

    event = await asyncio.wait_for(bus._queue.get(), timeout=1.0)
    for h in bus._handlers.get(type(event), []):
        await h(event)

    assert calls == ["a", "b"]


async def test_event_bus_type_isolation():
    """Handlers only fire for their registered event type."""
    bus = EventBus()
    detection_calls = 0
    sound_calls = 0

    async def on_detection(event):
        nonlocal detection_calls
        detection_calls += 1

    async def on_sound(event):
        nonlocal sound_calls
        sound_calls += 1

    bus.on(DetectionEvent, on_detection)
    bus.on(SoundEvent, on_sound)

    await bus.emit(DetectionEvent(label="cat"))
    event = await asyncio.wait_for(bus._queue.get(), timeout=1.0)
    for h in bus._handlers.get(type(event), []):
        await h(event)

    assert detection_calls == 1
    assert sound_calls == 0


def test_event_defaults():
    e = TranscriptionEvent(text="hello")
    assert e.text == "hello"
    assert e.is_final is True
    assert e.language == "pt-BR"
    assert e.timestamp > 0


def test_detection_event_fields():
    e = DetectionEvent(label="person", confidence=0.9, bbox=(10, 20, 100, 200))
    assert e.label == "person"
    assert e.bbox == (10, 20, 100, 200)


def test_emit_nowait():
    bus = EventBus()
    bus.emit_nowait(SystemEvent(kind="test"))
    assert bus._queue.qsize() == 1
