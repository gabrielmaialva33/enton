from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enton.config import Settings


@dataclass(slots=True)
class SensoryState:
    camera_online: bool = False
    mic_online: bool = False
    tts_ready: bool = False
    stt_ready: bool = False
    llm_ready: bool = False
    active_providers: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        parts = []
        if self.camera_online:
            parts.append("camera ON")
        else:
            parts.append("camera OFF")
        if self.mic_online:
            parts.append("mic ON")
        else:
            parts.append("mic OFF")
        if self.tts_ready:
            parts.append(f"voice via {self.active_providers.get('tts', '?')}")
        if self.stt_ready:
            parts.append(f"ears via {self.active_providers.get('stt', '?')}")
        if self.llm_ready:
            parts.append(f"brain via {self.active_providers.get('llm', '?')}")
        return ", ".join(parts) if parts else "all systems offline"


@dataclass(slots=True)
class Mood:
    engagement: float = 0.5
    social: float = 0.3
    _last_update: float = field(default_factory=time.time)

    DECAY_RATE = 0.02  # per minute

    def tick(self) -> None:
        now = time.time()
        elapsed_min = (now - self._last_update) / 60.0
        self.engagement = max(0.0, self.engagement - self.DECAY_RATE * elapsed_min)
        self.social = max(0.0, self.social - self.DECAY_RATE * elapsed_min * 0.5)
        self._last_update = now

    def on_interaction(self) -> None:
        self.engagement = min(1.0, self.engagement + 0.15)
        self.social = min(1.0, self.social + 0.2)

    def on_detection(self, label: str) -> None:
        if label == "cat":
            self.engagement = min(1.0, self.engagement + 0.3)
        elif label == "person":
            self.social = min(1.0, self.social + 0.15)

    def on_error(self) -> None:
        self.engagement = max(0.0, self.engagement - 0.1)

    def on_idle(self) -> None:
        self.engagement = max(0.0, self.engagement - 0.05)

    @property
    def label(self) -> str:
        avg = (self.engagement + self.social) / 2
        if avg >= 0.7:
            return "empolgado"
        if avg >= 0.4:
            return "tranquilo"
        if avg >= 0.2:
            return "entediado"
        return "largado"


class SelfModel:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._boot_time = time.time()
        self.mood = Mood()
        self.senses = SensoryState()
        self._interactions_count = 0
        self._detections_count = 0
        self._errors_count = 0

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._boot_time

    @property
    def uptime_human(self) -> str:
        s = int(self.uptime_seconds)
        if s < 60:
            return f"{s}s"
        if s < 3600:
            return f"{s // 60}min"
        h, m = divmod(s, 3600)
        return f"{h}h{m // 60}min"

    def record_interaction(self) -> None:
        self._interactions_count += 1
        self.mood.on_interaction()

    def record_detection(self, label: str) -> None:
        self._detections_count += 1
        self.mood.on_detection(label)

    def record_error(self) -> None:
        self._errors_count += 1
        self.mood.on_error()

    def introspect(self) -> str:
        self.mood.tick()
        eng = self.mood.engagement
        soc = self.mood.social
        parts = [
            f"I am Enton. Running for {self.uptime_human}.",
            f"Mood: {self.mood.label} (engagement={eng:.1f}, social={soc:.1f}).",
            f"Senses: {self.senses.summary()}.",
            f"Stats: {self._interactions_count} chats, {self._detections_count} detections.",
        ]
        if self._errors_count > 0:
            parts.append(f"Errors so far: {self._errors_count}.")
        return " ".join(parts)
