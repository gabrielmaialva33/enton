from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enton.core.config import Settings


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
        self._last_activity = "none"
        self._last_emotion = "neutral"
        self._recent_sounds: deque[tuple[str, float, float]] = deque(maxlen=10)

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

    def record_activity(self, activity: str) -> None:
        self._last_activity = activity
        # Activity labels come from activity.py in PT-BR
        low = activity.lower()
        if "acenando" in low or "maos pra cima" in low:
            self.mood.social = min(1.0, self.mood.social + 0.1)
        elif "no celular" in low:
            self.mood.engagement = max(0.0, self.mood.engagement - 0.03)

    def record_emotion(self, emotion: str) -> None:
        self._last_emotion = emotion
        low = emotion.lower()
        if low in ("feliz", "happy", "surpreso", "surprised"):
            self.mood.engagement = min(1.0, self.mood.engagement + 0.1)
        elif low in ("triste", "sad", "irritado", "angry", "medo", "fear"):
            self.mood.social = max(0.0, self.mood.social - 0.1)

    @property
    def last_emotion(self) -> str:
        return self._last_emotion

    @property
    def last_activity(self) -> str:
        return self._last_activity

    def record_sound(self, label: str, confidence: float) -> None:
        self._recent_sounds.append((label, confidence, time.time()))
        # Alert sounds boost engagement
        if label.lower() in ("alarme", "sirene", "vidro quebrando"):
            self.mood.engagement = min(1.0, self.mood.engagement + 0.2)

    @property
    def recent_sounds(self) -> list[tuple[str, float, float]]:
        return list(self._recent_sounds)

    def record_error(self) -> None:
        self._errors_count += 1
        self.mood.on_error()

    @staticmethod
    def _get_vram_info() -> str:
        """Get GPU VRAM usage if available."""
        try:
            import torch

            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                free = total - used
                return f"VRAM: {used:.1f}GB/{total:.0f}GB (free: {free:.1f}GB)"
        except Exception:
            pass
        return "VRAM: unknown"

    def introspect(self) -> str:
        self.mood.tick()
        eng = self.mood.engagement
        soc = self.mood.social
        last_act = self._last_activity
        last_emo = self._last_emotion

        parts = [
            f"I am Enton. Running for {self.uptime_human}.",
            f"Mood: {self.mood.label} (engagement={eng:.1f}, social={soc:.1f}).",
            f"User emotion: {last_emo}. User activity: {last_act}.",
            f"Senses: {self.senses.summary()}.",
            self._get_vram_info(),
            f"Stats: {self._interactions_count} chats, {self._detections_count} detections.",
        ]
        if self._recent_sounds:
            recent = [s[0] for s in list(self._recent_sounds)[-3:]]
            parts.append(f"Recent sounds: {', '.join(recent)}.")
        if self._errors_count > 0:
            parts.append(f"Errors so far: {self._errors_count}.")
        return " ".join(parts)
