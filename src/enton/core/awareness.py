"""AwarenessStateMachine — cognitive mode management.

Six awareness levels control which subsystems are active and how
resources are allocated.  Transitions are triggered by mood, presence,
time, and system events.

    DORMANT  ->  SENTINEL  ->  ATTENTIVE  ->  FOCUSED
                     ^              |
                     |         CREATIVE (dream)
                     +--- (any) <--- ALERT
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enton.core.events import EventBus
    from enton.core.self_model import SelfModel

logger = logging.getLogger(__name__)


class AwarenessLevel(Enum):
    DORMANT = auto()     # system idle, minimal processing
    SENTINEL = auto()    # background watching, motion/sound only
    ATTENTIVE = auto()   # someone present, all perception active
    FOCUSED = auto()     # active conversation, full cognition
    CREATIVE = auto()    # dream mode / background processing
    ALERT = auto()       # emergency: loud sound, error cascade


@dataclass(slots=True, frozen=True)
class AwarenessConfig:
    """What each awareness level activates."""

    vision_fps: int
    audio: bool
    llm: bool
    tts: bool
    dream: bool


# Pre-built configs per level
LEVEL_CONFIGS: dict[AwarenessLevel, AwarenessConfig] = {
    AwarenessLevel.DORMANT: AwarenessConfig(
        vision_fps=0, audio=False, llm=False, tts=False, dream=False,
    ),
    AwarenessLevel.SENTINEL: AwarenessConfig(
        vision_fps=2, audio=True, llm=False, tts=False, dream=False,
    ),
    AwarenessLevel.ATTENTIVE: AwarenessConfig(
        vision_fps=10, audio=True, llm=True, tts=True, dream=False,
    ),
    AwarenessLevel.FOCUSED: AwarenessConfig(
        vision_fps=15, audio=True, llm=True, tts=True, dream=False,
    ),
    AwarenessLevel.CREATIVE: AwarenessConfig(
        vision_fps=1, audio=False, llm=True, tts=False, dream=True,
    ),
    AwarenessLevel.ALERT: AwarenessConfig(
        vision_fps=30, audio=True, llm=True, tts=True, dream=False,
    ),
}

# Minimum seconds between transitions (debounce)
_MIN_TRANSITION_GAP = 2.0


@dataclass(slots=True)
class AwarenessStateMachine:
    """Manages Enton's cognitive mode based on environment and mood."""

    _state: AwarenessLevel = AwarenessLevel.SENTINEL
    _last_transition: float = field(default_factory=time.time)
    _state_enter_time: float = field(default_factory=time.time)
    _transition_count: int = 0

    # -- properties --

    @property
    def state(self) -> AwarenessLevel:
        return self._state

    @property
    def config(self) -> AwarenessConfig:
        return LEVEL_CONFIGS[self._state]

    @property
    def time_in_state(self) -> float:
        return time.time() - self._state_enter_time

    @property
    def is_dreaming(self) -> bool:
        return self._state == AwarenessLevel.CREATIVE

    @property
    def is_active(self) -> bool:
        return self._state in (
            AwarenessLevel.ATTENTIVE,
            AwarenessLevel.FOCUSED,
            AwarenessLevel.ALERT,
        )

    # -- transitions --

    def transition(
        self,
        new_state: AwarenessLevel,
        reason: str = "",
        bus: EventBus | None = None,
    ) -> bool:
        """Transition to a new awareness level. Returns True if transitioned."""
        if new_state == self._state:
            return False

        # debounce
        elapsed = time.time() - self._last_transition
        if elapsed < _MIN_TRANSITION_GAP:
            return False

        old = self._state
        self._state = new_state
        self._last_transition = time.time()
        self._state_enter_time = time.time()
        self._transition_count += 1

        logger.info(
            "Awareness: %s -> %s (%s) [#%d]",
            old.name, new_state.name, reason, self._transition_count,
        )

        if bus is not None:
            from enton.core.events import SystemEvent
            bus.emit_nowait(SystemEvent(
                kind="awareness_change",
                detail=f"{old.name}->{new_state.name}: {reason}",
            ))

        return True

    def evaluate(self, self_model: SelfModel, bus: EventBus | None = None) -> None:
        """Evaluate transitions based on current state. Call periodically."""
        mood = self_model.mood
        t = self.time_in_state

        if self._state == AwarenessLevel.DORMANT:
            # wake up on any sound or detection
            if mood.social > 0.1 or mood.engagement > 0.2:
                self.transition(AwarenessLevel.SENTINEL, "wakeup", bus)

        elif self._state == AwarenessLevel.SENTINEL:
            if mood.social > 0.3:
                self.transition(AwarenessLevel.ATTENTIVE, "person detected", bus)
            elif t > 600:  # 10min idle
                self.transition(AwarenessLevel.CREATIVE, "idle->dream", bus)

        elif self._state == AwarenessLevel.ATTENTIVE:
            if mood.engagement > 0.6:
                self.transition(AwarenessLevel.FOCUSED, "high engagement", bus)
            elif mood.social < 0.1 and t > 60:
                self.transition(AwarenessLevel.SENTINEL, "no one around", bus)

        elif self._state == AwarenessLevel.FOCUSED:
            if mood.engagement < 0.3 and t > 30:
                self.transition(AwarenessLevel.ATTENTIVE, "engagement dropped", bus)

        elif self._state == AwarenessLevel.CREATIVE:
            if mood.social > 0.2:
                self.transition(
                    AwarenessLevel.ATTENTIVE, "interaction during dream", bus,
                )
            elif t > 300:  # 5min dream max
                self.transition(AwarenessLevel.SENTINEL, "dream complete", bus)

        elif self._state == AwarenessLevel.ALERT and t > 60:
                self.transition(AwarenessLevel.ATTENTIVE, "alert timeout", bus)

    def trigger_alert(self, reason: str, bus: EventBus | None = None) -> None:
        """Force transition to ALERT (e.g. loud sound, unknown person)."""
        self.transition(AwarenessLevel.ALERT, reason, bus)

    def on_interaction(self, bus: EventBus | None = None) -> None:
        """User interaction — ensure we're at least ATTENTIVE."""
        if self._state in (
            AwarenessLevel.DORMANT,
            AwarenessLevel.SENTINEL,
            AwarenessLevel.CREATIVE,
        ):
            self.transition(AwarenessLevel.ATTENTIVE, "user interaction", bus)

    # -- serialization --

    def to_dict(self) -> dict:
        return {
            "state": self._state.name,
            "time_in_state": round(self.time_in_state, 1),
            "transitions": self._transition_count,
        }

    def from_dict(self, data: dict) -> None:
        name = data.get("state", "SENTINEL")
        try:
            self._state = AwarenessLevel[name]
        except KeyError:
            self._state = AwarenessLevel.SENTINEL

    def summary(self) -> str:
        cfg = self.config
        return (
            f"[{self._state.name}] "
            f"vision={cfg.vision_fps}fps audio={'on' if cfg.audio else 'off'} "
            f"llm={'on' if cfg.llm else 'off'} dream={'on' if cfg.dream else 'off'} "
            f"(in state {self.time_in_state:.0f}s, {self._transition_count} transitions)"
        )
