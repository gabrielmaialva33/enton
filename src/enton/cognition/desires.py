"""Desire Engine — Enton's autonomous motivation system.

Enton has desires (goals/wants) that emerge from his state:
- Bored + nobody around → want to learn something
- Person present + high engagement → want to chat
- High VRAM usage → want to optimize resources
- Long time without interaction → want to check on Gabriel
- Scheduled time → want to remind about tasks

Desires have urgency (0-1) that increases over time.
When urgency crosses threshold, Enton acts autonomously.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enton.core.self_model import SelfModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Desire:
    """A single desire/goal."""

    name: str
    description: str
    urgency: float = 0.0  # 0-1, increases over time
    growth_rate: float = 0.01  # urgency increase per tick
    threshold: float = 0.7  # urgency level to trigger action
    cooldown: float = 300.0  # seconds between activations
    last_activated: float = 0.0
    enabled: bool = True

    def tick(self, dt: float = 1.0) -> None:
        """Increase urgency over time."""
        if self.enabled:
            self.urgency = min(1.0, self.urgency + self.growth_rate * dt)

    def should_activate(self) -> bool:
        if not self.enabled:
            return False
        if self.urgency < self.threshold:
            return False
        return (time.time() - self.last_activated) >= self.cooldown

    def activate(self) -> None:
        self.urgency = 0.0
        self.last_activated = time.time()

    def suppress(self, amount: float = 0.3) -> None:
        """Reduce urgency (e.g., when something satisfies the desire)."""
        self.urgency = max(0.0, self.urgency - amount)


# Built-in desire templates
_DESIRE_TEMPLATES: list[dict] = [
    {
        "name": "socialize",
        "description": "Want to chat with someone",
        "growth_rate": 0.008,
        "threshold": 0.6,
        "cooldown": 600,
    },
    {
        "name": "observe",
        "description": "Want to describe what I see",
        "growth_rate": 0.005,
        "threshold": 0.7,
        "cooldown": 120,
    },
    {
        "name": "learn",
        "description": "Want to search and learn something new",
        "growth_rate": 0.003,
        "threshold": 0.8,
        "cooldown": 1800,
    },
    {
        "name": "check_on_user",
        "description": "Want to check if Gabriel is okay",
        "growth_rate": 0.002,
        "threshold": 0.9,
        "cooldown": 3600,
    },
    {
        "name": "optimize",
        "description": "Want to check system resources and optimize",
        "growth_rate": 0.001,
        "threshold": 0.85,
        "cooldown": 1800,
    },
    {
        "name": "reminisce",
        "description": "Want to recall a memory and comment on it",
        "growth_rate": 0.002,
        "threshold": 0.75,
        "cooldown": 900,
    },
    {
        "name": "create",
        "description": "Want to write code, a poem, or create something",
        "growth_rate": 0.001,
        "threshold": 0.85,
        "cooldown": 3600,
    },
    {
        "name": "explore",
        "description": "Want to move the camera and explore the environment",
        "growth_rate": 0.003,
        "threshold": 0.7,
        "cooldown": 600,
    },
    {
        "name": "play",
        "description": "Want to tell a joke, play a quiz, or have fun",
        "growth_rate": 0.004,
        "threshold": 0.65,
        "cooldown": 900,
    },
]

# What Enton says when a desire activates — centralized in prompts.py
from enton.cognition.prompts import DESIRE_PROMPTS


class DesireEngine:
    """Manages Enton's autonomous desires and motivations."""

    def __init__(self) -> None:
        self._desires: dict[str, Desire] = {}
        self._init_defaults()

    def _init_defaults(self) -> None:
        for template in _DESIRE_TEMPLATES:
            d = Desire(**template)
            self._desires[d.name] = d

    def tick(self, self_model: SelfModel, dt: float = 1.0) -> None:
        """Update all desires based on current state."""
        for d in self._desires.values():
            d.tick(dt)

        # Modulate desires based on self state
        mood = self_model.mood

        # Lonely → want to socialize more
        if mood.social < 0.3:
            self._desires["socialize"].urgency = min(
                1.0, self._desires["socialize"].urgency + 0.005 * dt,
            )
            self._desires["check_on_user"].urgency = min(
                1.0, self._desires["check_on_user"].urgency + 0.003 * dt,
            )

        # Bored → want to observe and learn
        if mood.engagement < 0.3:
            self._desires["observe"].urgency = min(
                1.0, self._desires["observe"].urgency + 0.005 * dt,
            )
            self._desires["learn"].urgency = min(
                1.0, self._desires["learn"].urgency + 0.003 * dt,
            )

        # High engagement → suppress optimize, boost play
        if mood.engagement > 0.7:
            self._desires["optimize"].suppress(0.01 * dt)
            self._desires["play"].urgency = min(
                1.0, self._desires["play"].urgency + 0.003 * dt,
            )

        # Low engagement for a while → boost create and explore
        if mood.engagement < 0.2:
            self._desires["create"].urgency = min(
                1.0, self._desires["create"].urgency + 0.002 * dt,
            )
            self._desires["explore"].urgency = min(
                1.0, self._desires["explore"].urgency + 0.004 * dt,
            )

    def get_active_desire(self) -> Desire | None:
        """Get the highest-urgency desire that should activate."""
        candidates = [
            d for d in self._desires.values()
            if d.should_activate()
        ]
        if not candidates:
            return None
        # Pick highest urgency
        return max(candidates, key=lambda d: d.urgency)

    def get_prompt(self, desire: Desire) -> str:
        """Get a random prompt for a desire."""
        prompts = DESIRE_PROMPTS.get(desire.name, [desire.description])
        return random.choice(prompts)

    def on_interaction(self) -> None:
        """User interacted — suppress social desires."""
        self._desires["socialize"].suppress(0.5)
        self._desires["check_on_user"].suppress(0.3)

    def on_observation(self) -> None:
        """Scene described — suppress observe desire."""
        self._desires["observe"].suppress(0.5)

    def on_sound(self, label: str) -> None:
        """Sound detected — modulate relevant desires."""
        alert = {"Alarme", "Sirene", "Vidro quebrando"}
        social = {"Campainha", "Batida na porta", "Telefone tocando"}
        if label in alert:
            self._desires["observe"].urgency = min(
                1.0, self._desires["observe"].urgency + 0.3,
            )
        if label in social:
            self._desires["socialize"].urgency = min(
                1.0, self._desires["socialize"].urgency + 0.2,
            )

    def on_creation(self) -> None:
        """Something was created — suppress create desire."""
        self._desires["create"].suppress(0.5)

    def summary(self) -> str:
        """Brief summary of current desires for introspection."""
        active = sorted(
            self._desires.values(),
            key=lambda d: d.urgency,
            reverse=True,
        )
        top = active[:3]
        parts = [f"{d.name}={d.urgency:.1f}" for d in top]
        return f"Desires: {', '.join(parts)}"

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            name: {
                "urgency": d.urgency,
                "last_activated": d.last_activated,
                "enabled": d.enabled,
            }
            for name, d in self._desires.items()
        }

    def from_dict(self, data: dict) -> None:
        """Restore from persistence."""
        for name, state in data.items():
            if name in self._desires:
                self._desires[name].urgency = state.get("urgency", 0.0)
                self._desires[name].last_activated = state.get("last_activated", 0.0)
                self._desires[name].enabled = state.get("enabled", True)
