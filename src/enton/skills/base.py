"""Base skill protocol — all skills implement this interface."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from enton.core.events import Event, EventBus


@runtime_checkable
class Skill(Protocol):
    """A composable behavior that reacts to events.

    Skills combine perception + cognition + action into reusable behaviors.
    They subscribe to specific events via the EventBus and trigger actions.
    """

    @property
    def name(self) -> str: ...

    def attach(self, bus: EventBus) -> None:
        """Register event handlers on the bus."""
        ...

    async def handle(self, event: Event) -> None:
        """Process a relevant event."""
        ...
