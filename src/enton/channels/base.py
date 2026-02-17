"""Base channel abstraction — all platforms implement this."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from time import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from enton.core.events import EventBus

logger = logging.getLogger(__name__)


class MessageType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    COMMAND = "command"
    REACTION = "reaction"


@dataclass(slots=True)
class ChannelMessage:
    """Unified message format across all channels."""

    channel: str  # "telegram", "discord", "whatsapp", "web", "voice"
    sender_id: str  # platform-specific user ID
    sender_name: str  # display name
    text: str = ""
    message_type: MessageType = MessageType.TEXT
    media: bytes | None = None  # image/audio/video bytes
    media_url: str = ""  # remote URL if not inline
    reply_to: str = ""  # message ID being replied to
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time)
    message_id: str = ""  # platform-specific message ID

    @property
    def has_media(self) -> bool:
        return self.media is not None or bool(self.media_url)


class BaseChannel(ABC):
    """Abstract base for all messaging channels.

    Lifecycle:
        1. __init__(bus, config) — setup
        2. start() — connect to platform, begin listening
        3. send(msg) — send a message to the platform
        4. stop() — disconnect, cleanup
    """

    name: str = "base"

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """Connect to platform and start listening for messages."""

    @abstractmethod
    async def stop(self) -> None:
        """Disconnect and cleanup resources."""

    @abstractmethod
    async def send(self, message: ChannelMessage) -> None:
        """Send a message to the platform."""

    async def send_text(self, target_id: str, text: str) -> None:
        """Convenience: send a text message."""
        msg = ChannelMessage(
            channel=self.name,
            sender_id="enton",
            sender_name="Enton",
            text=text,
            metadata={"target_id": target_id},
        )
        await self.send(msg)

    @property
    def is_running(self) -> bool:
        return self._running
