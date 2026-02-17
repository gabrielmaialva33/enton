"""Voice channel — wraps existing STT/TTS into channel abstraction.

This bridges the existing Ears (STT) → Brain → Voice (TTS) pipeline
into the unified channel system, so voice becomes just another channel.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from enton.channels.base import BaseChannel, ChannelMessage, MessageType

if TYPE_CHECKING:
    from enton.action.voice import Voice as VoiceEngine
    from enton.core.events import EventBus

logger = logging.getLogger(__name__)


class VoiceChannel(BaseChannel):
    """Voice channel — STT input, TTS output."""

    name = "voice"

    def __init__(self, bus: EventBus, voice_engine: VoiceEngine) -> None:
        super().__init__(bus)
        self._voice = voice_engine

    async def start(self) -> None:
        self._running = True
        logger.info("Voice channel active (STT/TTS bridge)")

    async def stop(self) -> None:
        self._running = False

    async def send(self, message: ChannelMessage) -> None:
        """Send message via TTS (speak it out loud)."""
        if message.text and self._voice:
            await self._voice.say(message.text)
