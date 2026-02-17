"""Channel Manager — orchestrates all messaging channels.

Routes incoming ChannelMessages to the brain and dispatches responses
back to the originating channel. Integrates with EventBus for full
event-driven flow.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from enton.channels.base import BaseChannel, ChannelMessage
from enton.core.events import ChannelMessageEvent

if TYPE_CHECKING:
    from enton.cognition.brain import EntonBrain
    from enton.core.events import EventBus
    from enton.core.memory import Memory

logger = logging.getLogger(__name__)


class ChannelManager:
    """Manages channel lifecycle and message routing."""

    def __init__(
        self,
        bus: EventBus,
        brain: EntonBrain,
        memory: Memory,
    ) -> None:
        self.bus = bus
        self.brain = brain
        self.memory = memory
        self._channels: dict[str, BaseChannel] = {}
        self._reply_queues: dict[str, asyncio.Queue[ChannelMessage]] = {}

    def register(self, channel: BaseChannel) -> None:
        """Register a channel for management."""
        self._channels[channel.name] = channel
        self._reply_queues[channel.name] = asyncio.Queue()
        logger.info("Channel registered: %s", channel.name)

    def get(self, name: str) -> BaseChannel | None:
        return self._channels.get(name)

    @property
    def active_channels(self) -> list[str]:
        return [n for n, ch in self._channels.items() if ch.is_running]

    async def start_all(self) -> None:
        """Start all registered channels concurrently."""
        tasks = []
        for name, channel in self._channels.items():
            logger.info("Starting channel: %s", name)
            tasks.append(channel.start())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self) -> None:
        """Stop all running channels."""
        tasks = []
        for name, channel in self._channels.items():
            if channel.is_running:
                logger.info("Stopping channel: %s", name)
                tasks.append(channel.stop())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def handle_message(self, event: ChannelMessageEvent) -> str:
        """Process an incoming channel message through the brain.

        Returns the brain response text.
        """
        msg = event.message
        logger.info(
            "[%s] %s: %s",
            msg.channel, msg.sender_name, msg.text[:80],
        )

        # Build context-aware system prompt
        system = (
            f"Voce e o Enton, um assistente AI zoeiro brasileiro. "
            f"Voce esta respondendo via {msg.channel}. "
            f"O usuario {msg.sender_name} disse algo. "
            f"Responda de forma natural, breve e zoeira em pt-BR."
        )

        # Use VLM if message has image
        if msg.has_media and msg.media:
            response = await self.brain.describe_scene(
                msg.media, system=system,
            )
        else:
            response = await self.brain.think_agent(
                msg.text, system=system,
            )

        # Send response back through the channel
        if response:
            channel = self._channels.get(msg.channel)
            if channel and channel.is_running:
                reply = ChannelMessage(
                    channel=msg.channel,
                    sender_id="enton",
                    sender_name="Enton",
                    text=response,
                    reply_to=msg.message_id,
                    metadata={"target_id": msg.sender_id},
                )
                await channel.send(reply)

        return response or ""

    async def broadcast(self, text: str, exclude: str = "") -> None:
        """Send a message to all active channels."""
        for name, channel in self._channels.items():
            if name == exclude or not channel.is_running:
                continue
            try:
                await channel.send_text("", text)
            except Exception:
                logger.warning("Broadcast failed for %s", name, exc_info=True)

    async def run(self) -> None:
        """Main loop — listen for channel events via EventBus."""
        self.bus.on(ChannelMessageEvent, self._on_channel_message)
        await self.start_all()
        logger.info(
            "ChannelManager running — active: %s",
            ", ".join(self.active_channels) or "none",
        )
        # Keep alive (channels run their own loops)
        try:
            while True:
                await asyncio.sleep(60)
        finally:
            await self.stop_all()

    async def _on_channel_message(self, event: ChannelMessageEvent) -> None:
        """EventBus handler for incoming channel messages."""
        await self.handle_message(event)
