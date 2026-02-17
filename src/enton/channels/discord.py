"""Discord channel — pycord-based async Discord bot.

Requires: pip install py-cord>=2.6
Config: DISCORD_BOT_TOKEN, DISCORD_ALLOWED_GUILDS (comma-separated IDs)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from enton.channels.base import BaseChannel, ChannelMessage, MessageType
from enton.core.events import ChannelMessageEvent

if TYPE_CHECKING:
    from enton.core.events import EventBus

logger = logging.getLogger(__name__)


class DiscordChannel(BaseChannel):
    """Discord bot channel via py-cord."""

    name = "discord"

    def __init__(
        self,
        bus: EventBus,
        token: str,
        allowed_guilds: list[str] | None = None,
    ) -> None:
        super().__init__(bus)
        self._token = token
        self._allowed_guilds = set(allowed_guilds or [])
        self._bot = None
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()

    async def start(self) -> None:
        try:
            import discord
        except ImportError:
            logger.warning("py-cord not installed — Discord channel disabled")
            return

        intents = discord.Intents.default()
        intents.message_content = True
        self._bot = discord.Bot(intents=intents)
        bus = self.bus
        allowed = self._allowed_guilds
        ready_event = self._ready

        @self._bot.event
        async def on_ready() -> None:
            logger.info("Discord bot ready: %s", self._bot.user)
            ready_event.set()

        @self._bot.event
        async def on_message(dc_msg: discord.Message) -> None:
            if dc_msg.author == self._bot.user:
                return
            if dc_msg.author.bot:
                return
            if allowed and str(dc_msg.guild.id) not in allowed:
                return

            # Handle attachments (images)
            media = None
            msg_type = MessageType.TEXT
            if dc_msg.attachments:
                att = dc_msg.attachments[0]
                if att.content_type and att.content_type.startswith("image/"):
                    msg_type = MessageType.IMAGE
                    media = await att.read()

            msg = ChannelMessage(
                channel="discord",
                sender_id=str(dc_msg.author.id),
                sender_name=dc_msg.author.display_name,
                text=dc_msg.content or "",
                message_type=msg_type,
                media=media,
                message_id=str(dc_msg.id),
                metadata={
                    "target_id": str(dc_msg.channel.id),
                    "guild_id": str(dc_msg.guild.id) if dc_msg.guild else "",
                },
            )

            await bus.emit(ChannelMessageEvent(message=msg))

        self._running = True
        logger.info("Discord channel starting...")
        self._task = asyncio.create_task(self._bot.start(self._token))

    async def stop(self) -> None:
        self._running = False
        if self._bot:
            await self._bot.close()
        if self._task:
            self._task.cancel()
        logger.info("Discord channel stopped")

    async def send(self, message: ChannelMessage) -> None:
        if not self._bot:
            return

        await self._ready.wait()

        channel_id = message.metadata.get("target_id")
        if not channel_id:
            logger.warning("Discord send: no target channel_id")
            return

        try:
            channel = self._bot.get_channel(int(channel_id))
            if not channel:
                channel = await self._bot.fetch_channel(int(channel_id))

            text = message.text or "..."
            # Discord limit: 2000 chars
            for i in range(0, len(text), 2000):
                await channel.send(text[i : i + 2000])
        except Exception:
            logger.exception("Discord send error")
