"""Telegram channel — aiogram-based async Telegram bot.

Requires: pip install aiogram>=3.15
Config: TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USERS (comma-separated IDs)
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


class TelegramChannel(BaseChannel):
    """Telegram bot channel via aiogram 3.x."""

    name = "telegram"

    def __init__(
        self,
        bus: EventBus,
        token: str,
        allowed_users: list[str] | None = None,
    ) -> None:
        super().__init__(bus)
        self._token = token
        self._allowed_users = set(allowed_users or [])
        self._bot = None
        self._dp = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        try:
            from aiogram import Bot, Dispatcher
            from aiogram.types import Message as TgMessage
        except ImportError:
            logger.warning("aiogram not installed — Telegram channel disabled")
            return

        self._bot = Bot(token=self._token)
        self._dp = Dispatcher()

        @self._dp.message()
        async def on_message(tg_msg: TgMessage) -> None:
            if not tg_msg.text and not tg_msg.photo:
                return

            user_id = str(tg_msg.from_user.id) if tg_msg.from_user else "unknown"
            user_name = tg_msg.from_user.full_name if tg_msg.from_user else "unknown"

            # Access control
            if self._allowed_users and user_id not in self._allowed_users:
                await tg_msg.reply("Acesso negado. Fala com o Gabriel.")
                return

            # Handle photo messages
            media = None
            msg_type = MessageType.TEXT
            if tg_msg.photo:
                msg_type = MessageType.IMAGE
                photo = tg_msg.photo[-1]  # largest resolution
                file = await self._bot.get_file(photo.file_id)
                bio = await self._bot.download_file(file.file_path)
                media = bio.read() if bio else None

            msg = ChannelMessage(
                channel=self.name,
                sender_id=user_id,
                sender_name=user_name,
                text=tg_msg.text or tg_msg.caption or "",
                message_type=msg_type,
                media=media,
                message_id=str(tg_msg.message_id),
                metadata={"chat_id": str(tg_msg.chat.id)},
            )

            await self.bus.emit(ChannelMessageEvent(message=msg))

        self._running = True
        logger.info("Telegram channel starting (polling)...")

        # Run polling in background task
        self._task = asyncio.create_task(
            self._dp.start_polling(self._bot),
        )

    async def stop(self) -> None:
        self._running = False
        if self._dp:
            await self._dp.stop_polling()
        if self._bot:
            await self._bot.session.close()
        if self._task:
            self._task.cancel()
        logger.info("Telegram channel stopped")

    async def send(self, message: ChannelMessage) -> None:
        if not self._bot:
            return

        chat_id = message.metadata.get("target_id") or message.metadata.get("chat_id")
        if not chat_id:
            logger.warning("Telegram send: no target chat_id")
            return

        try:
            if message.media and message.message_type == MessageType.IMAGE:
                from aiogram.types import BufferedInputFile

                photo = BufferedInputFile(message.media, filename="enton.jpg")
                await self._bot.send_photo(
                    chat_id=int(chat_id),
                    photo=photo,
                    caption=message.text[:1024] if message.text else None,
                )
            else:
                # Split long messages (Telegram limit: 4096 chars)
                text = message.text or "..."
                for i in range(0, len(text), 4096):
                    await self._bot.send_message(
                        chat_id=int(chat_id),
                        text=text[i : i + 4096],
                    )
        except Exception:
            logger.exception("Telegram send error")
