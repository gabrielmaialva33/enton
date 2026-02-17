"""ChannelTools — Agno toolkit for multi-platform messaging.

Gives the brain the ability to send messages through any registered channel
(Telegram, Discord, Web, Voice) via tool calling.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from agno.tools import Toolkit

if TYPE_CHECKING:
    from enton.channels.manager import ChannelManager

logger = logging.getLogger(__name__)


class ChannelTools(Toolkit):
    """Tools for sending messages through Enton's channels."""

    def __init__(self, channel_manager: ChannelManager) -> None:
        super().__init__(name="channel_tools")
        self._cm = channel_manager
        self.register(self.send_message)
        self.register(self.broadcast_message)
        self.register(self.list_channels)

    def send_message(self, channel: str, target_id: str, text: str) -> str:
        """Send a message to a specific user/chat on a channel.

        Args:
            channel: Channel name ("telegram", "discord", "web", "voice")
            target_id: Chat/user/channel ID on the platform
            text: Message text to send
        """
        ch = self._cm.get(channel)
        if not ch:
            available = ", ".join(self._cm.active_channels)
            return f"Canal '{channel}' nao encontrado. Disponiveis: {available}"
        if not ch.is_running:
            return f"Canal '{channel}' nao esta ativo."

        try:
            loop = asyncio.get_running_loop()
            _ = loop.create_task(ch.send_text(target_id, text))
            return f"Mensagem enviada via {channel} para {target_id}"
        except Exception as e:
            return f"Erro ao enviar: {e}"

    def broadcast_message(self, text: str) -> str:
        """Broadcast a message to ALL active channels simultaneously.

        Args:
            text: Message text to broadcast
        """
        active = self._cm.active_channels
        if not active:
            return "Nenhum canal ativo para broadcast."

        try:
            loop = asyncio.get_running_loop()
            _ = loop.create_task(self._cm.broadcast(text))
            return f"Broadcast enviado para: {', '.join(active)}"
        except Exception as e:
            return f"Erro no broadcast: {e}"

    def list_channels(self) -> str:
        """List all registered channels and their status."""
        lines = []
        for name in sorted(self._cm._channels.keys()):
            ch = self._cm._channels[name]
            status = "ATIVO" if ch.is_running else "INATIVO"
            lines.append(f"  {name}: {status}")
        if not lines:
            return "Nenhum canal registrado."
        return "Canais:\n" + "\n".join(lines)
