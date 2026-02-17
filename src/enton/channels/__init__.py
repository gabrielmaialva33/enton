"""Enton Channels — Multi-platform messaging abstraction.

Channels connect Enton to the outside world: Telegram, Discord, WhatsApp,
Web, CLI, and any future platform. Each channel translates platform-specific
messages into a unified `ChannelMessage` format and routes them through the
central `ChannelManager`.

Architecture (Nanobot/AstrBot-inspired):

    Platform → Channel.listen() → ChannelMessage → EventBus → Brain
    Brain response → ChannelManager.reply() → Channel.send() → Platform
"""

from enton.channels.base import BaseChannel, ChannelMessage, MessageType

__all__ = ["BaseChannel", "ChannelMessage", "MessageType"]
