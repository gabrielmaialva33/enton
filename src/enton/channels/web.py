"""Web channel — FastAPI WebSocket for browser-based chat.

Requires: pip install fastapi uvicorn
Config: WEB_CHANNEL_PORT (default: 8765)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from enton.channels.base import BaseChannel, ChannelMessage, MessageType
from enton.core.events import ChannelMessageEvent

if TYPE_CHECKING:
    from enton.core.events import EventBus

logger = logging.getLogger(__name__)


class WebChannel(BaseChannel):
    """WebSocket-based web chat channel."""

    name = "web"

    def __init__(
        self,
        bus: EventBus,
        host: str = "0.0.0.0",
        port: int = 8765,
    ) -> None:
        super().__init__(bus)
        self._host = host
        self._port = port
        self._app = None
        self._server = None
        self._connections: dict[str, object] = {}  # ws_id → websocket

    async def start(self) -> None:
        try:
            from fastapi import FastAPI, WebSocket, WebSocketDisconnect
            import uvicorn
        except ImportError:
            logger.warning("fastapi/uvicorn not installed — Web channel disabled")
            return

        self._app = FastAPI(title="Enton Web Channel")
        bus = self.bus
        connections = self._connections

        @self._app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket) -> None:
            await ws.accept()
            ws_id = str(id(ws))
            connections[ws_id] = ws
            logger.info("Web client connected: %s", ws_id)

            try:
                while True:
                    data = await ws.receive_text()
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        payload = {"text": data}

                    msg = ChannelMessage(
                        channel="web",
                        sender_id=payload.get("user_id", ws_id),
                        sender_name=payload.get("user_name", "WebUser"),
                        text=payload.get("text", ""),
                        message_id=ws_id,
                        metadata={"ws_id": ws_id},
                    )
                    await bus.emit(ChannelMessageEvent(message=msg))
            except WebSocketDisconnect:
                connections.pop(ws_id, None)
                logger.info("Web client disconnected: %s", ws_id)

        @self._app.get("/health")
        async def health():
            return {"status": "ok", "clients": len(connections)}

        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._running = True
        logger.info("Web channel starting on ws://%s:%d/ws", self._host, self._port)
        asyncio.create_task(self._server.serve())

    async def stop(self) -> None:
        self._running = False
        if self._server:
            self._server.should_exit = True
        # Close all WebSocket connections
        for ws_id, ws in list(self._connections.items()):
            try:
                await ws.close()
            except Exception:
                pass
        self._connections.clear()
        logger.info("Web channel stopped")

    async def send(self, message: ChannelMessage) -> None:
        ws_id = message.metadata.get("ws_id") or message.metadata.get("target_id")

        payload = json.dumps({
            "sender": message.sender_name,
            "text": message.text,
            "channel": message.channel,
            "timestamp": message.timestamp,
        })

        if ws_id and ws_id in self._connections:
            # Send to specific client
            try:
                await self._connections[ws_id].send_text(payload)
            except Exception:
                self._connections.pop(ws_id, None)
        else:
            # Broadcast to all
            dead = []
            for wid, ws in self._connections.items():
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(wid)
            for wid in dead:
                self._connections.pop(wid, None)
