"""Microsoft Edge TTS provider — free cloud neural TTS.

Uses edge-tts package for high-quality PT-BR speech synthesis.
No API key required, uses Microsoft's free neural voice API.

Requires: pip install edge-tts
"""
from __future__ import annotations

import asyncio
import io
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from enton.core.config import Settings

logger = logging.getLogger(__name__)


class EdgeTTS:
    """Microsoft Edge TTS — free cloud fallback for PT-BR."""

    def __init__(self, settings: Settings) -> None:
        self._voice = settings.edge_tts_voice
        self.sample_rate: int = 24000

    async def synthesize(self, text: str) -> np.ndarray:
        import edge_tts
        import soundfile as sf

        communicate = edge_tts.Communicate(text, self._voice)

        # Collect MP3 bytes from stream
        mp3_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_bytes += chunk["data"]

        if not mp3_bytes:
            return np.array([], dtype=np.float32)

        # Decode MP3 → numpy float32
        audio, sr = await asyncio.get_running_loop().run_in_executor(
            None, lambda: sf.read(io.BytesIO(mp3_bytes)),
        )
        self.sample_rate = sr

        # Stereo → mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return audio.astype(np.float32)

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        yield await self.synthesize(text)
