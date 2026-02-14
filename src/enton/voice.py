from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

from enton.config import Provider

if TYPE_CHECKING:
    from enton.config import Settings
    from enton.providers.base import TTSProvider

logger = logging.getLogger(__name__)


class Voice:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._providers: dict[Provider, TTSProvider] = {}
        self._primary = settings.tts_provider
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._speaking = False
        self._init_providers(settings)

    def _init_providers(self, s: Settings) -> None:
        if s.nvidia_api_key:
            try:
                from enton.providers.nvidia import NvidiaTTS

                self._providers[Provider.NVIDIA] = NvidiaTTS(s)
            except Exception:
                logger.warning("NVIDIA TTS unavailable")

        try:
            from enton.providers.google import GoogleTTS

            self._providers[Provider.GOOGLE] = GoogleTTS(s)
        except Exception:
            logger.warning("Google TTS unavailable")

        try:
            from enton.providers.local import LocalTTS

            self._providers[Provider.LOCAL] = LocalTTS(s)
        except Exception:
            logger.warning("Local TTS unavailable")

    def _get_provider(self) -> tuple[Provider, TTSProvider]:
        if self._primary in self._providers:
            return self._primary, self._providers[self._primary]
        for name, provider in self._providers.items():
            return name, provider
        raise RuntimeError("No TTS provider available")

    async def say(self, text: str) -> None:
        await self._queue.put(text)

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    async def run(self) -> None:
        while True:
            text = await self._queue.get()
            if not text.strip():
                continue
            self._speaking = True
            try:
                await self._speak(text)
            except Exception:
                logger.exception("TTS failed")
            finally:
                self._speaking = False

    async def _speak(self, text: str) -> None:
        name, provider = self._get_provider()
        try:
            audio = await provider.synthesize(text)
            await self._play(audio)
            logger.info("Voice [%s]: %s", name, text[:60])
        except Exception:
            logger.warning("TTS [%s] failed, trying fallback", name)
            if name != Provider.LOCAL and Provider.LOCAL in self._providers:
                audio = await self._providers[Provider.LOCAL].synthesize(text)
                await self._play(audio)
            else:
                raise

    async def _play(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        loop = asyncio.get_running_loop()
        sample_rate = self._settings.sample_rate
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Kokoro outputs at 24000 Hz
        if sample_rate != 24000 and audio.size > 0:
            pass  # let sounddevice handle it

        def _play_sync():
            sd.play(audio, samplerate=24000, blocking=True)

        await loop.run_in_executor(None, _play_sync)
