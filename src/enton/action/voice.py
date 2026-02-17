from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

from enton.core.config import Provider
from enton.providers.edge_tts_provider import EdgeTTS
from enton.providers.google import GoogleTTS
from enton.providers.local import LocalTTS
from enton.providers.nvidia import NvidiaTTS
from enton.providers.qwen_tts import Qwen3TTS

if TYPE_CHECKING:
    from enton.core.config import Settings
    from enton.providers.base import TTSProvider

logger = logging.getLogger(__name__)


class Voice:
    def __init__(self, settings: Settings, ears=None) -> None:
        self._settings = settings
        self._providers: dict[Provider, TTSProvider] = {}
        self._primary = settings.tts_provider
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._speaking = False
        self._ears = ears
        self._init_providers(settings)

    def _init_providers(self, s: Settings) -> None:
        # Qwen3-TTS — primary local GPU with voice design
        try:
            self._providers[Provider.QWEN3] = Qwen3TTS(s)
        except Exception:
            logger.warning("Qwen3 TTS unavailable")

        # Edge-TTS — free cloud fallback
        try:
            self._providers[Provider.EDGE] = EdgeTTS(s)
        except Exception:
            logger.warning("Edge TTS unavailable")

        if s.nvidia_api_key:
            try:
                self._providers[Provider.NVIDIA] = NvidiaTTS(s)
            except Exception:
                logger.warning("NVIDIA TTS unavailable")

        if s.google_project:
            try:
                self._providers[Provider.GOOGLE] = GoogleTTS(s)
            except Exception:
                logger.warning("Google TTS unavailable")

        if s.kokoro_voice:
            try:
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
            if self._ears:
                self._ears.muted = True
            try:
                await self._speak(text)
            except Exception:
                logger.exception("TTS failed")
            finally:
                self._speaking = False
                if self._ears:
                    self._ears.muted = False

    # Fallback order for TTS providers
    _FALLBACK_ORDER = [Provider.QWEN3, Provider.EDGE, Provider.LOCAL, Provider.GOOGLE]

    async def _speak(self, text: str) -> None:
        name, provider = self._get_provider()
        try:
            audio = await provider.synthesize(text)
            sr = getattr(provider, "sample_rate", 24000)
            await self._play(audio, sample_rate=sr)
            logger.info("Voice [%s]: %s", name, text[:60])
        except Exception:
            logger.warning("TTS [%s] failed, trying fallback", name)
            for fallback in self._FALLBACK_ORDER:
                if fallback != name and fallback in self._providers:
                    try:
                        fb = self._providers[fallback]
                        audio = await fb.synthesize(text)
                        sr = getattr(fb, "sample_rate", 24000)
                        await self._play(audio, sample_rate=sr)
                        logger.info("Voice fallback [%s]: %s", fallback, text[:60])
                        return
                    except Exception:
                        logger.warning("TTS fallback [%s] also failed", fallback)
            raise

    async def _play(self, audio: np.ndarray, sample_rate: int = 24000) -> None:
        if audio.size == 0:
            return
        loop = asyncio.get_running_loop()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        rate = sample_rate

        def _play_sync():
            sd.play(audio, samplerate=rate, blocking=True)

        await loop.run_in_executor(None, _play_sync)
