from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from enton.config import Provider
from enton.events import EventBus, TranscriptionEvent

if TYPE_CHECKING:
    from enton.config import Settings
    from enton.providers.base import STTProvider

logger = logging.getLogger(__name__)


class Ears:
    def __init__(self, settings: Settings, bus: EventBus) -> None:
        self._settings = settings
        self._bus = bus
        self._providers: dict[Provider, STTProvider] = {}
        self._primary = settings.stt_provider
        self._init_providers(settings)

    def _init_providers(self, s: Settings) -> None:
        if s.nvidia_api_key:
            try:
                from enton.providers.nvidia import NvidiaSTT

                self._providers[Provider.NVIDIA] = NvidiaSTT(s)
            except Exception:
                logger.warning("NVIDIA STT unavailable")

        try:
            from enton.providers.google import GoogleSTT

            self._providers[Provider.GOOGLE] = GoogleSTT(s)
        except Exception:
            logger.warning("Google STT unavailable")

        try:
            from enton.providers.local import LocalSTT

            self._providers[Provider.LOCAL] = LocalSTT(s)
        except Exception:
            logger.warning("Local STT unavailable")

    def _get_provider(self) -> tuple[Provider, STTProvider]:
        if self._primary in self._providers:
            return self._primary, self._providers[self._primary]
        for name, provider in self._providers.items():
            return name, provider
        raise RuntimeError("No STT provider available")

    async def transcribe(self, audio: np.ndarray) -> str:
        name, provider = self._get_provider()
        try:
            text = await provider.transcribe(audio, self._settings.sample_rate)
            if text.strip():
                await self._bus.emit(TranscriptionEvent(text=text))
                logger.info("Ears [%s]: %s", name, text[:80])
            return text
        except Exception:
            logger.exception("STT [%s] failed", name)
            if name != Provider.LOCAL and Provider.LOCAL in self._providers:
                return await self._providers[Provider.LOCAL].transcribe(
                    audio, self._settings.sample_rate
                )
            return ""

    async def run(self) -> None:
        """Continuous mic capture loop. Phase 2: streaming mic → STT → events."""
        import asyncio

        import sounddevice as sd

        logger.info("Ears listening on default mic (sample_rate=%d)", self._settings.sample_rate)
        chunk_duration = 3.0  # seconds per chunk
        chunk_samples = int(self._settings.sample_rate * chunk_duration)

        while True:
            loop = asyncio.get_running_loop()

            def _record():
                return sd.rec(
                    chunk_samples,
                    samplerate=self._settings.sample_rate,
                    channels=1,
                    dtype=np.float32,
                    blocking=True,
                )

            audio = await loop.run_in_executor(None, _record)
            audio = audio.squeeze()

            # Skip silence
            if np.abs(audio).max() < 0.01:
                continue

            await self.transcribe(audio)
