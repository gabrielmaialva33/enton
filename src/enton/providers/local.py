from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from enton.core.config import Settings

logger = logging.getLogger(__name__)


class LocalSTT:
    """faster-whisper local STT fallback."""

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.whisper_model
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_name,
                device="cuda",
                compute_type="float16",
            )
        return self._model

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        import asyncio

        model = self._ensure_model()
        loop = asyncio.get_running_loop()

        def _transcribe():
            segments, _ = model.transcribe(audio, language="pt", beam_size=5)
            return " ".join(seg.text for seg in segments)

        return await loop.run_in_executor(None, _transcribe)

    async def stream(self) -> AsyncIterator[str]:
        raise NotImplementedError("Local STT streaming — Phase 2")


class LocalTTS:
    """Kokoro local TTS fallback."""

    def __init__(self, settings: Settings) -> None:
        self._lang = settings.kokoro_lang
        self._voice = settings.kokoro_voice
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is None:
            import torch
            from kokoro import KPipeline

            self._pipeline = KPipeline(lang_code=self._lang)
            if hasattr(self._pipeline, "model") and self._pipeline.model is not None:
                self._pipeline.model = self._pipeline.model.to(torch.device("cpu"))
        return self._pipeline

    async def synthesize(self, text: str) -> np.ndarray:
        import asyncio

        pipeline = self._ensure_pipeline()
        loop = asyncio.get_running_loop()

        def _synth():
            chunks = []
            for _, _, audio in pipeline(text, voice=self._voice, speed=1.0):
                chunks.append(audio)
            if not chunks:
                return np.array([], dtype=np.float32)
            return np.concatenate(chunks)

        return await loop.run_in_executor(None, _synth)

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        yield await self.synthesize(text)
