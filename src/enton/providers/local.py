from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from enton.config import Settings

logger = logging.getLogger(__name__)


class LocalLLM:
    """Ollama local LLM fallback."""

    def __init__(self, settings: Settings) -> None:
        self._model = settings.ollama_model

    async def generate(
        self, prompt: str, *, system: str = "", history: list[dict] | None = None
    ) -> str:
        import ollama

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        response = await ollama.AsyncClient().chat(model=self._model, messages=messages)
        return response.message.content or ""

    async def generate_with_image(
        self,
        prompt: str,
        image: bytes,
        *,
        system: str = "",
        mime_type: str = "image/jpeg",
    ) -> str:
        import base64

        import ollama

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": prompt,
                "images": [base64.b64encode(image).decode()],
            }
        )

        response = await ollama.AsyncClient().chat(model="llava:7b", messages=messages)
        return response.message.content or ""


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
            from kokoro import KPipeline

            self._pipeline = KPipeline(lang_code=self._lang)
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
