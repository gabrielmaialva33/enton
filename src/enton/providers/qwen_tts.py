"""Qwen3-TTS local GPU provider with Voice Design.

Uses qwen-tts package for high-quality PT-BR speech synthesis with
customizable voice design via natural language instructions.

Requires: pip install qwen-tts (pulls torch automatically)
VRAM: ~5GB for 0.6B model, ~7GB for 1.7B model
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from enton.core.config import Settings

logger = logging.getLogger(__name__)


class Qwen3TTS:
    """Qwen3-TTS local GPU provider — voice design + synthesis."""

    def __init__(self, settings: Settings) -> None:
        self._model_id = settings.qwen3_tts_model
        self._voice_instruct = settings.qwen3_tts_voice_instruct
        self._device = settings.qwen3_tts_device
        self._model = None
        self._custom_voice = None
        self.sample_rate: int = 24000

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        import torch  # noqa: PLC0415
        from qwen_tts import Qwen3TTSModel  # noqa: PLC0415

        logger.info("Loading Qwen3-TTS model: %s", self._model_id)
        self._model = Qwen3TTSModel.from_pretrained(
            self._model_id,
            device_map=self._device,
            torch_dtype=torch.bfloat16,
        )

        # Voice design — create custom voice once, cache it
        if self._voice_instruct:
            logger.info("Designing custom voice: %s", self._voice_instruct[:60])
            self._custom_voice = self._model.generate_voice_design(
                text="Teste de voz para design de personalidade.",
                language="Portuguese",
                instruct=self._voice_instruct,
            )
            logger.info("Custom voice designed successfully")

        return self._model

    async def synthesize(self, text: str) -> np.ndarray:
        loop = asyncio.get_running_loop()

        def _synth():
            model = self._ensure_model()

            if self._custom_voice is not None:
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    custom_voice=self._custom_voice,
                )
            else:
                wavs, sr = model(text)

            self.sample_rate = sr

            if isinstance(wavs, list):
                if not wavs:
                    return np.array([], dtype=np.float32)
                combined = np.concatenate(wavs)
            else:
                combined = wavs

            if not isinstance(combined, np.ndarray):
                combined = np.array(combined, dtype=np.float32)
            elif combined.dtype != np.float32:
                combined = combined.astype(np.float32)

            return combined

        return await loop.run_in_executor(None, _synth)

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        yield await self.synthesize(text)
