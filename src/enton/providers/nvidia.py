from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from enton.core.config import Settings

logger = logging.getLogger(__name__)

NVIDIA_GRPC_URL = "grpc.nvcf.nvidia.com:443"


class NvidiaSTT:
    """NVIDIA Riva Canary/Parakeet STT via gRPC API."""

    def __init__(self, settings: Settings) -> None:
        import riva.client

        auth = riva.client.Auth(
            ssl_cert=None,
            use_ssl=True,
            uri=NVIDIA_GRPC_URL,
            metadata_args=[
                ["function-id", self._function_id(settings.nvidia_stt_model)],
                ["authorization", f"Bearer {settings.nvidia_api_key}"],
            ],
        )
        self._asr = riva.client.ASRService(auth)
        self._sample_rate = settings.sample_rate

    @staticmethod
    def _function_id(model: str) -> str:
        ids = {
            "parakeet-1.1b-rnnt-multilingual-asr": "71203149-d3b7-4460-8231-1be2543a1fca",
            "parakeet-tdt-0.6b-v2": "a]bc12345-placeholder",
        }
        return ids.get(model, model)

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        import asyncio

        audio_bytes = (audio * 32767).astype(np.int16).tobytes()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._asr.offline_recognize(audio_bytes, self._sample_rate),
        )
        texts = []
        for result in response.results:
            if result.alternatives:
                texts.append(result.alternatives[0].transcript)
        return " ".join(texts)

    async def stream(self) -> AsyncIterator[str]:
        raise NotImplementedError("NVIDIA STT streaming — Phase 2")


class NvidiaTTS:
    """NVIDIA Riva Magpie TTS via gRPC API."""

    def __init__(self, settings: Settings) -> None:
        import riva.client

        auth = riva.client.Auth(
            ssl_cert=None,
            use_ssl=True,
            uri=NVIDIA_GRPC_URL,
            metadata_args=[
                ["function-id", "0149dedb-2be8-4195-b9a0-e57e0e14f972"],
                ["authorization", f"Bearer {settings.nvidia_api_key}"],
            ],
        )
        self._tts = riva.client.SpeechSynthesisService(auth)
        self._voice = settings.nvidia_tts_voice
        self._sample_rate = settings.sample_rate

    async def synthesize(self, text: str) -> np.ndarray:
        import asyncio

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._tts.synthesize(
                text,
                voice_name=self._voice,
                language_code="en-US",
                sample_rate_hz=self._sample_rate,
                encoding=0,  # LINEAR16
            ),
        )
        audio_data = np.frombuffer(response.audio, dtype=np.int16)
        return audio_data.astype(np.float32) / 32767.0

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        yield await self.synthesize(text)
