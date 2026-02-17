from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from enton.core.config import Settings

logger = logging.getLogger(__name__)


class GoogleLLM:
    def __init__(self, settings: Settings) -> None:
        from google import genai

        self._client = genai.Client(
            vertexai=True,
            project=settings.google_project,
            location=settings.google_location,
        )
        self._model = settings.google_brain_model
        self._vision_model = settings.google_vision_model

    async def generate(
        self, prompt: str, *, system: str = "", history: list[dict] | None = None
    ) -> str:
        from google.genai import types

        contents = []
        if history:
            for msg in history:
                contents.append(
                    types.Content(
                        role=msg["role"],
                        parts=[types.Part.from_text(text=msg["content"])],
                    )
                )
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        )

        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=0.9,
            max_output_tokens=256,
        )

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        return response.text or ""

    async def generate_with_image(
        self,
        prompt: str,
        image: bytes,
        *,
        system: str = "",
        mime_type: str = "image/jpeg",
    ) -> str:
        from google.genai import types

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image, mime_type=mime_type),
                    types.Part.from_text(text=prompt),
                ],
            )
        ]

        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=0.9,
            max_output_tokens=512,
        )

        response = await self._client.aio.models.generate_content(
            model=self._vision_model,
            contents=contents,
            config=config,
        )
        return response.text or ""


class GoogleSTT:
    def __init__(self, settings: Settings) -> None:
        from google.cloud.speech_v2 import SpeechAsyncClient
        from google.cloud.speech_v2.types import cloud_speech

        self._client = SpeechAsyncClient()
        self._project = settings.google_project
        self._recognizer = f"projects/{settings.google_project}/locations/global/recognizers/_"
        self._config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=["pt-BR", "en-US"],
            model="chirp_2",
        )
        self._sample_rate = settings.sample_rate

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        from google.cloud.speech_v2.types import cloud_speech

        audio_bytes = (audio * 32767).astype(np.int16).tobytes()

        request = cloud_speech.RecognizeRequest(
            recognizer=self._recognizer,
            config=self._config,
            content=audio_bytes,
        )

        response = await self._client.recognize(request=request)
        texts = []
        for result in response.results:
            if result.alternatives:
                texts.append(result.alternatives[0].transcript)
        return " ".join(texts)

    async def stream(self) -> AsyncIterator[str]:
        raise NotImplementedError("Google STT streaming requires gRPC bidirectional — Phase 2")


class GoogleTTS:
    def __init__(self, settings: Settings) -> None:
        from google.cloud import texttospeech_v1 as tts

        self._client = tts.TextToSpeechAsyncClient()
        self._voice = tts.VoiceSelectionParams(
            language_code="pt-BR",
            name="pt-BR-Neural2-B",
        )
        self._audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16,
            sample_rate_hertz=settings.sample_rate,
        )

    async def synthesize(self, text: str) -> np.ndarray:
        from google.cloud import texttospeech_v1 as tts

        request = tts.SynthesizeSpeechRequest(
            input=tts.SynthesisInput(text=text),
            voice=self._voice,
            audio_config=self._audio_config,
        )
        response = await self._client.synthesize_speech(request=request)
        audio_data = np.frombuffer(response.audio_content, dtype=np.int16)
        return audio_data.astype(np.float32) / 32767.0

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        yield await self.synthesize(text)
