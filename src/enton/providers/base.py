from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@runtime_checkable
class STTProvider(Protocol):
    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str: ...
    async def stream(self) -> AsyncIterator[str]:
        yield ""  # pragma: no cover


@runtime_checkable
class TTSProvider(Protocol):
    async def synthesize(self, text: str) -> np.ndarray: ...
    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        yield np.array([])  # pragma: no cover


@runtime_checkable
class LLMProvider(Protocol):
    async def generate(
        self, prompt: str, *, system: str = "", history: list[dict] | None = None
    ) -> str: ...
    async def generate_with_image(
        self, prompt: str, image: bytes, *, system: str = "", mime_type: str = "image/jpeg"
    ) -> str: ...
    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        *,
        system: str = "",
        history: list[dict] | None = None,
    ) -> dict: ...  # Returns {"content": str, "tool_calls": list}
