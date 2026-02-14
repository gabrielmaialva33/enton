from __future__ import annotations

import logging
import re
from collections import deque
from typing import TYPE_CHECKING

from enton.config import Provider

if TYPE_CHECKING:
    from enton.config import Settings
    from enton.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class Brain:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._providers: dict[Provider, LLMProvider] = {}
        self._primary = settings.brain_provider
        self._history: deque[dict] = deque(maxlen=settings.memory_size)
        self._init_providers(settings)

    def _init_providers(self, s: Settings) -> None:
        try:
            from enton.providers.google import GoogleLLM

            self._providers[Provider.GOOGLE] = GoogleLLM(s)
        except Exception:
            logger.warning("Google LLM unavailable")

        try:
            from enton.providers.local import LocalLLM

            self._providers[Provider.LOCAL] = LocalLLM(s)
        except Exception:
            logger.warning("Local LLM unavailable")

    def _get_provider(self) -> tuple[Provider, LLMProvider]:
        if self._primary in self._providers:
            return self._primary, self._providers[self._primary]
        for name, provider in self._providers.items():
            return name, provider
        raise RuntimeError("No LLM provider available")

    @staticmethod
    def _clean_response(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    async def think(self, prompt: str, *, system: str = "") -> str:
        name, provider = self._get_provider()
        try:
            raw = await provider.generate(prompt, system=system, history=list(self._history))
            response = self._clean_response(raw)
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": response})
            logger.info("Brain [%s]: %s", name, response[:80])
            return response
        except Exception:
            logger.exception("Brain [%s] failed", name)
            if name != Provider.LOCAL and Provider.LOCAL in self._providers:
                return await self._providers[Provider.LOCAL].generate(
                    prompt, system=system, history=list(self._history)
                )
            raise

    async def describe_scene(self, image: bytes, *, system: str = "") -> str:
        name, provider = self._get_provider()
        try:
            return await provider.generate_with_image(
                "Describe what you see briefly in Portuguese.", image, system=system
            )
        except Exception:
            logger.exception("Vision LLM [%s] failed", name)
            if Provider.LOCAL in self._providers:
                return await self._providers[Provider.LOCAL].generate_with_image(
                    "Describe what you see briefly in Portuguese.", image, system=system
                )
            return ""

    def clear_history(self) -> None:
        self._history.clear()
