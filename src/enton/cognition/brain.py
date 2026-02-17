from __future__ import annotations

import logging
import re
from collections import deque
from typing import TYPE_CHECKING

from enton.core.config import Provider

if TYPE_CHECKING:
    from enton.core.config import Settings
    from enton.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class Brain:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._providers: dict[Provider, LLMProvider] = {}
        self._primary = settings.brain_provider
        self._history: deque[dict] = deque(maxlen=settings.memory_size)
        self._vlm = None  # QwenVL on-demand
        self._init_providers(settings)

    # Ordered fallback chain (7 providers!)
    _FALLBACK_ORDER = [
        Provider.LOCAL, Provider.NVIDIA, Provider.HUGGINGFACE,
        Provider.GROQ, Provider.OPENROUTER, Provider.AIMLAPI,
        Provider.GOOGLE,
    ]

    def _init_providers(self, s: Settings) -> None:
        # Local (Ollama) — always try
        try:
            from enton.providers.local import LocalLLM

            self._providers[Provider.LOCAL] = LocalLLM(s)
        except Exception:
            logger.warning("Local LLM unavailable")

        # NVIDIA NIM — round-robin multi-key
        nvidia_keys = [k.strip() for k in s.nvidia_api_keys.split(",") if k.strip()]
        if not nvidia_keys and s.nvidia_api_key:
            nvidia_keys = [s.nvidia_api_key]
        if nvidia_keys:
            try:
                from enton.providers.openai_compat import OpenAICompatLLM

                self._providers[Provider.NVIDIA] = OpenAICompatLLM(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_keys=nvidia_keys,
                    model=s.nvidia_nim_model,
                    vision_model=s.nvidia_nim_vision_model,
                    name="nvidia-nim",
                )
                rpm = len(nvidia_keys) * 40
                logger.info("NVIDIA NIM: %d keys, %d RPM", len(nvidia_keys), rpm)
            except Exception:
                logger.warning("NVIDIA NIM unavailable")

        # HuggingFace Inference API (Pro account)
        if s.huggingface_token:
            try:
                from enton.providers.openai_compat import OpenAICompatLLM

                self._providers[Provider.HUGGINGFACE] = OpenAICompatLLM(
                    base_url="https://api-inference.huggingface.co/v1",
                    api_keys=[s.huggingface_token],
                    model=s.huggingface_model,
                    vision_model=s.huggingface_vision_model,
                    name="huggingface",
                )
            except Exception:
                logger.warning("HuggingFace Inference unavailable")

        # Groq — free tier
        if s.groq_api_key:
            try:
                from enton.providers.openai_compat import OpenAICompatLLM

                self._providers[Provider.GROQ] = OpenAICompatLLM(
                    base_url="https://api.groq.com/openai/v1",
                    api_keys=[s.groq_api_key],
                    model=s.groq_model,
                    name="groq",
                )
            except Exception:
                logger.warning("Groq LLM unavailable")

        # OpenRouter — free multi-provider router
        if s.openrouter_api_key:
            try:
                from enton.providers.openai_compat import OpenAICompatLLM

                self._providers[Provider.OPENROUTER] = OpenAICompatLLM(
                    base_url="https://openrouter.ai/api/v1",
                    api_keys=[s.openrouter_api_key],
                    model=s.openrouter_model,
                    vision_model=s.openrouter_vision_model,
                    name="openrouter",
                )
            except Exception:
                logger.warning("OpenRouter unavailable")

        # AIML API
        if s.aimlapi_api_key:
            try:
                from enton.providers.openai_compat import OpenAICompatLLM

                self._providers[Provider.AIMLAPI] = OpenAICompatLLM(
                    base_url="https://api.aimlapi.com/v1",
                    api_keys=[s.aimlapi_api_key],
                    model=s.aimlapi_model,
                    name="aimlapi",
                )
            except Exception:
                logger.warning("AIML API unavailable")

        # Google Gemini — cloud
        try:
            from enton.providers.google import GoogleLLM

            self._providers[Provider.GOOGLE] = GoogleLLM(s)
        except Exception:
            logger.warning("Google LLM unavailable")

        available = ", ".join(str(p) for p in self._providers)
        logger.info("Brain providers: [%s] (primary: %s)", available, self._primary)

    def _get_provider(self) -> tuple[Provider, LLMProvider]:
        if self._primary in self._providers:
            return self._primary, self._providers[self._primary]
        # Fall through the chain
        for p in self._FALLBACK_ORDER:
            if p in self._providers:
                return p, self._providers[p]
        raise RuntimeError("No LLM provider available")

    @staticmethod
    def _clean_response(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    def _get_fallback_chain(
        self, exclude: Provider | None = None,
    ) -> list[tuple[Provider, LLMProvider]]:
        """Get all available providers in fallback order, excluding one."""
        return [
            (p, self._providers[p])
            for p in self._FALLBACK_ORDER
            if p != exclude and p in self._providers
        ]

    async def think(self, prompt: str, *, system: str = "") -> str:
        name, provider = self._get_provider()
        chain = [(name, provider), *self._get_fallback_chain(exclude=name)]

        for pname, prov in chain:
            try:
                raw = await prov.generate(
                    prompt, system=system, history=list(self._history),
                )
                response = self._clean_response(raw)
                self._history.append({"role": "user", "content": prompt})
                self._history.append({"role": "assistant", "content": response})
                logger.info("Brain [%s]: %s", pname, response[:80])
                return response
            except Exception:
                logger.warning("Brain [%s] failed, trying next", pname)

        raise RuntimeError("All LLM providers failed")

    def _get_vlm(self):
        """Lazy-load QwenVL transformers provider."""
        if self._vlm is None:
            try:
                from enton.providers.qwen_vl import QwenVL

                self._vlm = QwenVL(
                    model_id=self._settings.vlm_transformers_model,
                    device=self._settings.yolo_device,
                )
            except Exception:
                logger.debug("QwenVL transformers provider unavailable")
        return self._vlm

    async def describe_scene(self, image: bytes, *, system: str = "") -> str:
        """Describe scene via VLM. Chain: Ollama → NVIDIA → Transformers → Google."""
        prompt = "Descreva brevemente o que você vê em português."
        if system:
            prompt = f"{system}\n\n{prompt}"

        # 1) Try all providers with vision support in order
        vision_order = [
            Provider.LOCAL, Provider.NVIDIA, Provider.HUGGINGFACE,
            Provider.OPENROUTER, Provider.GOOGLE,
        ]
        for p in vision_order:
            if p in self._providers:
                try:
                    return await self._providers[p].generate_with_image(prompt, image)
                except Exception:
                    logger.warning("VLM [%s] failed, trying next", p)

        # 2) Try transformers VLM (on-demand, last resort local)
        vlm = self._get_vlm()
        if vlm is not None:
            try:
                return await vlm.describe(prompt, image)
            except Exception:
                logger.warning("Transformers VLM failed")

        return ""

    async def _try_generate_with_tools(
        self, prompt: str, tools: list[dict], *, system: str = "",
    ) -> dict | None:
        """Try all providers in fallback order for tool calling."""
        name, provider = self._get_provider()
        chain = [(name, provider), *self._get_fallback_chain(exclude=name)]

        for pname, prov in chain:
            try:
                return await prov.generate_with_tools(
                    prompt, tools=tools, system=system,
                    history=list(self._history),
                )
            except Exception:
                logger.warning("Brain [%s] tools failed, trying next", pname)
        return None

    async def think_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        tool_functions: dict[str, callable],
        *,
        system: str = "",
    ) -> str:
        # Multi-turn: prompt → generate → tool_call → execute → result → ...
        current_prompt = prompt
        content = ""

        for i in range(self._settings.brain_max_turns):
            response = await self._try_generate_with_tools(
                current_prompt, tools, system=system,
            )
            if response is None:
                return "Erro: todos os providers falharam."

            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])

            self._history.append({"role": "user", "content": current_prompt})

            if content:
                content = self._clean_response(content)
                self._history.append({"role": "assistant", "content": content})
                logger.info("Brain: %s", content[:80])

            if not tool_calls:
                return content or ""

            # Execute tools
            tool_results = []
            for tc in tool_calls:
                func_name = tc["name"]
                args = tc["arguments"]
                logger.info("Tool Call: %s(%s)", func_name, args)

                result = f"Error: Tool {func_name} not found"
                if func_name in tool_functions:
                    try:
                        import asyncio

                        func = tool_functions[func_name]
                        if asyncio.iscoroutinefunction(func):
                            result = await func(**args)
                        else:
                            result = func(**args)
                    except Exception as e:
                        result = f"Error executing {func_name}: {e}"
                        logger.error(result)

                tool_results.append(f"Tool '{func_name}' returned: {result}")

            current_prompt = "\n".join(tool_results)

            if i == self._settings.brain_max_turns - 1:
                logger.warning("Brain reached max turns")
                return content or current_prompt

        return content

    async def think_agent(self, prompt: str, *, system: str = "") -> str:
        """Thinks using all registered tools."""
        from enton.core.tools import registry

        return await self.think_with_tools(
            prompt,
            tools=registry.schemas,
            tool_functions=registry.get_all_tools(),
            system=system,
        )

    def clear_history(self) -> None:
        self._history.clear()

