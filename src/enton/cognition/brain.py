"""EntonBrain — Agno Agent with multi-provider fallback chain.

Wraps an Agno Agent that handles tool calling, memory, and knowledge
natively. Provider fallback is implemented by swapping agent.model
on failure (Agno doesn't have built-in fallback).

Fallback order:
  LOCAL → NVIDIA(x4) → HuggingFace → Groq → OpenRouter → AIMLAPI → Google
  → Claude Code CLI → Gemini CLI  (last resort — subprocess)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.ollama import Ollama

if TYPE_CHECKING:
    from agno.knowledge import Knowledge as AgentKnowledge
    from agno.models.base import Model
    from agno.tools import Toolkit

    from enton.core.config import Settings

logger = logging.getLogger(__name__)

_DB_PATH = str(Path.home() / ".enton" / "sessions.db")


class EntonBrain:
    """Enton's cognitive center — Agno Agent with cascading fallback."""

    def __init__(
        self,
        settings: Settings,
        toolkits: list[Toolkit],
        instructions: str | list[str] | None = None,
        knowledge: AgentKnowledge | None = None,
    ) -> None:
        self._settings = settings
        self._models = self._init_models(settings)
        self._vision_models = self._init_vision_models(settings)
        self._cli_providers = self._init_cli_providers(settings)
        self._vlm = None  # QwenVL transformers (last resort)

        Path.home().joinpath(".enton").mkdir(parents=True, exist_ok=True)

        self._agent = Agent(
            name="Enton",
            model=self._models[0] if self._models else Ollama(id="qwen2.5:14b"),
            tools=toolkits,
            instructions=instructions,
            db=SqliteDb(db_file=_DB_PATH),
            knowledge=knowledge,
            search_knowledge=knowledge is not None,
            add_history_to_context=True,
            num_history_runs=settings.memory_size,
            tool_call_limit=settings.brain_max_turns,
            retries=2,
            stream=False,
            telemetry=False,
            markdown=False,
        )

        self._dynamic_toolkits: dict[str, Toolkit] = {}
        self._error_handler: Any = None  # ErrorLoopBack (set via set_error_handler)

        names = [getattr(m, "id", str(m)) for m in self._models]
        cli_names = [p.id for p in self._cli_providers if p.available]
        all_names = names + cli_names
        logger.info("Brain models: [%s]", ", ".join(all_names))

    # ------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------

    @staticmethod
    def _init_models(s: Settings) -> list[Model]:
        """Build ordered list of Agno models for fallback chain."""
        models: list[Model] = []

        # 1. LOCAL (Ollama) — always primary
        models.append(Ollama(id=s.ollama_model))

        # 2. NVIDIA NIM — one instance per key (natural rate-limit fallback)
        nvidia_keys = [k.strip() for k in s.nvidia_api_keys.split(",") if k.strip()]
        if not nvidia_keys and s.nvidia_api_key:
            nvidia_keys = [s.nvidia_api_key]
        if nvidia_keys:
            try:
                from agno.models.nvidia import Nvidia

                for key in nvidia_keys:
                    models.append(Nvidia(id=s.nvidia_nim_model, api_key=key))
            except ImportError:
                logger.debug("agno.models.nvidia not available")

        # 3. HuggingFace Inference API
        if s.huggingface_token:
            from agno.models.openai.like import OpenAILike

            models.append(
                OpenAILike(
                    id=s.huggingface_model,
                    api_key=s.huggingface_token,
                    base_url="https://api-inference.huggingface.co/v1",
                )
            )

        # 4. Groq
        if s.groq_api_key:
            from agno.models.groq import Groq

            models.append(Groq(id=s.groq_model, api_key=s.groq_api_key))

        # 5. OpenRouter
        if s.openrouter_api_key:
            try:
                from agno.models.openrouter import OpenRouter

                models.append(
                    OpenRouter(
                        id=s.openrouter_model,
                        api_key=s.openrouter_api_key,
                    )
                )
            except ImportError:
                from agno.models.openai.like import OpenAILike

                models.append(
                    OpenAILike(
                        id=s.openrouter_model,
                        api_key=s.openrouter_api_key,
                        base_url="https://openrouter.ai/api/v1",
                    )
                )

        # 6. AIML API
        if s.aimlapi_api_key:
            from agno.models.openai.like import OpenAILike

            models.append(
                OpenAILike(
                    id=s.aimlapi_model,
                    api_key=s.aimlapi_api_key,
                    base_url="https://api.aimlapi.com/v1",
                )
            )

        # 7. Google Gemini
        if s.google_project:
            try:
                from agno.models.google import Gemini

                models.append(Gemini(id=s.google_brain_model))
            except ImportError:
                logger.debug("agno.models.google not available")

        return models

    @staticmethod
    def _init_cli_providers(s: Settings) -> list:
        """Build CLI-based AI providers (last resort fallback)."""
        from enton.providers.claude_code import ClaudeCodeProvider
        from enton.providers.gemini_cli import GeminiCliProvider

        providers = []

        if s.claude_code_enabled:
            p = ClaudeCodeProvider(
                model=s.claude_code_model,
                timeout=s.claude_code_timeout,
                max_turns=s.claude_code_max_turns,
            )
            if p.available:
                providers.append(p)
                logger.info("CLI provider: Claude Code (%s)", p.id)
            else:
                logger.debug("Claude Code CLI not installed")

        if s.gemini_cli_enabled:
            p = GeminiCliProvider(
                model=s.gemini_cli_model,
                timeout=s.gemini_cli_timeout,
                yolo=s.gemini_cli_yolo,
            )
            if p.available:
                providers.append(p)
                logger.info("CLI provider: Gemini CLI (%s)", p.id)
            else:
                logger.debug("Gemini CLI not installed")

        return providers

    @staticmethod
    def _init_vision_models(s: Settings) -> list[Model]:
        """Models that support vision (image input)."""
        models: list[Model] = []

        # Ollama VLM
        models.append(Ollama(id=s.ollama_vlm_model))

        # NVIDIA NIM vision
        nvidia_keys = [k.strip() for k in s.nvidia_api_keys.split(",") if k.strip()]
        if nvidia_keys:
            try:
                from agno.models.nvidia import Nvidia

                models.append(
                    Nvidia(
                        id=s.nvidia_nim_vision_model,
                        api_key=nvidia_keys[0],
                    )
                )
            except ImportError:
                pass

        # HuggingFace vision
        if s.huggingface_token and s.huggingface_vision_model:
            from agno.models.openai.like import OpenAILike

            models.append(
                OpenAILike(
                    id=s.huggingface_vision_model,
                    api_key=s.huggingface_token,
                    base_url="https://api-inference.huggingface.co/v1",
                )
            )

        # OpenRouter vision
        if s.openrouter_api_key and s.openrouter_vision_model:
            from agno.models.openai.like import OpenAILike

            models.append(
                OpenAILike(
                    id=s.openrouter_vision_model,
                    api_key=s.openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                )
            )

        # Google Gemini vision
        if s.google_project:
            try:
                from agno.models.google import Gemini

                models.append(Gemini(id=s.google_vision_model))
            except ImportError:
                pass

        return models

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        """Strip <think> reasoning tags from response."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    def set_error_handler(self, handler: Any) -> None:
        """Attach an ErrorLoopBack handler for error-aware retries."""
        self._error_handler = handler

    async def think(self, prompt: str, *, system: str = "") -> str:
        """Run prompt through agent with full tool calling + fallback.

        Fallback order:
        1. Agno models (Ollama → NVIDIA → HF → Groq → OpenRouter → AIMLAPI → Google)
           - With error loop-back: retry same provider with error context before falling back
        2. CLI providers (Claude Code → Gemini CLI) — subprocess, no tool calling
        """
        # Update instructions if system override provided
        original_instructions = self._agent.instructions
        if system:
            self._agent.instructions = [system]

        try:
            # Tier 1: Agno models (full tool calling)
            for model in self._models:
                mid = getattr(model, "id", "?")
                self._agent.model = model

                # With error loop-back: retry same provider with error context
                if self._error_handler:
                    result, error = await self._error_handler.execute(
                        self._arun_safe,
                        prompt,
                        provider_id=mid,
                    )
                    if result:
                        return result
                    if error:
                        logger.warning(
                            "Brain [%s] failed after loop-back: %s",
                            mid,
                            error.message[:80],
                        )
                    continue

                # Without error handler: original behavior
                try:
                    response = await self._agent.arun(prompt)
                    content = response.content or ""
                    content = self._clean(content)
                    logger.info("Brain [%s]: %s", mid, content[:80])
                    return content
                except Exception:
                    logger.warning("Brain [%s] failed, trying next", mid)

            # Tier 2: CLI providers (subprocess, text-only — no tool calling)
            for cli in self._cli_providers:
                try:
                    content = await cli.generate(prompt, system=system)
                    if content:
                        content = self._clean(content)
                        logger.info("Brain CLI [%s]: %s", cli.id, content[:80])
                        return content
                except Exception:
                    logger.warning("Brain CLI [%s] failed, trying next", cli.id)
        finally:
            if system:
                self._agent.instructions = original_instructions

        return "Erro: todos os providers falharam."

    async def _arun_safe(self, prompt: str) -> str:
        """Wrapper for agent.arun() that returns clean text or raises."""
        response = await self._agent.arun(prompt)
        content = response.content or ""
        content = self._clean(content)
        mid = getattr(self._agent.model, "id", "?")
        logger.info("Brain [%s]: %s", mid, content[:80])
        return content

    async def think_agent(self, prompt: str, *, system: str = "") -> str:
        """Alias for think() — Agno handles tool calling natively."""
        return await self.think(prompt, system=system)

    async def describe_scene(self, image: bytes, *, system: str = "") -> str:
        """Describe scene via VLM with fallback chain."""
        from agno.media import Image as AgnoImage

        prompt = system or "Descreva brevemente o que você vê em português."
        img = AgnoImage(content=image)

        for model in self._vision_models:
            mid = getattr(model, "id", "?")
            try:
                vlm_agent = Agent(
                    name="EntonVLM",
                    model=model,
                    markdown=False,
                    telemetry=False,
                )
                response = await vlm_agent.arun(prompt, images=[img])
                content = response.content or ""
                logger.info("VLM [%s]: %s", mid, content[:80])
                return self._clean(content)
            except Exception as exc:
                logger.warning("VLM [%s] failed: %s", mid, exc)

        # Last resort: local transformers VLM
        vlm = self._get_vlm()
        if vlm is not None:
            try:
                return await vlm.describe(prompt, image)
            except Exception:
                logger.warning("Transformers VLM failed")

        return ""

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

    # ------------------------------------------------------------------
    # Dynamic toolkit management (v0.4.0)
    # ------------------------------------------------------------------

    def register_toolkit(self, toolkit: Toolkit, name: str) -> None:
        """Register a dynamic toolkit with the Agent.

        Safe to call between arun() invocations — Agno rebuilds its
        tool list automatically on the next arun() call.
        """
        self._agent.add_tool(toolkit)
        self._dynamic_toolkits[name] = toolkit
        logger.info("Registered dynamic toolkit: %s", name)

    def unregister_toolkit(self, name: str) -> bool:
        """Remove a dynamic toolkit from the Agent."""
        toolkit = self._dynamic_toolkits.pop(name, None)
        if toolkit is None:
            return False
        if self._agent.tools and toolkit in self._agent.tools:
            self._agent.tools.remove(toolkit)
        logger.info("Unregistered dynamic toolkit: %s", name)
        return True

    def clear_history(self) -> None:
        """Clear conversation history by deleting the session."""
        self._agent.delete_session()

    @property
    def agent(self) -> Agent:
        """Direct access to Agno Agent (for session_state, etc.)."""
        return self._agent

    @property
    def cli_providers(self) -> list:
        """CLI-based AI providers (Claude Code, Gemini CLI)."""
        return self._cli_providers
