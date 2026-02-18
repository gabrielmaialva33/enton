"""Gemini CLI as an AI provider for Enton.

Wraps `gemini -p` (headless mode) for:
- Brain fallback chain (when all API providers fail)
- Explicit task delegation via AIDelegateTools
- Research tasks with Google-grounded search
- Full agentic coding with --yolo mode

Requires: gemini CLI installed (`npm i -g @google/gemini-cli`)
Auth: Uses existing GEMINI_API_KEY or Google OAuth from environment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_ENTON_SYSTEM = (
    "Você é um sub-agente do Enton, um robô AI brasileiro zoeiro. "
    "Responda em português BR informal e direto. "
    "Seja eficiente — retorne APENAS a resposta útil, sem explicações "
    "desnecessárias ou disclaimers. O Enton delegou esta tarefa pra você."
)


@dataclass(frozen=True, slots=True)
class GeminiResult:
    """Parsed result from Gemini CLI."""

    content: str
    is_error: bool = False
    raw_output: str = ""


class GeminiCliProvider:
    """Gemini CLI provider — headless mode via subprocess.

    Usage:
        provider = GeminiCliProvider()
        if provider.available:
            result = await provider.generate("Pesquisa sobre X")
    """

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        timeout: float = 120.0,
        yolo: bool = False,
        system_prompt: str = "",
    ) -> None:
        self._model = model
        self._timeout = timeout
        self._yolo = yolo
        self._system = system_prompt or _ENTON_SYSTEM
        self._binary: str | None = None
        self._checked = False

    @property
    def available(self) -> bool:
        """Check if gemini CLI is installed."""
        if not self._checked:
            self._binary = shutil.which("gemini")
            self._checked = True
        return self._binary is not None

    @property
    def id(self) -> str:
        return f"gemini-cli:{self._model}"

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        timeout: float | None = None,
        working_dir: str | None = None,
    ) -> str:
        """Run gemini -p and return the response text."""
        result = await self._run(
            prompt,
            system=system,
            timeout=timeout,
            working_dir=working_dir,
        )
        return result.content

    async def generate_json(
        self,
        prompt: str,
        *,
        system: str = "",
        timeout: float | None = None,
        working_dir: str | None = None,
    ) -> GeminiResult:
        """Run gemini -p and return full parsed result."""
        return await self._run(
            prompt,
            system=system,
            timeout=timeout,
            working_dir=working_dir,
        )

    async def code_task(
        self,
        task: str,
        *,
        working_dir: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """Delegate a full agentic coding task to Gemini CLI.

        Uses --yolo for auto-approving tool usage.
        """
        return await self.generate(
            task,
            timeout=timeout or 300.0,
            working_dir=working_dir,
        )

    async def research(
        self,
        topic: str,
        *,
        timeout: float | None = None,
    ) -> str:
        """Research a topic using Gemini's Google-grounded search."""
        prompt = (
            f"Pesquise profundamente sobre: {topic}\n"
            "Use todas as ferramentas de busca disponíveis. "
            "Retorne um resumo completo e detalhado em português BR."
        )
        return await self.generate(prompt, timeout=timeout or 180.0)

    async def _run(
        self,
        prompt: str,
        *,
        system: str = "",
        timeout: float | None = None,
        working_dir: str | None = None,
        extra_args: list[str] | None = None,
    ) -> GeminiResult:
        """Core subprocess execution."""
        if not self.available:
            return GeminiResult(content="", is_error=True)

        effective_system = system or self._system

        # Gemini CLI uses env var for system prompt
        env_overrides: dict[str, str] = {}
        if effective_system:
            # Inject via prepending to prompt (Gemini doesn't have --system flag)
            prompt = f"[INSTRUÇÃO DO SISTEMA: {effective_system}]\n\n{prompt}"

        cmd = [
            self._binary,
            "-p",
            prompt,
            "--output-format",
            "json",
            "-m",
            self._model,
        ]

        if self._yolo:
            cmd.append("--yolo")

        if extra_args:
            cmd.extend(extra_args)

        env_timeout = timeout or self._timeout

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=env_overrides or None,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=env_timeout,
            )

            raw = stdout.decode(errors="replace")

            if proc.returncode != 0:
                err = stderr.decode(errors="replace").strip()
                logger.warning("Gemini CLI failed (rc=%d): %s", proc.returncode, err[:200])
                return GeminiResult(content="", is_error=True, raw_output=raw)

            return self._parse_output(raw)

        except TimeoutError:
            logger.warning("Gemini CLI timed out after %.0fs", env_timeout)
            proc.kill()
            return GeminiResult(content="", is_error=True)
        except Exception:
            logger.exception("Gemini CLI subprocess error")
            return GeminiResult(content="", is_error=True)

    @staticmethod
    def _parse_output(raw: str) -> GeminiResult:
        """Parse JSON output from gemini -p --output-format json."""
        try:
            data = json.loads(raw.strip())
            content = data.get("response", "")
            if not content:
                # Fallback fields
                content = data.get("result", data.get("text", ""))
            return GeminiResult(content=content, raw_output=raw)
        except (json.JSONDecodeError, KeyError):
            # Fallback: treat entire output as text (non-JSON mode)
            text = raw.strip()
            if text:
                return GeminiResult(content=text, raw_output=raw)
            return GeminiResult(content="", is_error=True, raw_output=raw)
