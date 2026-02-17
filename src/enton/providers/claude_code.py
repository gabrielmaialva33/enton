"""Claude Code CLI as an AI provider for Enton.

Wraps `claude -p` (headless mode) for:
- Brain fallback chain (when all API providers fail)
- Explicit task delegation via AIDelegateTools
- Full agentic coding tasks with tool permissions

Requires: claude CLI installed (`npm i -g @anthropic-ai/claude-code`)
Auth: Uses existing ANTHROPIC_API_KEY from environment.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Enton's personality injection for Claude Code
_ENTON_SYSTEM = (
    "Você é um sub-agente do Enton, um robô AI brasileiro zoeiro. "
    "Responda em português BR informal e direto. "
    "Seja eficiente — retorne APENAS a resposta útil, sem explicações "
    "desnecessárias ou disclaimers. O Enton delegou esta tarefa pra você."
)


@dataclass(frozen=True, slots=True)
class ClaudeResult:
    """Parsed result from Claude Code CLI."""

    content: str
    session_id: str = ""
    cost_usd: float = 0.0
    num_turns: int = 0
    duration_ms: int = 0
    is_error: bool = False


class ClaudeCodeProvider:
    """Claude Code CLI provider — headless mode via subprocess.

    Usage:
        provider = ClaudeCodeProvider()
        if provider.available:
            result = await provider.generate("Explica esse código")
    """

    def __init__(
        self,
        *,
        model: str = "sonnet",
        timeout: float = 120.0,
        max_turns: int = 10,
        system_prompt: str = "",
    ) -> None:
        self._model = model
        self._timeout = timeout
        self._max_turns = max_turns
        self._system = system_prompt or _ENTON_SYSTEM
        self._binary: str | None = None
        self._checked = False

    @property
    def available(self) -> bool:
        """Check if claude CLI is installed."""
        if not self._checked:
            self._binary = shutil.which("claude")
            self._checked = True
        return self._binary is not None

    @property
    def id(self) -> str:
        return f"claude-code:{self._model}"

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        timeout: float | None = None,
        working_dir: str | None = None,
    ) -> str:
        """Run claude -p and return the response text."""
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
    ) -> ClaudeResult:
        """Run claude -p and return full parsed result."""
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
        allowed_tools: list[str] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Delegate a full agentic coding task to Claude Code.

        Claude Code will read files, write code, run commands, etc.
        """
        extra = []
        if allowed_tools:
            for t in allowed_tools:
                extra.extend(["--allowedTools", t])
        return await self.generate(
            task,
            timeout=timeout or 300.0,
            working_dir=working_dir,
        )

    async def _run(
        self,
        prompt: str,
        *,
        system: str = "",
        timeout: float | None = None,
        working_dir: str | None = None,
        extra_args: list[str] | None = None,
    ) -> ClaudeResult:
        """Core subprocess execution."""
        if not self.available:
            return ClaudeResult(content="", is_error=True)

        cmd = [
            self._binary,
            "-p", prompt,
            "--output-format", "json",
            "--no-session-persistence",
            "--model", self._model,
            "--max-turns", str(self._max_turns),
        ]

        effective_system = system or self._system
        if effective_system:
            cmd.extend(["--append-system-prompt", effective_system])

        if extra_args:
            cmd.extend(extra_args)

        env_timeout = timeout or self._timeout

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=env_timeout,
            )

            if proc.returncode != 0:
                err = stderr.decode(errors="replace").strip()
                logger.warning("Claude Code failed (rc=%d): %s", proc.returncode, err[:200])
                return ClaudeResult(content="", is_error=True)

            return self._parse_output(stdout.decode(errors="replace"))

        except TimeoutError:
            logger.warning("Claude Code timed out after %.0fs", env_timeout)
            proc.kill()
            return ClaudeResult(content="", is_error=True)
        except Exception:
            logger.exception("Claude Code subprocess error")
            return ClaudeResult(content="", is_error=True)

    @staticmethod
    def _parse_output(raw: str) -> ClaudeResult:
        """Parse JSON output from claude -p --output-format json."""
        try:
            data = json.loads(raw.strip())
            return ClaudeResult(
                content=data.get("result", ""),
                session_id=data.get("session_id", ""),
                cost_usd=data.get("total_cost_usd", 0.0),
                num_turns=data.get("num_turns", 0),
                duration_ms=data.get("duration_ms", 0),
                is_error=data.get("is_error", False),
            )
        except (json.JSONDecodeError, KeyError):
            # Fallback: treat entire output as text
            text = raw.strip()
            if text:
                return ClaudeResult(content=text)
            return ClaudeResult(content="", is_error=True)
