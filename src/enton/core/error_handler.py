"""Error Loop-Back Handler — auto-retry with error context fed back to LLM.

Inspired by Open Interpreter's error loop-back pattern: when a brain call
or tool execution fails, capture the error + traceback, feed it back as
context to the LLM, and let it self-correct its approach.

Integrates with:
- EntonBrain: wraps think()/think_agent() with error-aware retry
- ContextEngine: stores error history as temporary context
- MetaCognitiveEngine: records error traces for strategy learning

Pattern:
  1. Execute brain call
  2. If fails → capture error (type, message, traceback)
  3. Build error context prompt ("you tried X, it failed because Y")
  4. Retry with SAME provider + error context (before falling back)
  5. If still fails → fall back to next provider (existing behavior)
  6. After N total retries → give up gracefully
"""

from __future__ import annotations

import logging
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from enton.core.context_engine import ContextEngine

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ErrorRecord:
    """A single error occurrence with context."""

    error_type: str  # exception class name
    message: str
    provider: str = ""
    prompt_snippet: str = ""  # first 200 chars of prompt
    traceback_snippet: str = ""  # last 500 chars of traceback
    timestamp: float = field(default_factory=time.time)
    retry_attempt: int = 0
    resolved: bool = False
    resolution: str = ""  # what fixed it

    def summary(self) -> str:
        return (
            f"[{self.error_type}] {self.message[:100]} "
            f"(provider={self.provider}, attempt={self.retry_attempt})"
        )


from enton.cognition.prompts import ERROR_LOOPBACK_PROMPT as LOOPBACK_PROMPT


class ErrorLoopBack:
    """Error-aware retry handler for brain calls.

    Usage:
        handler = ErrorLoopBack(context_engine=ctx)

        # Wrap a brain call
        result = await handler.execute(
            brain.think, prompt, system=system,
            provider_id="ollama:qwen2.5:14b",
        )
    """

    def __init__(
        self,
        context_engine: ContextEngine | None = None,
        max_retries_per_provider: int = 1,
        max_total_retries: int = 3,
        error_ttl: float = 120.0,  # errors expire from context after 2min
    ) -> None:
        self._context = context_engine
        self._max_per_provider = max_retries_per_provider
        self._max_total = max_total_retries
        self._error_ttl = error_ttl
        self._history: deque[ErrorRecord] = deque(maxlen=50)
        self._consecutive_failures = 0

    # ------------------------------------------------------------------ #
    # Core execution with loop-back
    # ------------------------------------------------------------------ #

    async def execute(
        self,
        func: Any,
        *args: Any,
        provider_id: str = "",
        **kwargs: Any,
    ) -> tuple[str, ErrorRecord | None]:
        """Execute an async function with error loop-back.

        Returns (result, last_error_or_none).
        On success, last_error is None.
        On total failure, result is empty string.
        """
        last_error: ErrorRecord | None = None
        original_prompt = str(args[0]) if args else ""

        for attempt in range(1, self._max_total + 1):
            try:
                # If previous attempt failed, inject error context
                if last_error and args:
                    enhanced_prompt = self._build_loopback_prompt(
                        original_prompt,
                        last_error,
                        attempt,
                    )
                    args = (enhanced_prompt, *args[1:])

                result = await func(*args, **kwargs)

                # Success — record resolution if we recovered
                if last_error:
                    last_error.resolved = True
                    last_error.resolution = f"Resolved on attempt {attempt}"
                    self._consecutive_failures = 0
                    self._inject_context(
                        f"error_resolved_{attempt}",
                        f"Erro anterior resolvido na tentativa {attempt}",
                        priority=0.3,
                    )
                    logger.info(
                        "Error loop-back: resolved on attempt %d (provider=%s)",
                        attempt,
                        provider_id,
                    )

                self._consecutive_failures = 0
                return result, None

            except Exception as exc:
                error = self._capture_error(
                    exc,
                    provider_id,
                    original_prompt,
                    attempt,
                )
                self._history.append(error)
                self._consecutive_failures += 1
                last_error = error

                # Store error in context engine for cross-call awareness
                self._inject_context(
                    f"error_{attempt}",
                    error.summary(),
                    priority=0.7,
                )

                logger.warning(
                    "Error loop-back [%d/%d]: %s (provider=%s)",
                    attempt,
                    self._max_total,
                    error.message[:80],
                    provider_id,
                )

        # All retries exhausted
        return "", last_error

    async def execute_with_fallback(
        self,
        providers: list[tuple[str, Any, tuple, dict]],
    ) -> tuple[str, str]:
        """Execute across multiple providers with per-provider loop-back.

        Args:
            providers: List of (provider_id, func, args, kwargs) tuples.

        Returns (result, provider_that_succeeded).
        """
        for provider_id, func, args, kwargs in providers:
            result, error = await self.execute(
                func,
                *args,
                provider_id=provider_id,
                **kwargs,
            )
            if result:
                return result, provider_id
            if error:
                logger.info(
                    "Provider %s exhausted, trying next",
                    provider_id,
                )

        return "", ""

    # ------------------------------------------------------------------ #
    # Error analysis
    # ------------------------------------------------------------------ #

    def _capture_error(
        self,
        exc: Exception,
        provider_id: str,
        prompt: str,
        attempt: int,
    ) -> ErrorRecord:
        """Capture exception details into an ErrorRecord."""
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        tb_str = "".join(tb)

        return ErrorRecord(
            error_type=type(exc).__name__,
            message=str(exc)[:500],
            provider=provider_id,
            prompt_snippet=prompt[:200],
            traceback_snippet=tb_str[-500:],
            retry_attempt=attempt,
        )

    def _build_loopback_prompt(
        self,
        original_prompt: str,
        error: ErrorRecord,
        attempt: int,
    ) -> str:
        """Build enhanced prompt with error context for retry."""
        # Analyze error pattern for hints
        context_hint = self._error_hints(error)

        return LOOPBACK_PROMPT.format(
            error_type=error.error_type,
            error_message=error.message[:300],
            provider=error.provider,
            attempt=attempt,
            max_attempts=self._max_total,
            original_prompt=original_prompt[:500],
            context_hint=context_hint,
        )

    def _error_hints(self, error: ErrorRecord) -> str:
        """Generate hints based on error patterns."""
        hints: list[str] = []
        msg = error.message.lower()
        etype = error.error_type.lower()

        # Rate limit
        if "429" in msg or "rate" in msg or "limit" in msg:
            hints.append("DICA: Rate limit atingido. Simplifique a chamada.")

        # Timeout
        if "timeout" in etype or "timeout" in msg:
            hints.append("DICA: Timeout. Tente uma abordagem mais rapida.")

        # Tool not found
        if "tool" in msg and ("not found" in msg or "unknown" in msg):
            hints.append(
                "DICA: Ferramenta nao encontrada. Use outra ferramenta "
                "disponivel ou resolva sem ferramentas."
            )

        # JSON/parsing errors
        if "json" in msg or "parse" in msg or "decode" in msg:
            hints.append("DICA: Erro de parsing. Retorne texto simples em vez de JSON.")

        # Connection errors
        if "connection" in msg or "connect" in msg or "refused" in msg:
            hints.append("DICA: Servico indisponivel. Evite dependencias externas.")

        # Permission denied
        if "permission" in msg or "denied" in msg or "forbidden" in msg:
            hints.append("DICA: Sem permissao. Tente um caminho/recurso diferente.")

        # Similar errors in history — pattern detection
        similar = self._find_similar_errors(error)
        if similar >= 3:
            hints.append(
                f"ALERTA: Este tipo de erro ja ocorreu {similar}x recentemente. "
                "Mude completamente a estrategia."
            )

        return "\n".join(hints) if hints else ""

    def _find_similar_errors(self, error: ErrorRecord) -> int:
        """Count recent similar errors (same type + provider)."""
        cutoff = time.time() - 300  # last 5 minutes
        return sum(
            1
            for e in self._history
            if e.error_type == error.error_type
            and e.provider == error.provider
            and e.timestamp > cutoff
        )

    # ------------------------------------------------------------------ #
    # Context integration
    # ------------------------------------------------------------------ #

    def _inject_context(
        self,
        key: str,
        content: str,
        priority: float = 0.5,
    ) -> None:
        """Inject error info into context engine (if available)."""
        if self._context:
            self._context.set(
                key=key,
                content=content,
                category="error",
                priority=priority,
                ttl=self._error_ttl,
            )

    # ------------------------------------------------------------------ #
    # Analytics
    # ------------------------------------------------------------------ #

    @property
    def recent_errors(self) -> list[ErrorRecord]:
        """Last 10 errors."""
        return list(self._history)[-10:]

    @property
    def error_rate(self) -> float:
        """Error rate over last 20 records."""
        recent = list(self._history)[-20:]
        if not recent:
            return 0.0
        return sum(1 for e in recent if not e.resolved) / len(recent)

    @property
    def is_degraded(self) -> bool:
        """True if experiencing persistent failures."""
        return self._consecutive_failures >= 5

    def stats(self) -> dict:
        """Error handler statistics."""
        total = len(self._history)
        resolved = sum(1 for e in self._history if e.resolved)
        by_type: dict[str, int] = {}
        for e in self._history:
            by_type[e.error_type] = by_type.get(e.error_type, 0) + 1

        return {
            "total_errors": total,
            "resolved": resolved,
            "resolution_rate": resolved / total if total else 1.0,
            "consecutive_failures": self._consecutive_failures,
            "is_degraded": self.is_degraded,
            "error_rate": round(self.error_rate, 3),
            "by_type": by_type,
        }

    def summary(self) -> str:
        """One-liner for logging."""
        s = self.stats()
        return (
            f"Errors: {s['total_errors']} total, "
            f"{s['resolved']} resolved, "
            f"rate={s['error_rate']:.1%}, "
            f"degraded={s['is_degraded']}"
        )
