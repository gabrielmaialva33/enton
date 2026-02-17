"""MetaCognitiveEngine — self-monitoring wrapper for Brain calls.

Wraps every Brain call with monitoring: tracks strategy selection,
confidence assessment, latency, token usage, and success rates.
Injects metacognitive insights into the SelfModel introspection.

Inspired by: SAFLA, MUSE (arXiv 2411.13537), rewire.it patterns.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass(slots=True)
class ReasoningTrace:
    """Record of a single Brain call."""

    query: str
    strategy: str  # "agent", "direct", "vlm", "dream"
    provider: str = ""
    confidence: float = 0.5
    latency_ms: float = 0.0
    response_len: int = 0
    success: bool = True
    error: str = ""
    retry_count: int = 0
    timestamp: float = field(default_factory=time.time)


class MetaCognitiveEngine:
    """Monitors and improves Enton's reasoning quality over time."""

    MAX_TRACES = 200

    def __init__(self) -> None:
        self._traces: deque[ReasoningTrace] = deque(maxlen=self.MAX_TRACES)
        self._strategy_scores: dict[str, float] = {
            "agent": 0.5,
            "direct": 0.5,
            "vlm": 0.5,
            "dream": 0.5,
        }
        self._total_calls = 0
        self._total_errors = 0
        self._total_latency_ms = 0.0

    # -- recording --

    def record(self, trace: ReasoningTrace) -> None:
        """Record a completed reasoning trace."""
        self._traces.append(trace)
        self._total_calls += 1
        self._total_latency_ms += trace.latency_ms
        if not trace.success:
            self._total_errors += 1

        # update strategy score with exponential moving average
        alpha = 0.15
        key = trace.strategy
        reward = 1.0 if trace.success else 0.0
        old = self._strategy_scores.get(key, 0.5)
        self._strategy_scores[key] = (1 - alpha) * old + alpha * reward

    def begin_trace(self, query: str, strategy: str = "agent") -> ReasoningTrace:
        """Start timing a new call. Call record() when done."""
        return ReasoningTrace(
            query=query[:200],
            strategy=strategy,
            timestamp=time.time(),
        )

    def end_trace(
        self,
        trace: ReasoningTrace,
        response: str,
        *,
        provider: str = "",
        success: bool = True,
        error: str = "",
    ) -> ReasoningTrace:
        """Finalize a trace with results and record it."""
        trace.latency_ms = (time.time() - trace.timestamp) * 1000
        trace.response_len = len(response)
        trace.provider = provider
        trace.success = success
        trace.error = error[:200]
        trace.confidence = self._assess_confidence(trace, response)
        self.record(trace)
        return trace

    # -- confidence assessment --

    def _assess_confidence(self, trace: ReasoningTrace, response: str) -> float:
        """Heuristic confidence scoring (0-1)."""
        score = 0.7  # baseline

        # penalize retries
        score -= trace.retry_count * 0.15

        # penalize slow responses
        if trace.latency_ms > 10000:
            score -= 0.2
        elif trace.latency_ms > 5000:
            score -= 0.1

        # penalize very short responses
        if len(response) < 10:
            score -= 0.3
        elif len(response) < 30:
            score -= 0.1

        # penalize errors
        if not trace.success:
            score -= 0.4

        # penalize "fallou" / error patterns
        if response and ("erro" in response.lower() or "failed" in response.lower()):
            score -= 0.15

        return max(0.0, min(1.0, score))

    # -- strategy selection --

    def best_strategy(self) -> str:
        """Return the strategy with highest historical success."""
        return max(self._strategy_scores, key=self._strategy_scores.get)

    def should_use_tools(self, query: str) -> bool:
        """Heuristic: should this query use tool-calling (agent) or direct?"""
        tool_keywords = [
            "arquivo", "file", "busca", "search", "sistema", "system",
            "shell", "comando", "run", "execute", "camera", "ptz",
            "lembra", "memory", "lembrete", "reminder", "descreva", "describe",
        ]
        q_lower = query.lower()
        return any(kw in q_lower for kw in tool_keywords)

    # -- analytics --

    @property
    def recent_traces(self) -> list[ReasoningTrace]:
        return list(self._traces)[-10:]

    @property
    def success_rate(self) -> float:
        if self._total_calls == 0:
            return 1.0
        return 1.0 - (self._total_errors / self._total_calls)

    @property
    def avg_latency_ms(self) -> float:
        if self._total_calls == 0:
            return 0.0
        return self._total_latency_ms / self._total_calls

    @property
    def avg_confidence(self) -> float:
        recent = list(self._traces)[-20:]
        if not recent:
            return 0.5
        return sum(t.confidence for t in recent) / len(recent)

    # -- introspection --

    def introspect(self) -> str:
        """Generate metacognitive summary for context injection."""
        if not self._traces:
            return "No reasoning history yet."

        recent = list(self._traces)[-10:]
        errors = sum(1 for t in recent if not t.success)
        best = self.best_strategy()
        providers_used = set(t.provider for t in recent if t.provider)

        parts = [
            f"Calls: {self._total_calls} total",
            f"success rate: {self.success_rate:.0%}",
            f"avg latency: {self.avg_latency_ms:.0f}ms",
            f"avg confidence: {self.avg_confidence:.2f}",
            f"best strategy: {best}",
        ]
        if errors > 0:
            parts.append(f"recent errors: {errors}/10")
        if providers_used:
            parts.append(f"providers: {', '.join(providers_used)}")

        return " | ".join(parts)

    def provider_stats(self) -> dict[str, dict]:
        """Per-provider success rates and latencies."""
        stats: dict[str, dict] = {}
        for trace in self._traces:
            if not trace.provider:
                continue
            p = trace.provider
            if p not in stats:
                stats[p] = {"calls": 0, "errors": 0, "total_ms": 0.0}
            stats[p]["calls"] += 1
            stats[p]["total_ms"] += trace.latency_ms
            if not trace.success:
                stats[p]["errors"] += 1

        for _p, s in stats.items():
            s["success_rate"] = 1.0 - (s["errors"] / s["calls"]) if s["calls"] else 0
            s["avg_ms"] = s["total_ms"] / s["calls"] if s["calls"] else 0

        return stats

    # -- serialization --

    def to_dict(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "success_rate": round(self.success_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_confidence": round(self.avg_confidence, 3),
            "strategy_scores": {k: round(v, 3) for k, v in self._strategy_scores.items()},
        }
