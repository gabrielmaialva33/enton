"""Tests for MetaCognitiveEngine."""

from __future__ import annotations

import time

import pytest

from enton.cognition.metacognition import (
    MetaCognitiveEngine,
    ReasoningTrace,
)


def test_record_updates_stats():
    mc = MetaCognitiveEngine()
    trace = ReasoningTrace(
        query="test",
        strategy="agent",
        success=True,
        latency_ms=500,
    )
    mc.record(trace)
    assert mc._total_calls == 1
    assert mc._total_errors == 0


def test_record_error_updates_stats():
    mc = MetaCognitiveEngine()
    trace = ReasoningTrace(
        query="test",
        strategy="agent",
        success=False,
        latency_ms=100,
    )
    mc.record(trace)
    assert mc._total_errors == 1
    assert mc.success_rate == 0.0


def test_success_rate():
    mc = MetaCognitiveEngine()
    for i in range(8):
        mc.record(ReasoningTrace(query=f"q{i}", strategy="agent", success=True))
    for i in range(2):
        mc.record(ReasoningTrace(query=f"e{i}", strategy="agent", success=False))
    assert mc.success_rate == pytest.approx(0.8)


def test_strategy_scores_update():
    mc = MetaCognitiveEngine()
    # record many successes for "agent"
    for _ in range(10):
        mc.record(ReasoningTrace(query="q", strategy="agent", success=True))
    # record failures for "direct"
    for _ in range(10):
        mc.record(ReasoningTrace(query="q", strategy="direct", success=False))
    assert mc._strategy_scores["agent"] > mc._strategy_scores["direct"]
    assert mc.best_strategy() == "agent"


def test_begin_end_trace():
    mc = MetaCognitiveEngine()
    trace = mc.begin_trace("what is 2+2?", strategy="direct")
    assert trace.query == "what is 2+2?"
    time.sleep(0.01)
    mc.end_trace(trace, "4", provider="ollama", success=True)
    assert trace.latency_ms > 0
    assert trace.confidence > 0
    assert trace.provider == "ollama"
    assert mc._total_calls == 1


def test_confidence_high_for_good_response():
    mc = MetaCognitiveEngine()
    trace = mc.begin_trace("q")
    mc.end_trace(trace, "This is a good detailed response about the topic.")
    assert trace.confidence >= 0.5


def test_confidence_low_for_short_response():
    mc = MetaCognitiveEngine()
    trace = mc.begin_trace("q")
    mc.end_trace(trace, "ok")
    assert trace.confidence < 0.5


def test_confidence_low_for_error():
    mc = MetaCognitiveEngine()
    trace = mc.begin_trace("q")
    mc.end_trace(trace, "Erro: todos os providers falharam.", success=False)
    assert trace.confidence < 0.3


def test_introspect_empty():
    mc = MetaCognitiveEngine()
    assert "No reasoning history" in mc.introspect()


def test_introspect_with_data():
    mc = MetaCognitiveEngine()
    for _ in range(5):
        mc.record(
            ReasoningTrace(
                query="q",
                strategy="agent",
                success=True,
                latency_ms=200,
                provider="ollama",
            )
        )
    result = mc.introspect()
    assert "Calls: 5" in result
    assert "ollama" in result


def test_should_use_tools():
    mc = MetaCognitiveEngine()
    assert mc.should_use_tools("leia o arquivo config.py") is True
    assert mc.should_use_tools("como esta o sistema?") is True
    assert mc.should_use_tools("oi tudo bem?") is False


def test_provider_stats():
    mc = MetaCognitiveEngine()
    mc.record(
        ReasoningTrace(
            query="q",
            strategy="agent",
            provider="ollama",
            success=True,
            latency_ms=100,
        )
    )
    mc.record(
        ReasoningTrace(
            query="q",
            strategy="agent",
            provider="ollama",
            success=True,
            latency_ms=200,
        )
    )
    mc.record(
        ReasoningTrace(
            query="q",
            strategy="agent",
            provider="groq",
            success=False,
            latency_ms=5000,
        )
    )
    stats = mc.provider_stats()
    assert stats["ollama"]["calls"] == 2
    assert stats["ollama"]["success_rate"] == 1.0
    assert stats["groq"]["success_rate"] == 0.0


def test_to_dict():
    mc = MetaCognitiveEngine()
    mc.record(ReasoningTrace(query="q", strategy="agent", success=True, latency_ms=100))
    d = mc.to_dict()
    assert d["total_calls"] == 1
    assert "strategy_scores" in d


def test_avg_latency():
    mc = MetaCognitiveEngine()
    mc.record(ReasoningTrace(query="q", strategy="agent", latency_ms=100))
    mc.record(ReasoningTrace(query="q", strategy="agent", latency_ms=300))
    assert mc.avg_latency_ms == pytest.approx(200.0)
