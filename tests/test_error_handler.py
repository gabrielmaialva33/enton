"""Tests for ErrorLoopBack handler."""
from __future__ import annotations

import asyncio

import pytest

from enton.core.context_engine import ContextEngine
from enton.core.error_handler import ErrorLoopBack, ErrorRecord


@pytest.fixture
def ctx():
    return ContextEngine(max_tokens=4000)


@pytest.fixture
def handler(ctx):
    return ErrorLoopBack(context_engine=ctx, max_total_retries=3)


class TestErrorRecord:
    def test_summary(self):
        rec = ErrorRecord(
            error_type="ValueError",
            message="something went wrong",
            provider="ollama",
            retry_attempt=1,
        )
        assert "ValueError" in rec.summary()
        assert "ollama" in rec.summary()


class TestErrorLoopBack:
    def test_success_on_first_try(self, handler):
        async def ok_fn(prompt):
            return "success"

        result, error = asyncio.get_event_loop().run_until_complete(
            handler.execute(ok_fn, "test")
        )
        assert result == "success"
        assert error is None
        assert handler.stats()["total_errors"] == 0

    def test_fails_after_retries(self, handler):
        async def fail_fn(prompt):
            raise RuntimeError("always fails")

        result, error = asyncio.get_event_loop().run_until_complete(
            handler.execute(fail_fn, "test", provider_id="test_provider")
        )
        assert result == ""
        assert error is not None
        assert error.error_type == "RuntimeError"
        assert handler.stats()["total_errors"] == 3  # max_total_retries

    def test_recovers_on_retry(self, handler):
        calls = 0

        async def flaky_fn(prompt):
            nonlocal calls
            calls += 1
            if calls <= 1:
                raise ConnectionError("timeout")
            return "recovered"

        result, error = asyncio.get_event_loop().run_until_complete(
            handler.execute(flaky_fn, "test", provider_id="flaky")
        )
        assert result == "recovered"
        assert error is None
        assert handler.stats()["resolved"] == 1

    def test_injects_context(self, handler, ctx):
        async def fail_fn(prompt):
            raise ValueError("bad input")

        asyncio.get_event_loop().run_until_complete(
            handler.execute(fail_fn, "test")
        )
        # Should have injected error context
        error_ctx = ctx.get("error_1")
        assert error_ctx is not None
        assert "ValueError" in error_ctx

    def test_error_hints_rate_limit(self, handler):
        rec = ErrorRecord(
            error_type="HTTPError",
            message="429 Too Many Requests",
            provider="nvidia",
        )
        hints = handler._error_hints(rec)
        assert "Rate limit" in hints

    def test_error_hints_timeout(self, handler):
        rec = ErrorRecord(
            error_type="TimeoutError",
            message="Request timed out",
            provider="nvidia",
        )
        hints = handler._error_hints(rec)
        assert "Timeout" in hints

    def test_is_degraded(self, handler):
        assert not handler.is_degraded
        handler._consecutive_failures = 5
        assert handler.is_degraded

    def test_stats_structure(self, handler):
        s = handler.stats()
        assert "total_errors" in s
        assert "resolved" in s
        assert "error_rate" in s
        assert "is_degraded" in s
        assert "by_type" in s
