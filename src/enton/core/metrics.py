"""TimescaleDB metrics collector for time-series data.

Records mood, detections/min, FPS, VRAM, latencies, etc.
Falls back gracefully if TimescaleDB is unavailable.
"""

from __future__ import annotations

import asyncio
import logging

import asyncpg

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS enton_metrics (
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metric TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    tags JSONB DEFAULT '{}'
);
SELECT create_hypertable('enton_metrics', 'ts', if_not_exists => TRUE);
"""


class MetricsCollector:
    """Async metrics writer to TimescaleDB."""

    def __init__(self, dsn: str, interval: float = 10.0) -> None:
        self._dsn = dsn
        self._interval = interval
        self._pool = None
        self._collectors: list[tuple[str, callable]] = []

    async def _ensure_pool(self):
        if self._pool is not None:
            return self._pool

        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=3)
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE)
        logger.info("TimescaleDB metrics pool ready")
        return self._pool

    def register(self, name: str, fn: callable) -> None:
        """Register a metric collector function.

        fn should return a float value when called.
        """
        self._collectors.append((name, fn))

    async def record(self, metric: str, value: float, tags: dict | None = None) -> None:
        """Record a single metric point."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO enton_metrics (metric, value, tags) VALUES ($1, $2, $3)",
                    metric,
                    value,
                    tags or {},
                )
        except Exception:
            logger.debug("Failed to record metric %s", metric)

    async def run(self) -> None:
        """Periodically collect and record all registered metrics."""
        # Wait a bit for everything to initialize
        await asyncio.sleep(5.0)

        try:
            await self._ensure_pool()
        except Exception:
            logger.warning("TimescaleDB unavailable, metrics disabled")
            return

        logger.info(
            "Metrics loop started (interval=%.0fs, collectors=%d)",
            self._interval,
            len(self._collectors),
        )

        while True:
            for name, fn in self._collectors:
                try:
                    value = fn()
                    if value is not None:
                        await self.record(name, float(value))
                except Exception:
                    logger.debug("Metric collector %s failed", name)

            await asyncio.sleep(self._interval)

    async def query(self, metric: str, hours: float = 1.0, limit: int = 100) -> list[dict]:
        """Query recent metrics."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT ts, value, tags FROM enton_metrics
                    WHERE metric = $1 AND ts > NOW() - $2::interval
                    ORDER BY ts DESC LIMIT $3
                    """,
                    metric,
                    f"{hours} hours",
                    limit,
                )
                return [dict(r) for r in rows]
        except Exception:
            logger.debug("Metrics query failed")
            return []
