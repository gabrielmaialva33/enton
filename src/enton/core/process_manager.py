"""Process Manager — Enton's background task orchestrator.

Inspired by wcgw's persistent shell sessions and OpenHands' event-sourced
task tracking. Enton can run background tasks, monitor their status,
capture output, and auto-retry on failure.

Pattern: Named async tasks with status tracking + output buffering.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ManagedTask:
    """A tracked background task."""

    id: str
    name: str
    command: str
    status: TaskStatus = TaskStatus.PENDING
    output: str = ""
    error: str = ""
    exit_code: int | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    retries: int = 0
    max_retries: int = 1
    timeout: float = 300.0  # 5 min default
    _task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def elapsed(self) -> float:
        if self.started_at == 0:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def is_done(self) -> bool:
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        )

    def summary(self) -> str:
        elapsed_str = f"{self.elapsed:.1f}s" if self.started_at else "n/a"
        output_preview = self.output[:80] if self.output else ""
        return (
            f"[{self.id[:8]}] {self.name} — {self.status} "
            f"({elapsed_str})"
            + (f"\n  > {output_preview}" if output_preview else "")
            + (f"\n  ! {self.error[:80]}" if self.error else "")
        )


class ProcessManager:
    """Manages background async tasks with tracking and retry.

    Usage:
        pm = ProcessManager()
        task_id = await pm.submit("build", "make -j$(nproc)")
        status = pm.get(task_id)
        output = pm.output(task_id)
        await pm.cancel(task_id)
    """

    def __init__(self, max_concurrent: int = 10) -> None:
        self._tasks: dict[str, ManagedTask] = {}
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def submit(
        self,
        name: str,
        command: str,
        timeout: float = 300.0,
        max_retries: int = 1,
        cwd: str = "",
    ) -> str:
        """Submit a background shell command.

        Returns task ID for tracking.
        """
        task_id = uuid.uuid4().hex[:12]
        mt = ManagedTask(
            id=task_id,
            name=name,
            command=command,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._tasks[task_id] = mt

        # Launch in background
        mt._task = asyncio.create_task(
            self._run_task(mt, cwd=cwd),
            name=f"pm:{task_id}:{name}",
        )
        logger.info("Task submitted: %s (%s)", name, task_id)
        return task_id

    async def submit_async(
        self,
        name: str,
        coro: asyncio.coroutines,
        timeout: float = 300.0,
    ) -> str:
        """Submit an async coroutine as a tracked task."""
        task_id = uuid.uuid4().hex[:12]
        mt = ManagedTask(
            id=task_id,
            name=name,
            command=f"<async:{name}>",
            timeout=timeout,
        )
        self._tasks[task_id] = mt
        mt._task = asyncio.create_task(
            self._run_coro(mt, coro),
            name=f"pm:{task_id}:{name}",
        )
        return task_id

    async def _run_task(self, mt: ManagedTask, cwd: str = "") -> None:
        """Execute shell command with retry logic."""
        async with self._semaphore:
            for attempt in range(mt.max_retries + 1):
                mt.status = TaskStatus.RUNNING
                mt.started_at = time.time()
                mt.retries = attempt

                try:
                    proc = await asyncio.create_subprocess_shell(
                        mt.command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=cwd or None,
                    )
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=mt.timeout,
                    )
                    mt.output = stdout.decode(errors="replace").strip()
                    mt.error = stderr.decode(errors="replace").strip()
                    mt.exit_code = proc.returncode or 0
                    mt.finished_at = time.time()

                    if mt.exit_code == 0:
                        mt.status = TaskStatus.COMPLETED
                        logger.info(
                            "Task completed: %s (%.1fs)",
                            mt.name,
                            mt.elapsed,
                        )
                        return

                    # Non-zero exit — retry?
                    if attempt < mt.max_retries:
                        logger.warning(
                            "Task %s failed (exit=%d), retry %d/%d",
                            mt.name,
                            mt.exit_code,
                            attempt + 1,
                            mt.max_retries,
                        )
                        await asyncio.sleep(2**attempt)  # exponential backoff
                        continue

                    mt.status = TaskStatus.FAILED
                    logger.warning(
                        "Task failed: %s (exit=%d, retries exhausted)",
                        mt.name,
                        mt.exit_code,
                    )

                except TimeoutError:
                    mt.finished_at = time.time()
                    mt.error = f"Timeout after {mt.timeout}s"
                    mt.status = TaskStatus.FAILED
                    logger.warning("Task timeout: %s", mt.name)
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    return

                except Exception as e:
                    mt.finished_at = time.time()
                    mt.error = str(e)
                    mt.status = TaskStatus.FAILED
                    logger.exception("Task error: %s", mt.name)
                    return

    async def _run_coro(
        self,
        mt: ManagedTask,
        coro: asyncio.coroutines,
    ) -> None:
        """Execute async coroutine with tracking."""
        async with self._semaphore:
            mt.status = TaskStatus.RUNNING
            mt.started_at = time.time()
            try:
                result = await asyncio.wait_for(coro, timeout=mt.timeout)
                mt.output = str(result) if result is not None else ""
                mt.exit_code = 0
                mt.status = TaskStatus.COMPLETED
            except TimeoutError:
                mt.error = f"Timeout after {mt.timeout}s"
                mt.status = TaskStatus.FAILED
            except Exception as e:
                mt.error = str(e)
                mt.status = TaskStatus.FAILED
            finally:
                mt.finished_at = time.time()

    def get(self, task_id: str) -> ManagedTask | None:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def output(self, task_id: str) -> str:
        """Get task output."""
        task = self._tasks.get(task_id)
        if not task:
            return f"Task '{task_id}' nao encontrada."
        return task.output or task.error or "(sem output ainda)"

    async def cancel(self, task_id: str) -> bool:
        """Cancel a running task."""
        task = self._tasks.get(task_id)
        if not task or not task._task:
            return False
        if task.is_done:
            return False
        task._task.cancel()
        task.status = TaskStatus.CANCELLED
        task.finished_at = time.time()
        logger.info("Task cancelled: %s", task.name)
        return True

    def list_tasks(
        self,
        status: TaskStatus | None = None,
        limit: int = 20,
    ) -> list[ManagedTask]:
        """List tasks, optionally filtered by status."""
        tasks = sorted(
            self._tasks.values(),
            key=lambda t: t.created_at,
            reverse=True,
        )
        if status:
            tasks = [t for t in tasks if t.status == status]
        return tasks[:limit]

    def cleanup(self, max_age: float = 3600.0) -> int:
        """Remove old completed/failed tasks. Returns count removed."""
        now = time.time()
        to_remove = [
            tid for tid, t in self._tasks.items() if t.is_done and (now - t.finished_at) > max_age
        ]
        for tid in to_remove:
            del self._tasks[tid]
        return len(to_remove)

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)

    def summary(self) -> str:
        """Quick status summary."""
        running = sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)
        pending = sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)
        done = sum(1 for t in self._tasks.values() if t.is_done)
        return f"Tasks: {running} running, {pending} pending, {done} done"
