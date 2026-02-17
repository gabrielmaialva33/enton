"""Tests for ProcessManager — Enton's background task orchestrator.

Covers TaskStatus enum, ManagedTask dataclass, and ProcessManager methods
including submit, submit_async, cancel, timeout, retry, cleanup, and more.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from enton.core.process_manager import ManagedTask, ProcessManager, TaskStatus


# ---------------------------------------------------------------------------
# TaskStatus enum
# ---------------------------------------------------------------------------


class TestTaskStatus:
    def test_enum_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.CANCELLED == "cancelled"

    def test_enum_member_count(self):
        assert len(TaskStatus) == 5

    def test_str_enum_is_string(self):
        assert isinstance(TaskStatus.PENDING, str)
        assert f"status={TaskStatus.RUNNING}" == "status=running"


# ---------------------------------------------------------------------------
# ManagedTask dataclass
# ---------------------------------------------------------------------------


class TestManagedTask:
    def test_creation_defaults(self):
        mt = ManagedTask(id="abc123", name="test", command="echo hi")
        assert mt.id == "abc123"
        assert mt.name == "test"
        assert mt.command == "echo hi"
        assert mt.status == TaskStatus.PENDING
        assert mt.output == ""
        assert mt.error == ""
        assert mt.exit_code is None
        assert mt.started_at == 0.0
        assert mt.finished_at == 0.0
        assert mt.retries == 0
        assert mt.max_retries == 1
        assert mt.timeout == 300.0
        assert mt._task is None

    def test_elapsed_not_started(self):
        mt = ManagedTask(id="a", name="t", command="x")
        assert mt.elapsed == 0.0

    def test_elapsed_running(self):
        mt = ManagedTask(id="a", name="t", command="x")
        mt.started_at = time.time() - 2.0
        # finished_at is still 0.0 so elapsed uses time.time()
        assert mt.elapsed >= 2.0

    def test_elapsed_finished(self):
        mt = ManagedTask(id="a", name="t", command="x")
        mt.started_at = 100.0
        mt.finished_at = 105.5
        assert mt.elapsed == pytest.approx(5.5)

    def test_is_done_pending(self):
        mt = ManagedTask(id="a", name="t", command="x")
        assert mt.is_done is False

    def test_is_done_running(self):
        mt = ManagedTask(id="a", name="t", command="x", status=TaskStatus.RUNNING)
        assert mt.is_done is False

    def test_is_done_completed(self):
        mt = ManagedTask(id="a", name="t", command="x", status=TaskStatus.COMPLETED)
        assert mt.is_done is True

    def test_is_done_failed(self):
        mt = ManagedTask(id="a", name="t", command="x", status=TaskStatus.FAILED)
        assert mt.is_done is True

    def test_is_done_cancelled(self):
        mt = ManagedTask(id="a", name="t", command="x", status=TaskStatus.CANCELLED)
        assert mt.is_done is True

    def test_summary_pending(self):
        mt = ManagedTask(id="abcdef123456", name="build", command="make")
        s = mt.summary()
        assert "[abcdef12]" in s
        assert "build" in s
        assert "pending" in s
        assert "(n/a)" in s

    def test_summary_with_output(self):
        mt = ManagedTask(id="abcdef123456", name="build", command="make")
        mt.status = TaskStatus.COMPLETED
        mt.started_at = 100.0
        mt.finished_at = 102.3
        mt.output = "build success"
        s = mt.summary()
        assert "completed" in s
        assert "2.3s" in s
        assert "build success" in s

    def test_summary_with_error(self):
        mt = ManagedTask(id="abcdef123456", name="fail", command="false")
        mt.status = TaskStatus.FAILED
        mt.started_at = 100.0
        mt.finished_at = 100.5
        mt.error = "command exited with 1"
        s = mt.summary()
        assert "failed" in s
        assert "!" in s
        assert "command exited" in s


# ---------------------------------------------------------------------------
# ProcessManager — async tests
# ---------------------------------------------------------------------------


@pytest.fixture
def pm():
    return ProcessManager(max_concurrent=5)


class TestProcessManagerSubmit:
    @pytest.mark.asyncio
    async def test_submit_echo(self, pm):
        tid = await pm.submit("echo-test", "echo hello")
        assert isinstance(tid, str)
        assert len(tid) == 12

        task = pm.get(tid)
        assert task is not None
        assert task._task is not None
        await task._task
        assert task.status == TaskStatus.COMPLETED
        assert task.output == "hello"
        assert task.exit_code == 0

    @pytest.mark.asyncio
    async def test_submit_failing_command(self, pm):
        tid = await pm.submit("fail-test", "false", max_retries=0)
        task = pm.get(tid)
        await task._task
        assert task.status == TaskStatus.FAILED
        assert task.exit_code != 0

    @pytest.mark.asyncio
    async def test_submit_multiline_output(self, pm):
        tid = await pm.submit("multi", "echo 'line1'; echo 'line2'")
        task = pm.get(tid)
        await task._task
        assert task.status == TaskStatus.COMPLETED
        assert "line1" in task.output
        assert "line2" in task.output

    @pytest.mark.asyncio
    async def test_submit_captures_stderr(self, pm):
        tid = await pm.submit("stderr-test", "echo err >&2; exit 1", max_retries=0)
        task = pm.get(tid)
        await task._task
        assert task.status == TaskStatus.FAILED
        assert "err" in task.error


class TestProcessManagerGet:
    @pytest.mark.asyncio
    async def test_get_existing_task(self, pm):
        tid = await pm.submit("x", "echo ok")
        task = pm.get(tid)
        assert task is not None
        assert task.id == tid

    @pytest.mark.asyncio
    async def test_get_nonexistent_task(self, pm):
        assert pm.get("doesnotexist") is None


class TestProcessManagerOutput:
    @pytest.mark.asyncio
    async def test_output_success(self, pm):
        tid = await pm.submit("out-test", "echo 'the output'")
        task = pm.get(tid)
        await task._task
        assert pm.output(tid) == "the output"

    @pytest.mark.asyncio
    async def test_output_nonexistent(self, pm):
        result = pm.output("nope123")
        assert "nao encontrada" in result

    @pytest.mark.asyncio
    async def test_output_no_output_yet(self, pm):
        """A task with no output and no error returns the fallback string."""
        tid = await pm.submit("slow", "sleep 5")
        task = pm.get(tid)
        # Force empty output/error to exercise the fallback path
        original_output = task.output
        original_error = task.error
        task.output = ""
        task.error = ""
        assert pm.output(tid) == "(sem output ainda)"
        task.output = original_output
        task.error = original_error
        await pm.cancel(tid)


class TestProcessManagerListTasks:
    @pytest.mark.asyncio
    async def test_list_all(self, pm):
        await pm.submit("a", "echo 1")
        await pm.submit("b", "echo 2")
        await pm.submit("c", "echo 3")
        tasks = pm.list_tasks()
        assert len(tasks) == 3

    @pytest.mark.asyncio
    async def test_list_filtered_by_status(self, pm):
        tid1 = await pm.submit("ok", "echo done")
        tid2 = await pm.submit("fail", "false", max_retries=0)

        await pm.get(tid1)._task
        await pm.get(tid2)._task

        completed = pm.list_tasks(status=TaskStatus.COMPLETED)
        failed = pm.list_tasks(status=TaskStatus.FAILED)
        assert len(completed) >= 1
        assert len(failed) >= 1
        assert all(t.status == TaskStatus.COMPLETED for t in completed)
        assert all(t.status == TaskStatus.FAILED for t in failed)

    @pytest.mark.asyncio
    async def test_list_with_limit(self, pm):
        for i in range(5):
            await pm.submit(f"t{i}", f"echo {i}")
        tasks = pm.list_tasks(limit=2)
        assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_list_sorted_most_recent_first(self, pm):
        await pm.submit("first", "echo 1")
        await asyncio.sleep(0.02)
        await pm.submit("second", "echo 2")
        tasks = pm.list_tasks()
        assert tasks[0].name == "second"
        assert tasks[1].name == "first"


class TestProcessManagerCancel:
    @pytest.mark.asyncio
    async def test_cancel_running_task(self, pm):
        tid = await pm.submit("long", "sleep 30")
        await asyncio.sleep(0.05)
        result = await pm.cancel(tid)
        assert result is True
        task = pm.get(tid)
        assert task.status == TaskStatus.CANCELLED
        assert task.is_done is True
        assert task.finished_at > 0

    @pytest.mark.asyncio
    async def test_cancel_already_done_task(self, pm):
        tid = await pm.submit("quick", "echo fast")
        task = pm.get(tid)
        await task._task
        assert task.status == TaskStatus.COMPLETED
        result = await pm.cancel(tid)
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(self, pm):
        result = await pm.cancel("ghost")
        assert result is False


class TestProcessManagerTimeout:
    @pytest.mark.asyncio
    async def test_timeout_kills_task(self, pm):
        tid = await pm.submit("timeout-test", "sleep 30", timeout=0.3)
        task = pm.get(tid)
        await task._task
        assert task.status == TaskStatus.FAILED
        assert "Timeout" in task.error
        assert task.finished_at > 0

    @pytest.mark.asyncio
    async def test_fast_command_no_timeout(self, pm):
        tid = await pm.submit("fast", "echo ok", timeout=5.0)
        task = pm.get(tid)
        await task._task
        assert task.status == TaskStatus.COMPLETED


class TestProcessManagerRetry:
    @pytest.mark.asyncio
    async def test_retry_exhausted(self, pm):
        """Command 'false' always fails — retries should be exhausted."""
        tid = await pm.submit("retry-fail", "false", max_retries=2)
        task = pm.get(tid)
        await task._task
        assert task.status == TaskStatus.FAILED
        assert task.retries == 2

    @pytest.mark.asyncio
    async def test_no_retry_when_zero(self, pm):
        tid = await pm.submit("no-retry", "false", max_retries=0)
        task = pm.get(tid)
        await task._task
        assert task.status == TaskStatus.FAILED
        assert task.retries == 0

    @pytest.mark.asyncio
    async def test_success_no_retry_needed(self, pm):
        tid = await pm.submit("ok", "echo hi", max_retries=3)
        task = pm.get(tid)
        await task._task
        assert task.status == TaskStatus.COMPLETED
        assert task.retries == 0


class TestProcessManagerSubmitAsync:
    @pytest.mark.asyncio
    async def test_submit_async_success(self, pm):
        async def my_coro():
            await asyncio.sleep(0.05)
            return 42

        tid = await pm.submit_async("coro-test", my_coro())
        task = pm.get(tid)
        assert task is not None
        assert task.command == "<async:coro-test>"
        await task._task
        assert task.status == TaskStatus.COMPLETED
        assert task.output == "42"
        assert task.exit_code == 0

    @pytest.mark.asyncio
    async def test_submit_async_returns_none(self, pm):
        async def noop():
            await asyncio.sleep(0.01)

        tid = await pm.submit_async("noop", noop())
        task = pm.get(tid)
        await task._task
        assert task.status == TaskStatus.COMPLETED
        assert task.output == ""

    @pytest.mark.asyncio
    async def test_submit_async_failure(self, pm):
        async def bad():
            raise ValueError("boom")

        tid = await pm.submit_async("bad-coro", bad())
        task = pm.get(tid)
        await task._task
        assert task.status == TaskStatus.FAILED
        assert "boom" in task.error

    @pytest.mark.asyncio
    async def test_submit_async_timeout(self, pm):
        async def slow():
            await asyncio.sleep(30)

        tid = await pm.submit_async("slow-coro", slow(), timeout=0.2)
        task = pm.get(tid)
        await task._task
        assert task.status == TaskStatus.FAILED
        assert "Timeout" in task.error


class TestProcessManagerCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_removes_old_done_tasks(self, pm):
        tid = await pm.submit("old", "echo old")
        task = pm.get(tid)
        await task._task
        task.finished_at = time.time() - 7200  # 2 hours ago

        removed = pm.cleanup(max_age=3600.0)
        assert removed == 1
        assert pm.get(tid) is None

    @pytest.mark.asyncio
    async def test_cleanup_keeps_recent_done_tasks(self, pm):
        tid = await pm.submit("recent", "echo recent")
        task = pm.get(tid)
        await task._task
        removed = pm.cleanup(max_age=3600.0)
        assert removed == 0
        assert pm.get(tid) is not None

    @pytest.mark.asyncio
    async def test_cleanup_keeps_running_tasks(self, pm):
        tid = await pm.submit("running", "sleep 60")
        await asyncio.sleep(0.05)
        removed = pm.cleanup(max_age=0)
        assert removed == 0
        await pm.cancel(tid)

    @pytest.mark.asyncio
    async def test_cleanup_multiple_mixed(self, pm):
        tid1 = await pm.submit("old-ok", "echo 1")
        tid2 = await pm.submit("old-fail", "false", max_retries=0)
        tid3 = await pm.submit("recent-ok", "echo 2")

        await pm.get(tid1)._task
        await pm.get(tid2)._task
        await pm.get(tid3)._task

        pm.get(tid1).finished_at = time.time() - 5000
        pm.get(tid2).finished_at = time.time() - 5000

        removed = pm.cleanup(max_age=3600.0)
        assert removed == 2
        assert pm.get(tid1) is None
        assert pm.get(tid2) is None
        assert pm.get(tid3) is not None


class TestProcessManagerActiveCountAndSummary:
    @pytest.mark.asyncio
    async def test_active_count_zero_initially(self, pm):
        assert pm.active_count == 0

    @pytest.mark.asyncio
    async def test_active_count_with_running_tasks(self, pm):
        tid1 = await pm.submit("s1", "sleep 10")
        tid2 = await pm.submit("s2", "sleep 10")
        await asyncio.sleep(0.1)
        assert pm.active_count >= 1
        await pm.cancel(tid1)
        await pm.cancel(tid2)

    @pytest.mark.asyncio
    async def test_summary_format(self, pm):
        tid1 = await pm.submit("ok", "echo hi")
        tid2 = await pm.submit("long", "sleep 10")

        await pm.get(tid1)._task
        await asyncio.sleep(0.05)

        s = pm.summary()
        assert "Tasks:" in s
        assert "running" in s
        assert "done" in s
        assert "pending" in s
        await pm.cancel(tid2)

    @pytest.mark.asyncio
    async def test_summary_all_done(self, pm):
        tid = await pm.submit("q", "echo done")
        await pm.get(tid)._task
        s = pm.summary()
        assert "1 done" in s
        assert "0 running" in s


class TestProcessManagerConcurrency:
    @pytest.mark.asyncio
    async def test_max_concurrent_respected(self):
        pm = ProcessManager(max_concurrent=2)
        tids = []
        for i in range(4):
            tid = await pm.submit(f"c{i}", "sleep 0.3")
            tids.append(tid)
        await asyncio.sleep(0.05)
        # With semaphore=2, at most 2 should be running simultaneously
        assert pm.active_count <= 2
        for tid in tids:
            task = pm.get(tid)
            if not task.is_done:
                await pm.cancel(tid)
        for tid in tids:
            task = pm.get(tid)
            if task._task and not task._task.done():
                try:
                    await task._task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_submit_with_cwd(self, pm):
        tid = await pm.submit("pwd-test", "pwd", cwd="/tmp")
        task = pm.get(tid)
        await task._task
        assert task.status == TaskStatus.COMPLETED
        assert "/tmp" in task.output
