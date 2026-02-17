"""Tests for enhanced ShellTools — CWD tracking + background commands."""

from __future__ import annotations

import pytest

from enton.skills._shell_state import ShellState
from enton.skills.shell_toolkit import ShellTools


@pytest.fixture()
def shell(tmp_path):
    state = ShellState(cwd=tmp_path)
    return ShellTools(state)


# --- CWD tracking ---


@pytest.mark.asyncio()
async def test_cwd_persists_after_cd(shell, tmp_path):
    sub = tmp_path / "mydir"
    sub.mkdir()

    await shell.run_command(f"cd {sub}")
    assert str(sub) in await shell.get_cwd()


@pytest.mark.asyncio()
async def test_cwd_with_compound_commands(shell, tmp_path):
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)

    await shell.run_command(f"cd {tmp_path / 'a'} && cd b")
    assert str(sub) in await shell.get_cwd()


@pytest.mark.asyncio()
async def test_cwd_shown_in_output(shell):
    result = await shell.run_command("echo hi")
    assert "[cwd:" in result
    assert "Exit code: 0" in result


@pytest.mark.asyncio()
async def test_get_cwd_returns_path(shell, tmp_path):
    cwd = await shell.get_cwd()
    assert cwd == str(tmp_path)


# --- Command execution ---


@pytest.mark.asyncio()
async def test_run_command_basic(shell):
    result = await shell.run_command("echo hello")
    assert "hello" in result
    assert "Exit code: 0" in result


@pytest.mark.asyncio()
async def test_run_command_dangerous_blocked(shell):
    result = await shell.run_command("rm -rf /")
    assert "BLOQUEADO" in result


@pytest.mark.asyncio()
async def test_run_command_stderr(shell):
    result = await shell.run_command("ls /nonexistent_path_xyz 2>&1 || true")
    # Should complete without error
    assert "Exit code:" in result


# --- Background commands ---


@pytest.mark.asyncio()
async def test_background_lifecycle(shell):
    result = await shell.run_background("sleep 0.1 && echo done")
    assert "Background iniciado" in result

    bg_id = result.split("ID: ")[1].split("\n")[0]

    # Wait a bit for it to finish
    import asyncio
    await asyncio.sleep(0.3)

    status = await shell.check_background(bg_id)
    assert "concluido" in status
    assert "done" in status

    stop = await shell.stop_background(bg_id)
    assert "parado" in stop or "removido" in stop


@pytest.mark.asyncio()
async def test_background_not_found(shell):
    result = await shell.check_background("nonexistent")
    assert "nao encontrado" in result


@pytest.mark.asyncio()
async def test_background_dangerous_blocked(shell):
    result = await shell.run_background("rm -rf /")
    assert "BLOQUEADO" in result


# --- Shared state ---


def test_shared_state():
    state = ShellState()
    shell = ShellTools(state)
    from enton.skills.file_toolkit import FileTools
    files = FileTools(state)
    assert shell._state is files._state
