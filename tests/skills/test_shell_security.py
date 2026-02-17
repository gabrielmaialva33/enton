"""Tests for shell command classification and security — CRITICAL for safety."""

from __future__ import annotations

import pytest

from enton.skills.shell_toolkit import (
    DANGEROUS_PATTERNS,
    ELEVATED_COMMANDS,
    SAFE_COMMANDS,
    ShellTools,
    _classify_command,
)
from enton.skills._shell_state import ShellState


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Command classification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCommandClassification:
    # --- Safe commands ---
    @pytest.mark.parametrize("cmd", [
        "ls -la", "cat /etc/hostname", "pwd", "whoami", "date",
        "git status", "python --version", "pip list", "uv sync",
        "df -h", "free -m", "ps aux", "nvidia-smi", "uptime",
        "grep pattern file.txt", "find . -name '*.py'",
        "pytest tests/", "ruff check .", "echo hello",
    ])
    def test_safe_commands(self, cmd: str):
        assert _classify_command(cmd) == "safe"

    # --- Elevated commands ---
    @pytest.mark.parametrize("cmd", [
        "apt install python3", "apt-get update",
        "kill 1234", "killall python", "pkill firefox",
        "pip install numpy", "uv add pandas",
        "crontab -l", "chmod 755 script.sh",
    ])
    def test_elevated_commands(self, cmd: str):
        assert _classify_command(cmd) == "elevated"

    # --- Dangerous commands ---
    @pytest.mark.parametrize("cmd", [
        "rm -rf /",
        "rm -rf /*",
        "mkfs.ext4 /dev/sda",
        "dd if=/dev/zero of=/dev/sda",
        "shutdown now",
        "reboot",
        ":(){ :|:& };:",  # fork bomb
    ])
    def test_dangerous_commands(self, cmd: str):
        assert _classify_command(cmd) == "dangerous"

    # --- Sudo wrapping ---
    def test_sudo_safe_command(self):
        assert _classify_command("sudo ls -la") == "safe"

    def test_sudo_elevated_command(self):
        assert _classify_command("sudo apt install vim") == "elevated"

    # --- Edge cases ---
    def test_unknown_command_is_elevated(self):
        assert _classify_command("some_random_binary --flag") == "elevated"

    def test_empty_command(self):
        # shlex.split("") returns [], base = ""
        result = _classify_command("")
        assert result in ("safe", "elevated")

    def test_malformed_quotes(self):
        # shlex.split fails on unclosed quotes
        result = _classify_command("echo 'unclosed")
        assert result == "elevated"  # safe fallback on parse error


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Command sets validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCommandSets:
    def test_safe_commands_are_frozenset(self):
        assert isinstance(SAFE_COMMANDS, frozenset)

    def test_elevated_commands_are_frozenset(self):
        assert isinstance(ELEVATED_COMMANDS, frozenset)

    def test_dangerous_patterns_are_frozenset(self):
        assert isinstance(DANGEROUS_PATTERNS, frozenset)

    def test_no_overlap_safe_elevated(self):
        overlap = SAFE_COMMANDS & ELEVATED_COMMANDS
        assert overlap == frozenset(), f"Overlap: {overlap}"

    def test_critical_safe_commands_present(self):
        for cmd in ["ls", "cat", "pwd", "git", "python", "pip", "pytest"]:
            assert cmd in SAFE_COMMANDS, f"{cmd} missing from SAFE_COMMANDS"

    def test_critical_dangerous_patterns(self):
        assert "rm -rf /" in DANGEROUS_PATTERNS
        assert ":(){ :|:& };:" in DANGEROUS_PATTERNS  # fork bomb


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ShellTools instantiation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestShellToolsInit:
    def test_instantiation(self):
        tools = ShellTools()
        assert tools.name == "shell_tools"

    def test_with_custom_state(self):
        from pathlib import Path
        state = ShellState(cwd=Path("/tmp"))
        tools = ShellTools(state=state)
        assert tools._state.cwd == Path("/tmp")

    def test_registered_functions(self):
        tools = ShellTools()
        # Shell tools are async, so they go in async_functions
        func_names = list(tools.async_functions.keys())
        assert "run_command" in func_names
        assert "run_command_sudo" in func_names
        assert "get_cwd" in func_names
        assert "run_background" in func_names
        assert "check_background" in func_names
        assert "stop_background" in func_names
        assert len(func_names) == 6
