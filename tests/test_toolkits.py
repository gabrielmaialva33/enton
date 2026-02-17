"""Tests for Agno Toolkits (pure logic, no hardware)."""

from __future__ import annotations

import datetime

from enton.skills.search_toolkit import _extract_url, _strip_tags
from enton.skills.shell_toolkit import _classify_command

# --- Shell safety classification ---


def test_classify_safe_commands():
    assert _classify_command("ls -la") == "safe"
    assert _classify_command("git status") == "safe"
    assert _classify_command("nvidia-smi") == "safe"
    assert _classify_command("docker ps") == "safe"
    assert _classify_command("python script.py") == "safe"
    assert _classify_command("uv sync") == "safe"


def test_classify_elevated_commands():
    assert _classify_command("apt install vim") == "elevated"
    assert _classify_command("kill 1234") == "elevated"
    assert _classify_command("pip install flask") == "elevated"


def test_classify_dangerous_commands():
    assert _classify_command("rm -rf /") == "dangerous"
    assert _classify_command("rm -rf /*") == "dangerous"
    assert _classify_command("mkfs /dev/sda") == "dangerous"
    assert _classify_command("shutdown now") == "dangerous"


def test_classify_sudo_wrapping():
    assert _classify_command("sudo ls") == "safe"
    assert _classify_command("sudo apt update") == "elevated"


def test_classify_unknown_defaults_elevated():
    assert _classify_command("some_unknown_binary --flag") == "elevated"


# --- Search helpers ---


def test_strip_tags():
    assert _strip_tags("<b>Hello</b> world") == "Hello world"
    assert _strip_tags("No &amp; tags") == "No & tags"
    assert _strip_tags("") == ""


def test_extract_url_direct():
    assert _extract_url("https://example.com") == "https://example.com"


def test_extract_url_ddg_redirect():
    raw = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&rut=abc"
    assert _extract_url(raw) == "https://example.com"


# --- SystemTools instantiation ---


def test_system_tools_registers():
    from enton.skills.system_toolkit import SystemTools

    st = SystemTools()
    assert st.name == "system_tools"


def test_system_tools_get_time():
    from enton.skills.system_toolkit import SystemTools

    st = SystemTools()
    result = st.get_time()
    # Should be a valid ISO datetime
    datetime.datetime.fromisoformat(result)


def test_system_tools_get_stats():
    from enton.skills.system_toolkit import SystemTools

    st = SystemTools()
    result = st.get_system_stats()
    assert "CPU" in result
    assert "Memory" in result


def test_system_tools_list_processes():
    from enton.skills.system_toolkit import SystemTools

    st = SystemTools()
    result = st.list_processes(limit=3)
    assert "PID" in result or "Nenhum" in result


# --- Toolkit registration ---


def test_ptz_tools_registers():
    from enton.skills.ptz_toolkit import PTZTools

    pt = PTZTools()
    assert pt.name == "ptz_tools"


def test_shell_tools_registers():
    from enton.skills.shell_toolkit import ShellTools

    st = ShellTools()
    assert st.name == "shell_tools"


def test_search_tools_registers():
    from enton.skills.search_toolkit import SearchTools

    st = SearchTools()
    assert st.name == "search_tools"


def test_memory_tools_registers():
    from unittest.mock import MagicMock

    from enton.skills.memory_toolkit import MemoryTools

    mem = MagicMock()
    mt = MemoryTools(mem)
    assert mt.name == "memory_tools"


def test_planner_tools_registers():
    from unittest.mock import MagicMock

    from enton.skills.planner_toolkit import PlannerTools

    planner = MagicMock()
    pt = PlannerTools(planner)
    assert pt.name == "planner_tools"


def test_face_tools_registers():
    from unittest.mock import MagicMock

    from enton.skills.face_toolkit import FaceTools

    vision = MagicMock()
    ft = FaceTools(vision, face_recognizer=None)
    assert ft.name == "face_tools"


def test_describe_tools_registers():
    from unittest.mock import MagicMock

    from enton.skills.describe_toolkit import DescribeTools

    vision = MagicMock()
    dt = DescribeTools(vision)
    assert dt.name == "describe_tools"
