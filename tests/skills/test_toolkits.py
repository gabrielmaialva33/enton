"""Test toolkit instantiation and tool registration for all Agno toolkits.

Each test verifies:
  1. The toolkit class can be imported
  2. It can be instantiated (with MagicMock deps when needed)
  3. It is an instance of agno.tools.Toolkit (for Agno-based toolkits)
  4. It has the expected tools registered (sync in self.functions,
     async in self.async_functions)
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _all_registered_tools(instance) -> set[str]:
    """Return the union of sync and async tool names registered on a Toolkit."""
    registered = set(instance.functions.keys())
    if hasattr(instance, "async_functions"):
        registered |= set(instance.async_functions.keys())
    return registered


def _assert_agno_toolkit(instance, expected_tools: list[str]) -> None:
    """Shared assertions for any agno Toolkit subclass."""
    from agno.tools import Toolkit

    assert isinstance(instance, Toolkit), (
        f"{type(instance).__name__} is not a Toolkit subclass"
    )
    registered = _all_registered_tools(instance)
    for tool_name in expected_tools:
        assert tool_name in registered, (
            f"Tool '{tool_name}' not found in {type(instance).__name__}. "
            f"Registered: {sorted(registered)}"
        )


# ---------------------------------------------------------------------------
# Tests — Agno Toolkit subclasses
# ---------------------------------------------------------------------------


class TestShellTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.shell_toolkit import ShellTools
        except ImportError:
            pytest.skip("ShellTools dependencies not installed")

        toolkit = ShellTools()
        _assert_agno_toolkit(toolkit, [
            "run_command",
            "run_command_sudo",
            "get_cwd",
            "run_background",
            "check_background",
            "stop_background",
        ])


class TestCodingTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.coding_toolkit import CodingTools
        except ImportError:
            pytest.skip("CodingTools dependencies not installed")

        toolkit = CodingTools()
        _assert_agno_toolkit(toolkit, [
            "code_run",
            "code_reference",
            "code_languages",
            "code_benchmark",
        ])


class TestSystemTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.system_toolkit import SystemTools
        except ImportError:
            pytest.skip("SystemTools dependencies not installed")

        toolkit = SystemTools()
        _assert_agno_toolkit(toolkit, [
            "get_system_stats",
            "get_time",
            "list_processes",
        ])


class TestSearchTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.search_toolkit import SearchTools
        except ImportError:
            pytest.skip("SearchTools dependencies not installed")

        toolkit = SearchTools()
        _assert_agno_toolkit(toolkit, [
            "search_web",
        ])


class TestDesktopTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.desktop_toolkit import DesktopTools
        except ImportError:
            pytest.skip("DesktopTools dependencies not installed")

        toolkit = DesktopTools(brain=None)
        _assert_agno_toolkit(toolkit, [
            "screenshot",
            "screenshot_analyze",
            "ocr_screen",
            "click",
            "type_text",
            "press_key",
            "clipboard_get",
            "clipboard_set",
            "active_window",
            "list_windows",
            "focus_window",
            "notify",
            "mouse_move",
            "screen_size",
        ])


class TestBrowserTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.browser_toolkit import BrowserTools
        except ImportError:
            pytest.skip("BrowserTools dependencies not installed")

        toolkit = BrowserTools()
        _assert_agno_toolkit(toolkit, [
            "browse_url",
            "web_screenshot",
            "web_search",
            "extract_text",
            "download_file",
        ])


class TestMediaTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.media_toolkit import MediaTools
        except ImportError:
            pytest.skip("MediaTools dependencies not installed")

        toolkit = MediaTools()
        _assert_agno_toolkit(toolkit, [
            "download_video",
            "download_audio",
            "media_info",
            "play_media",
            "player_control",
            "volume_get",
            "volume_set",
            "list_audio_sinks",
            "tts_speak",
        ])


class TestNetworkTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.network_toolkit import NetworkTools
        except ImportError:
            pytest.skip("NetworkTools dependencies not installed")

        toolkit = NetworkTools()
        _assert_agno_toolkit(toolkit, [
            "network_scan",
            "port_scan",
            "ping",
            "local_ip",
            "bluetooth_devices",
            "bluetooth_connect",
            "bluetooth_disconnect",
            "tailscale_status",
            "wifi_networks",
            "dns_lookup",
        ])


class TestPTZTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.ptz_toolkit import PTZTools
        except ImportError:
            pytest.skip("PTZTools dependencies not installed")

        toolkit = PTZTools()
        _assert_agno_toolkit(toolkit, [
            "camera_move",
            "camera_motor_move",
        ])


class TestMemoryTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.memory_toolkit import MemoryTools
        except ImportError:
            pytest.skip("MemoryTools dependencies not installed")

        toolkit = MemoryTools(memory=MagicMock())
        _assert_agno_toolkit(toolkit, [
            "search_memory",
            "recall_recent",
            "what_do_you_know_about_user",
        ])


class TestKnowledgeTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.knowledge_toolkit import KnowledgeTools
        except ImportError:
            pytest.skip("KnowledgeTools dependencies not installed")

        toolkit = KnowledgeTools(crawler=MagicMock())
        _assert_agno_toolkit(toolkit, [
            "learn_from_url",
            "search_knowledge",
            "learn_about_topic",
        ])


class TestVisualMemoryTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.visual_memory_toolkit import VisualMemoryTools
        except ImportError:
            pytest.skip("VisualMemoryTools dependencies not installed")

        toolkit = VisualMemoryTools(visual_memory=MagicMock())
        _assert_agno_toolkit(toolkit, [
            "search_visual_memory",
            "recall_recent_scenes",
        ])


class TestPlannerTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.planner_toolkit import PlannerTools
        except ImportError:
            pytest.skip("PlannerTools dependencies not installed")

        toolkit = PlannerTools(planner=MagicMock())
        _assert_agno_toolkit(toolkit, [
            "add_reminder",
            "add_recurring_reminder",
            "list_reminders",
            "cancel_reminder",
            "add_todo",
            "complete_todo",
            "list_todos",
        ])


class TestForgeTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.forge_toolkit import ForgeTools
        except ImportError:
            pytest.skip("ForgeTools dependencies not installed")

        toolkit = ForgeTools(forge=MagicMock(), registry=MagicMock())
        _assert_agno_toolkit(toolkit, [
            "create_tool",
            "list_dynamic_tools",
            "retire_tool",
            "tool_stats",
        ])


class TestChannelTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.channel_toolkit import ChannelTools
        except ImportError:
            pytest.skip("ChannelTools dependencies not installed")

        toolkit = ChannelTools(channel_manager=MagicMock())
        _assert_agno_toolkit(toolkit, [
            "send_message",
            "broadcast_message",
            "list_channels",
        ])


class TestExtensionTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.extension_toolkit import ExtensionTools
        except ImportError:
            pytest.skip("ExtensionTools dependencies not installed")

        toolkit = ExtensionTools(registry=MagicMock())
        _assert_agno_toolkit(toolkit, [
            "extension_list",
            "extension_enable",
            "extension_disable",
            "extension_install",
            "extension_stats",
        ])


class TestProcessTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.process_toolkit import ProcessTools
        except ImportError:
            pytest.skip("ProcessTools dependencies not installed")

        toolkit = ProcessTools(manager=MagicMock())
        _assert_agno_toolkit(toolkit, [
            "task_run",
            "task_status",
            "task_list",
            "task_output",
            "task_cancel",
            "task_summary",
        ])


class TestWorkspaceTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.workspace_toolkit import WorkspaceTools
        except ImportError:
            pytest.skip("WorkspaceTools dependencies not installed")

        # Pass a MagicMock for hardware to avoid detect_hardware() side effects
        toolkit = WorkspaceTools(
            workspace=Path("/tmp/enton-test-workspace"),
            hardware=MagicMock(),
        )
        _assert_agno_toolkit(toolkit, [
            "workspace_info",
            "workspace_list",
            "hardware_status",
            "hardware_gpu",
            "hardware_full",
            "project_create",
            "project_list",
            "disk_usage",
        ])


class TestGcpTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.gcp_toolkit import GcpTools
        except ImportError:
            pytest.skip("GcpTools dependencies not installed")

        toolkit = GcpTools()
        _assert_agno_toolkit(toolkit, [
            "gcp_status",
            "gcp_vm_presets",
            "gcp_vm_create",
            "gcp_vm_list",
            "gcp_vm_ssh",
            "gcp_vm_delete",
            "gcp_vm_start",
            "gcp_vm_stop",
            "gcp_run_code",
            "gcp_billing",
        ])


class TestSubAgentTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.sub_agent_toolkit import SubAgentTools
        except ImportError:
            pytest.skip("SubAgentTools dependencies not installed")

        toolkit = SubAgentTools(orchestrator=MagicMock())
        _assert_agno_toolkit(toolkit, [
            "delegate_task",
            "auto_delegate",
            "agent_consensus",
            "list_agents",
        ])


class TestDescribeTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.describe_toolkit import DescribeTools
        except ImportError:
            pytest.skip("DescribeTools dependencies not installed")

        toolkit = DescribeTools(vision=MagicMock(), brain=None)
        _assert_agno_toolkit(toolkit, [
            "describe_scene",
            "what_do_you_see",
        ])


class TestBlobTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.blob_toolkit import BlobTools
        except ImportError:
            pytest.skip("BlobTools dependencies not installed")

        toolkit = BlobTools(blob_store=MagicMock())
        _assert_agno_toolkit(toolkit, [
            "search_blobs",
            "recent_blobs",
            "blob_stats",
        ])


class TestN8nTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.n8n_toolkit import N8nTools
        except ImportError:
            pytest.skip("N8nTools dependencies not installed")

        toolkit = N8nTools()
        _assert_agno_toolkit(toolkit, [
            "trigger_automation",
        ])


class TestScreenpipeTools:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.screenpipe_toolkit import ScreenpipeTools
        except ImportError:
            pytest.skip("ScreenpipeTools dependencies not installed")

        toolkit = ScreenpipeTools()
        _assert_agno_toolkit(toolkit, [
            "search_screen",
            "get_recent_activity",
        ])


# ---------------------------------------------------------------------------
# Tests — Legacy toolkits (non-Agno, use get_tools() pattern)
# ---------------------------------------------------------------------------


class TestCryptoToolkit:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.crypto_toolkit import CryptoToolkit
        except ImportError:
            pytest.skip("CryptoToolkit dependencies not installed")

        # Use a temp path to avoid creating the wallet in the real project dir
        import tempfile, os
        wallet_path = os.path.join(tempfile.mkdtemp(), "paper_wallet.json")
        toolkit = CryptoToolkit(wallet_path=wallet_path)

        assert hasattr(toolkit, "get_tools"), "CryptoToolkit must expose get_tools()"
        tools = toolkit.get_tools()
        assert isinstance(tools, list)
        tool_names = {t["name"] for t in tools}
        for expected in ["get_crypto_price"]:
            assert expected in tool_names, (
                f"Tool '{expected}' not in CryptoToolkit.get_tools(). "
                f"Found: {sorted(tool_names)}"
            )

    def test_has_name(self):
        try:
            from enton.skills.crypto_toolkit import CryptoToolkit
        except ImportError:
            pytest.skip("CryptoToolkit dependencies not installed")

        import tempfile, os
        wallet_path = os.path.join(tempfile.mkdtemp(), "paper_wallet.json")
        toolkit = CryptoToolkit(wallet_path=wallet_path)
        assert toolkit.name == "crypto_toolkit"


class TestGodModeToolkit:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.god_mode_toolkit import GodModeToolkit
        except ImportError:
            pytest.skip("GodModeToolkit dependencies not installed")

        toolkit = GodModeToolkit()

        assert hasattr(toolkit, "get_tools"), "GodModeToolkit must expose get_tools()"
        tools = toolkit.get_tools()
        assert isinstance(tools, list)
        tool_names = {t["name"] for t in tools}
        for expected in ["list_heavy_processes", "kill_process"]:
            assert expected in tool_names, (
                f"Tool '{expected}' not in GodModeToolkit.get_tools(). "
                f"Found: {sorted(tool_names)}"
            )

    def test_has_name(self):
        try:
            from enton.skills.god_mode_toolkit import GodModeToolkit
        except ImportError:
            pytest.skip("GodModeToolkit dependencies not installed")

        toolkit = GodModeToolkit()
        assert toolkit.name == "god_mode_toolkit"


class TestNeurosurgeonToolkit:
    def test_instantiation_and_tools(self):
        try:
            from enton.skills.neurosurgeon_toolkit import NeurosurgeonToolkit
        except ImportError:
            pytest.skip("NeurosurgeonToolkit dependencies not installed")

        toolkit = NeurosurgeonToolkit()

        assert hasattr(toolkit, "get_tools"), (
            "NeurosurgeonToolkit must expose get_tools()"
        )
        tools = toolkit.get_tools()
        assert isinstance(tools, list)
        tool_names = {t["name"] for t in tools}
        for expected in ["read_enton_source", "backup_module"]:
            assert expected in tool_names, (
                f"Tool '{expected}' not in NeurosurgeonToolkit.get_tools(). "
                f"Found: {sorted(tool_names)}"
            )

    def test_has_name(self):
        try:
            from enton.skills.neurosurgeon_toolkit import NeurosurgeonToolkit
        except ImportError:
            pytest.skip("NeurosurgeonToolkit dependencies not installed")

        toolkit = NeurosurgeonToolkit()
        assert toolkit.name == "neurosurgeon_toolkit"


# ---------------------------------------------------------------------------
# Parametrized sanity check — every Agno toolkit is a Toolkit subclass
# ---------------------------------------------------------------------------


_AGNO_TOOLKIT_IMPORTS = [
    ("enton.skills.shell_toolkit", "ShellTools"),
    ("enton.skills.coding_toolkit", "CodingTools"),
    ("enton.skills.system_toolkit", "SystemTools"),
    ("enton.skills.search_toolkit", "SearchTools"),
    ("enton.skills.desktop_toolkit", "DesktopTools"),
    ("enton.skills.browser_toolkit", "BrowserTools"),
    ("enton.skills.media_toolkit", "MediaTools"),
    ("enton.skills.network_toolkit", "NetworkTools"),
    ("enton.skills.ptz_toolkit", "PTZTools"),
    ("enton.skills.memory_toolkit", "MemoryTools"),
    ("enton.skills.knowledge_toolkit", "KnowledgeTools"),
    ("enton.skills.visual_memory_toolkit", "VisualMemoryTools"),
    ("enton.skills.planner_toolkit", "PlannerTools"),
    ("enton.skills.forge_toolkit", "ForgeTools"),
    ("enton.skills.channel_toolkit", "ChannelTools"),
    ("enton.skills.extension_toolkit", "ExtensionTools"),
    ("enton.skills.process_toolkit", "ProcessTools"),
    ("enton.skills.workspace_toolkit", "WorkspaceTools"),
    ("enton.skills.gcp_toolkit", "GcpTools"),
    ("enton.skills.sub_agent_toolkit", "SubAgentTools"),
    ("enton.skills.describe_toolkit", "DescribeTools"),
    ("enton.skills.blob_toolkit", "BlobTools"),
    ("enton.skills.n8n_toolkit", "N8nTools"),
    ("enton.skills.screenpipe_toolkit", "ScreenpipeTools"),
]


@pytest.mark.parametrize("module_path,class_name", _AGNO_TOOLKIT_IMPORTS)
def test_agno_toolkit_is_subclass(module_path: str, class_name: str):
    """Every Agno-based toolkit must be a subclass of agno.tools.Toolkit."""
    import importlib

    from agno.tools import Toolkit

    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        pytest.skip(f"Could not import {module_path}")

    cls = getattr(mod, class_name, None)
    assert cls is not None, f"{class_name} not found in {module_path}"
    assert issubclass(cls, Toolkit), (
        f"{class_name} is not a subclass of agno.tools.Toolkit"
    )


@pytest.mark.parametrize("module_path,class_name", _AGNO_TOOLKIT_IMPORTS)
def test_agno_toolkit_has_registered_tools(module_path: str, class_name: str):
    """Every Agno toolkit must register at least one tool (sync or async)."""
    import importlib

    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        pytest.skip(f"Could not import {module_path}")

    cls = getattr(mod, class_name)

    # Build kwargs to satisfy __init__ signatures with MagicMock
    init_mock_kwargs = _mock_kwargs_for(class_name)
    instance = cls(**init_mock_kwargs)

    registered = _all_registered_tools(instance)
    assert len(registered) > 0, (
        f"{class_name} registered 0 tools (sync+async) -- expected at least 1"
    )


# ---------------------------------------------------------------------------
# Mock factory for toolkit constructors
# ---------------------------------------------------------------------------


def _mock_kwargs_for(class_name: str) -> dict:
    """Return minimal MagicMock kwargs to instantiate a toolkit class."""
    mapping: dict[str, dict] = {
        "ShellTools": {},
        "CodingTools": {},
        "SystemTools": {},
        "SearchTools": {},
        "DesktopTools": {"brain": None},
        "BrowserTools": {},
        "MediaTools": {},
        "NetworkTools": {},
        "PTZTools": {},
        "MemoryTools": {"memory": MagicMock()},
        "KnowledgeTools": {"crawler": MagicMock()},
        "VisualMemoryTools": {"visual_memory": MagicMock()},
        "PlannerTools": {"planner": MagicMock()},
        "ForgeTools": {"forge": MagicMock(), "registry": MagicMock()},
        "ChannelTools": {"channel_manager": MagicMock()},
        "ExtensionTools": {"registry": MagicMock()},
        "ProcessTools": {"manager": MagicMock()},
        "WorkspaceTools": {
            "workspace": Path("/tmp/enton-test-workspace"),
            "hardware": MagicMock(),
        },
        "GcpTools": {},
        "SubAgentTools": {"orchestrator": MagicMock()},
        "DescribeTools": {"vision": MagicMock(), "brain": None},
        "BlobTools": {"blob_store": MagicMock()},
        "N8nTools": {},
        "ScreenpipeTools": {},
    }
    return mapping.get(class_name, {})
