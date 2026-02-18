"""Tests for AndroidBridge and AndroidTools."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proc(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    """Create a fake asyncio.subprocess result."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    return proc


# ---------------------------------------------------------------------------
# find_adb
# ---------------------------------------------------------------------------


class TestFindAdb:
    def test_hint_path_exists(self, tmp_path):
        from enton.providers.android_bridge import find_adb

        fake_adb = tmp_path / "adb"
        fake_adb.touch()
        assert find_adb(str(fake_adb)) == str(fake_adb)

    def test_hint_path_not_exists(self):
        from enton.providers.android_bridge import find_adb

        result = find_adb("/nonexistent/adb")
        # Falls through to PATH / SDK check — may or may not find adb
        assert result is None or isinstance(result, str)

    def test_empty_hint_uses_shutil(self):
        from enton.providers.android_bridge import find_adb

        with patch("enton.providers.android_bridge.shutil.which", return_value="/usr/bin/adb"):
            assert find_adb("") == "/usr/bin/adb"


# ---------------------------------------------------------------------------
# AndroidBridge
# ---------------------------------------------------------------------------


class TestAndroidBridge:
    def _make_bridge(self, serial: str = ""):
        from enton.providers.android_bridge import AndroidBridge

        return AndroidBridge(adb_path="/usr/bin/adb", device_serial=serial)

    @pytest.mark.asyncio
    async def test_exec_builds_command_with_serial(self):
        bridge = self._make_bridge(serial="ABC123")
        proc = _make_proc(stdout=b"hello")

        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            out, _err, _rc = await bridge._exec("shell", "echo hello")

        args = mock_exec.call_args[0]
        assert args == ("/usr/bin/adb", "-s", "ABC123", "shell", "echo hello")
        assert out == "hello"

    @pytest.mark.asyncio
    async def test_exec_builds_command_without_serial(self):
        bridge = self._make_bridge(serial="")
        proc = _make_proc(stdout=b"ok")

        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            out, _, _ = await bridge._exec("devices")

        args = mock_exec.call_args[0]
        assert args == ("/usr/bin/adb", "devices")
        assert out == "ok"

    @pytest.mark.asyncio
    async def test_exec_raw_returns_bytes(self):
        bridge = self._make_bridge()
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        proc = _make_proc(stdout=png)

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            out, _, _ = await bridge._exec("exec-out", "screencap", "-p", raw=True)

        assert isinstance(out, bytes)
        assert out == png

    @pytest.mark.asyncio
    async def test_is_connected_true(self):
        bridge = self._make_bridge()
        proc = _make_proc(stdout=b"device")

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            assert await bridge.is_connected() is True

    @pytest.mark.asyncio
    async def test_is_connected_false(self):
        bridge = self._make_bridge()
        proc = _make_proc(stdout=b"", returncode=1)

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            assert await bridge.is_connected() is False

    @pytest.mark.asyncio
    async def test_shell_returns_output(self):
        bridge = self._make_bridge()
        proc = _make_proc(stdout=b"hello from phone")

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await bridge.shell("echo hello from phone")

        assert result == "hello from phone"

    @pytest.mark.asyncio
    async def test_shell_includes_error(self):
        bridge = self._make_bridge()
        proc = _make_proc(stdout=b"", stderr=b"not found", returncode=1)

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await bridge.shell("bad_command")

        assert "ERRO" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_screenshot_returns_bytes(self):
        bridge = self._make_bridge()
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        proc = _make_proc(stdout=png)

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            data = await bridge.screenshot()

        assert data == png

    @pytest.mark.asyncio
    async def test_tap_sends_correct_command(self):
        bridge = self._make_bridge()
        proc = _make_proc(stdout=b"")

        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            result = await bridge.tap(100, 200)

        args = mock_exec.call_args[0]
        assert "input tap 100 200" in " ".join(args)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_swipe_sends_correct_command(self):
        bridge = self._make_bridge()
        proc = _make_proc(stdout=b"")

        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            await bridge.swipe(10, 20, 300, 400, 500)

        args = mock_exec.call_args[0]
        assert "input swipe 10 20 300 400 500" in " ".join(args)

    @pytest.mark.asyncio
    async def test_keyevent_sends_command(self):
        bridge = self._make_bridge()
        proc = _make_proc(stdout=b"")

        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            await bridge.keyevent("HOME")

        args = mock_exec.call_args[0]
        assert "input keyevent HOME" in " ".join(args)

    @pytest.mark.asyncio
    async def test_list_packages(self):
        bridge = self._make_bridge()
        proc = _make_proc(
            stdout=b"package:com.whatsapp\npackage:com.google.chrome\npackage:com.instagram.android",
        )

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            pkgs = await bridge.list_packages()

        assert "com.whatsapp" in pkgs
        assert "com.google.chrome" in pkgs
        assert len(pkgs) == 3

    @pytest.mark.asyncio
    async def test_list_packages_filtered(self):
        bridge = self._make_bridge()
        proc = _make_proc(
            stdout=b"package:com.whatsapp\npackage:com.google.chrome\npackage:com.instagram.android",
        )

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            pkgs = await bridge.list_packages("google")

        assert pkgs == ["com.google.chrome"]

    @pytest.mark.asyncio
    async def test_device_info(self):
        bridge = self._make_bridge()

        _responses = {
            "ro.product.model": b"Galaxy A01 Core",
            "ro.product.brand": b"samsung",
            "ro.build.version.release": b"10",
            "ro.build.version.sdk": b"29",
            "ro.serialno": b"RQ8N70E19KH",
            "ro.build.display.id": b"QQ3A.200805.001",
            "dumpsys battery": b"  level: 85",
            "wm size": b"Physical size: 720x1480",
        }

        async def fake_exec(*args, **kwargs):
            cmd = " ".join(args)
            for key, val in _responses.items():
                if key in cmd:
                    return _make_proc(stdout=val)
            return _make_proc(stdout=b"")

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            info = await bridge.device_info()

        assert info["model"] == "Galaxy A01 Core"
        assert info["brand"] == "samsung"
        assert info["battery"] == "85%"
        assert info["screen"] == "720x1480"

    @pytest.mark.asyncio
    async def test_open_url(self):
        bridge = self._make_bridge()
        proc = _make_proc(stdout=b"Starting: Intent")

        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            await bridge.open_url("https://google.com")

        args = mock_exec.call_args[0]
        assert "android.intent.action.VIEW" in " ".join(args)
        assert "https://google.com" in " ".join(args)


# ---------------------------------------------------------------------------
# AndroidTools (Toolkit)
# ---------------------------------------------------------------------------


class TestAndroidTools:
    def _make_tools(self):
        from enton.providers.android_bridge import AndroidBridge
        from enton.skills.android_toolkit import AndroidTools

        bridge = MagicMock(spec=AndroidBridge)
        tools = AndroidTools(bridge)
        return tools, bridge

    @pytest.mark.asyncio
    async def test_phone_status_connected(self):
        tools, bridge = self._make_tools()
        bridge.is_connected = AsyncMock(return_value=True)
        bridge.device_info = AsyncMock(return_value={"model": "Galaxy A01", "battery": "85%"})

        result = await tools.phone_status()
        assert "conectado" in result.lower()
        assert "Galaxy A01" in result

    @pytest.mark.asyncio
    async def test_phone_status_disconnected(self):
        tools, bridge = self._make_tools()
        bridge.is_connected = AsyncMock(return_value=False)

        result = await tools.phone_status()
        assert "NAO" in result

    @pytest.mark.asyncio
    async def test_phone_shell(self):
        tools, bridge = self._make_tools()
        bridge.shell = AsyncMock(return_value="hello")

        result = await tools.phone_shell("echo hello")
        assert result == "hello"
        bridge.shell.assert_called_once_with("echo hello")

    @pytest.mark.asyncio
    async def test_phone_screenshot_saves_file(self, tmp_path):
        tools, bridge = self._make_tools()
        png = b"\x89PNG" + b"\x00" * 100
        bridge.screenshot = AsyncMock(return_value=png)
        save_path = str(tmp_path / "shot.png")

        result = await tools.phone_screenshot(save_path=save_path)
        assert "salvo" in result.lower()
        assert (tmp_path / "shot.png").read_bytes() == png

    @pytest.mark.asyncio
    async def test_phone_screenshot_empty(self):
        tools, bridge = self._make_tools()
        bridge.screenshot = AsyncMock(return_value=b"")

        result = await tools.phone_screenshot()
        assert "vazio" in result.lower() or "erro" in result.lower()

    @pytest.mark.asyncio
    async def test_phone_tap(self):
        tools, bridge = self._make_tools()
        bridge.tap = AsyncMock(return_value="ok")

        result = await tools.phone_tap(100, 200)
        assert result == "ok"
        bridge.tap.assert_called_once_with(100, 200)

    @pytest.mark.asyncio
    async def test_phone_apps_with_filter(self):
        tools, bridge = self._make_tools()
        bridge.list_packages = AsyncMock(return_value=["com.google.chrome"])

        result = await tools.phone_apps(filter_text="google")
        assert "google" in result.lower()
        assert "chrome" in result.lower()

    @pytest.mark.asyncio
    async def test_phone_apps_empty(self):
        tools, bridge = self._make_tools()
        bridge.list_packages = AsyncMock(return_value=[])

        result = await tools.phone_apps(filter_text="xyz")
        assert "nenhum" in result.lower()

    @pytest.mark.asyncio
    async def test_phone_key(self):
        tools, bridge = self._make_tools()
        bridge.keyevent = AsyncMock(return_value="ok")

        result = await tools.phone_key("HOME")
        assert result == "ok"
        bridge.keyevent.assert_called_once_with("HOME")

    @pytest.mark.asyncio
    async def test_phone_open_url(self):
        tools, bridge = self._make_tools()
        bridge.open_url = AsyncMock(return_value="ok")

        result = await tools.phone_open_url("https://example.com")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_phone_error_handling(self):
        tools, bridge = self._make_tools()
        bridge.shell = AsyncMock(side_effect=TimeoutError("ADB timeout"))

        result = await tools.phone_shell("slow_command")
        assert "erro" in result.lower()
        assert "timeout" in result.lower()

    @pytest.mark.asyncio
    async def test_phone_push_file(self):
        tools, bridge = self._make_tools()
        bridge.push = AsyncMock(return_value="1 file pushed")

        result = await tools.phone_push_file("/tmp/test.txt", "/sdcard/test.txt")
        assert "pushed" in result.lower()

    @pytest.mark.asyncio
    async def test_phone_pull_file(self):
        tools, bridge = self._make_tools()
        bridge.pull = AsyncMock(return_value="1 file pulled")

        result = await tools.phone_pull_file("/sdcard/photo.jpg", "/tmp/photo.jpg")
        assert "pulled" in result.lower()
