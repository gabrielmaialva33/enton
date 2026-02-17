"""Android Debug Bridge — async subprocess wrapper for ADB.

Provides a low-level async interface to control Android devices via
the ADB command-line tool. No Python ADB library needed — just the
system ``adb`` binary.

Requires: ADB installed (``apt install adb`` or Android SDK platform-tools).
"""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_SDK_ADB = Path.home() / "Android" / "Sdk" / "platform-tools" / "adb"


def _extract_field(line: str, field: str) -> str:
    """Extract a field value from ADB content query output.

    Lines look like: ``Row: 0 display_name=John, data1=+55..., mimetype=...``
    """
    if f"{field}=" not in line:
        return ""
    part = line.split(f"{field}=")[1]
    # Value ends at next comma-space or end of line
    val = part.split(", ")[0].strip()
    if val == "NULL":
        return ""
    return val


def find_adb(hint: str = "") -> str | None:
    """Locate the ADB binary. Priority: hint > PATH > ~/Android/Sdk/."""
    if hint:
        p = Path(hint)
        if p.is_file():
            return str(p)
    # Try system PATH
    found = shutil.which("adb")
    if found:
        return found
    # Try Android SDK default location
    if _SDK_ADB.is_file():
        return str(_SDK_ADB)
    return None


class AndroidBridge:
    """Low-level ADB bridge — async subprocess wrapper."""

    def __init__(
        self,
        adb_path: str,
        device_serial: str = "",
        timeout: float = 15.0,
        wifi_host: str = "",
        wifi_port: int = 5555,
    ) -> None:
        self._adb = adb_path
        self._serial = device_serial
        self._timeout = timeout
        self._wifi_host = wifi_host
        self._wifi_port = wifi_port

    # ------------------------------------------------------------------
    # Core exec
    # ------------------------------------------------------------------

    async def _exec(
        self, *args: str, timeout: float | None = None, raw: bool = False,
    ) -> tuple[bytes | str, str, int]:
        """Run an ADB command and return (stdout, stderr, returncode).

        When *raw* is True, stdout is returned as ``bytes`` (useful for
        binary data like screenshots).
        """
        cmd = [self._adb]
        if self._serial:
            cmd += ["-s", self._serial]
        cmd += list(args)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout or self._timeout,
        )
        err = stderr.decode(errors="replace").strip()
        if raw:
            return stdout, err, proc.returncode or 0
        return stdout.decode(errors="replace").strip(), err, proc.returncode or 0

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def is_connected(self) -> bool:
        """Check whether a device is reachable via ADB."""
        out, _err, rc = await self._exec("get-state")
        return rc == 0 and "device" in str(out)

    async def get_ip(self) -> str:
        """Get the phone's WiFi IP address."""
        out, _, rc = await self._exec("shell", "ip addr show wlan0")
        if rc != 0:
            return ""
        for raw_line in str(out).splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("inet "):
                # "inet 10.42.0.66/24 ..." → "10.42.0.66"
                return stripped.split()[1].split("/")[0]
        return ""

    async def enable_tcpip(self, port: int = 0) -> str:
        """Switch device to TCP/IP mode (requires USB connection first).

        After this, the device accepts ADB connections over WiFi on the given port.
        """
        p = port or self._wifi_port
        out, err, rc = await self._exec("tcpip", str(p))
        if rc != 0:
            return f"ERRO: {err or out}"
        # Give device time to restart ADB in TCP mode
        await asyncio.sleep(2)
        # Auto-discover IP if not set
        ip = await self.get_ip()
        if ip and not self._wifi_host:
            self._wifi_host = ip
        return f"TCP/IP mode ativo na porta {p}. IP do celular: {ip or '(use get_ip)'}"

    async def connect_wifi(self, host: str = "", port: int = 0) -> str:
        """Connect to device over WiFi (TCP/IP must be enabled first)."""
        h = host or self._wifi_host
        p = port or self._wifi_port
        if not h:
            return "ERRO: IP do celular nao definido. Use enable_tcpip() via USB primeiro."
        target = f"{h}:{p}"
        out, err, _rc = await self._exec("connect", target)
        result = str(out)
        if "connected" in result.lower():
            # Update serial to use WiFi target for subsequent commands
            self._serial = target
            logger.info("ADB WiFi connected: %s", target)
            return f"Conectado via WiFi: {target}"
        return f"ERRO: {result} {err}"

    async def disconnect_wifi(self, host: str = "", port: int = 0) -> str:
        """Disconnect WiFi ADB and revert to USB."""
        h = host or self._wifi_host
        p = port or self._wifi_port
        if h:
            target = f"{h}:{p}"
            await self._exec("disconnect", target)
        # Clear WiFi serial so we fall back to USB
        if self._serial and ":" in self._serial:
            self._serial = ""
        return "Desconectado do WiFi ADB. Usando USB."

    async def auto_connect(self) -> str:
        """Try USB first, then WiFi. Returns connection status."""
        # Try current connection (USB or already-connected WiFi)
        if await self.is_connected():
            mode = "WiFi" if self._serial and ":" in self._serial else "USB"
            return f"Conectado via {mode}"
        # Try WiFi if host is configured
        if self._wifi_host:
            result = await self.connect_wifi()
            if "Conectado" in result:
                return result
        return "ERRO: celular nao encontrado (USB nem WiFi)"

    async def device_info(self) -> dict[str, str]:
        """Collect device properties (model, Android version, battery, etc)."""
        props = {}
        for key, prop in [
            ("model", "ro.product.model"),
            ("brand", "ro.product.brand"),
            ("android", "ro.build.version.release"),
            ("sdk", "ro.build.version.sdk"),
            ("serial", "ro.serialno"),
            ("display", "ro.build.display.id"),
        ]:
            out, _, _ = await self._exec("shell", f"getprop {prop}")
            if out:
                props[key] = str(out)
        # Battery level
        bat, _, _ = await self._exec("shell", "dumpsys battery | grep level")
        if bat:
            props["battery"] = str(bat).split(":")[-1].strip() + "%"
        # Screen resolution
        size, _, _ = await self._exec("shell", "wm size")
        if size:
            props["screen"] = str(size).replace("Physical size: ", "")
        return props

    # ------------------------------------------------------------------
    # Shell
    # ------------------------------------------------------------------

    async def shell(self, command: str, timeout: float | None = None) -> str:
        """Execute a shell command on the device."""
        out, err, rc = await self._exec("shell", command, timeout=timeout)
        result = str(out)
        if err and rc != 0:
            result += f"\nERRO: {err}"
        return result

    # ------------------------------------------------------------------
    # Screen capture
    # ------------------------------------------------------------------

    async def screenshot(self) -> bytes:
        """Take a screenshot and return PNG bytes."""
        out, err, rc = await self._exec("exec-out", "screencap", "-p", raw=True)
        if rc != 0:
            logger.warning("Screenshot failed: %s", err)
            return b""
        return out  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    async def tap(self, x: int, y: int) -> str:
        """Tap the screen at (x, y)."""
        out, _, _ = await self._exec("shell", f"input tap {x} {y}")
        return str(out) or "ok"

    async def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300,
    ) -> str:
        """Swipe from (x1,y1) to (x2,y2)."""
        out, _, _ = await self._exec(
            "shell", f"input swipe {x1} {y1} {x2} {y2} {duration_ms}",
        )
        return str(out) or "ok"

    async def input_text(self, text: str) -> str:
        """Type text into the currently focused field."""
        # ADB input text needs spaces escaped
        safe = text.replace(" ", "%s").replace("&", "\\&").replace("<", "\\<")
        out, _, _ = await self._exec("shell", f"input text '{safe}'")
        return str(out) or "ok"

    async def keyevent(self, key: str) -> str:
        """Send a keyevent (HOME, BACK, ENTER, POWER, VOLUME_UP, etc)."""
        out, _, _ = await self._exec("shell", f"input keyevent {key}")
        return str(out) or "ok"

    # ------------------------------------------------------------------
    # Apps
    # ------------------------------------------------------------------

    async def list_packages(self, filter_str: str = "") -> list[str]:
        """List installed packages, optionally filtered."""
        out, _, _ = await self._exec("shell", "pm list packages")
        packages = [line.replace("package:", "") for line in str(out).splitlines()]
        if filter_str:
            packages = [p for p in packages if filter_str.lower() in p.lower()]
        return sorted(packages)

    async def install_apk(self, apk_path: str) -> str:
        """Install an APK file on the device."""
        out, err, rc = await self._exec("install", "-r", apk_path, timeout=60.0)
        return str(out) if rc == 0 else f"ERRO: {err or out}"

    async def uninstall(self, package: str) -> str:
        """Uninstall a package from the device."""
        out, err, rc = await self._exec("uninstall", package)
        return str(out) if rc == 0 else f"ERRO: {err or out}"

    async def open_app(self, package: str) -> str:
        """Launch an app by package name."""
        out, _, _ = await self._exec(
            "shell", f"monkey -p {package} -c android.intent.category.LAUNCHER 1",
        )
        return str(out) or "ok"

    # ------------------------------------------------------------------
    # Intents
    # ------------------------------------------------------------------

    async def open_url(self, url: str) -> str:
        """Open a URL in the device browser."""
        out, _, _ = await self._exec(
            "shell", f"am start -a android.intent.action.VIEW -d '{url}'",
        )
        return str(out) or "ok"

    # ------------------------------------------------------------------
    # File transfer
    # ------------------------------------------------------------------

    async def push(self, local_path: str, remote_path: str) -> str:
        """Push a file from PC to device."""
        out, err, rc = await self._exec("push", local_path, remote_path, timeout=60.0)
        return str(out) if rc == 0 else f"ERRO: {err or out}"

    async def pull(self, remote_path: str, local_path: str) -> str:
        """Pull a file from device to PC."""
        out, err, rc = await self._exec("pull", remote_path, local_path, timeout=60.0)
        return str(out) if rc == 0 else f"ERRO: {err or out}"

    # ------------------------------------------------------------------
    # Content Providers (contacts, SMS, calendar, call log)
    # ------------------------------------------------------------------

    async def content_query(
        self, uri: str, projection: str = "", where: str = "", sort: str = "",
    ) -> str:
        """Query an Android content provider via `adb shell content query`."""
        cmd = f"content query --uri {uri}"
        if projection:
            cmd += f" --projection {projection}"
        if where:
            cmd += f" --where \"{where}\""
        if sort:
            cmd += f" --sort \"{sort}\""
        out, err, rc = await self._exec("shell", cmd, timeout=20.0)
        if rc != 0 and err:
            return f"ERRO: {err}"
        return str(out)

    async def contacts(self, limit: int = 50) -> list[dict[str, str]]:
        """List device contacts (name + number). No root needed."""
        raw = await self.content_query(
            "content://com.android.contacts/data",
            projection="display_name:data1:mimetype",
        )
        contacts: dict[str, str] = {}
        for line in raw.splitlines():
            if "vnd.android.cursor.item/phone" not in line:
                continue
            name = _extract_field(line, "display_name")
            number = _extract_field(line, "data1")
            if name and number:
                contacts[name] = number
        result = [{"name": k, "number": v} for k, v in contacts.items()]
        return result[:limit]

    async def sms_list(self, limit: int = 20) -> list[dict[str, str]]:
        """Read recent SMS messages. May require permission grant."""
        raw = await self.content_query(
            "content://sms/",
            projection="address:body:date:type",
            sort="date DESC",
        )
        messages: list[dict[str, str]] = []
        for line in raw.splitlines():
            if "Row:" not in line:
                continue
            addr = _extract_field(line, "address")
            body = _extract_field(line, "body")
            sms_type = _extract_field(line, "type")
            direction = "recebido" if sms_type == "1" else "enviado"
            if addr and body:
                messages.append({"from": addr, "text": body, "dir": direction})
            if len(messages) >= limit:
                break
        return messages

    async def send_sms(self, number: str, message: str) -> str:
        """Send an SMS via ADB shell service call."""
        # Use am to send via default SMS app intent
        out, _, _ = await self._exec(
            "shell",
            f"am start -a android.intent.action.SENDTO -d sms:{number}"
            f" --es sms_body '{message}' --ez exit_on_sent true",
        )
        return str(out) or "Intent SMS aberto"

    async def call_log(self, limit: int = 20) -> list[dict[str, str]]:
        """Read recent call log entries."""
        raw = await self.content_query(
            "content://call_log/calls",
            projection="number:name:type:duration:date",
            sort="date DESC",
        )
        calls: list[dict[str, str]] = []
        type_map = {"1": "recebida", "2": "feita", "3": "perdida"}
        for line in raw.splitlines():
            if "Row:" not in line:
                continue
            number = _extract_field(line, "number")
            name = _extract_field(line, "name")
            call_type = type_map.get(_extract_field(line, "type"), "outro")
            duration = _extract_field(line, "duration")
            if number:
                calls.append({
                    "number": number,
                    "name": name or "Desconhecido",
                    "type": call_type,
                    "duration": f"{duration}s",
                })
            if len(calls) >= limit:
                break
        return calls

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    async def notifications(self, limit: int = 15) -> list[dict[str, str]]:
        """Get current notifications from the notification bar."""
        out, _, _ = await self._exec(
            "shell", "dumpsys notification --noredact", timeout=20.0,
        )
        notifs: list[dict[str, str]] = []
        lines = str(out).splitlines()
        pkg = title = text = ""
        for raw_line in lines:
            stripped = raw_line.strip()
            if "pkg=" in stripped:
                pkg = stripped.split("pkg=")[1].split()[0]
            if "android.title=String(" in stripped:
                title = stripped.split("android.title=String(")[1].rstrip(")")
            if "android.text=String(" in stripped:
                text = stripped.split("android.text=String(")[1].rstrip(")")
                if pkg and (title or text):
                    notifs.append({"app": pkg, "title": title, "text": text})
                    title = text = ""
                if len(notifs) >= limit:
                    break
        return notifs

    # ------------------------------------------------------------------
    # Location
    # ------------------------------------------------------------------

    async def location(self) -> dict[str, str]:
        """Get last known GPS/network location."""
        out, _, _ = await self._exec("shell", "dumpsys location", timeout=10.0)
        result: dict[str, str] = {}
        raw = str(out)
        # GPS status
        gps_status, _, _ = await self._exec(
            "shell", "settings get secure location_providers_allowed",
        )
        result["providers"] = str(gps_status) or "off"
        # Parse last known locations
        in_section = False
        for line in raw.splitlines():
            if "Last Known Locations" in line or "last location=" in line.lower():
                in_section = True
            if in_section and "Location[" in line:
                # Location[fused -24.003975,-48.346278 ...] or BR locale with comma
                loc_part = line.split("Location[")[1].split("]")[0]
                parts = loc_part.split()
                if len(parts) >= 2:
                    provider = parts[0]
                    coords = parts[1]
                    result[f"location_{provider}"] = coords
                    # Handle BR locale (comma as decimal): -24,003975,-48,346278
                    # vs standard: -24.003975,-48.346278
                    # Strategy: find the split between lat and lon
                    # A negative lon after lat means pattern: num,num,-num,num
                    m = re.match(
                        r"(-?\d+[.,]\d+),(-?\d+[.,]\d+)", coords,
                    )
                    if m:
                        lat_s = m.group(1).replace(",", ".")
                        lon_s = m.group(2).replace(",", ".")
                        result.setdefault("lat", lat_s)
                        result.setdefault("lon", lon_s)
        return result

    # ------------------------------------------------------------------
    # Network info
    # ------------------------------------------------------------------

    async def network_info(self) -> dict[str, str]:
        """Get WiFi/mobile network info."""
        info: dict[str, str] = {}
        # WiFi IP
        ip = await self.get_ip()
        if ip:
            info["wifi_ip"] = ip
        # WiFi SSID
        wifi_out, _, _ = await self._exec(
            "shell", "dumpsys wifi | grep 'mWifiInfo'",
        )
        wifi_str = str(wifi_out)
        if "SSID:" in wifi_str:
            ssid = wifi_str.split("SSID:")[1].split(",")[0].strip()
            info["ssid"] = ssid
        if "Link speed:" in wifi_str:
            speed = wifi_str.split("Link speed:")[1].split(",")[0].strip()
            info["link_speed"] = speed
        # Connection type
        conn_out, _, _ = await self._exec(
            "shell", "dumpsys connectivity | grep 'ActiveDefaultNetwork'",
        )
        conn_str = str(conn_out)
        if conn_str:
            info["connection"] = conn_str.strip()
        # Mobile data
        mobile_out, _, _ = await self._exec(
            "shell", "settings get global mobile_data",
        )
        info["mobile_data"] = "on" if str(mobile_out).strip() == "1" else "off"
        return info

    # ------------------------------------------------------------------
    # Clipboard
    # ------------------------------------------------------------------

    async def get_clipboard(self) -> str:
        """Read clipboard text (Android 10+, may need grant)."""
        # Try via service call (varies by Android version)
        out, _, _ = await self._exec(
            "shell", "am broadcast -a clipper.get 2>/dev/null || echo ''",
        )
        return str(out).strip()

    async def set_clipboard(self, text: str) -> str:
        """Set clipboard text via input command hack."""
        safe = text.replace("'", "'\\''")
        out, _, _ = await self._exec(
            "shell", f"am broadcast -a clipper.set -e text '{safe}' 2>/dev/null"
            " || echo 'clipboard set requires Clipper app'",
        )
        return str(out).strip()

    # ------------------------------------------------------------------
    # Calendar
    # ------------------------------------------------------------------

    async def calendar_events(self, limit: int = 20) -> list[dict[str, str]]:
        """Read calendar events."""
        raw = await self.content_query(
            "content://com.android.calendar/events",
            projection="title:dtstart:dtend:eventLocation:description",
        )
        events: list[dict[str, str]] = []
        for line in raw.splitlines():
            if "Row:" not in line:
                continue
            title = _extract_field(line, "title")
            location = _extract_field(line, "eventLocation")
            if title:
                events.append({"title": title, "location": location or ""})
            if len(events) >= limit:
                break
        return events
