"""Robot Body Toolkit — physical control of Enton via Yoosee SC-B21 camera.

Exposes the Enton body hardware (IP camera with persistent backdoor) as tools
for the brain. Works through 3 channels:

1. **ONVIF SOAP** (port 5000) — PTZ pan/tilt motor, preset positions
2. **Telnet root** (ports 23 + 2323) — direct shell no password, GPIOs, filesystem
3. **RTSP** (port 554) — dual-lens video stream (used by `perception/vision.py`)

Hardware docs: see `docs/hardware/robot_body.md`.

Brain usage examples:
- "look right" → pan_right()
- "go home" → goto_home()
- "turn IR LED on" → gpio_set(19, 1)
- "listen to room" → capture_audio(duration=5)
"""

from __future__ import annotations

import asyncio
import shlex
import time
from dataclasses import dataclass

import httpx
from agno.tools import Toolkit

from enton.core.config import settings

_SOAP_ENV = """<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
  <s:Body>{body}</s:Body>
</s:Envelope>"""

_PTZ_NS = 'xmlns="http://www.onvif.org/ver20/ptz/wsdl"'
_TT_NS = 'xmlns:tt="http://www.onvif.org/ver10/schema"'


@dataclass
class RobotBodyConfig:
    """Physical body config. Loads from settings (.env) with defaults."""

    ip: str = "192.168.1.5"
    onvif_port: int = 5000
    telnet_port: int = 23
    rtsp_port: int = 554
    # Real profile token discovered via GetProfiles on media_service.
    # PROFILE_000 was fake (returned 200 OK but never moved the motor).
    profile: str = "IPCProfilesToken0"
    rtsp_path: str = "/cam/realmonitor"  # no-auth video-only
    # Auth stream with audio (PCM A-law 16kHz mono). Password is set by user in Yoosee app.
    rtsp_auth_path: str = "/onvif2"  # /onvif1 main (1920x2160) /onvif2 sub (640x720)
    rtsp_user: str = "admin"
    rtsp_password: str = ""
    ptz_min_interval: float = 0.6  # throttle to avoid IPC DoS
    # enton_bridge TCP daemon (see scripts/enton_bridge/). Single entry point
    # that dispatches motor/gpio/exec over loopback; avoids HTTP+telnet overhead
    # and bundles PTZ burst logic on the camera side.
    bridge_port: int = 9999
    bridge_timeout: float = 8.0  # seconds; PAN bursts of 20 iters take ~10s


class BridgeUnavailableError(RuntimeError):
    """Raised when enton_bridge TCP is unreachable or returned a malformed reply."""


class RobotBodyTools(Toolkit):
    """Physical control of the Enton body (Yoosee SC-B21 camera)."""

    def __init__(self, cfg: RobotBodyConfig | None = None):
        super().__init__(name="robot_body")
        self.cfg = cfg or RobotBodyConfig(
            ip=getattr(settings, "robot_body_ip", "192.168.1.5"),
            rtsp_user=getattr(settings, "robot_body_rtsp_user", "admin"),
            rtsp_password=getattr(settings, "robot_body_rtsp_password", ""),
            rtsp_auth_path=getattr(settings, "robot_body_rtsp_auth_path", "/onvif2"),
            bridge_port=getattr(settings, "robot_body_bridge_port", 9999),
        )
        self._last_ptz = 0.0
        self._client: httpx.AsyncClient | None = None
        self._bridge_available: bool | None = None  # cached probe result

        self.register(self.pan_right)
        self.register(self.pan_left)
        self.register(self.tilt_up)
        self.register(self.tilt_down)
        self.register(self.stop_motor)
        self.register(self.goto_home)
        self.register(self.shell_exec)
        self.register(self.gpio_set)
        self.register(self.gpio_read)
        self.register(self.get_pos)
        self.register(self.capture_audio)
        self.register(self.get_body_status)

    # ─── internal ───────────────────────────────────────────────────────

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=5.0)
        return self._client

    async def _throttle(self) -> None:
        """Prevent PTZ flood (>500ms between requests = guaranteed IPC crash)."""
        elapsed = time.monotonic() - self._last_ptz
        if elapsed < self.cfg.ptz_min_interval:
            await asyncio.sleep(self.cfg.ptz_min_interval - elapsed)
        self._last_ptz = time.monotonic()

    # ─── enton_bridge TCP client ────────────────────────────────────────

    async def _bridge_send(self, command: str, timeout: float | None = None) -> str:
        """Send a single line command to the enton_bridge TCP daemon and return the reply.

        Raises BridgeUnavailableError if the daemon is unreachable or the handshake fails.
        The bridge replies with one line: "OK ..." on success, "ERR ..." on failure.
        """
        t = timeout if timeout is not None else self.cfg.bridge_timeout
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.cfg.ip, self.cfg.bridge_port),
                timeout=2.0,
            )
        except (OSError, TimeoutError) as e:
            self._bridge_available = False
            raise BridgeUnavailableError(f"connect {self.cfg.ip}:{self.cfg.bridge_port}: {e}") from e

        try:
            # Drain banner: "OK enton_bridge v0.1 ready\n"
            try:
                banner = await asyncio.wait_for(reader.readline(), timeout=2.0)
            except TimeoutError as e:
                raise BridgeUnavailableError("banner timeout") from e
            if not banner.startswith(b"OK "):
                raise BridgeUnavailableError(f"bad banner: {banner!r}")

            writer.write(command.encode("utf-8") + b"\n")
            await writer.drain()
            try:
                reply = await asyncio.wait_for(reader.readline(), timeout=t)
            except TimeoutError as e:
                raise BridgeUnavailableError(f"reply timeout for {command!r}") from e
            self._bridge_available = True
            return reply.decode("utf-8", errors="replace").rstrip("\r\n")
        finally:
            try:
                writer.write(b"BYE\n")
                await writer.drain()
            except Exception:
                pass
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _bridge_try(self, command: str, timeout: float | None = None) -> str | None:
        """Convenience wrapper: None when bridge is unavailable, "OK ..."/"ERR ..." otherwise."""
        try:
            return await self._bridge_send(command, timeout=timeout)
        except BridgeUnavailableError:
            return None

    async def _soap(self, action: str, body: str, service: str = "ptz") -> str:
        await self._throttle()
        url = f"http://{self.cfg.ip}:{self.cfg.onvif_port}/onvif/{service}_service"
        action_url = f"http://www.onvif.org/ver20/{service}/wsdl/{action}"
        headers = {"Content-Type": f'application/soap+xml;charset=UTF-8;action="{action_url}"'}
        client = await self._http()
        r = await client.post(url, content=_SOAP_ENV.format(body=body), headers=headers)
        r.raise_for_status()
        return r.text

    async def _continuous_move(self, x: float, y: float, duration: float) -> None:
        body = f"""<ContinuousMove {_PTZ_NS}>
          <ProfileToken>{self.cfg.profile}</ProfileToken>
          <Velocity><PanTilt x="{x}" y="{y}" {_TT_NS}/></Velocity>
        </ContinuousMove>"""
        await self._soap("ContinuousMove", body)
        # No Stop: Gwell firmware freezes the IPC on Stop (watchdog reboot).
        # Each ContinuousMove already moves ~2-3 steps and halts on its own;
        # for longer motion just re-send ContinuousMove every ~600ms.
        if duration > 0:
            await asyncio.sleep(duration)
            await self._continuous_move_halt()

    async def _continuous_move_halt(self) -> None:
        """Halt the motor without calling ONVIF Stop (which freezes IPC).
        Sends ContinuousMove with zero velocity — firmware accepts and motor stops.
        """
        body = f"""<ContinuousMove {_PTZ_NS}>
          <ProfileToken>{self.cfg.profile}</ProfileToken>
          <Velocity><PanTilt x="0" y="0" {_TT_NS}/></Velocity>
        </ContinuousMove>"""
        await self._soap("ContinuousMove", body)

    # ─── PTZ tools ──────────────────────────────────────────────────────

    async def _pan_via_bridge(self, direction: str, iters: int, speed: float) -> str | None:
        """Burst PAN through enton_bridge. Returns reply line or None if unavailable.
        The bridge internally re-sends ContinuousMove every 500ms (firmware needs
        sustained requests to actually move the motor).
        """
        return await self._bridge_try(f"PAN {direction} {iters} {speed:.2f}")

    async def pan_right(self, speed: float = 0.5, duration: float = 1.0) -> str:
        """Pan the body to the right (positive pan).

        Args:
            speed: Velocity in range (0.0, 1.0]. Default 0.5.
            duration: Seconds before auto-halt. Default 1.0s.
        """
        iters = max(1, int(duration / 0.5))
        bridged = await self._pan_via_bridge("right", iters, speed)
        if bridged is not None:
            return bridged
        await self._continuous_move(x=abs(speed), y=0.0, duration=duration)
        return f"Panned right ({speed:.1f}x for {duration:.1f}s, onvif fallback)"

    async def pan_left(self, speed: float = 0.5, duration: float = 1.0) -> str:
        """Pan the body to the left (negative pan).

        Args:
            speed: Velocity (0.0, 1.0]. Default 0.5.
            duration: Seconds before halting. Default 1.0s.
        """
        iters = max(1, int(duration / 0.5))
        bridged = await self._pan_via_bridge("left", iters, speed)
        if bridged is not None:
            return bridged
        await self._continuous_move(x=-abs(speed), y=0.0, duration=duration)
        return f"Panned left ({speed:.1f}x for {duration:.1f}s, onvif fallback)"

    async def tilt_up(self, speed: float = 0.5, duration: float = 1.0) -> str:
        """Tilt up (positive tilt). speed (0,1], duration in seconds."""
        iters = max(1, int(duration / 0.5))
        bridged = await self._pan_via_bridge("up", iters, speed)
        if bridged is not None:
            return bridged
        await self._continuous_move(x=0.0, y=abs(speed), duration=duration)
        return f"Tilted up ({speed:.1f}x for {duration:.1f}s, onvif fallback)"

    async def tilt_down(self, speed: float = 0.5, duration: float = 1.0) -> str:
        """Tilt down (negative tilt). speed (0,1], duration in seconds."""
        iters = max(1, int(duration / 0.5))
        bridged = await self._pan_via_bridge("down", iters, speed)
        if bridged is not None:
            return bridged
        await self._continuous_move(x=0.0, y=-abs(speed), duration=duration)
        return f"Tilted down ({speed:.1f}x for {duration:.1f}s, onvif fallback)"

    async def stop_motor(self) -> str:
        """Halt the PTZ motor. Uses ContinuousMove x=0 y=0 instead of ONVIF Stop —
        the Stop handler on Gwell firmware freezes the IPC and triggers watchdog reboot.
        """
        bridged = await self._bridge_try("HALT")
        if bridged is not None:
            return bridged
        await self._continuous_move_halt()
        return "Motor halted (onvif fallback)"

    async def get_pos(self) -> str:
        """Read current motor position (H, V) via enton_bridge dmesg parse."""
        bridged = await self._bridge_try("POS", timeout=2.5)
        if bridged is not None:
            return bridged
        return "ERR bridge unavailable"

    async def goto_home(self) -> str:
        """Return to home position. Home is set via SetHomePosition."""
        body = f"""<GotoHomePosition {_PTZ_NS}>
          <ProfileToken>{self.cfg.profile}</ProfileToken>
        </GotoHomePosition>"""
        await self._soap("GotoHomePosition", body)
        return "Returned to home position"

    # ─── Shell / GPIO via telnet ───────────────────────────────────────

    async def shell_exec(self, command: str, timeout: float = 5.0) -> str:
        """Run a shell command on the camera via the telnet backdoor (root, no password).

        Args:
            command: Shell command to run on the camera (busybox sh). Ex: "ps", "ls /ipc".
            timeout: Timeout in seconds. Default 5.0.

        Returns:
            Command stdout, or error message.

        Warning:
            Shell is root. Be careful with destructive commands (rm, reboot, etc).
        """
        quoted = shlex.quote(command)
        script = (
            f"( sleep 0.5; echo {quoted}; sleep 0.8; echo exit ) | "
            f"timeout {int(timeout)} telnet {self.cfg.ip} {self.cfg.telnet_port}"
        )
        proc = await asyncio.create_subprocess_shell(
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout + 2)
        except TimeoutError:
            proc.kill()
            return "ERROR: telnet connect timeout"

        text = stdout.decode(errors="replace")
        lines = text.splitlines()
        # Strip banner/prompt
        filtered = [
            ln
            for ln in lines
            if not any(
                s in ln
                for s in ("Trying ", "Connected to", "Escape character", "Connection closed", "~ #")
            )
            and not ln.strip().startswith(command[:20])
        ]
        return "\n".join(filtered).strip() or "(sem output)"

    async def gpio_set(self, gpio: int, value: int) -> str:
        """Drive a camera GPIO (LEDs, motor direction pins, IR, etc).

        Args:
            gpio: GPIO number (available: 19, 37, 48, 67, 68, 98).
            value: 0 (low) or 1 (high).
        """
        if value not in (0, 1):
            return f"ERROR: value must be 0 or 1, got {value}"
        bridged = await self._bridge_try(f"GPIO_SET {gpio} {value}")
        if bridged is not None:
            return bridged
        await self.shell_exec(f"echo {value} > /sys/class/gpio/gpio{gpio}/value")
        return f"GPIO{gpio} → {value} (telnet fallback)"

    async def gpio_read(self, gpio: int) -> str:
        """Read the current value of a GPIO.

        Args:
            gpio: GPIO number (e.g. 56 = reset button).
        """
        bridged = await self._bridge_try(f"GPIO_GET {gpio}")
        if bridged is not None:
            return bridged
        result = await self.shell_exec(f"cat /sys/class/gpio/gpio{gpio}/value")
        return f"GPIO{gpio} = {result.strip()} (telnet fallback)"

    # ─── Audio ──────────────────────────────────────────────────────────

    async def capture_audio(self, duration: float = 3.0, out_path: str = "") -> str:
        """Capture mic audio from the camera via RTSP (PCM A-law 16kHz mono).

        Args:
            duration: Seconds to record. Default 3.0.
            out_path: Output path (.wav). Empty = temp file.

        Returns:
            Path to the recorded WAV file, or error message.

        Notes:
            Uses ffmpeg on the host to pull the auth RTSP stream (`/onvif2` with
            Yoosee-app credentials). Direct /dev/pcmC0D0c doesn't work — the IPC
            opens it exclusively.
        """
        import tempfile
        from pathlib import Path

        if not self.cfg.rtsp_password:
            return "ERROR: rtsp_password not set (export ROBOT_BODY_RTSP_PASSWORD in .env)"

        if not out_path:
            out_path = tempfile.mktemp(suffix=".wav", prefix="enton_mic_")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        url = (
            f"rtsp://{self.cfg.rtsp_user}:{self.cfg.rtsp_password}"
            f"@{self.cfg.ip}:{self.cfg.rtsp_port}{self.cfg.rtsp_auth_path}"
        )
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-rtsp_transport", "udp",
            "-i", url,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-t", str(duration),
            out_path,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=duration + 10)
        except TimeoutError:
            proc.kill()
            return "ERROR: audio capture timeout"

        if proc.returncode != 0:
            return f"ERROR ffmpeg: {stderr.decode(errors='replace')[:200]}"

        size = Path(out_path).stat().st_size if Path(out_path).exists() else 0
        return f"Recorded {duration:.1f}s at {out_path} ({size} bytes)"

    # ─── Status ─────────────────────────────────────────────────────────

    async def get_body_status(self) -> str:
        """Return current body status: IP, uptime, load, key processes, temperature."""
        out = await self.shell_exec(
            "uptime; cat /proc/loadavg; "
            "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null; "
            'echo "---"; '
            'ls /proc | grep "^[0-9]" | while read p; do '
            "  c=$(cat /proc/$p/comm 2>/dev/null); "
            '  case "$c" in ipc|main|telnetd|wpa_supplicant) echo "$p:$c";; '
            "esac; done"
        )
        return f"Body @ {self.cfg.ip}:\n{out}"

    # ─── RTSP info (video handling lives in perception/) ──────────────

    @property
    def rtsp_url(self) -> str:
        """RTSP URL for the primary stream (no auth, video only)."""
        return f"rtsp://{self.cfg.ip}:{self.cfg.rtsp_port}{self.cfg.rtsp_path}"

    async def aclose(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
