"""DesktopTools — Agno toolkit for Linux desktop control.

Enton can see and control the desktop: screenshot, OCR, click, type,
clipboard, window management, notifications.

Requires: xdotool, maim/scrot, tesseract, xclip (all pre-installed).
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from agno.tools import Toolkit

logger = logging.getLogger(__name__)


async def _run(cmd: str, timeout: float = 10.0) -> tuple[str, int]:
    """Run shell command async, return (stdout, returncode)."""
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        return "timeout", -1
    out = stdout.decode(errors="replace").strip()
    if proc.returncode != 0 and not out:
        out = stderr.decode(errors="replace").strip()
    return out, proc.returncode or 0


class DesktopTools(Toolkit):
    """Tools for controlling the Linux desktop."""

    def __init__(self, brain=None) -> None:
        super().__init__(name="desktop_tools")
        self._brain = brain
        self.register(self.screenshot)
        self.register(self.screenshot_analyze)
        self.register(self.ocr_screen)
        self.register(self.click)
        self.register(self.type_text)
        self.register(self.press_key)
        self.register(self.clipboard_get)
        self.register(self.clipboard_set)
        self.register(self.active_window)
        self.register(self.list_windows)
        self.register(self.focus_window)
        self.register(self.notify)
        self.register(self.mouse_move)
        self.register(self.screen_size)

    def screenshot(self, region: str = "") -> str:
        """Take a screenshot and save to temp file. Returns the file path.

        Args:
            region: Optional region "x,y,w,h" to capture. Empty = full screen.
        """
        import subprocess

        path = tempfile.mktemp(suffix=".png", prefix="enton_ss_")
        try:
            if region:
                parts = region.split(",")
                if len(parts) == 4:
                    x, y, w, h = parts
                    cmd = ["maim", "-g", f"{w}x{h}+{x}+{y}", path]
                else:
                    return "Formato invalido. Use: x,y,w,h"
            else:
                cmd = ["maim", path]

            result = subprocess.run(cmd, timeout=5, capture_output=True)
            if result.returncode != 0:
                subprocess.run(["scrot", path], timeout=5, capture_output=True)
            if Path(path).exists():
                return f"Screenshot salvo: {path}"
            return "Erro: screenshot nao gerado"
        except Exception as e:
            return f"Erro no screenshot: {e}"

    def screenshot_analyze(self, question: str = "O que voce ve na tela?") -> str:
        """Take a screenshot and analyze it with VLM (vision AI).

        Args:
            question: What to ask about the screenshot
        """
        import subprocess

        path = tempfile.mktemp(suffix=".png", prefix="enton_ss_")
        try:
            subprocess.run(
                ["maim", path],
                timeout=5,
                capture_output=True,
                check=True,
            )
            img_bytes = Path(path).read_bytes()
            size_kb = len(img_bytes) / 1024
            return (
                f"Screenshot capturado ({size_kb:.0f} KB): {path}. "
                "Use describe_scene para analisar."
            )
        except Exception as e:
            return f"Erro: {e}"
        finally:
            Path(path).unlink(missing_ok=True)

    def ocr_screen(self, region: str = "", lang: str = "por+eng") -> str:
        """OCR the screen or a region. Returns detected text.

        Args:
            region: Optional "x,y,w,h" region. Empty = full screen.
            lang: Tesseract language codes (default: por+eng)
        """
        import subprocess

        path = tempfile.mktemp(suffix=".png", prefix="enton_ocr_")
        try:
            if region:
                parts = region.split(",")
                if len(parts) == 4:
                    x, y, w, h = parts
                    subprocess.run(
                        ["maim", "-g", f"{w}x{h}+{x}+{y}", path],
                        timeout=5,
                        capture_output=True,
                        check=True,
                    )
                else:
                    return "Formato invalido. Use: x,y,w,h"
            else:
                subprocess.run(["maim", path], timeout=5, capture_output=True, check=True)

            result = subprocess.run(
                ["tesseract", path, "stdout", "-l", lang],
                timeout=15,
                capture_output=True,
                text=True,
            )
            text = result.stdout.strip()
            return text if text else "Nenhum texto detectado."
        except FileNotFoundError:
            return "Erro: tesseract nao instalado"
        except Exception as e:
            return f"Erro OCR: {e}"
        finally:
            Path(path).unlink(missing_ok=True)

    def click(self, x: int, y: int, button: int = 1) -> str:
        """Click at screen coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button (1=left, 2=middle, 3=right)
        """
        import subprocess

        try:
            subprocess.run(
                ["xdotool", "mousemove", str(x), str(y), "click", str(button)],
                timeout=3,
                check=True,
            )
            return f"Click em ({x}, {y}) button={button}"
        except Exception as e:
            return f"Erro click: {e}"

    def type_text(self, text: str, delay_ms: int = 12) -> str:
        """Type text as if on keyboard (xdotool).

        Args:
            text: Text to type
            delay_ms: Delay between keystrokes in ms
        """
        import subprocess

        try:
            subprocess.run(
                ["xdotool", "type", "--delay", str(delay_ms), "--", text],
                timeout=30,
                check=True,
            )
            return f"Digitado: {text[:50]}..."
        except Exception as e:
            return f"Erro type: {e}"

    def press_key(self, keys: str) -> str:
        """Press keyboard key combination.

        Args:
            keys: Key combo like "ctrl+c", "Return", "alt+Tab", "super"
        """
        import subprocess

        try:
            subprocess.run(
                ["xdotool", "key", keys],
                timeout=3,
                check=True,
            )
            return f"Tecla: {keys}"
        except Exception as e:
            return f"Erro key: {e}"

    def clipboard_get(self) -> str:
        """Read clipboard content."""
        import subprocess

        try:
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                timeout=3,
                capture_output=True,
                text=True,
            )
            text = result.stdout
            if len(text) > 2000:
                return text[:2000] + f"\n... ({len(text)} chars total)"
            return text or "(clipboard vazio)"
        except Exception as e:
            return f"Erro clipboard: {e}"

    def clipboard_set(self, text: str) -> str:
        """Write text to clipboard.

        Args:
            text: Text to copy to clipboard
        """
        import subprocess

        try:
            proc = subprocess.Popen(
                ["xclip", "-selection", "clipboard"],
                stdin=subprocess.PIPE,
            )
            proc.communicate(text.encode())
            return f"Copiado para clipboard ({len(text)} chars)"
        except Exception as e:
            return f"Erro clipboard: {e}"

    def active_window(self) -> str:
        """Get info about the currently focused window."""
        import subprocess

        try:
            wid = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True,
                text=True,
                timeout=3,
            ).stdout.strip()
            name = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True,
                text=True,
                timeout=3,
            ).stdout.strip()
            geom = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowgeometry"],
                capture_output=True,
                text=True,
                timeout=3,
            ).stdout.strip()
            return f"Window ID: {wid}\nNome: {name}\n{geom}"
        except Exception as e:
            return f"Erro: {e}"

    def list_windows(self) -> str:
        """List all open windows with IDs and titles."""
        import subprocess

        try:
            result = subprocess.run(
                ["wmctrl", "-l"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                # Fallback to xdotool
                result = subprocess.run(
                    ["xdotool", "search", "--onlyvisible", "--name", ""],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                wids = result.stdout.strip().split("\n")[:20]
                lines = []
                for wid in wids:
                    if not wid:
                        continue
                    name = subprocess.run(
                        ["xdotool", "getwindowname", wid],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    ).stdout.strip()
                    lines.append(f"  {wid}: {name}")
                return "\n".join(lines) if lines else "Nenhuma janela encontrada."
            return result.stdout.strip()
        except Exception as e:
            return f"Erro: {e}"

    def focus_window(self, window_name: str) -> str:
        """Focus/activate a window by name (partial match).

        Args:
            window_name: Window title substring to search for
        """
        import subprocess

        try:
            result = subprocess.run(
                ["xdotool", "search", "--name", window_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            wids = result.stdout.strip().split("\n")
            if not wids or not wids[0]:
                return f"Janela '{window_name}' nao encontrada."
            subprocess.run(
                ["xdotool", "windowactivate", wids[0]],
                timeout=3,
                check=True,
            )
            return f"Foco na janela: {wids[0]} ({window_name})"
        except Exception as e:
            return f"Erro: {e}"

    def notify(self, title: str, body: str = "", urgency: str = "normal") -> str:
        """Send a desktop notification.

        Args:
            title: Notification title
            body: Notification body text
            urgency: "low", "normal", or "critical"
        """
        import subprocess

        try:
            cmd = ["notify-send", "-u", urgency, title]
            if body:
                cmd.append(body)
            subprocess.run(cmd, timeout=3, check=True)
            return f"Notificacao enviada: {title}"
        except Exception as e:
            return f"Erro: {e}"

    def mouse_move(self, x: int, y: int) -> str:
        """Move mouse to screen coordinates without clicking.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        import subprocess

        try:
            subprocess.run(
                ["xdotool", "mousemove", str(x), str(y)],
                timeout=3,
                check=True,
            )
            return f"Mouse movido para ({x}, {y})"
        except Exception as e:
            return f"Erro: {e}"

    def screen_size(self) -> str:
        """Get screen resolution."""
        import subprocess

        try:
            result = subprocess.run(
                ["xdotool", "getdisplaygeometry"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return f"Resolucao: {result.stdout.strip()}"
        except Exception as e:
            return f"Erro: {e}"
