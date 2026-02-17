"""BrowserTools — Agno toolkit for web browser automation.

Enton can browse the web: open URLs, screenshot pages, extract text,
search, download content. Uses subprocess chromium in headless mode.

Requires: chromium-browser or google-chrome (pre-installed).
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from agno.tools import Toolkit

logger = logging.getLogger(__name__)


def _find_browser() -> str | None:
    """Find available browser binary."""
    import shutil

    for name in ("chromium-browser", "chromium", "google-chrome", "google-chrome-stable"):
        path = shutil.which(name)
        if path:
            return path
    return None


class BrowserTools(Toolkit):
    """Tools for web browsing and automation."""

    def __init__(self, workspace: Path | None = None) -> None:
        super().__init__(name="browser_tools")
        self._browser = _find_browser()
        self._workspace = workspace or Path(tempfile.gettempdir())
        self._downloads = self._workspace / "downloads"
        self._downloads.mkdir(parents=True, exist_ok=True)

        self.register(self.browse_url)
        self.register(self.web_screenshot)
        self.register(self.web_search)
        self.register(self.extract_text)
        self.register(self.download_file)

    def browse_url(self, url: str) -> str:
        """Open a URL in the default browser (visible to user).

        Args:
            url: URL to open
        """
        try:
            subprocess.Popen(
                ["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return f"Abrindo no browser: {url}"
        except Exception as e:
            return f"Erro: {e}"

    def web_screenshot(self, url: str, width: int = 1920, height: int = 1080) -> str:
        """Screenshot a webpage using headless browser. Returns file path.

        Args:
            url: URL to screenshot
            width: Viewport width
            height: Viewport height
        """
        if not self._browser:
            return "Erro: nenhum browser encontrado (chromium/chrome)"

        path = tempfile.mktemp(suffix=".png", prefix="enton_web_")
        try:
            subprocess.run(
                [
                    self._browser,
                    "--headless=new",
                    "--disable-gpu",
                    "--no-sandbox",
                    f"--window-size={width},{height}",
                    f"--screenshot={path}",
                    url,
                ],
                timeout=30,
                capture_output=True,
            )
            if Path(path).exists():
                size = Path(path).stat().st_size
                return f"Screenshot da pagina salvo: {path} ({size} bytes)"
            return "Erro: screenshot nao foi gerado"
        except subprocess.TimeoutExpired:
            return "Timeout ao carregar pagina (30s)"
        except Exception as e:
            return f"Erro: {e}"

    def web_search(self, query: str) -> str:
        """Open a web search in the browser.

        Args:
            query: Search query
        """
        import urllib.parse

        encoded = urllib.parse.quote(query)
        url = f"https://www.google.com/search?q={encoded}"
        try:
            subprocess.Popen(
                ["xdg-open", url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return f"Pesquisando: {query}"
        except Exception as e:
            return f"Erro: {e}"

    def extract_text(self, url: str) -> str:
        """Extract text content from a webpage (headless, no JS rendering needed).

        Args:
            url: URL to extract text from
        """
        try:
            result = subprocess.run(
                [
                    self._browser or "chromium-browser",
                    "--headless=new",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--dump-dom",
                    url,
                ],
                timeout=30,
                capture_output=True,
                text=True,
            )
            html = result.stdout
            # Simple HTML to text (strip tags)
            import re
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 5000:
                text = text[:5000] + f"\n... ({len(text)} chars total)"
            return text if text else "Nenhum texto extraido."
        except subprocess.TimeoutExpired:
            return "Timeout (30s)"
        except Exception as e:
            return f"Erro: {e}"

    def download_file(self, url: str, filename: str = "") -> str:
        """Download a file from URL to workspace/downloads.

        Args:
            url: URL to download
            filename: Optional filename (auto-detect if empty)
        """
        if not filename:
            filename = url.rsplit("/", maxsplit=1)[-1].split("?")[0] or "download"

        dest = self._downloads / filename
        try:
            result = subprocess.run(
                ["aria2c", "-x", "16", "-s", "16", "-d", str(self._downloads), "-o", filename, url],
                timeout=120,
                capture_output=True,
                text=True,
            )
            if dest.exists():
                size = dest.stat().st_size
                return f"Download completo: {dest} ({size / 1024:.0f} KB)"
            # Fallback to wget
            result = subprocess.run(
                ["wget", "-q", "-O", str(dest), url],
                timeout=120,
                capture_output=True,
            )
            if dest.exists():
                return f"Download completo: {dest}"
            return f"Erro no download: {result.stderr.decode()[:200]}"
        except subprocess.TimeoutExpired:
            return "Timeout no download (120s)"
        except Exception as e:
            return f"Erro: {e}"
