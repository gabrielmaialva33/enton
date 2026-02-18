"""BrowserTools — Agno toolkit for web browser automation.

Enton can browse the web: open URLs, screenshot pages, extract text,
search, download content. Uses Crawl4AI/Playwright for high-fidelity automation.
"""

from __future__ import annotations

import base64
import logging
import subprocess
import tempfile
from pathlib import Path

from agno.tools import Toolkit

from enton.core.crawler_engine import Crawl4AIEngine

logger = logging.getLogger(__name__)


class BrowserTools(Toolkit):
    """Tools for web browsing and automation."""

    def __init__(self, workspace: Path | None = None) -> None:
        super().__init__(name="browser_tools")
        self._workspace = workspace or Path(tempfile.gettempdir())
        self._downloads = self._workspace / "downloads"
        self._downloads.mkdir(parents=True, exist_ok=True)
        self._crawler = Crawl4AIEngine()

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
                ["xdg-open", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return f"Abrindo no browser: {url}"
        except Exception as e:
            return f"Erro: {e}"

    async def web_screenshot(self, url: str) -> str:
        """Screenshot a webpage using headless browser. Returns file path.

        Args:
            url: URL to screenshot
        """
        path = tempfile.mktemp(suffix=".png", prefix="enton_web_")
        try:
            result = await self._crawler.crawl(url, screenshot=True)
            if result.get("error"):
                return f"Erro ao tirar screenshot: {result['error']}"

            b64_data = result.get("screenshot")
            if not b64_data:
                return "Erro: screenshot nao retornado pelo crawler"

            with open(path, "wb") as f:
                f.write(base64.b64decode(b64_data))

            size = Path(path).stat().st_size
            return f"Screenshot da pagina salvo: {path} ({size} bytes)"
        except Exception as e:
            return f"Erro: {e}"

    def web_search(self, query: str) -> str:
        """Open a web search in the browser (user visible).

        Args:
            query: Search query
        """
        import urllib.parse

        encoded = urllib.parse.quote(query)
        url = f"https://www.google.com/search?q={encoded}"
        try:
            subprocess.Popen(
                ["xdg-open", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return f"Pesquisando: {query}"
        except Exception as e:
            return f"Erro: {e}"

    async def extract_text(self, url: str) -> str:
        """Extract markdown content from a webpage.

        Args:
            url: URL to extract text from
        """
        try:
            result = await self._crawler.crawl(url)
            if result.get("error"):
                return f"Erro ao extrair texto: {result['error']}"

            markdown = result.get("markdown", "")
            if not markdown:
                return "Nenhum texto extraido (conteudo vazio)."

            return markdown
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
            # Try curl first, then wget
            try:
                subprocess.run(
                    ["curl", "-L", "-o", str(dest), url],
                    timeout=120,
                    check=True,
                    capture_output=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                subprocess.run(
                    ["wget", "-q", "-O", str(dest), url],
                    timeout=120,
                    check=True,
                    capture_output=True,
                )

            if dest.exists():
                size = dest.stat().st_size
                return f"Download completo: {dest} ({size / 1024:.0f} KB)"
            return "Erro no download: arquivo nao criado"
        except subprocess.TimeoutExpired:
            return "Timeout no download (120s)"
        except Exception as e:
            return f"Erro: {e}"
