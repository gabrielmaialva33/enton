"""Crawl4AI Engine — High-performance AI crawler.

Wraps crawl4ai to provide:
- Headless browser crawling (Playwright)
- Markdown extraction
- Anti-bot evasion
- Structured data extraction
"""

from __future__ import annotations

import logging
from typing import Any

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

logger = logging.getLogger(__name__)


class Crawl4AIEngine:
    """High-performance crawler engine using Crawl4AI."""

    def __init__(self, headless: bool = True) -> None:
        self.headless = headless
        self._browser_config = BrowserConfig(
            headless=self.headless,
            verbose=False,
            # user_agent_mode="random", # crawl4ai specific
        )
        self._run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=10,
        )

    async def crawl(self, url: str, screenshot: bool = False) -> dict[str, Any]:
        """Crawl a single URL and return processed result."""
        try:
            run_config = self._run_config.clone()
            if screenshot:
                run_config.screenshot = True

            async with AsyncWebCrawler(config=self._browser_config) as crawler:
                result = await crawler.arun(url=url, config=run_config)

                if not result.success:
                    logger.warning(f"Crawl failed for {url}: {result.error_message}")
                    return {"error": result.error_message, "url": url}

                return {
                    "url": result.url,
                    "title": result.metadata.get("title", ""),
                    "markdown": result.markdown,
                    "html": result.html,
                    "links": list(result.links.keys()) if hasattr(result.links, "keys") else [],
                    "media": result.media,
                    "metadata": result.metadata,
                    "screenshot": result.screenshot,  # Base64 string
                }
        except Exception as e:
            logger.exception(f"Crawl exception for {url}")
            return {"error": str(e), "url": url}

    async def crawl_many(self, urls: list[str]) -> list[dict[str, Any]]:
        """Crawl multiple URLs in parallel."""
        try:
            async with AsyncWebCrawler(config=self._browser_config) as crawler:
                results = await crawler.arun_many(urls, config=self._run_config)

                processed = []
                for res in results:
                    if not res.success:
                        processed.append({"error": res.error_message, "url": res.url})
                        continue

                    processed.append(
                        {
                            "url": res.url,
                            "title": res.metadata.get("title", ""),
                            "markdown": res.markdown,
                            "links": list(res.links.keys()) if hasattr(res.links, "keys") else [],
                            "metadata": res.metadata,
                        }
                    )
                return processed
        except Exception as e:
            logger.exception("Bulk crawl exception")
            return [{"error": str(e)} for _ in urls]
