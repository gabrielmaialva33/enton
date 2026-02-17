"""Tests for KnowledgeCrawler."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from enton.core.knowledge_crawler import KnowledgeCrawler, KnowledgeTriple


def test_knowledge_triple_creation():
    t = KnowledgeTriple(
        subject="Python", predicate="is", obj="a programming language",
        source_url="https://python.org",
    )
    assert t.subject == "Python"
    assert t.confidence == 1.0


def test_crawler_init():
    kc = KnowledgeCrawler(qdrant_url="http://fake:6333")
    assert kc._brain is None
    assert kc._triple_count == 0


@pytest.mark.asyncio()
async def test_crawl_url():
    kc = KnowledgeCrawler()
    html = "<html><body><p>Python is great.</p></body></html>"

    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        text = await kc.crawl_url("https://example.com")

    assert "Python is great" in text


@pytest.mark.asyncio()
async def test_crawl_url_failure():
    kc = KnowledgeCrawler()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=Exception("Network error"))

    with patch("httpx.AsyncClient", return_value=mock_client):
        text = await kc.crawl_url("https://bad-url.com")

    assert text == ""


@pytest.mark.asyncio()
async def test_extract_triples():
    brain = MagicMock()
    brain.think = AsyncMock(return_value='[{"subject":"Python","predicate":"is","obj":"fast"}]')
    kc = KnowledgeCrawler(brain=brain)

    triples = await kc.extract_triples("Python is fast.", source_url="https://x.com")
    assert len(triples) == 1
    assert triples[0].subject == "Python"
    assert triples[0].source_url == "https://x.com"


@pytest.mark.asyncio()
async def test_extract_triples_bad_json():
    brain = MagicMock()
    brain.think = AsyncMock(return_value="I don't know, sorry!")
    kc = KnowledgeCrawler(brain=brain)

    triples = await kc.extract_triples("Some text")
    assert triples == []


@pytest.mark.asyncio()
async def test_extract_triples_markdown_fences():
    brain = MagicMock()
    brain.think = AsyncMock(
        return_value='```json\n[{"subject":"A","predicate":"B","obj":"C"}]\n```',
    )
    kc = KnowledgeCrawler(brain=brain)

    triples = await kc.extract_triples("text")
    assert len(triples) == 1
    assert triples[0].subject == "A"


@pytest.mark.asyncio()
async def test_extract_triples_no_brain():
    kc = KnowledgeCrawler(brain=None)
    triples = await kc.extract_triples("some text")
    assert triples == []


@pytest.mark.asyncio()
async def test_learn_from_url():
    kc = KnowledgeCrawler()
    kc._brain = MagicMock()
    kc._brain.think = AsyncMock(
        return_value='[{"subject":"X","predicate":"is","obj":"Y"}]',
    )

    with (
        patch.object(kc, "crawl_url", return_value="X is Y"),
        patch.object(kc, "_store_triples", new_callable=AsyncMock),
    ):
        triples = await kc.learn_from_url("https://example.com")

    assert len(triples) == 1


@pytest.mark.asyncio()
async def test_learn_from_url_empty():
    kc = KnowledgeCrawler()
    with patch.object(kc, "crawl_url", return_value=""):
        triples = await kc.learn_from_url("https://bad.com")
    assert triples == []


@pytest.mark.asyncio()
async def test_search_no_qdrant():
    kc = KnowledgeCrawler()
    results = await kc.search("Python")
    assert results == []


@pytest.mark.asyncio()
async def test_search_with_results():
    kc = KnowledgeCrawler()
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.payload = {
        "subject": "Python", "predicate": "is",
        "obj": "fast", "source_url": "https://x.com",
    }
    mock_result.score = 0.85
    mock_client.search.return_value = [mock_result]
    kc._qdrant = mock_client

    mock_embedder = MagicMock()
    mock_embedder.get_embedding.return_value = [0.1] * 768
    kc._embedder = mock_embedder

    results = await kc.search("Python speed")
    assert len(results) == 1
    assert results[0]["subject"] == "Python"
