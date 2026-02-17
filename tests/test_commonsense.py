"""Tests for CommonsenseKB."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from enton.core.commonsense import COMMONSENSE_COLLECTION, CommonsenseKB


def test_init():
    kb = CommonsenseKB(qdrant_url="http://fake:6333")
    assert kb._available is None


def test_not_available_by_default():
    kb = CommonsenseKB()
    # Without Qdrant running, should be unavailable
    with patch("qdrant_client.QdrantClient", side_effect=Exception):
        assert kb.available is False


def test_available_when_collection_exists():
    kb = CommonsenseKB()
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.name = COMMONSENSE_COLLECTION
    mock_client.get_collections.return_value.collections = [mock_collection]

    with patch("qdrant_client.QdrantClient", return_value=mock_client):
        assert kb.available is True


def test_not_available_when_collection_missing():
    kb = CommonsenseKB()
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []

    with patch("qdrant_client.QdrantClient", return_value=mock_client):
        assert kb.available is False


@pytest.mark.asyncio()
async def test_search_unavailable():
    kb = CommonsenseKB()
    kb._available = False
    results = await kb.search("cats can fly")
    assert results == []


@pytest.mark.asyncio()
async def test_search_with_results():
    kb = CommonsenseKB()
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.payload = {
        "subject": "cat", "predicate": "has", "obj": "four legs",
    }
    mock_result.score = 0.9
    mock_client.search.return_value = [mock_result]
    kb._qdrant = mock_client
    kb._available = True

    mock_embedder = MagicMock()
    mock_embedder.get_embedding.return_value = [0.1] * 768
    kb._embedder = mock_embedder

    results = await kb.search("cat anatomy")
    assert len(results) == 1
    assert results[0]["subject"] == "cat"


@pytest.mark.asyncio()
async def test_what_is():
    kb = CommonsenseKB()
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.payload = {
        "subject": "dog", "predicate": "is", "obj": "a mammal",
    }
    mock_result.score = 0.8
    mock_client.search.return_value = [mock_result]
    kb._qdrant = mock_client
    kb._available = True

    mock_embedder = MagicMock()
    mock_embedder.get_embedding.return_value = [0.1] * 768
    kb._embedder = mock_embedder

    facts = await kb.what_is("dog")
    assert len(facts) == 1
    assert "dog is a mammal" in facts[0]
