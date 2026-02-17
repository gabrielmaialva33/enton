#!/usr/bin/env python3
"""ETL script: download ASCENT++ TSV, embed assertions, upsert to Qdrant.

Usage:
    python scripts/load_commonsense.py [--limit 100000] [--batch-size 500]

Downloads ASCENT++ commonsense assertions, embeds using nomic-embed-text
via Ollama, and stores in Qdrant collection 'enton_commonsense'.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

ASCENT_URL = (
    "https://raw.githubusercontent.com/phze22/ASCENT-pp/"
    "main/data/ascent_pp.csv"
)
COLLECTION = "enton_commonsense"
EMBED_DIM = 768
DATA_DIR = Path.home() / ".enton" / "data"


def download_ascent(path: Path) -> Path:
    """Download ASCENT++ CSV if not cached."""
    if path.exists():
        logger.info("Using cached %s", path)
        return path

    import httpx

    logger.info("Downloading ASCENT++ from %s ...", ASCENT_URL)
    path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.stream("GET", ASCENT_URL, follow_redirects=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=65536):
                f.write(chunk)

    logger.info("Downloaded to %s", path)
    return path


def parse_csv(path: Path, limit: int) -> list[dict]:
    """Parse ASCENT++ CSV into list of {subject, predicate, obj, text}."""
    import csv

    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(records) >= limit:
                break
            subj = row.get("arg1", row.get("subject", "")).strip()
            pred = row.get("relation", row.get("predicate", "")).strip()
            obj = row.get("arg2", row.get("object", "")).strip()
            if subj and pred and obj:
                text = f"{subj} {pred} {obj}"
                records.append({
                    "subject": subj,
                    "predicate": pred,
                    "obj": obj,
                    "text": text,
                })

    logger.info("Parsed %d assertions", len(records))
    return records


def batch_embed(texts: list[str], batch_size: int) -> list[list[float]]:
    """Embed texts using nomic-embed-text via Ollama."""
    from agno.embedder.ollama import OllamaEmbedder

    embedder = OllamaEmbedder(id="nomic-embed-text", dimensions=EMBED_DIM)
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for text in batch:
            emb = embedder.get_embedding(text)
            all_embeddings.append(emb if emb else [0.0] * EMBED_DIM)

        done = min(i + batch_size, len(texts))
        logger.info("Embedded %d/%d (%.0f%%)", done, len(texts), done / len(texts) * 100)

    return all_embeddings


def upsert_to_qdrant(
    records: list[dict],
    embeddings: list[list[float]],
    url: str,
    batch_size: int,
) -> None:
    """Upsert records + embeddings to Qdrant."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(url=url, timeout=30)

    # Recreate collection
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION in collections:
        client.delete_collection(COLLECTION)
        logger.info("Deleted existing collection '%s'", COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    logger.info("Created collection '%s'", COLLECTION)

    # Batch upsert
    for i in range(0, len(records), batch_size):
        batch_r = records[i : i + batch_size]
        batch_e = embeddings[i : i + batch_size]
        points = [
            PointStruct(
                id=i + j + 1,
                vector=emb,
                payload={
                    "subject": r["subject"],
                    "predicate": r["predicate"],
                    "obj": r["obj"],
                },
            )
            for j, (r, emb) in enumerate(zip(batch_r, batch_e))
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        done = min(i + batch_size, len(records))
        logger.info("Upserted %d/%d", done, len(records))

    logger.info("Done! %d assertions in '%s'", len(records), COLLECTION)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load ASCENT++ to Qdrant")
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    args = parser.parse_args()

    t0 = time.time()

    csv_path = DATA_DIR / "ascent_pp.csv"
    download_ascent(csv_path)
    records = parse_csv(csv_path, limit=args.limit)

    if not records:
        logger.error("No records parsed!")
        sys.exit(1)

    texts = [r["text"] for r in records]
    embeddings = batch_embed(texts, batch_size=args.batch_size)
    upsert_to_qdrant(records, embeddings, args.qdrant_url, batch_size=args.batch_size)

    elapsed = time.time() - t0
    logger.info("Total time: %.1fs", elapsed)


if __name__ == "__main__":
    main()
