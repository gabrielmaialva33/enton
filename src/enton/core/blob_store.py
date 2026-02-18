"""BlobStore — external HD binary memory for Enton.

Manages binary files (images, audio, video, faces, snapshots) on an
external HD with Qdrant metadata indexing and transparent fallback.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)

BLOB_COLLECTION = "enton_blobs"
EMBED_DIM = 768  # nomic-embed-text

_MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".bin": "application/octet-stream",
}


class BlobType(StrEnum):
    IMAGE = "images"
    AUDIO = "audio"
    VIDEO = "video"
    FACE = "faces"
    SNAPSHOT = "snapshots"


@dataclass(frozen=True, slots=True)
class BlobMeta:
    blob_id: str
    blob_type: BlobType
    path: str
    size_bytes: int
    timestamp: float = field(default_factory=time.time)
    mime_type: str = "application/octet-stream"
    camera_id: str = ""
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class BlobStore:
    """Binary memory store backed by external HD + Qdrant index."""

    def __init__(
        self,
        root: str,
        fallback: str,
        qdrant_url: str = "http://localhost:6333",
    ) -> None:
        self._root = Path(root)
        self._fallback = Path(fallback)
        self._qdrant_url = qdrant_url
        self._qdrant: QdrantClient | None = None
        self._embedder: Any = None
        self._hd_available: bool | None = None
        self._last_check: float = 0.0
        self._check_interval = 60.0
        self._ensure_dirs()

    # -- properties --

    @property
    def available(self) -> bool:
        """True if HD is mounted and writable (cached 60s)."""
        now = time.time()
        if self._hd_available is None or (now - self._last_check) > self._check_interval:
            self._hd_available = self._check_hd()
            self._last_check = now
        return self._hd_available

    @property
    def active_root(self) -> Path:
        return self._root if self.available else self._fallback

    # -- public API --

    async def store(
        self,
        data: bytes,
        blob_type: BlobType,
        *,
        extension: str = ".bin",
        camera_id: str = "",
        tags: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> BlobMeta:
        """Write binary data to the store. Returns metadata."""
        blob_id = self._make_id()
        filename = f"{blob_id}{extension}"
        dest = self._resolve_path(blob_type, filename)
        dest.parent.mkdir(parents=True, exist_ok=True)

        await asyncio.to_thread(dest.write_bytes, data)

        meta = BlobMeta(
            blob_id=blob_id,
            blob_type=blob_type,
            path=str(dest),
            size_bytes=len(data),
            timestamp=time.time(),
            mime_type=_MIME_MAP.get(extension, "application/octet-stream"),
            camera_id=camera_id,
            tags=tags or [],
            extra=extra or {},
        )
        await self._index(meta)
        logger.debug("BlobStore: stored %s (%d bytes)", dest.name, len(data))
        return meta

    async def store_file(
        self,
        source: str | Path,
        blob_type: BlobType,
        *,
        move: bool = False,
        camera_id: str = "",
        tags: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> BlobMeta:
        """Copy/move an existing file into the store."""
        src = Path(source)
        blob_id = self._make_id()
        filename = f"{blob_id}{src.suffix}"
        dest = self._resolve_path(blob_type, filename)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if move:
            await asyncio.to_thread(shutil.move, str(src), str(dest))
        else:
            await asyncio.to_thread(shutil.copy2, str(src), str(dest))

        size = await asyncio.to_thread(lambda: dest.stat().st_size)
        meta = BlobMeta(
            blob_id=blob_id,
            blob_type=blob_type,
            path=str(dest),
            size_bytes=size,
            timestamp=time.time(),
            mime_type=_MIME_MAP.get(src.suffix, "application/octet-stream"),
            camera_id=camera_id,
            tags=tags or [],
            extra=extra or {},
        )
        await self._index(meta)
        return meta

    async def search(
        self,
        query: str,
        blob_type: BlobType | None = None,
        n: int = 10,
    ) -> list[BlobMeta]:
        """Semantic search via Qdrant."""
        embedding = await self._embed_text(query)
        if not embedding or not self._init_qdrant():
            return []

        try:
            filt = None
            if blob_type:
                filt = Filter(
                    must=[
                        FieldCondition(key="blob_type", match=MatchValue(value=blob_type.value)),
                    ]
                )
            response = self._qdrant.query_points(
                collection_name=BLOB_COLLECTION,
                query=embedding,
                limit=n,
                query_filter=filt,
            )
            return [self._payload_to_meta(r.payload) for r in response.points]
        except Exception:
            logger.warning("BlobStore search failed")
            return []

    async def recent(
        self,
        blob_type: BlobType | None = None,
        n: int = 10,
    ) -> list[BlobMeta]:
        """Get N most recent blobs via Qdrant scroll."""
        if not self._init_qdrant():
            return []
        try:
            filt = None
            if blob_type:
                filt = Filter(
                    must=[
                        FieldCondition(key="blob_type", match=MatchValue(value=blob_type.value)),
                    ]
                )
            # Fetch more than needed, sort in Python by timestamp
            fetch_limit = min(n * 3, 100)
            results, _ = self._qdrant.scroll(
                collection_name=BLOB_COLLECTION,
                scroll_filter=filt,
                limit=fetch_limit,
            )
            metas = [self._payload_to_meta(r.payload) for r in results]
            return sorted(metas, key=lambda m: m.timestamp, reverse=True)[:n]
        except Exception:
            logger.warning("BlobStore recent query failed")
            return []

    async def stats(self) -> dict[str, Any]:
        """Storage stats: free space, counts by type."""
        root = self.active_root
        try:
            usage = await asyncio.to_thread(shutil.disk_usage, str(root))
            free_gb = round(usage.free / (1024**3), 1)
        except Exception:
            free_gb = -1

        counts: dict[str, int] = {}
        for bt in BlobType:
            d = root / bt.value
            if d.exists():
                counts[bt.value] = len(list(d.iterdir()))
            else:
                counts[bt.value] = 0

        return {
            "root": str(root),
            "hd_mounted": self.available,
            "free_gb": free_gb,
            "counts": counts,
        }

    # -- internals --

    def _make_id(self) -> str:
        ts = int(time.time() * 1000)
        short = uuid.uuid4().hex[:6]
        return f"{ts}_{short}"

    def _resolve_path(self, blob_type: BlobType, filename: str) -> Path:
        return self.active_root / blob_type.value / filename

    def _check_hd(self) -> bool:
        try:
            # Check if mount point parent exists (HD mounted)
            parent = self._root.parent
            if not parent.exists():
                logger.warning("BlobStore: HD not mounted at %s, using fallback", parent)
                return False
            # Create our subdirectory if mount point is writable
            self._root.mkdir(parents=True, exist_ok=True)
            test = self._root / ".probe"
            test.write_text("ok")
            test.unlink()
            return True
        except Exception:
            logger.warning("BlobStore: HD not writable at %s, using fallback", self._root)
            return False

    def _ensure_dirs(self) -> None:
        root = self._root if self._check_hd() else self._fallback
        for bt in BlobType:
            (root / bt.value).mkdir(parents=True, exist_ok=True)
        status = "HD" if root == self._root else "fallback"
        logger.info("BlobStore: ready at %s (%s)", root, status)

    def _init_qdrant(self) -> bool:
        if self._qdrant is not None:
            return True
        try:
            client = QdrantClient(url=self._qdrant_url, timeout=5)
            collections = [c.name for c in client.get_collections().collections]
            if BLOB_COLLECTION not in collections:
                client.create_collection(
                    collection_name=BLOB_COLLECTION,
                    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
                )
                logger.info("Created Qdrant collection '%s'", BLOB_COLLECTION)
            # Create payload indexes for filtering and ordering
            try:
                client.create_payload_index(
                    collection_name=BLOB_COLLECTION,
                    field_name="timestamp",
                    field_schema=PayloadSchemaType.FLOAT,
                )
                client.create_payload_index(
                    collection_name=BLOB_COLLECTION,
                    field_name="blob_type",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # indexes may already exist
            self._qdrant = client
            return True
        except Exception:
            logger.debug("Qdrant unavailable for BlobStore")
            return False

    def _init_embedder(self) -> bool:
        if self._embedder is not None:
            return True
        try:
            from agno.knowledge.embedder.ollama import OllamaEmbedder

            self._embedder = OllamaEmbedder(id="nomic-embed-text", dimensions=EMBED_DIM)
            return True
        except Exception:
            logger.debug("OllamaEmbedder unavailable for BlobStore")
            return False

    async def _embed_text(self, text: str) -> list[float] | None:
        if not self._init_embedder():
            return None
        try:
            result = await asyncio.to_thread(self._embedder.get_embedding, text)
            return result
        except Exception:
            logger.debug("BlobStore embedding failed")
            return None

    async def _index(self, meta: BlobMeta) -> None:
        """Index blob metadata in Qdrant."""
        if not self._init_qdrant():
            return

        # Build searchable text from tags + type + camera
        search_text = " ".join(
            [
                meta.blob_type.value,
                meta.camera_id,
                *meta.tags,
                *[f"{k}={v}" for k, v in meta.extra.items() if isinstance(v, str)],
            ]
        )
        embedding = await self._embed_text(search_text)
        if not embedding:
            return

        payload = {
            "blob_id": meta.blob_id,
            "blob_type": meta.blob_type.value,
            "path": meta.path,
            "size_bytes": meta.size_bytes,
            "timestamp": meta.timestamp,
            "mime_type": meta.mime_type,
            "camera_id": meta.camera_id,
            "tags": meta.tags,
        }

        try:
            self._qdrant.upsert(
                collection_name=BLOB_COLLECTION,
                points=[
                    PointStruct(
                        id=uuid.uuid4().hex,
                        vector=embedding,
                        payload=payload,
                    )
                ],
            )
        except Exception:
            logger.debug("BlobStore Qdrant indexing failed")

    @staticmethod
    def _payload_to_meta(payload: dict) -> BlobMeta:
        return BlobMeta(
            blob_id=payload.get("blob_id", ""),
            blob_type=BlobType(payload.get("blob_type", "images")),
            path=payload.get("path", ""),
            size_bytes=payload.get("size_bytes", 0),
            timestamp=payload.get("timestamp", 0.0),
            mime_type=payload.get("mime_type", ""),
            camera_id=payload.get("camera_id", ""),
            tags=payload.get("tags", []),
        )
