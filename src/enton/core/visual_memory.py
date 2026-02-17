"""Visual Episodic Memory — SigLIP embeddings + Qdrant visual collection.

Embeds keyframes on significant vision events (new objects, scene changes).
Stores embeddings in Qdrant collection 'enton_visual' with metadata.
Saves JPEG thumbnails to ~/.enton/frames/ for retrieval.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import open_clip
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

FRAMES_DIR = Path.home() / ".enton" / "frames"
VISUAL_COLLECTION = "enton_visual"
EMBED_DIM = 512  # SigLIP ViT-B-16 output dimension
MIN_EMBED_INTERVAL = 30.0  # seconds between embeds per camera


@dataclass(frozen=True, slots=True)
class VisualEpisode:
    """A single visual memory snapshot."""

    timestamp: float = field(default_factory=time.time)
    camera_id: str = "main"
    detections: list[str] = field(default_factory=list)
    thumbnail_path: str = ""
    embedding: list[float] = field(default_factory=list)


class VisualMemory:
    """Manages visual episodic memories with SigLIP + Qdrant."""

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        siglip_model: str = "ViT-B-16-SigLIP",
        siglip_pretrained: str = "webli",
        frames_dir: Path = FRAMES_DIR,
        blob_store: Any = None,
    ) -> None:
        self._qdrant_url = qdrant_url
        self._siglip_model_name = siglip_model
        self._siglip_pretrained = siglip_pretrained
        self._frames_dir = frames_dir
        self._blob_store = blob_store
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._qdrant: Any = None
        self._last_embed: dict[str, float] = {}  # camera_id → timestamp
        self._last_labels: dict[str, set[str]] = {}  # camera_id → label set
        self._episode_count = 0

    # -- initialization --

    def _ensure_frames_dir(self) -> None:
        self._frames_dir.mkdir(parents=True, exist_ok=True)

    def _init_qdrant(self) -> bool:
        """Initialize Qdrant collection for visual embeddings (lazy)."""
        if self._qdrant is not None:
            return True
        try:
            client = QdrantClient(url=self._qdrant_url, timeout=5)
            collections = [c.name for c in client.get_collections().collections]
            if VISUAL_COLLECTION not in collections:
                client.create_collection(
                    collection_name=VISUAL_COLLECTION,
                    vectors_config=VectorParams(
                        size=EMBED_DIM, distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection '%s'", VISUAL_COLLECTION)
            self._qdrant = client
            return True
        except Exception:
            logger.warning("Qdrant unavailable for visual memory")
            return False

    def _load_model(self) -> bool:
        """Load SigLIP model via open-clip-torch (lazy, synchronous)."""
        if self._model is not None:
            return True
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(
                self._siglip_model_name,
                pretrained=self._siglip_pretrained,
                device=device,
            )
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self._siglip_model_name)
            logger.info(
                "SigLIP '%s' loaded on %s", self._siglip_model_name, device,
            )
            return True
        except Exception:
            logger.warning("SigLIP model unavailable")
            return False

    # -- embedding --

    async def embed_frame(self, frame_bgr: np.ndarray) -> list[float]:
        """Embed a BGR numpy frame using SigLIP. Returns embedding vector."""
        if not self._load_model():
            return []
        try:
            # BGR → RGB → PIL
            rgb = frame_bgr[:, :, ::-1]
            img = Image.fromarray(rgb)
            tensor = self._preprocess(img).unsqueeze(0)
            device = next(self._model.parameters()).device
            tensor = tensor.to(device)

            with torch.no_grad():
                features = self._model.encode_image(tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            return features[0].cpu().tolist()
        except Exception:
            logger.warning("Failed to embed frame", exc_info=True)
            return []

    async def embed_text(self, text: str) -> list[float]:
        """Embed a text query using SigLIP text encoder."""
        if not self._load_model():
            return []
        try:
            tokens = self._tokenizer([text])
            device = next(self._model.parameters()).device
            tokens = tokens.to(device)

            with torch.no_grad():
                features = self._model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            return features[0].cpu().tolist()
        except Exception:
            logger.warning("Failed to embed text", exc_info=True)
            return []

    # -- novelty detection --

    def _should_embed(self, detections: list[str], camera_id: str) -> bool:
        """Check if scene is novel enough to embed."""
        now = time.time()
        last_t = self._last_embed.get(camera_id, 0.0)
        if now - last_t < MIN_EMBED_INTERVAL:
            return False

        current = set(detections)
        prev = self._last_labels.get(camera_id, set())
        return not (current == prev and last_t > 0)

    # -- remember --

    async def remember_scene(
        self,
        frame_bgr: np.ndarray,
        detections: list[str],
        camera_id: str = "main",
    ) -> VisualEpisode | None:
        """Save a keyframe as a visual episode."""
        if not self._should_embed(detections, camera_id):
            return None

        embedding = await self.embed_frame(frame_bgr)
        if not embedding:
            return None

        now = time.time()
        self._last_embed[camera_id] = now
        self._last_labels[camera_id] = set(detections)

        self._episode_count += 1

        # Save thumbnail
        thumbnail_path = await self._save_thumbnail(frame_bgr, now, camera_id)

        episode = VisualEpisode(
            timestamp=now,
            camera_id=camera_id,
            detections=list(detections),
            thumbnail_path=thumbnail_path,
            embedding=embedding,
        )

        # Store in Qdrant
        if self._init_qdrant():
            try:
                self._qdrant.upsert(
                    collection_name=VISUAL_COLLECTION,
                    points=[PointStruct(
                        id=self._episode_count,
                        vector=embedding,
                        payload={
                            "timestamp": now,
                            "camera_id": camera_id,
                            "detections": detections,
                            "thumbnail_path": thumbnail_path,
                        },
                    )],
                )
            except Exception:
                logger.warning("Failed to store visual episode in Qdrant")

        logger.info(
            "Visual memory #%d: %s (%s)",
            self._episode_count, detections, camera_id,
        )
        return episode

    async def _save_thumbnail(
        self, frame_bgr: np.ndarray, timestamp: float, camera_id: str = "",
    ) -> str:
        """Save JPEG thumbnail, return path string."""
        try:
            # Resize to 320px wide for storage
            h, w = frame_bgr.shape[:2]
            if w > 320:
                scale = 320 / w
                frame_bgr = cv2.resize(
                    frame_bgr, (320, int(h * scale)),
                )

            _, jpeg_buf = cv2.imencode(
                ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70],
            )
            jpeg_bytes = jpeg_buf.tobytes()

            # Use BlobStore (external HD) when available
            if self._blob_store is not None:
                try:
                    from enton.core.blob_store import BlobType

                    meta = await self._blob_store.store(
                        jpeg_bytes,
                        BlobType.IMAGE,
                        extension=".jpg",
                        camera_id=camera_id,
                        tags=["thumbnail", "visual_memory"],
                    )
                    return meta.path
                except Exception:
                    logger.debug("BlobStore save failed, falling back to local")

            # Fallback: direct local save
            import asyncio

            self._ensure_frames_dir()
            fname = f"{int(timestamp * 1000)}.jpg"
            path = self._frames_dir / fname
            await asyncio.to_thread(path.write_bytes, jpeg_bytes)
            return str(path)
        except Exception:
            logger.warning("Failed to save thumbnail")
            return ""

    # -- search --

    async def search(self, query: str, n: int = 5) -> list[dict]:
        """Semantic search: embed query text, search Qdrant visual collection."""
        embedding = await self.embed_text(query)
        if not embedding or not self._init_qdrant():
            return []

        try:
            response = self._qdrant.query_points(
                collection_name=VISUAL_COLLECTION,
                query=embedding,
                limit=n,
            )
            return [
                {
                    "timestamp": r.payload.get("timestamp", 0),
                    "camera_id": r.payload.get("camera_id", "?"),
                    "detections": r.payload.get("detections", []),
                    "thumbnail_path": r.payload.get("thumbnail_path", ""),
                    "score": r.score,
                }
                for r in response.points
            ]
        except Exception:
            logger.warning("Visual search failed")
            return []

    async def recent_scenes(self, n: int = 5) -> list[dict]:
        """Return N most recent visual episodes from Qdrant."""
        if not self._init_qdrant():
            return []

        try:
            results = self._qdrant.scroll(
                collection_name=VISUAL_COLLECTION,
                limit=n,
                order_by="timestamp",
            )
            points = results[0] if results else []
            return [
                {
                    "timestamp": p.payload.get("timestamp", 0),
                    "camera_id": p.payload.get("camera_id", "?"),
                    "detections": p.payload.get("detections", []),
                    "thumbnail_path": p.payload.get("thumbnail_path", ""),
                }
                for p in reversed(points)
            ]
        except Exception:
            logger.warning("Recent scenes query failed")
            return []
