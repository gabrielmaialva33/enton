"""Face recognition using InsightFace buffalo_s.

Detects and identifies faces by comparing embeddings against a persistent
database stored at ~/.enton/faces.pkl.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DB_PATH = Path.home() / ".enton" / "faces.pkl"
_SIMILARITY_THRESHOLD = 0.45  # cosine similarity threshold for match


class FaceResult:
    __slots__ = ("identity", "confidence", "bbox")

    def __init__(
        self, identity: str, confidence: float, bbox: tuple[int, int, int, int]
    ) -> None:
        self.identity = identity
        self.confidence = confidence
        self.bbox = bbox


class FaceRecognizer:
    """InsightFace-based face detector + recognizer."""

    def __init__(self, device: str = "cuda:0") -> None:
        self._app = None
        self._device = device
        self._db: dict[str, np.ndarray] = {}
        self._load_db()

    def _ensure_model(self):
        if self._app is not None:
            return self._app

        from insightface.app import FaceAnalysis

        ctx_id = 0
        if "cuda" in self._device:
            parts = self._device.split(":")
            ctx_id = int(parts[1]) if len(parts) > 1 else 0

        self._app = FaceAnalysis(
            name="buffalo_s",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=ctx_id, det_size=(320, 320))
        logger.info("InsightFace buffalo_s loaded (ctx=%d)", ctx_id)
        return self._app

    def _load_db(self) -> None:
        if _DB_PATH.exists():
            try:
                with open(_DB_PATH, "rb") as f:
                    self._db = pickle.load(f)
                logger.info(
                    "Face DB loaded: %d identities", len(self._db)
                )
            except Exception:
                logger.exception("Failed to load face DB")
                self._db = {}

    def _save_db(self) -> None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_DB_PATH, "wb") as f:
            pickle.dump(self._db, f)
        logger.info("Face DB saved: %d identities", len(self._db))

    def enroll(self, name: str, frame: np.ndarray) -> str:
        """Enroll a face from the current frame. Returns status message."""
        app = self._ensure_model()
        faces = app.get(frame)
        if not faces:
            return "Nenhum rosto detectado no frame."
        if len(faces) > 1:
            return (
                f"{len(faces)} rostos detectados. "
                "Mostre apenas 1 rosto para cadastrar."
            )

        face = faces[0]
        embedding = face.normed_embedding
        self._db[name.lower()] = embedding
        self._save_db()
        return f"Rosto de '{name}' cadastrado com sucesso!"

    def identify(self, frame: np.ndarray) -> list[FaceResult]:
        """Detect and identify all faces in the frame."""
        app = self._ensure_model()
        faces = app.get(frame)
        results = []

        for face in faces:
            bbox = tuple(int(c) for c in face.bbox[:4])
            embedding = face.normed_embedding

            best_name = "unknown"
            best_score = 0.0

            for name, db_emb in self._db.items():
                score = float(np.dot(embedding, db_emb))
                if score > best_score:
                    best_score = score
                    best_name = name

            if best_score < _SIMILARITY_THRESHOLD:
                best_name = "unknown"
                best_score = 0.0

            results.append(
                FaceResult(
                    identity=best_name,
                    confidence=best_score,
                    bbox=bbox,
                )
            )

        return results

    def list_enrolled(self) -> list[str]:
        """List all enrolled identities."""
        return list(self._db.keys())

    def remove(self, name: str) -> bool:
        """Remove an enrolled identity."""
        key = name.lower()
        if key in self._db:
            del self._db[key]
            self._save_db()
            return True
        return False

    @property
    def db_size(self) -> int:
        return len(self._db)
