"""Facial emotion recognition using ViT (dima806/facial_emotions_image_detection)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_MODEL_ID = "dima806/facial_emotions_image_detection"

# Translate emotion labels to pt-BR
_LABEL_MAP = {
    "happy": "Feliz",
    "sad": "Triste",
    "angry": "Irritado",
    "surprise": "Surpreso",
    "fear": "Assustado",
    "disgust": "Enojado",
    "neutral": "Neutro",
}

# Color per emotion (BGR)
_COLOR_MAP = {
    "happy": (0, 255, 180),
    "sad": (255, 120, 50),
    "angry": (50, 50, 255),
    "surprise": (0, 230, 255),
    "fear": (200, 100, 255),
    "disgust": (0, 180, 0),
    "neutral": (180, 180, 180),
}


@dataclass
class FaceEmotion:
    label: str  # pt-BR
    label_en: str
    score: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (face crop region)
    color: tuple[int, int, int]


class EmotionRecognizer:
    """Lightweight face emotion classifier using HuggingFace ViT pipeline."""

    def __init__(self, device: str = "cuda:0", interval_frames: int = 5) -> None:
        self._device = device
        self._pipeline = None
        self._interval = interval_frames
        self._frame_count = 0
        self._cache: list[FaceEmotion] = []

    def _ensure_pipeline(self):
        if self._pipeline is None:
            import torch
            from transformers import pipeline

            self._pipeline = pipeline(
                "image-classification",
                model=_MODEL_ID,
                device=self._device,
                dtype=torch.float16,
            )
            logger.info("Emotion model loaded: %s on %s", _MODEL_ID, self._device)
        return self._pipeline

    def _crop_face(
        self, frame: np.ndarray, kpts, margin: float = 0.6,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        """Crop face region from keypoints (eyes, nose, ears)."""
        # Use eye/nose/ear keypoints to estimate face bbox
        face_kpt_ids = [0, 1, 2, 3, 4]  # nose, l_eye, r_eye, l_ear, r_ear
        points = []
        for idx in face_kpt_ids:
            if float(kpts[idx][2]) > 0.3:
                points.append((float(kpts[idx][0]), float(kpts[idx][1])))

        if len(points) < 2:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        spread = max(max(xs) - min(xs), max(ys) - min(ys), 30)
        half = spread * (0.5 + margin)

        fh, fw = frame.shape[:2]
        x1 = max(0, int(cx - half))
        y1 = max(0, int(cy - half * 1.2))  # more room above (forehead)
        x2 = min(fw, int(cx + half))
        y2 = min(fh, int(cy + half * 0.8))  # less below (chin)

        if x2 - x1 < 20 or y2 - y1 < 20:
            return None

        crop = frame[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2)

    def classify(
        self, frame: np.ndarray, keypoints_list: list,
    ) -> list[FaceEmotion]:
        """Classify emotions for all detected persons.

        Runs inference every `interval_frames` and caches results in between.
        """
        self._frame_count += 1
        if self._frame_count % self._interval != 0 and self._cache:
            return self._cache

        if not keypoints_list:
            self._cache = []
            return self._cache

        pipe = self._ensure_pipeline()
        results: list[FaceEmotion] = []

        for kpts in keypoints_list:
            crop_result = self._crop_face(frame, kpts)
            if crop_result is None:
                continue

            crop_bgr, bbox = crop_result
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)

            try:
                preds = pipe(pil_img)
                top = preds[0]
                label_en = top["label"]
                results.append(FaceEmotion(
                    label=_LABEL_MAP.get(label_en, label_en),
                    label_en=label_en,
                    score=top["score"],
                    bbox=bbox,
                    color=_COLOR_MAP.get(label_en, (180, 180, 180)),
                ))
            except Exception:
                logger.debug("Emotion inference failed for face crop", exc_info=True)

        self._cache = results
        return results
