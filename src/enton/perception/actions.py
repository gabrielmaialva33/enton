"""Reconhecimento de acoes temporais via VideoMAE (Kinetics-400)."""

from __future__ import annotations

import logging
from collections import deque

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Top Kinetics-400 action labels com traducoes PT-BR
ACTION_LABELS_PT: dict[str, str] = {
    "drinking": "bebendo",
    "eating": "comendo",
    "reading": "lendo",
    "writing": "escrevendo",
    "typing": "digitando",
    "talking on phone": "falando no telefone",
    "waving": "acenando",
    "clapping": "aplaudindo",
    "stretching": "se espreguicando",
    "yawning": "bocejando",
    "sitting": "sentado",
    "standing": "em pe",
    "walking": "andando",
    "running": "correndo",
    "dancing": "dancando",
    "cooking": "cozinhando",
    "cleaning": "limpando",
    "playing guitar": "tocando guitarra",
    "playing piano": "tocando piano",
    "exercising": "se exercitando",
}


class ActionRecognizer:
    """Reconhecimento de acoes temporais via VideoMAE.

    Mantem buffer circular de frames e classifica acoes a cada N frames.
    Modelo carregado on-demand para economizar VRAM.
    """

    MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
    NUM_FRAMES = 16
    CLASSIFY_EVERY = 32  # classifica a cada N frames (~2s a 15fps)
    MIN_CONFIDENCE = 0.3

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._model = None
        self._processor = None
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=64)
        self._frame_count = 0
        self._last_actions: list[tuple[str, str, float]] = []  # (en, pt, conf)
        self._loaded = False

    def _ensure_loaded(self) -> bool:
        """Carrega modelo no primeiro uso. Retorna True se carregou com sucesso."""
        if self._loaded:
            return True
        try:
            from transformers import AutoImageProcessor, VideoMAEForVideoClassification

            self._processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self._model = (
                VideoMAEForVideoClassification.from_pretrained(
                    self.MODEL_ID,
                    torch_dtype=torch.float16,
                    attn_implementation="sdpa",
                )
                .to(self._device)
                .eval()
            )
            self._loaded = True
            logger.info("VideoMAE loaded: %s (fp16+SDPA)", self.MODEL_ID)
            return True
        except Exception as e:
            logger.warning("Failed to load VideoMAE: %s", e)
            return False

    def unload(self) -> None:
        """Libera VRAM."""
        if self._model is not None:
            del self._model, self._processor
            self._model = self._processor = None
            self._loaded = False
            torch.cuda.empty_cache()
            logger.info("VideoMAE unloaded")

    def feed_frame(self, frame: np.ndarray) -> None:
        """Adiciona um frame BGR ao buffer."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._frame_buffer.append(rgb)
        self._frame_count += 1

    def _sample_frames(self) -> list[np.ndarray] | None:
        """Amostra NUM_FRAMES uniformemente do buffer."""
        buf = list(self._frame_buffer)
        if len(buf) < self.NUM_FRAMES:
            return None
        indices = np.linspace(0, len(buf) - 1, self.NUM_FRAMES, dtype=int)
        return [buf[i] for i in indices]

    def should_classify(self) -> bool:
        """Verifica se e hora de rodar classificacao."""
        return (
            self._frame_count % self.CLASSIFY_EVERY == 0
            and len(self._frame_buffer) >= self.NUM_FRAMES
        )

    @torch.no_grad()
    def classify(self) -> list[tuple[str, str, float]]:
        """Roda classificacao VideoMAE nos frames bufferizados.

        Retorna lista de (action_en, action_pt, confidence).
        """
        frames = self._sample_frames()
        if frames is None:
            return []

        if not self._ensure_loaded():
            return []

        try:
            inputs = self._processor(frames, return_tensors="pt")
            inputs = {k: v.to(self._device, torch.float16) for k, v in inputs.items()}

            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits[0], dim=-1)
            top5 = probs.topk(5)

            results = []
            for idx, prob in zip(top5.indices, top5.values, strict=True):
                label_en = self._model.config.id2label[idx.item()]
                conf = prob.item()
                if conf < self.MIN_CONFIDENCE:
                    continue
                label_pt = ACTION_LABELS_PT.get(label_en, label_en)
                results.append((label_en, label_pt, conf))

            self._last_actions = results
            return results
        except Exception as e:
            logger.warning("VideoMAE inference failed: %s", e)
            return []

    @property
    def last_actions(self) -> list[tuple[str, str, float]]:
        """Retorna as ultimas acoes classificadas."""
        return self._last_actions
