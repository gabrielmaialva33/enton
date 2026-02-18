"""Sound event detection using CLAP (Contrastive Language-Audio Pretraining).

Detects ambient sounds like doorbells, alarms, dogs barking, etc.
Uses text embeddings for open-set classification — new classes can be
added without retraining.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

logger = logging.getLogger(__name__)

# Default sound classes with Portuguese labels
DEFAULT_SOUND_CLASSES = {
    "doorbell ringing": "Campainha",
    "dog barking": "Cachorro latindo",
    "cat meowing": "Gato miando",
    "alarm sounding": "Alarme",
    "glass breaking": "Vidro quebrando",
    "applause": "Aplausos",
    "music playing": "Música",
    "baby crying": "Bebê chorando",
    "phone ringing": "Telefone tocando",
    "knock on door": "Batida na porta",
    "thunder": "Trovão",
    "siren": "Sirene",
}


class SoundResult:
    __slots__ = ("confidence", "label", "label_en")

    def __init__(self, label: str, label_en: str, confidence: float) -> None:
        self.label = label
        self.label_en = label_en
        self.confidence = confidence


class SoundDetector:
    """CLAP-based ambient sound detector."""

    def __init__(
        self,
        threshold: float = 0.3,
        classes: dict[str, str] | None = None,
    ) -> None:
        self._threshold = threshold
        self._classes = classes or DEFAULT_SOUND_CLASSES
        self._model = None
        self._processor = None
        self._text_embeds = None

    def _ensure_model(self):
        if self._model is not None:
            return

        model_id = "laion/clap-htsat-unfused"
        logger.info("Loading CLAP model: %s", model_id)
        self._processor = ClapProcessor.from_pretrained(model_id)
        self._model = ClapModel.from_pretrained(model_id)
        self._model.eval()

        # Pre-compute text embeddings for all classes
        self._precompute_text_embeddings()
        logger.info("CLAP loaded with %d sound classes", len(self._classes))

    def _precompute_text_embeddings(self):
        texts = list(self._classes.keys())
        inputs = self._processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = self._model.get_text_features(**inputs)
            self._text_embeds = out if isinstance(out, torch.Tensor) else out.pooler_output
            self._text_embeds = self._text_embeds / self._text_embeds.norm(dim=-1, keepdim=True)

    def classify(self, audio: np.ndarray, sample_rate: int = 48000) -> list[SoundResult]:
        """Classify ambient sounds in an audio chunk.

        Args:
            audio: Float32 audio array.
            sample_rate: Sample rate of the audio.

        Returns:
            List of detected sounds above threshold.
        """
        self._ensure_model()

        inputs = self._processor(audio=audio, sampling_rate=sample_rate, return_tensors="pt")
        with torch.no_grad():
            out = self._model.get_audio_features(**inputs)
            audio_embeds = out if isinstance(out, torch.Tensor) else out.pooler_output
            audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarities = (audio_embeds @ self._text_embeds.T).squeeze(0)
        probs = similarities.softmax(dim=-1).numpy()

        results = []
        en_labels = list(self._classes.keys())
        pt_labels = list(self._classes.values())

        for i, prob in enumerate(probs):
            if prob >= self._threshold:
                results.append(
                    SoundResult(
                        label=pt_labels[i],
                        label_en=en_labels[i],
                        confidence=float(prob),
                    )
                )

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    async def classify_async(
        self, audio: np.ndarray, sample_rate: int = 48000
    ) -> list[SoundResult]:
        """Async wrapper for classify."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.classify, audio, sample_rate)
