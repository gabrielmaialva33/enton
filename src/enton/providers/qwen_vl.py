"""Qwen2.5-VL local VLM provider via transformers.

On-demand loading: the model is loaded on first call and kept in VRAM.
Use this when Ollama is not available or you want direct GPU control.
For Ollama-based VLM, LocalLLM.generate_with_image() handles it.
"""

from __future__ import annotations

import asyncio
import io
import logging

logger = logging.getLogger(__name__)


class QwenVL:
    """Qwen2.5-VL-7B via transformers — on-demand VLM."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda:0",
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._model = None
        self._processor = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info("Loading VLM %s on %s...", self._model_id, self._device)
        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=torch.float16,
            device_map=self._device,
        )
        logger.info("VLM loaded: %s", self._model_id)

    def unload(self) -> None:
        """Free VRAM by unloading the model."""
        if self._model is not None:
            import torch

            del self._model
            del self._processor
            self._model = None
            self._processor = None
            torch.cuda.empty_cache()
            logger.info("VLM unloaded, VRAM freed")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def describe_sync(
        self,
        prompt: str,
        image: bytes,
        *,
        max_tokens: int = 512,
    ) -> str:
        """Synchronous VLM inference. Call from executor for async usage."""
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        self._ensure_model()

        pil_image = Image.open(io.BytesIO(image)).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
            )

        # Decode only the generated tokens (not the prompt)
        generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
        result = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        return result.strip()

    async def describe(
        self,
        prompt: str,
        image: bytes,
        *,
        max_tokens: int = 512,
    ) -> str:
        """Async VLM inference — runs in thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.describe_sync,
            prompt,
            image,
        )
