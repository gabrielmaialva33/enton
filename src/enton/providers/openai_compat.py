"""OpenAI-compatible LLM provider with round-robin key rotation.

Works with NVIDIA NIM, Groq, Together, and any OpenAI-compatible API.
Supports: text generation, tool calling, and vision (multimodal).
"""
from __future__ import annotations

import base64
import json
import logging

logger = logging.getLogger(__name__)


class OpenAICompatLLM:
    """OpenAI-compatible LLM with round-robin key rotation."""

    def __init__(
        self,
        base_url: str,
        api_keys: list[str],
        model: str,
        vision_model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.9,
        name: str = "openai-compat",
    ) -> None:
        from openai import AsyncOpenAI

        self._name = name
        self._model = model
        self._vision_model = vision_model or model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._key_idx = 0

        # Pre-create async clients for each key (round-robin)
        self._clients = [
            AsyncOpenAI(base_url=base_url, api_key=k.strip())
            for k in api_keys
            if k.strip()
        ]
        if not self._clients:
            raise ValueError(f"{name}: no valid API keys provided")

        logger.info(
            "%s: %d keys, model=%s, vision=%s",
            name, len(self._clients), model, self._vision_model,
        )

    def _next_client(self):
        client = self._clients[self._key_idx % len(self._clients)]
        self._key_idx += 1
        return client

    async def generate(
        self, prompt: str, *, system: str = "", history: list[dict] | None = None
    ) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        client = self._next_client()
        response = await client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""

    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        *,
        system: str = "",
        history: list[dict] | None = None,
    ) -> dict:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        client = self._next_client()
        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools or None,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            msg = response.choices[0].message
            content = msg.content or ""
            tool_calls = []

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    tool_calls.append({
                        "name": tc.function.name,
                        "arguments": args,
                    })

            return {"content": content, "tool_calls": tool_calls}
        except Exception:
            logger.exception("%s generate_with_tools failed", self._name)
            return {"content": "", "tool_calls": []}

    async def generate_with_image(
        self,
        prompt: str,
        image: bytes,
        *,
        system: str = "",
        mime_type: str = "image/jpeg",
    ) -> str:
        b64 = base64.b64encode(image).decode()
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        })

        client = self._next_client()
        response = await client.chat.completions.create(
            model=self._vision_model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""
