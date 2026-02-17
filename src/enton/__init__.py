"""Enton — Autonomous AI robot assistant.

Architecture:
    core/        — EventBus, config, self-model, memory
    perception/  — Vision (YOLO), ears (STT), activity, emotion, overlay
    cognition/   — Brain (LLM), persona, fuser
    action/      — Voice (TTS), motor (future)
    skills/      — Composable behaviors (greet, react, follow)
    providers/   — Multi-backend (Google, NVIDIA, local)
"""

__version__ = "0.2.0"
