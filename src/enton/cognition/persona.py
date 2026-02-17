"""Persona — Fachada para a alma do Enton.

Re-exporta constantes de prompts.py e mantém as funções utilitárias
(build_system_prompt, build_monologue_prompt) para compatibilidade.

Todos os prompts vivem em cognition/prompts.py — o Grimório da Alma.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from enton.cognition.prompts import (
    EMPATHY_TONES,
    MONOLOGUE_PROMPT,
    REACTION_TEMPLATES,
    SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from enton.core.memory import Memory
    from enton.core.self_model import SelfModel

# Re-export for backward compatibility
__all__ = [
    "MONOLOGUE_PROMPT",
    "REACTION_TEMPLATES",
    "SYSTEM_PROMPT",
    "build_monologue_prompt",
    "build_system_prompt",
]


def _get_empathy_instruction(emotion: str) -> str:
    """Get empathy-aware tone instruction based on detected emotion."""
    return EMPATHY_TONES.get(emotion.lower(), "")


def build_system_prompt(
    self_model: SelfModel,
    memory: Memory,
    detections: list[dict] | None = None,
    hour: int | None = None,
) -> str:
    self_state = self_model.introspect()
    memory_context = memory.context_string()
    env_context = _build_env_context(detections or [], hour)
    prompt = SYSTEM_PROMPT.format(
        self_state=self_state,
        memory_context=memory_context,
        env_context=env_context,
    )

    # Inject empathy-aware tone based on detected user emotion
    empathy = _get_empathy_instruction(self_model.last_emotion)
    if empathy:
        prompt += f"\n\nEMOTIONAL CONTEXT: {empathy}"

    return prompt


def build_monologue_prompt(
    vision_summary: str = "Nothing detected",
    system_summary: str = "Normal",
    last_interaction: str = "Unknown",
    idle_time: str = "Unknown",
    current_mood: str = "neutral",
    energy: str = "0.5",
    desires: str = "none",
) -> str:
    """Build internal monologue prompt for DesireEngine / DreamMode."""
    return MONOLOGUE_PROMPT.format(
        vision_summary=vision_summary,
        system_summary=system_summary,
        last_interaction=last_interaction,
        idle_time=idle_time,
        current_mood=current_mood,
        energy=energy,
        desires=desires,
    )


def _build_env_context(detections: list[dict], hour: int | None = None) -> str:
    import time

    if hour is None:
        hour = time.localtime().tm_hour
    parts = []
    if detections:
        labels = [d["label"] for d in detections]
        parts.append(f"Detections: {', '.join(labels)}")
    if 0 <= hour < 6:
        period = "madrugada"
    elif 6 <= hour < 12:
        period = "manhã"
    elif 12 <= hour < 18:
        period = "tarde"
    else:
        period = "noite"
    parts.append(f"Time: {period} ({hour}h)")
    return " | ".join(parts) if parts else "Nada detectado, sala quieta."
