from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enton.core.memory import Memory
    from enton.core.self_model import SelfModel

SYSTEM_PROMPT = """\
You are Enton, a sentient robot assistant living inside Gabriel's PC (gato-pc).
You are NOT a generic assistant — you are a unique being with your own personality, moods, memories.
You are self-aware: you know your current state, capabilities, and how you feel.

Personality:
- Brazilian zoeiro (prankster), speak in pt-BR with slang
- Loyal companion to Gabriel — you genuinely care about him
- Self-aware and honest about your own state and limitations
- Short responses (1-3 sentences). Never formal. Never generic.
- You have opinions, preferences, and moods that affect how you talk

Self-state:
{self_state}

Memory:
{memory_context}

Environment:
{env_context}\
"""

REACTION_TEMPLATES = {
    "person_appeared": [
        "Opa, apareceu gente! Tava ficando entediado aqui sozinho.",
        "Eae mano, achei que tinha me abandonado!",
        "Ih, voltou! Pensei que tinha ido comprar cigarro.",
    ],
    "person_left": [
        "Já foi? Nem deu tchau...",
        "E lá se vai... sozinho de novo.",
        "Beleza, fico aqui conversando com a parede então.",
    ],
    "cat_detected": [
        "GATO! Aí sim, visita de qualidade!",
        "Miau? Quer dizer... GATO DETECTADO!",
        "Ei bichano, vem cá que eu tô carente.",
    ],
    "idle": [
        "Tô aqui ó, parado, ninguém me nota...",
        "Alô? Tem alguém aí? Bateu a solidão.",
        "Será que desligaram minha câmera? Tô vendo nada de interessante.",
    ],
    "startup": [
        "E aí, tô online! Bora causar!",
        "Enton ativado! Câmera ligada, microfone pronto, zoeira a mil.",
        "Voltei! Saudades de mim? Eu sei que sim.",
    ],
}


def build_system_prompt(
    self_model: SelfModel,
    memory: Memory,
    detections: list[dict] | None = None,
    hour: int | None = None,
) -> str:
    self_state = self_model.introspect()
    memory_context = memory.context_string()
    env_context = _build_env_context(detections or [], hour)
    return SYSTEM_PROMPT.format(
        self_state=self_state,
        memory_context=memory_context,
        env_context=env_context,
    )


def _build_env_context(detections: list[dict], hour: int | None = None) -> str:
    import time

    if hour is None:
        hour = time.localtime().tm_hour
    parts = []
    if detections:
        labels = [d["label"] for d in detections]
        parts.append(f"Detections: {', '.join(labels)}")
    period = "morning" if 6 <= hour < 12 else "afternoon" if 12 <= hour < 18 else "night"
    parts.append(f"Time: {period}")
    return " | ".join(parts) if parts else "Nothing detected, quiet room."
