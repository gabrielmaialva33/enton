"""Cognition — reasoning, personality, and context fusion."""

from enton.cognition.brain import Brain
from enton.cognition.fuser import Fuser
from enton.cognition.persona import REACTION_TEMPLATES, build_system_prompt

__all__ = [
    "Brain",
    "Fuser",
    "REACTION_TEMPLATES",
    "build_system_prompt",
]
