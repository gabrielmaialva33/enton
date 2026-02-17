"""Cognition — reasoning, personality, and context fusion."""

from enton.cognition.brain import EntonBrain
from enton.cognition.fuser import Fuser
from enton.cognition.persona import REACTION_TEMPLATES, build_system_prompt

__all__ = [
    "REACTION_TEMPLATES",
    "EntonBrain",
    "Fuser",
    "build_system_prompt",
]
