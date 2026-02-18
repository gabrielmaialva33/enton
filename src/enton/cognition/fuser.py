"""Fuser — merges all perception streams into a unified natural-language context.

Inspired by OM1's Natural Language Data Bus (NLDB). Instead of passing raw
detection/activity/emotion data to the Brain, the Fuser produces a human-readable
paragraph that captures the full scene, making the LLM's job easier and responses
more contextual.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enton.core.events import ActivityEvent, DetectionEvent, EmotionEvent


class Fuser:
    """Fuses perception events into a coherent scene description."""

    def fuse(
        self,
        detections: list[DetectionEvent],
        activities: list[ActivityEvent],
        emotions: list[EmotionEvent],
    ) -> str:
        """Produce a natural-language summary of the current scene."""
        parts: list[str] = []

        # --- object summary ---
        counts: dict[str, int] = {}
        for d in detections:
            counts[d.label] = counts.get(d.label, 0) + 1
        if counts:
            obj_parts = []
            for label, n in sorted(counts.items(), key=lambda x: -x[1]):
                if n > 1:
                    obj_parts.append(f"{n}x {label}")
                else:
                    obj_parts.append(label)
            parts.append(f"Objetos detectados: {', '.join(obj_parts)}.")

        # --- person details (activity + emotion) ---
        n_persons = counts.get("person", 0)
        if n_persons:
            person_descs: list[str] = []
            for i in range(n_persons):
                desc_parts: list[str] = []
                # activity
                matching_act = [a for a in activities if a.person_index == i]
                if matching_act:
                    desc_parts.append(matching_act[0].activity.lower())
                # emotion
                matching_emo = [e for e in emotions if e.person_index == i]
                if matching_emo:
                    emo = matching_emo[0]
                    desc_parts.append(f"parecendo {emo.emotion.lower()} ({emo.score:.0%})")
                if desc_parts:
                    person_descs.append(f"Pessoa {i + 1}: {', '.join(desc_parts)}")
            if person_descs:
                parts.append(" | ".join(person_descs) + ".")

        if not parts:
            parts.append("Nenhum objeto detectado. Ambiente calmo.")

        return " ".join(parts)
