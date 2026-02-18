"""Tests for Enton's Prompts — validation of all prompt constants."""

import re

import pytest

from enton.cognition.prompts import (
    AGENTIC_TOOL_PROMPT,
    CHANNEL_MESSAGE_SYSTEM,
    CONSCIOUSNESS_LEARN_VOCALIZE,
    DESCRIBE_TOOL_DEFAULT,
    DESCRIBE_TOOL_SYSTEM,
    DESIRE_CREATE_PROMPT,
    DESIRE_CREATE_SYSTEM,
    DESIRE_EXPLORE_PROMPT,
    DESIRE_EXPLORE_SYSTEM,
    DESIRE_LEARN_PROMPT,
    DESIRE_LEARN_SYSTEM,
    DESIRE_OBSERVE_SYSTEM,
    DESIRE_OPTIMIZE_PROMPT,
    DESIRE_PLAY_PROMPT,
    DESIRE_PLAY_SYSTEM,
    DESIRE_PROMPTS,
    DREAM_CONSOLIDATE_PROMPT,
    DREAM_CONSOLIDATE_SYSTEM,
    DREAM_PROFILE_PROMPT,
    DREAM_PROFILE_SYSTEM,
    EMPATHY_TONES,
    ERROR_HINTS,
    ERROR_LOOPBACK_PROMPT,
    FORGE_CORRECTION_PROMPT,
    FORGE_SYSTEM_PROMPT,
    KNOWLEDGE_EXTRACT_PROMPT,
    KNOWLEDGE_EXTRACT_SYSTEM,
    MONOLOGUE_PROMPT,
    REACTION_TEMPLATES,
    SCENE_DESCRIBE_SYSTEM,
    SCENE_FALLBACK_PROMPT,
    SCENE_FALLBACK_SYSTEM,
    SOUND_REACTION_PROMPT,
    SOUND_REACTION_SYSTEM,
    SUBAGENT_CODING,
    SUBAGENT_PROMPTS,
    SUBAGENT_RESEARCH,
    SUBAGENT_SYSTEM,
    SUBAGENT_VISION,
    SYSTEM_PROMPT,
    URGENT_SOUND_REACTIONS,
)

# ---------------------------------------------------------------------------
#  All exported string constants (used for bulk checks)
# ---------------------------------------------------------------------------

ALL_STRING_CONSTANTS = [
    SYSTEM_PROMPT,
    MONOLOGUE_PROMPT,
    DESIRE_OBSERVE_SYSTEM,
    DESIRE_LEARN_PROMPT,
    DESIRE_LEARN_SYSTEM,
    DESIRE_CREATE_PROMPT,
    DESIRE_CREATE_SYSTEM,
    DESIRE_EXPLORE_PROMPT,
    DESIRE_EXPLORE_SYSTEM,
    DESIRE_PLAY_PROMPT,
    DESIRE_PLAY_SYSTEM,
    DESIRE_OPTIMIZE_PROMPT,
    SCENE_DESCRIBE_SYSTEM,
    SCENE_FALLBACK_SYSTEM,
    SCENE_FALLBACK_PROMPT,
    DESCRIBE_TOOL_SYSTEM,
    DESCRIBE_TOOL_DEFAULT,
    SOUND_REACTION_PROMPT,
    SOUND_REACTION_SYSTEM,
    CHANNEL_MESSAGE_SYSTEM,
    DREAM_CONSOLIDATE_PROMPT,
    DREAM_CONSOLIDATE_SYSTEM,
    DREAM_PROFILE_PROMPT,
    DREAM_PROFILE_SYSTEM,
    SUBAGENT_VISION,
    SUBAGENT_CODING,
    SUBAGENT_RESEARCH,
    SUBAGENT_SYSTEM,
    ERROR_LOOPBACK_PROMPT,
    FORGE_SYSTEM_PROMPT,
    FORGE_CORRECTION_PROMPT,
    KNOWLEDGE_EXTRACT_PROMPT,
    KNOWLEDGE_EXTRACT_SYSTEM,
    AGENTIC_TOOL_PROMPT,
    CONSCIOUSNESS_LEARN_VOCALIZE,
]


# ---------------------------------------------------------------------------
#  I. Type and existence checks
# ---------------------------------------------------------------------------


class TestExportedTypes:
    def test_system_prompt_is_str(self):
        assert isinstance(SYSTEM_PROMPT, str)

    def test_monologue_prompt_is_str(self):
        assert isinstance(MONOLOGUE_PROMPT, str)

    def test_reaction_templates_is_dict(self):
        assert isinstance(REACTION_TEMPLATES, dict)

    def test_empathy_tones_is_dict(self):
        assert isinstance(EMPATHY_TONES, dict)

    def test_desire_prompts_is_dict(self):
        assert isinstance(DESIRE_PROMPTS, dict)

    def test_urgent_sound_reactions_is_dict(self):
        assert isinstance(URGENT_SOUND_REACTIONS, dict)

    def test_error_hints_is_dict(self):
        assert isinstance(ERROR_HINTS, dict)

    def test_subagent_prompts_is_dict(self):
        assert isinstance(SUBAGENT_PROMPTS, dict)


# ---------------------------------------------------------------------------
#  II. SYSTEM_PROMPT placeholders
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_contains_self_state_placeholder(self):
        assert "{self_state}" in SYSTEM_PROMPT

    def test_contains_memory_context_placeholder(self):
        assert "{memory_context}" in SYSTEM_PROMPT

    def test_contains_env_context_placeholder(self):
        assert "{env_context}" in SYSTEM_PROMPT

    def test_format_works_with_required_placeholders(self):
        result = SYSTEM_PROMPT.format(
            self_state="happy",
            memory_context="saw a cat",
            env_context="GPU 45C",
        )
        assert "happy" in result
        assert "saw a cat" in result
        assert "GPU 45C" in result


# ---------------------------------------------------------------------------
#  III. MONOLOGUE_PROMPT placeholders
# ---------------------------------------------------------------------------


class TestMonologuePrompt:
    REQUIRED_PLACEHOLDERS = [
        "vision_summary",
        "system_summary",
        "last_interaction",
        "idle_time",
        "current_mood",
        "energy",
        "desires",
    ]

    @pytest.mark.parametrize("placeholder", REQUIRED_PLACEHOLDERS)
    def test_contains_required_placeholder(self, placeholder):
        assert f"{{{placeholder}}}" in MONOLOGUE_PROMPT


# ---------------------------------------------------------------------------
#  IV. REACTION_TEMPLATES
# ---------------------------------------------------------------------------


class TestReactionTemplates:
    REQUIRED_CATEGORIES = [
        "person_appeared",
        "person_left",
        "cat_detected",
        "idle",
        "startup",
        "face_recognized",
        "doorbell",
        "alarm",
        "tool_executed",
        "coding_late",
        "gpu_hot",
        "bad_commit",
    ]

    @pytest.mark.parametrize("category", REQUIRED_CATEGORIES)
    def test_has_required_category(self, category):
        assert category in REACTION_TEMPLATES, f"REACTION_TEMPLATES missing category: {category}"

    @pytest.mark.parametrize("category", REQUIRED_CATEGORIES)
    def test_category_is_nonempty_list(self, category):
        templates = REACTION_TEMPLATES[category]
        assert isinstance(templates, list)
        assert len(templates) > 0, f"Category '{category}' has no templates"

    def test_all_templates_are_strings(self):
        for category, templates in REACTION_TEMPLATES.items():
            for t in templates:
                assert isinstance(t, str), f"Non-string template in '{category}': {t!r}"


# ---------------------------------------------------------------------------
#  V. EMPATHY_TONES
# ---------------------------------------------------------------------------


class TestEmpathyTones:
    def test_has_at_least_3_entries(self):
        assert len(EMPATHY_TONES) >= 3

    def test_all_values_are_strings(self):
        for tone, desc in EMPATHY_TONES.items():
            assert isinstance(tone, str)
            assert isinstance(desc, str)
            assert len(desc) > 0


# ---------------------------------------------------------------------------
#  VI. DESIRE_PROMPTS — all 10 desire categories (9 in current source)
# ---------------------------------------------------------------------------


class TestDesirePrompts:
    EXPECTED_DESIRES = [
        "socialize",
        "observe",
        "learn",
        "check_on_user",
        "optimize",
        "reminisce",
        "create",
        "explore",
        "play",
    ]

    @pytest.mark.parametrize("desire", EXPECTED_DESIRES)
    def test_has_desire_category(self, desire):
        assert desire in DESIRE_PROMPTS, f"DESIRE_PROMPTS missing desire: {desire}"

    @pytest.mark.parametrize("desire", EXPECTED_DESIRES)
    def test_desire_is_nonempty_list(self, desire):
        prompts = DESIRE_PROMPTS[desire]
        assert isinstance(prompts, list)
        assert len(prompts) > 0

    def test_all_desire_prompts_are_strings(self):
        for _desire, prompts in DESIRE_PROMPTS.items():
            for p in prompts:
                assert isinstance(p, str)
                assert len(p) > 0


# ---------------------------------------------------------------------------
#  VII. Format placeholder smoke tests
# ---------------------------------------------------------------------------


class TestFormatPlaceholders:
    def test_gpu_hot_with_temp(self):
        for template in REACTION_TEMPLATES["gpu_hot"]:
            result = template.format(temp=85)
            assert "85" in result

    def test_face_recognized_with_name(self):
        for template in REACTION_TEMPLATES["face_recognized"]:
            result = template.format(name="Gabriel")
            assert "Gabriel" in result

    def test_coding_late_with_hour_and_name(self):
        for template in REACTION_TEMPLATES["coding_late"]:
            result = template.format(hour=3, name="Gabriel")
            # At least one of the placeholders should appear
            assert "3" in result or "Gabriel" in result

    def test_scene_fallback_prompt_format(self):
        result = SCENE_FALLBACK_PROMPT.format(scene_desc="a dark room")
        assert "a dark room" in result

    def test_consciousness_learn_vocalize_format(self):
        result = CONSCIOUSNESS_LEARN_VOCALIZE.format(topic="quantum computing")
        assert "quantum computing" in result

    def test_error_loopback_prompt_format(self):
        result = ERROR_LOOPBACK_PROMPT.format(
            error_type="TimeoutError",
            error_message="request timed out",
            provider="openai",
            attempt=2,
            max_attempts=3,
            original_prompt="search the web",
            context_hint="Try a simpler approach.",
        )
        assert "TimeoutError" in result
        assert "2" in result

    def test_forge_correction_prompt_format(self):
        result = FORGE_CORRECTION_PROMPT.format(
            task="build a timer",
            code="def timer(): pass",
            test_code="assert timer() is None",
            error="AssertionError",
        )
        assert "build a timer" in result


# ---------------------------------------------------------------------------
#  VIII. Bulk non-empty and no TODO/FIXME checks
# ---------------------------------------------------------------------------


class TestBulkValidation:
    # Match standalone TODO/FIXME markers (not substrings like "TODOS")
    _TODO_RE = re.compile(r"\bTODO\b", re.IGNORECASE)
    _FIXME_RE = re.compile(r"\bFIXME\b", re.IGNORECASE)

    @pytest.mark.parametrize("prompt", ALL_STRING_CONSTANTS)
    def test_prompt_is_nonempty_string(self, prompt):
        assert isinstance(prompt, str)
        assert len(prompt.strip()) > 0

    @pytest.mark.parametrize("prompt", ALL_STRING_CONSTANTS)
    def test_no_todo_or_fixme(self, prompt):
        assert not self._TODO_RE.search(prompt), f"Found TODO marker in prompt: {prompt[:80]}..."
        assert not self._FIXME_RE.search(prompt), f"Found FIXME marker in prompt: {prompt[:80]}..."

    def test_reaction_templates_no_todo(self):
        for category, templates in REACTION_TEMPLATES.items():
            for t in templates:
                assert not self._TODO_RE.search(t), f"Found TODO in '{category}': {t[:60]}"
                assert not self._FIXME_RE.search(t), f"Found FIXME in '{category}': {t[:60]}"

    def test_desire_prompts_no_todo(self):
        for desire, prompts in DESIRE_PROMPTS.items():
            for p in prompts:
                assert not self._TODO_RE.search(p), f"Found TODO in desire '{desire}': {p[:60]}"
                assert not self._FIXME_RE.search(p), f"Found FIXME in desire '{desire}': {p[:60]}"

    def test_empathy_tones_no_todo(self):
        for tone, desc in EMPATHY_TONES.items():
            assert not self._TODO_RE.search(desc), f"Found TODO in tone '{tone}': {desc[:60]}"
            assert not self._FIXME_RE.search(desc), f"Found FIXME in tone '{tone}': {desc[:60]}"
