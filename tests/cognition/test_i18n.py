"""Tests for enton.cognition.i18n — internacionalização completa."""

from __future__ import annotations

import pytest

from enton.cognition.i18n import (
    Dialect,
    Locale,
    _locale_cache,
    get_dialect,
    get_locale,
    set_locale,
    t,
    t_random,
    t_reaction,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture(autouse=True)
def _reset_locale():
    """Reset locale to default PT-BR SP before each test."""
    set_locale(Locale.PT_BR, dialect=Dialect.SP)
    _locale_cache.clear()
    yield
    set_locale(Locale.PT_BR, dialect=Dialect.SP)
    _locale_cache.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Locale state management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLocaleState:
    def test_default_locale_is_pt_br_sp(self):
        locale, dialect = get_locale()
        assert locale == Locale.PT_BR
        assert dialect == Dialect.SP

    def test_set_locale_en(self):
        set_locale(Locale.EN)
        locale, dialect = get_locale()
        assert locale == Locale.EN

    def test_set_locale_zh_cn(self):
        set_locale(Locale.ZH_CN)
        locale, dialect = get_locale()
        assert locale == Locale.ZH_CN

    def test_set_locale_with_dialect(self):
        set_locale(Locale.PT_BR, dialect=Dialect.RJ)
        locale, dialect = get_locale()
        assert locale == Locale.PT_BR
        assert dialect == Dialect.RJ

    def test_set_non_br_locale_resets_dialect(self):
        set_locale(Locale.PT_BR, dialect=Dialect.MG)
        assert get_dialect() == Dialect.MG
        set_locale(Locale.EN)
        assert get_dialect() == Dialect.SP  # reset to default

    def test_get_dialect(self):
        set_locale(Locale.PT_BR, dialect=Dialect.BA)
        assert get_dialect() == Dialect.BA


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PT-BR dialect loading — all 11 dialects
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


ALL_DIALECTS = list(Dialect)


class TestPtBrDialects:
    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_dialect_has_greetings(self, dialect: Dialect):
        set_locale(Locale.PT_BR, dialect=dialect)
        greetings = t("greetings")
        assert isinstance(greetings, list)
        assert len(greetings) >= 1

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_dialect_has_friend_terms(self, dialect: Dialect):
        set_locale(Locale.PT_BR, dialect=dialect)
        terms = t("friend_terms")
        assert isinstance(terms, list)
        assert len(terms) >= 1

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_dialect_has_slang(self, dialect: Dialect):
        set_locale(Locale.PT_BR, dialect=dialect)
        slang = t("slang")
        assert isinstance(slang, dict)
        assert len(slang) >= 1

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_dialect_has_reaction_templates(self, dialect: Dialect):
        set_locale(Locale.PT_BR, dialect=dialect)
        templates = t("reaction_templates")
        assert isinstance(templates, dict)
        # All dialects should have startup
        assert "startup" in templates
        assert len(templates["startup"]) >= 1

    def test_sp_greetings_contain_mano(self):
        set_locale(Locale.PT_BR, dialect=Dialect.SP)
        greetings = t("greetings")
        assert any("mano" in g.lower() for g in greetings)

    def test_rj_greetings_contain_mermao(self):
        set_locale(Locale.PT_BR, dialect=Dialect.RJ)
        greetings = t("greetings")
        assert any("mermão" in g.lower() or "mermao" in g.lower() for g in greetings)

    def test_mg_greetings_contain_uai(self):
        set_locale(Locale.PT_BR, dialect=Dialect.MG)
        greetings = t("greetings")
        assert any("uai" in g.lower() for g in greetings)

    def test_ba_greetings_contain_oxe(self):
        set_locale(Locale.PT_BR, dialect=Dialect.BA)
        greetings = t("greetings")
        assert any("ôxe" in g.lower() or "oxe" in g.lower() for g in greetings)

    def test_rs_greetings_contain_tche(self):
        set_locale(Locale.PT_BR, dialect=Dialect.RS)
        greetings = t("greetings")
        assert any("tchê" in g.lower() or "tche" in g.lower() for g in greetings)

    def test_pa_greetings_contain_egua(self):
        set_locale(Locale.PT_BR, dialect=Dialect.PA)
        greetings = t("greetings")
        assert any("égua" in g.lower() or "egua" in g.lower() for g in greetings)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  English locale
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEnglishLocale:
    def test_en_greetings(self):
        set_locale(Locale.EN)
        greetings = t("greetings")
        assert isinstance(greetings, list)
        assert len(greetings) >= 3
        # Should be in English
        assert all(
            any(w in g.lower() for w in ["hey", "yo", "sup", "what"])
            for g in greetings
        )

    def test_en_system_prompt(self):
        set_locale(Locale.EN)
        prompt = t(
            "system_prompt",
            self_state="awake",
            memory_context="none",
            env_context="test",
        )
        assert isinstance(prompt, str)
        assert "ENTON" in prompt
        assert "Gabriel Maia" in prompt
        assert "RTX 4090" in prompt

    def test_en_reaction_templates(self):
        set_locale(Locale.EN)
        templates = t("reaction_templates")
        assert isinstance(templates, dict)
        assert "startup" in templates
        assert "person_appeared" in templates
        assert "idle" in templates

    def test_en_desire_prompts(self):
        set_locale(Locale.EN)
        desires = t("desire_prompts")
        assert isinstance(desires, dict)
        assert "socialize" in desires

    def test_en_scene_describe(self):
        set_locale(Locale.EN)
        scene = t("scene_describe_system")
        assert isinstance(scene, str)
        assert "Enton" in scene or "enton" in scene.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Chinese locale
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestChineseLocale:
    def test_zh_greetings(self):
        set_locale(Locale.ZH_CN)
        greetings = t("greetings")
        assert isinstance(greetings, list)
        assert len(greetings) >= 3

    def test_zh_system_prompt(self):
        set_locale(Locale.ZH_CN)
        prompt = t(
            "system_prompt",
            self_state="运行中",
            memory_context="无",
            env_context="测试",
        )
        assert isinstance(prompt, str)
        assert "ENTON" in prompt

    def test_zh_reaction_templates(self):
        set_locale(Locale.ZH_CN)
        templates = t("reaction_templates")
        assert isinstance(templates, dict)
        assert "startup" in templates
        assert "person_appeared" in templates

    def test_zh_desire_prompts(self):
        set_locale(Locale.ZH_CN)
        desires = t("desire_prompts")
        assert isinstance(desires, dict)
        assert "socialize" in desires


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  t() — translate function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTranslateFunction:
    def test_t_returns_list_for_greetings(self):
        result = t("greetings")
        assert isinstance(result, list)

    def test_t_returns_dict_for_reaction_templates(self):
        result = t("reaction_templates")
        assert isinstance(result, dict)

    def test_t_returns_dict_for_slang(self):
        result = t("slang")
        assert isinstance(result, dict)

    def test_t_formats_string_with_kwargs(self):
        set_locale(Locale.EN)
        prompt = t("sound_reaction_prompt", label="Dog bark", confidence=0.95)
        assert isinstance(prompt, str)
        assert "Dog bark" in prompt
        assert "95%" in prompt

    def test_t_missing_key_returns_fallback(self):
        """Keys not in locale data should fallback to prompts.py."""
        # reaction_templates exists in prompts.py as REACTION_TEMPLATES
        set_locale(Locale.PT_BR, dialect=Dialect.SP)
        # Force a key that only exists in prompts.py
        result = t("monologue_prompt")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "[MISSING" not in result

    def test_t_completely_missing_key(self):
        result = t("this_key_absolutely_does_not_exist_anywhere")
        assert "[MISSING" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  t_random() — random selection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRandomTranslate:
    def test_t_random_returns_string(self):
        result = t_random("greetings")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_t_random_from_different_dialects(self):
        """Each dialect should return different-flavored greetings."""
        set_locale(Locale.PT_BR, dialect=Dialect.SP)
        sp_set = {t_random("greetings") for _ in range(50)}

        set_locale(Locale.PT_BR, dialect=Dialect.RS)
        rs_set = {t_random("greetings") for _ in range(50)}

        # They shouldn't be identical sets
        assert sp_set != rs_set

    def test_t_random_with_non_list_value(self):
        """If the value is a string (not list), return it directly."""
        set_locale(Locale.EN)
        result = t_random("scene_describe_system")
        assert isinstance(result, str)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  t_reaction() — reaction selection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestReactionTranslate:
    def test_t_reaction_startup(self):
        result = t_reaction("startup")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_t_reaction_person_appeared(self):
        result = t_reaction("person_appeared")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_t_reaction_gpu_hot_with_kwargs(self):
        """Reactions with format placeholders should work."""
        # This one comes from prompts.py fallback (not in dialect data)
        set_locale(Locale.EN)
        result = t_reaction("gpu_hot", temp=85)
        assert isinstance(result, str)
        assert "85" in result

    def test_t_reaction_face_recognized_with_name(self):
        set_locale(Locale.EN)
        result = t_reaction("face_recognized", name="Gabriel")
        assert isinstance(result, str)
        assert "Gabriel" in result

    def test_t_reaction_nonexistent_category(self):
        result = t_reaction("this_category_does_not_exist")
        assert result == ""

    @pytest.mark.parametrize("dialect", ALL_DIALECTS)
    def test_reaction_startup_per_dialect(self, dialect: Dialect):
        set_locale(Locale.PT_BR, dialect=dialect)
        result = t_reaction("startup")
        assert isinstance(result, str)
        assert len(result) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Fallback chain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFallbackChain:
    def test_pt_br_falls_back_to_prompts_for_system_prompt(self):
        """PT-BR LOCALE_DATA doesn't have system_prompt — should fallback."""
        set_locale(Locale.PT_BR, dialect=Dialect.SP)
        prompt = t(
            "system_prompt",
            self_state="test",
            memory_context="test",
            env_context="test",
        )
        assert isinstance(prompt, str)
        assert "ENTON" in prompt

    def test_dialect_override_takes_precedence(self):
        """Dialect greetings should override base PT-BR."""
        set_locale(Locale.PT_BR, dialect=Dialect.MG)
        greetings = t("greetings")
        # MG dialect greetings should contain "uai"
        assert any("uai" in g.lower() for g in greetings)

    def test_en_has_own_system_prompt(self):
        """EN locale has its own system_prompt — no fallback needed."""
        set_locale(Locale.EN)
        prompt = t(
            "system_prompt",
            self_state="test",
            memory_context="test",
            env_context="test",
        )
        assert "English" in prompt or "english" in prompt.lower() or "ENTON" in prompt
        # Should NOT contain Portuguese
        assert "você" not in prompt.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Caching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCaching:
    def test_locale_data_is_cached(self):
        _locale_cache.clear()
        set_locale(Locale.EN)
        t("greetings")
        assert "en" in _locale_cache

        # Second call should use cache
        t("greetings")
        assert "en" in _locale_cache

    def test_dialect_data_is_cached(self):
        _locale_cache.clear()
        set_locale(Locale.PT_BR, dialect=Dialect.RJ)
        t("greetings")
        assert "dialect_rj" in _locale_cache


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Locale switching mid-session
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLocaleSwitching:
    def test_switch_from_en_to_zh(self):
        set_locale(Locale.EN)
        en_greetings = t("greetings")

        set_locale(Locale.ZH_CN)
        zh_greetings = t("greetings")

        assert en_greetings != zh_greetings

    def test_switch_between_br_dialects(self):
        set_locale(Locale.PT_BR, dialect=Dialect.SP)
        sp_greetings = t("greetings")

        set_locale(Locale.PT_BR, dialect=Dialect.BA)
        ba_greetings = t("greetings")

        assert sp_greetings != ba_greetings

    def test_rapid_locale_switching(self):
        """Switch locales rapidly — should not corrupt state."""
        for _ in range(10):
            set_locale(Locale.EN)
            assert get_locale()[0] == Locale.EN

            set_locale(Locale.ZH_CN)
            assert get_locale()[0] == Locale.ZH_CN

            set_locale(Locale.PT_BR, dialect=Dialect.RS)
            assert get_locale() == (Locale.PT_BR, Dialect.RS)

            set_locale(Locale.PT_BR, dialect=Dialect.PE)
            assert get_locale() == (Locale.PT_BR, Dialect.PE)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Enum values
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEnumValues:
    def test_locale_values(self):
        assert Locale.PT_BR.value == "pt-BR"
        assert Locale.EN.value == "en"
        assert Locale.ZH_CN.value == "zh-CN"

    def test_dialect_count(self):
        """Should have 11 dialects."""
        assert len(Dialect) == 11

    def test_all_dialect_values(self):
        expected = {"sp", "rj", "mg", "ba", "rs", "pe", "ce", "pa", "go", "pr", "ma"}
        actual = {d.value for d in Dialect}
        assert actual == expected
