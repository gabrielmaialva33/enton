"""i18n — Internacionalização do Enton.

Sistema de locales com suporte a:
- PT-BR nativo com dialetos regionais (SP, RJ, MG, BA, RS, PE, CE, PA, GO, PR, MA)
- EN e ZH-CN para mercado internacional
- Runtime locale switching via set_locale()

Arquitetura:
  prompts.py = source of truth (PT-BR SP default)
  i18n/       = overrides por locale e dialeto regional

Uso:
  from enton.cognition.i18n import set_locale, t, get_dialect, Locale, Dialect

  set_locale(Locale.PT_BR, dialect=Dialect.RJ)
  greeting = t("greetings")  # retorna lista de cumprimentos cariocas

  set_locale(Locale.EN)
  greeting = t("greetings")  # retorna lista de greetings em EN
"""

from __future__ import annotations

import random
from enum import StrEnum
from typing import Any

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Enums
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class Locale(StrEnum):
    """Idiomas suportados."""

    PT_BR = "pt-BR"
    EN = "en"
    ZH_CN = "zh-CN"


class Dialect(StrEnum):
    """Dialetos regionais brasileiros."""

    SP = "sp"  # São Paulo (default)
    RJ = "rj"  # Rio de Janeiro
    MG = "mg"  # Minas Gerais
    BA = "ba"  # Bahia / Nordeste
    RS = "rs"  # Rio Grande do Sul
    PE = "pe"  # Pernambuco
    CE = "ce"  # Ceará
    PA = "pa"  # Pará
    GO = "go"  # Goiás
    PR = "pr"  # Paraná
    MA = "ma"  # Maranhão


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  State — locale global do Enton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_current_locale: Locale = Locale.PT_BR
_current_dialect: Dialect = Dialect.SP

# Lazy-loaded locale data
_locale_cache: dict[str, dict[str, Any]] = {}


def set_locale(locale: Locale, dialect: Dialect | None = None) -> None:
    """Define o locale ativo do Enton."""
    global _current_locale, _current_dialect
    _current_locale = locale
    if dialect is not None:
        _current_dialect = dialect
    elif locale != Locale.PT_BR:
        _current_dialect = Dialect.SP  # reset dialect for non-BR


def get_locale() -> tuple[Locale, Dialect]:
    """Retorna (locale, dialect) atual."""
    return _current_locale, _current_dialect


def get_dialect() -> Dialect:
    """Retorna o dialeto BR ativo."""
    return _current_dialect


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Locale data loaders (lazy)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _load_locale(locale: Locale) -> dict[str, Any]:
    """Carrega dados de um locale (com cache)."""
    key = locale.value
    if key in _locale_cache:
        return _locale_cache[key]

    if locale == Locale.PT_BR:
        from enton.cognition.i18n.pt_br import LOCALE_DATA
    elif locale == Locale.EN:
        from enton.cognition.i18n.en import LOCALE_DATA
    elif locale == Locale.ZH_CN:
        from enton.cognition.i18n.zh import LOCALE_DATA
    else:
        from enton.cognition.i18n.pt_br import LOCALE_DATA

    _locale_cache[key] = LOCALE_DATA
    return LOCALE_DATA


def _load_dialect(dialect: Dialect) -> dict[str, Any]:
    """Carrega dados de dialeto BR."""
    key = f"dialect_{dialect.value}"
    if key in _locale_cache:
        return _locale_cache[key]

    from enton.cognition.i18n.pt_br import DIALECTS

    data = DIALECTS.get(dialect.value, {})
    _locale_cache[key] = data
    return data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  API pública — t() é o translate do Enton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def t(key: str, **kwargs: Any) -> Any:
    """Retorna o valor traduzido para o locale/dialeto ativo.

    Para strings com placeholders, passa kwargs pro .format().
    Para listas, retorna a lista completa.
    Para dicts, retorna o dict completo.

    Fallback: PT-BR SP (prompts.py) se a key não existir no locale.

    Exemplos:
        t("greetings")  # list[str]
        t("system_prompt", self_state="...", ...)  # str formatada
        t("reaction_templates")  # dict[str, list[str]]
    """
    locale_data = _load_locale(_current_locale)

    # Para PT-BR, checar dialect override primeiro
    if _current_locale == Locale.PT_BR:
        dialect_data = _load_dialect(_current_dialect)
        value = dialect_data.get(key)
        if value is not None:
            return _format_value(value, kwargs)

    # Locale base
    value = locale_data.get(key)
    if value is not None:
        return _format_value(value, kwargs)

    # Fallback: importar direto do prompts.py
    return _fallback(key, kwargs)


def t_random(key: str, **kwargs: Any) -> str:
    """Retorna um item aleatório de uma lista traduzida.

    Útil pra reactions, greetings, etc.

    Exemplo:
        t_random("greetings")  # "Eae mano!"
    """
    value = t(key, **kwargs)
    if isinstance(value, list) and value:
        chosen = random.choice(value)
        if kwargs and isinstance(chosen, str):
            return chosen.format(**kwargs)
        return chosen
    if isinstance(value, str):
        return value
    return str(value)


def t_reaction(category: str, **kwargs: Any) -> str:
    """Retorna uma reação aleatória de uma categoria.

    Exemplo:
        t_reaction("person_appeared")
        t_reaction("gpu_hot", temp=85)
    """
    templates = t("reaction_templates")
    if isinstance(templates, dict) and category in templates:
        choices = templates[category]
        if choices:
            chosen = random.choice(choices)
            if kwargs:
                return chosen.format(**kwargs)
            return chosen
    return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers internos
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _format_value(value: Any, kwargs: dict[str, Any]) -> Any:
    """Aplica .format() se for string e tiver kwargs."""
    if isinstance(value, str) and kwargs:
        try:
            return value.format(**kwargs)
        except (KeyError, IndexError):
            return value
    return value


def _fallback(key: str, kwargs: dict[str, Any]) -> Any:
    """Fallback pro prompts.py original."""
    from enton.cognition import prompts

    # Mapeia keys i18n → constantes do prompts.py
    _KEY_MAP = {
        "system_prompt": "SYSTEM_PROMPT",
        "monologue_prompt": "MONOLOGUE_PROMPT",
        "reaction_templates": "REACTION_TEMPLATES",
        "empathy_tones": "EMPATHY_TONES",
        "desire_prompts": "DESIRE_PROMPTS",
        "urgent_sound_reactions": "URGENT_SOUND_REACTIONS",
        "scene_describe_system": "SCENE_DESCRIBE_SYSTEM",
        "sound_reaction_prompt": "SOUND_REACTION_PROMPT",
        "channel_message_system": "CHANNEL_MESSAGE_SYSTEM",
        "consciousness_learn_vocalize": "CONSCIOUSNESS_LEARN_VOCALIZE",
    }

    attr_name = _KEY_MAP.get(key, key.upper())
    value = getattr(prompts, attr_name, None)
    if value is not None:
        return _format_value(value, kwargs)
    return f"[MISSING: {key}]"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Exports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__all__ = [
    "Dialect",
    "Locale",
    "get_dialect",
    "get_locale",
    "set_locale",
    "t",
    "t_random",
    "t_reaction",
]
