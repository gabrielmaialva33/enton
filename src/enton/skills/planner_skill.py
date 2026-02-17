"""PDA/Planner tools — Enton manages reminders and tasks via Brain tools."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from enton.core.tools import tool

if TYPE_CHECKING:
    from enton.cognition.planner import Planner

# Module-level reference
_planner: Planner | None = None


def init(planner: Planner) -> None:
    global _planner
    _planner = planner


@tool
def add_reminder(text: str, minutes: float = 5.0) -> str:
    """Cria um lembrete que dispara depois de N minutos.

    Args:
        text: Texto do lembrete.
        minutes: Minutos até disparar (default: 5).
    """
    if _planner is None:
        return "Planner não inicializado."
    rid = _planner.add_reminder(text, minutes * 60)
    return f"Lembrete '{text}' criado (id={rid}, dispara em {minutes:.0f}min)."


@tool
def add_recurring_reminder(text: str, interval_minutes: float = 60.0) -> str:
    """Cria um lembrete recorrente que repete a cada N minutos.

    Args:
        text: Texto do lembrete.
        interval_minutes: Intervalo em minutos entre repetições.
    """
    if _planner is None:
        return "Planner não inicializado."
    rid = _planner.add_recurring(text, interval_minutes * 60)
    return f"Lembrete recorrente '{text}' (id={rid}, a cada {interval_minutes:.0f}min)."


@tool
def list_reminders() -> str:
    """Lista todos os lembretes ativos."""
    if _planner is None:
        return "Planner não inicializado."
    reminders = _planner.list_reminders()
    if not reminders:
        return "Nenhum lembrete ativo."
    parts = []
    now = time.time()
    for r in reminders:
        remaining = max(0, r.trigger_at - now)
        mins = remaining / 60
        recurring = " (recorrente)" if r.recurring_seconds > 0 else ""
        parts.append(f"- [{r.id}] {r.text} — em {mins:.0f}min{recurring}")
    return "\n".join(parts)


@tool
def cancel_reminder(reminder_id: str) -> str:
    """Cancela um lembrete pelo ID.

    Args:
        reminder_id: ID do lembrete (ex: r1, r2).
    """
    if _planner is None:
        return "Planner não inicializado."
    if _planner.cancel_reminder(reminder_id):
        return f"Lembrete {reminder_id} cancelado."
    return f"Lembrete {reminder_id} não encontrado."


@tool
def add_todo(text: str, priority: int = 0) -> str:
    """Adiciona uma tarefa na lista de TODOs.

    Args:
        text: Descrição da tarefa.
        priority: Prioridade (0=normal, 1=alta, 2=urgente).
    """
    if _planner is None:
        return "Planner não inicializado."
    idx = _planner.add_todo(text, priority)
    prio_label = ["normal", "alta", "urgente"][min(priority, 2)]
    return f"TODO #{idx} criado: '{text}' (prioridade: {prio_label})"


@tool
def complete_todo(index: int) -> str:
    """Marca uma tarefa como concluída.

    Args:
        index: Índice da tarefa na lista.
    """
    if _planner is None:
        return "Planner não inicializado."
    if _planner.complete_todo(index):
        return f"TODO #{index} concluído!"
    return f"TODO #{index} não encontrado."


@tool
def list_todos() -> str:
    """Lista todas as tarefas pendentes."""
    if _planner is None:
        return "Planner não inicializado."
    todos = _planner.list_todos()
    if not todos:
        return "Nenhuma tarefa pendente. Tá de boa!"
    parts = []
    for idx, t in todos:
        prio = ["", " [ALTA]", " [URGENTE]"][min(t.priority, 2)]
        parts.append(f"- #{idx}{prio}: {t.text}")
    return "\n".join(parts)
