"""PDA/Planner toolkit -- Agno-compatible Enton reminder and task tools."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from agno.tools import Toolkit

if TYPE_CHECKING:
    from enton.cognition.planner import Planner


class PlannerTools(Toolkit):
    """Manages reminders and to-do tasks via the Enton Planner subsystem."""

    def __init__(self, planner: Planner) -> None:
        super().__init__(name="planner_tools")
        self._planner = planner
        self.register(self.add_reminder)
        self.register(self.add_recurring_reminder)
        self.register(self.list_reminders)
        self.register(self.cancel_reminder)
        self.register(self.add_todo)
        self.register(self.complete_todo)
        self.register(self.list_todos)

    # ---- Reminders ----

    def add_reminder(self, text: str, minutes: float = 5.0) -> str:
        """Cria um lembrete que dispara depois de N minutos.

        Args:
            text: Texto do lembrete.
            minutes: Minutos ate disparar (default: 5).
        """
        rid = self._planner.add_reminder(text, minutes * 60)
        return f"Lembrete '{text}' criado (id={rid}, dispara em {minutes:.0f}min)."

    def add_recurring_reminder(self, text: str, interval_minutes: float = 60.0) -> str:
        """Cria um lembrete recorrente que repete a cada N minutos.

        Args:
            text: Texto do lembrete.
            interval_minutes: Intervalo em minutos entre repeticoes.
        """
        rid = self._planner.add_recurring(text, interval_minutes * 60)
        return f"Lembrete recorrente '{text}' (id={rid}, a cada {interval_minutes:.0f}min)."

    def list_reminders(self) -> str:
        """Lista todos os lembretes ativos."""
        reminders = self._planner.list_reminders()
        if not reminders:
            return "Nenhum lembrete ativo."
        parts: list[str] = []
        now = time.time()
        for r in reminders:
            remaining = max(0, r.trigger_at - now)
            mins = remaining / 60
            recurring = " (recorrente)" if r.recurring_seconds > 0 else ""
            parts.append(f"- [{r.id}] {r.text} -- em {mins:.0f}min{recurring}")
        return "\n".join(parts)

    def cancel_reminder(self, reminder_id: str) -> str:
        """Cancela um lembrete pelo ID.

        Args:
            reminder_id: ID do lembrete (ex: r1, r2).
        """
        if self._planner.cancel_reminder(reminder_id):
            return f"Lembrete {reminder_id} cancelado."
        return f"Lembrete {reminder_id} nao encontrado."

    # ---- Todos ----

    def add_todo(self, text: str, priority: int = 0) -> str:
        """Adiciona uma tarefa na lista de TODOs.

        Args:
            text: Descricao da tarefa.
            priority: Prioridade (0=normal, 1=alta, 2=urgente).
        """
        idx = self._planner.add_todo(text, priority)
        prio_label = ["normal", "alta", "urgente"][min(priority, 2)]
        return f"TODO #{idx} criado: '{text}' (prioridade: {prio_label})"

    def complete_todo(self, index: int) -> str:
        """Marca uma tarefa como concluida.

        Args:
            index: Indice da tarefa na lista.
        """
        if self._planner.complete_todo(index):
            return f"TODO #{index} concluido!"
        return f"TODO #{index} nao encontrado."

    def list_todos(self) -> str:
        """Lista todas as tarefas pendentes."""
        todos = self._planner.list_todos()
        if not todos:
            return "Nenhuma tarefa pendente. Ta de boa!"
        parts: list[str] = []
        for idx, t in todos:
            prio = ["", " [ALTA]", " [URGENTE]"][min(t.priority, 2)]
            parts.append(f"- #{idx}{prio}: {t.text}")
        return "\n".join(parts)
