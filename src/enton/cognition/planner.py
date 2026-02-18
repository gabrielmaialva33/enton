"""PDA / Planner — Enton's task and reminder system.

Manages:
- Reminders with timestamps
- Recurring routines (morning, night, etc.)
- Todo list that Enton can manage via tools
- Persisted to JSON file
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_PLANNER_FILE = Path.home() / ".enton" / "planner.json"


@dataclass(slots=True)
class Reminder:
    """A single reminder."""

    text: str
    trigger_at: float  # Unix timestamp
    recurring_seconds: float = 0  # 0 = one-shot
    active: bool = True
    id: str = ""

    def is_due(self) -> bool:
        return self.active and time.time() >= self.trigger_at

    def advance(self) -> None:
        """Mark as done or reschedule if recurring."""
        if self.recurring_seconds > 0:
            self.trigger_at += self.recurring_seconds
        else:
            self.active = False


@dataclass(slots=True)
class TodoItem:
    """A single task on the todo list."""

    text: str
    done: bool = False
    created_at: float = field(default_factory=time.time)
    priority: int = 0  # 0=normal, 1=high, 2=urgent


class Planner:
    """Enton's personal organizer."""

    def __init__(self) -> None:
        self._reminders: list[Reminder] = []
        self._todos: list[TodoItem] = []
        self._routines: dict[str, dict] = {}
        self._next_id = 1
        self._load()

    def _load(self) -> None:
        if _PLANNER_FILE.exists():
            try:
                data = json.loads(_PLANNER_FILE.read_text())
                for r in data.get("reminders", []):
                    self._reminders.append(Reminder(**r))
                for t in data.get("todos", []):
                    self._todos.append(TodoItem(**t))
                self._routines = data.get("routines", {})
                self._next_id = data.get("next_id", 1)
                logger.info(
                    "Planner loaded: %d reminders, %d todos",
                    len(self._reminders),
                    len(self._todos),
                )
            except Exception:
                logger.warning("Failed to load planner data")

    def save(self) -> None:
        _PLANNER_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "reminders": [asdict(r) for r in self._reminders],
            "todos": [asdict(t) for t in self._todos],
            "routines": self._routines,
            "next_id": self._next_id,
        }
        _PLANNER_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # --- Reminders ---

    def add_reminder(self, text: str, seconds_from_now: float) -> str:
        """Add a reminder that triggers after N seconds."""
        rid = f"r{self._next_id}"
        self._next_id += 1
        r = Reminder(
            text=text,
            trigger_at=time.time() + seconds_from_now,
            id=rid,
        )
        self._reminders.append(r)
        self.save()
        return rid

    def add_recurring(self, text: str, interval_seconds: float) -> str:
        """Add a recurring reminder."""
        rid = f"r{self._next_id}"
        self._next_id += 1
        r = Reminder(
            text=text,
            trigger_at=time.time() + interval_seconds,
            recurring_seconds=interval_seconds,
            id=rid,
        )
        self._reminders.append(r)
        self.save()
        return rid

    def get_due_reminders(self) -> list[Reminder]:
        """Get all reminders that are due now."""
        due = [r for r in self._reminders if r.is_due()]
        for r in due:
            r.advance()
        if due:
            self.save()
        return due

    def list_reminders(self) -> list[Reminder]:
        return [r for r in self._reminders if r.active]

    def cancel_reminder(self, rid: str) -> bool:
        for r in self._reminders:
            if r.id == rid:
                r.active = False
                self.save()
                return True
        return False

    # --- Todos ---

    def add_todo(self, text: str, priority: int = 0) -> int:
        idx = len(self._todos)
        self._todos.append(TodoItem(text=text, priority=priority))
        self.save()
        return idx

    def complete_todo(self, index: int) -> bool:
        if 0 <= index < len(self._todos):
            self._todos[index].done = True
            self.save()
            return True
        return False

    def list_todos(self, include_done: bool = False) -> list[tuple[int, TodoItem]]:
        result = []
        for i, t in enumerate(self._todos):
            if include_done or not t.done:
                result.append((i, t))
        return result

    # --- Routines ---

    def set_routine(self, name: str, hour: int, text: str) -> None:
        """Set a daily routine at a specific hour (0-23)."""
        self._routines[name] = {"hour": hour, "text": text, "last_run": ""}
        self.save()

    def get_due_routines(self, current_hour: int) -> list[dict]:
        """Get routines due this hour that haven't run today."""
        import datetime

        today = datetime.date.today().isoformat()
        due = []
        for name, r in self._routines.items():
            if r["hour"] == current_hour and r.get("last_run") != today:
                due.append({"name": name, "text": r["text"]})
                r["last_run"] = today
        if due:
            self.save()
        return due

    def summary(self) -> str:
        """Brief summary for introspection."""
        active_reminders = len(self.list_reminders())
        pending_todos = len(self.list_todos())
        return f"PDA: {active_reminders} reminders, {pending_todos} pending tasks"
