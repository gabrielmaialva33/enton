"""Tests for Planner (reminders, todos, routines)."""

from __future__ import annotations

import time

from enton.cognition.planner import Planner, Reminder


def test_reminder_is_due():
    r = Reminder(text="Test", trigger_at=time.time() - 1)
    assert r.is_due() is True


def test_reminder_not_due():
    r = Reminder(text="Test", trigger_at=time.time() + 9999)
    assert r.is_due() is False


def test_reminder_one_shot_advance():
    r = Reminder(text="Test", trigger_at=time.time() - 1)
    r.advance()
    assert r.active is False


def test_reminder_recurring_advance():
    r = Reminder(text="Test", trigger_at=time.time() - 1, recurring_seconds=60)
    old_trigger = r.trigger_at
    r.advance()
    assert r.active is True
    assert r.trigger_at == old_trigger + 60


def test_planner_add_reminder(tmp_planner_file):
    p = Planner()
    rid = p.add_reminder("Drink water", 60)
    assert rid.startswith("r")
    active = p.list_reminders()
    assert len(active) == 1
    assert active[0].text == "Drink water"


def test_planner_cancel_reminder(tmp_planner_file):
    p = Planner()
    rid = p.add_reminder("Cancel me", 60)
    assert p.cancel_reminder(rid) is True
    assert len(p.list_reminders()) == 0


def test_planner_cancel_nonexistent(tmp_planner_file):
    p = Planner()
    assert p.cancel_reminder("r999") is False


def test_planner_get_due(tmp_planner_file):
    p = Planner()
    p.add_reminder("Due now", -1)  # already past
    p.add_reminder("Future", 9999)
    due = p.get_due_reminders()
    assert len(due) == 1
    assert due[0].text == "Due now"


def test_planner_recurring(tmp_planner_file):
    p = Planner()
    p.add_recurring("Stretch", 300)
    active = p.list_reminders()
    assert len(active) == 1
    assert active[0].recurring_seconds == 300


def test_planner_add_todo(tmp_planner_file):
    p = Planner()
    idx = p.add_todo("Buy milk", priority=1)
    assert idx == 0
    todos = p.list_todos()
    assert len(todos) == 1
    assert todos[0][1].text == "Buy milk"
    assert todos[0][1].priority == 1


def test_planner_complete_todo(tmp_planner_file):
    p = Planner()
    p.add_todo("Task A")
    p.add_todo("Task B")
    assert p.complete_todo(0) is True
    # Only Task B should be pending
    pending = p.list_todos()
    assert len(pending) == 1
    assert pending[0][1].text == "Task B"


def test_planner_complete_invalid(tmp_planner_file):
    p = Planner()
    assert p.complete_todo(999) is False


def test_planner_list_todos_include_done(tmp_planner_file):
    p = Planner()
    p.add_todo("Done task")
    p.complete_todo(0)
    assert len(p.list_todos(include_done=False)) == 0
    assert len(p.list_todos(include_done=True)) == 1


def test_planner_routines(tmp_planner_file):
    p = Planner()
    p.set_routine("morning", 8, "Good morning!")
    due = p.get_due_routines(8)
    assert len(due) == 1
    assert due[0]["text"] == "Good morning!"
    # Should not fire again same day
    due2 = p.get_due_routines(8)
    assert len(due2) == 0


def test_planner_routine_wrong_hour(tmp_planner_file):
    p = Planner()
    p.set_routine("night", 22, "Good night!")
    assert len(p.get_due_routines(8)) == 0


def test_planner_persistence(tmp_planner_file):
    p1 = Planner()
    p1.add_reminder("Persist me", 999)
    p1.add_todo("Persist todo")

    p2 = Planner()
    assert len(p2.list_reminders()) == 1
    assert len(p2.list_todos()) == 1


def test_planner_summary(tmp_planner_file):
    p = Planner()
    p.add_reminder("Test", 60)
    p.add_todo("Test")
    s = p.summary()
    assert "1 reminders" in s
    assert "1 pending" in s
