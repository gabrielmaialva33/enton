"""Tests for Enton's Lifecycle — state persistence between boots."""

import json
import time
from unittest.mock import MagicMock

import pytest

import enton.core.lifecycle as lifecycle_mod
from enton.core.lifecycle import Lifecycle


@pytest.fixture(autouse=True)
def redirect_state_file(tmp_path, monkeypatch):
    """Redirect _STATE_FILE to a temporary directory for every test."""
    fake_state = tmp_path / "state.json"
    monkeypatch.setattr(lifecycle_mod, "_STATE_FILE", fake_state)
    return fake_state


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _make_self_model(engagement=0.7, social=0.4):
    sm = MagicMock()
    sm.mood.engagement = engagement
    sm.mood.social = social
    return sm


def _make_desires(data=None):
    de = MagicMock()
    de.to_dict.return_value = data or {"socialize": 0.5}
    return de


# ---------------------------------------------------------------------------
#  Init / Load
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_no_state_file(self, redirect_state_file):
        """Lifecycle initializes cleanly when there is no prior state."""
        lc = Lifecycle()
        assert lc._state == {}
        assert lc.boot_count == 0
        assert lc.total_uptime_hours == 0.0
        assert lc.last_shutdown == 0

    def test_init_loads_existing_state(self, redirect_state_file):
        """Lifecycle loads data from an existing state file."""
        state = {
            "boot_count": 5,
            "total_uptime_seconds": 7200,
            "last_shutdown": 1000000.0,
        }
        redirect_state_file.parent.mkdir(parents=True, exist_ok=True)
        redirect_state_file.write_text(json.dumps(state))

        lc = Lifecycle()
        assert lc.boot_count == 5
        assert lc.total_uptime_hours == pytest.approx(2.0)
        assert lc.last_shutdown == 1000000.0

    def test_init_handles_corrupt_json(self, redirect_state_file):
        """Lifecycle falls back to empty state if file is corrupted."""
        redirect_state_file.parent.mkdir(parents=True, exist_ok=True)
        redirect_state_file.write_text("{bad json!!")

        lc = Lifecycle()
        assert lc._state == {}
        assert lc.boot_count == 0


# ---------------------------------------------------------------------------
#  Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_boot_count_default(self):
        lc = Lifecycle()
        assert lc.boot_count == 0

    def test_total_uptime_hours_default(self):
        lc = Lifecycle()
        assert lc.total_uptime_hours == 0.0

    def test_last_shutdown_default(self):
        lc = Lifecycle()
        assert lc.last_shutdown == 0


# ---------------------------------------------------------------------------
#  time_asleep / time_asleep_human
# ---------------------------------------------------------------------------


class TestTimeAsleep:
    def test_time_asleep_no_prior_shutdown(self):
        """When there is no prior shutdown, time_asleep should be 0."""
        lc = Lifecycle()
        assert lc.time_asleep == 0

    def test_time_asleep_with_prior_shutdown(self, redirect_state_file):
        state = {"last_shutdown": time.time() - 300}
        redirect_state_file.parent.mkdir(parents=True, exist_ok=True)
        redirect_state_file.write_text(json.dumps(state))

        lc = Lifecycle()
        assert 299 <= lc.time_asleep <= 302

    def test_time_asleep_human_seconds(self, monkeypatch):
        lc = Lifecycle()
        monkeypatch.setattr(type(lc), "time_asleep", property(lambda self: 45.0))
        assert lc.time_asleep_human == "45s"

    def test_time_asleep_human_minutes(self, monkeypatch):
        lc = Lifecycle()
        monkeypatch.setattr(type(lc), "time_asleep", property(lambda self: 300.0))
        assert lc.time_asleep_human == "5min"

    def test_time_asleep_human_hours(self, monkeypatch):
        lc = Lifecycle()
        # 2h 30min = 9000 seconds
        monkeypatch.setattr(type(lc), "time_asleep", property(lambda self: 9000.0))
        assert lc.time_asleep_human == "2h30min"

    def test_time_asleep_human_days(self, monkeypatch):
        lc = Lifecycle()
        # 2d 3h = 2*86400 + 3*3600 = 183600
        monkeypatch.setattr(type(lc), "time_asleep", property(lambda self: 183600.0))
        assert lc.time_asleep_human == "2d3h"

    def test_time_asleep_human_zero(self, monkeypatch):
        lc = Lifecycle()
        monkeypatch.setattr(type(lc), "time_asleep", property(lambda self: 0.0))
        assert lc.time_asleep_human == "0s"


# ---------------------------------------------------------------------------
#  on_boot
# ---------------------------------------------------------------------------


class TestOnBoot:
    def test_on_boot_increments_boot_count(self):
        lc = Lifecycle()
        sm = _make_self_model()
        de = _make_desires()
        lc.on_boot(sm, de)
        assert lc.boot_count == 1

    def test_on_boot_increments_boot_count_multiple(self):
        lc = Lifecycle()
        sm = _make_self_model()
        de = _make_desires()
        lc.on_boot(sm, de)
        lc.on_boot(sm, de)
        lc.on_boot(sm, de)
        assert lc.boot_count == 3

    def test_on_boot_restores_mood(self, redirect_state_file):
        state = {"mood": {"engagement": 0.9, "social": 0.8}}
        redirect_state_file.parent.mkdir(parents=True, exist_ok=True)
        redirect_state_file.write_text(json.dumps(state))

        lc = Lifecycle()
        sm = _make_self_model()
        de = _make_desires()
        lc.on_boot(sm, de)

        assert sm.mood.engagement == 0.9
        assert sm.mood.social == 0.8

    def test_on_boot_restores_desires(self, redirect_state_file):
        desire_data = {"socialize": 0.9, "learn": 0.3}
        state = {"desires": desire_data}
        redirect_state_file.parent.mkdir(parents=True, exist_ok=True)
        redirect_state_file.write_text(json.dumps(state))

        lc = Lifecycle()
        sm = _make_self_model()
        de = _make_desires()
        lc.on_boot(sm, de)

        de.from_dict.assert_called_once_with(desire_data)

    def test_on_boot_wake_message_long_sleep(self, redirect_state_file, monkeypatch):
        """Sleeping > 1 day produces the 'saudade' message."""
        state = {"last_shutdown": time.time() - 100000}  # > 1 day
        redirect_state_file.parent.mkdir(parents=True, exist_ok=True)
        redirect_state_file.write_text(json.dumps(state))

        lc = Lifecycle()
        sm = _make_self_model()
        de = _make_desires()
        msg = lc.on_boot(sm, de)
        assert "dormi" in msg.lower()
        assert "saudade" in msg.lower()

    def test_on_boot_wake_message_medium_sleep(self, redirect_state_file):
        """Sleeping > 1 hour but < 1 day."""
        state = {"last_shutdown": time.time() - 7200}  # 2 hours
        redirect_state_file.parent.mkdir(parents=True, exist_ok=True)
        redirect_state_file.write_text(json.dumps(state))

        lc = Lifecycle()
        sm = _make_self_model()
        de = _make_desires()
        msg = lc.on_boot(sm, de)
        assert "voltei" in msg.lower()
        assert "offline" in msg.lower()

    def test_on_boot_wake_message_short_sleep(self, redirect_state_file):
        """Sleeping > 1 min but < 1 hour."""
        state = {"last_shutdown": time.time() - 120}  # 2 minutes
        redirect_state_file.parent.mkdir(parents=True, exist_ok=True)
        redirect_state_file.write_text(json.dumps(state))

        lc = Lifecycle()
        sm = _make_self_model()
        de = _make_desires()
        msg = lc.on_boot(sm, de)
        assert "rapidinho" in msg.lower()

    def test_on_boot_wake_message_first_boot(self):
        """First boot ever returns empty string."""
        lc = Lifecycle()
        sm = _make_self_model()
        de = _make_desires()
        msg = lc.on_boot(sm, de)
        assert msg == ""


# ---------------------------------------------------------------------------
#  on_shutdown
# ---------------------------------------------------------------------------


class TestOnShutdown:
    def test_on_shutdown_saves_state_file(self, redirect_state_file):
        lc = Lifecycle()
        sm = _make_self_model(engagement=0.6, social=0.2)
        de = _make_desires({"learn": 0.7})

        lc.on_shutdown(sm, de)

        assert redirect_state_file.exists()
        saved = json.loads(redirect_state_file.read_text())
        assert saved["mood"]["engagement"] == 0.6
        assert saved["mood"]["social"] == 0.2
        assert saved["desires"] == {"learn": 0.7}
        assert "last_shutdown" in saved
        assert "total_uptime_seconds" in saved

    def test_on_shutdown_accumulates_uptime(self, redirect_state_file):
        state = {"total_uptime_seconds": 1000}
        redirect_state_file.parent.mkdir(parents=True, exist_ok=True)
        redirect_state_file.write_text(json.dumps(state))

        lc = Lifecycle()
        sm = _make_self_model()
        de = _make_desires()

        lc.on_shutdown(sm, de)

        saved = json.loads(redirect_state_file.read_text())
        assert saved["total_uptime_seconds"] >= 1000

    def test_on_shutdown_records_last_shutdown_timestamp(self, redirect_state_file):
        lc = Lifecycle()
        sm = _make_self_model()
        de = _make_desires()

        before = time.time()
        lc.on_shutdown(sm, de)
        after = time.time()

        saved = json.loads(redirect_state_file.read_text())
        assert before <= saved["last_shutdown"] <= after


# ---------------------------------------------------------------------------
#  save_periodic
# ---------------------------------------------------------------------------


class TestSavePeriodic:
    def test_save_periodic_delegates_to_on_shutdown(self, redirect_state_file):
        lc = Lifecycle()
        sm = _make_self_model()
        de = _make_desires()

        lc.save_periodic(sm, de)

        assert redirect_state_file.exists()
        saved = json.loads(redirect_state_file.read_text())
        assert "mood" in saved
        assert "desires" in saved


# ---------------------------------------------------------------------------
#  summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_format(self):
        lc = Lifecycle()
        sm = _make_self_model()
        de = _make_desires()
        # Boot once so we have data
        lc.on_boot(sm, de)

        s = lc.summary()
        assert "Boot #1" in s
        assert "total uptime" in s
        assert "slept" in s

    def test_summary_fresh_state(self):
        lc = Lifecycle()
        s = lc.summary()
        assert "Boot #0" in s
        assert "0.0h" in s
        assert "0s" in s
