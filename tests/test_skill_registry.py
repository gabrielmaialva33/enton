"""Tests for SkillRegistry."""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from enton.skills.skill_protocol import SkillMetadata
from enton.skills.skill_registry import SkillRegistry

VALID_SKILL = '''\
from agno.tools import Toolkit

SKILL_NAME = "greet"
SKILL_DESCRIPTION = "Greeting tool"
SKILL_VERSION = "2.0"
SKILL_AUTHOR = "test"


class GreetTools(Toolkit):
    def __init__(self):
        super().__init__(name="greet_tools")
        self.register(self.greet)

    def greet(self, name: str) -> str:
        """Say hello."""
        return f"Hello, {name}!"
'''

FACTORY_SKILL = '''\
from agno.tools import Toolkit

SKILL_DESCRIPTION = "Factory tool"


class _Internal(Toolkit):
    def __init__(self):
        super().__init__(name="factory_tools")
        self.register(self.ping)

    def ping(self) -> str:
        """Ping."""
        return "pong"


def create_toolkit():
    return _Internal()
'''

BAD_SYNTAX = "def oops(:\n    pass"

NO_TOOLKIT = "x = 42\ndef hello(): return 'hi'"


@pytest.fixture()
def mock_brain():
    brain = MagicMock()
    brain.register_toolkit = MagicMock()
    brain.unregister_toolkit = MagicMock(return_value=True)
    return brain


@pytest.fixture()
def mock_bus():
    bus = MagicMock()
    bus.emit = AsyncMock()
    return bus


@pytest.fixture()
def registry(tmp_path, mock_brain, mock_bus):
    return SkillRegistry(brain=mock_brain, bus=mock_bus, skills_dir=tmp_path)


def test_skill_metadata_success_rate():
    m = SkillMetadata(name="test", file_path="/tmp/t.py")
    assert m.success_rate == 1.0  # no data → default 1.0
    m.success_count = 3
    m.failure_count = 1
    assert m.success_rate == 0.75


def test_skill_metadata_zero_total():
    m = SkillMetadata(name="x", file_path="/tmp/x.py", success_count=0, failure_count=0)
    assert m.success_rate == 1.0


def test_registry_init(registry):
    assert registry.list_skills() == []
    assert registry.loaded_skills == {}


@pytest.mark.asyncio()
async def test_load_valid_skill(tmp_path, registry, mock_brain):
    path = tmp_path / "greet.py"
    path.write_text(VALID_SKILL)
    result = await registry.load_skill(path)
    assert result is True
    assert "greet" in registry.list_skills()
    mock_brain.register_toolkit.assert_called_once()
    meta = registry.loaded_skills["greet"]
    assert meta.description == "Greeting tool"
    assert meta.version == "2.0"


@pytest.mark.asyncio()
async def test_load_factory_skill(tmp_path, registry, mock_brain):
    path = tmp_path / "factory.py"
    path.write_text(FACTORY_SKILL)
    result = await registry.load_skill(path)
    assert result is True
    assert "factory" in registry.list_skills()
    mock_brain.register_toolkit.assert_called_once()


@pytest.mark.asyncio()
async def test_load_bad_syntax(tmp_path, registry, mock_brain):
    path = tmp_path / "bad.py"
    path.write_text(BAD_SYNTAX)
    result = await registry.load_skill(path)
    assert result is False
    assert "bad" not in registry.list_skills()
    mock_brain.register_toolkit.assert_not_called()


@pytest.mark.asyncio()
async def test_load_no_toolkit(tmp_path, registry, mock_brain):
    path = tmp_path / "empty.py"
    path.write_text(NO_TOOLKIT)
    result = await registry.load_skill(path)
    assert result is False
    assert "empty" not in registry.list_skills()


@pytest.mark.asyncio()
async def test_unload_skill(tmp_path, registry, mock_brain, mock_bus):
    path = tmp_path / "greet.py"
    path.write_text(VALID_SKILL)
    await registry.load_skill(path)
    result = await registry.unload_skill("greet")
    assert result is True
    assert "greet" not in registry.list_skills()
    mock_brain.unregister_toolkit.assert_called_once_with("greet")
    # Module should be cleaned from sys.modules
    assert "enton_skill_greet" not in sys.modules


@pytest.mark.asyncio()
async def test_unload_nonexistent(registry):
    result = await registry.unload_skill("nope")
    assert result is False


@pytest.mark.asyncio()
async def test_reload_skill(tmp_path, registry, mock_brain):
    path = tmp_path / "greet.py"
    path.write_text(VALID_SKILL)
    await registry.load_skill(path)

    # Modify and reload
    path.write_text(VALID_SKILL.replace("2.0", "3.0"))
    result = await registry.reload_skill(path)
    assert result is True
    assert registry.loaded_skills["greet"].version == "3.0"


@pytest.mark.asyncio()
async def test_scan_existing(tmp_path, registry, mock_brain):
    (tmp_path / "a.py").write_text(VALID_SKILL)
    (tmp_path / "b.py").write_text(FACTORY_SKILL)
    (tmp_path / "_hidden.py").write_text(VALID_SKILL)  # should skip
    await registry._scan_existing()
    assert len(registry.list_skills()) == 2
    assert "_hidden" not in registry.list_skills()


@pytest.mark.asyncio()
async def test_emits_events(tmp_path, registry, mock_bus):
    path = tmp_path / "greet.py"
    path.write_text(VALID_SKILL)
    await registry.load_skill(path)
    assert mock_bus.emit.call_count == 1
    event = mock_bus.emit.call_args[0][0]
    assert event.kind == "loaded"
    assert event.name == "greet"

    await registry.unload_skill("greet")
    assert mock_bus.emit.call_count == 2
    event = mock_bus.emit.call_args[0][0]
    assert event.kind == "unloaded"


def test_record_outcome(tmp_path, registry):
    registry._loaded["test"] = SkillMetadata(name="test", file_path="/tmp/t.py")
    registry.record_outcome("test", success=True)
    registry.record_outcome("test", success=True)
    registry.record_outcome("test", success=False)
    assert registry._loaded["test"].success_count == 2
    assert registry._loaded["test"].failure_count == 1
    assert registry._loaded["test"].success_rate == pytest.approx(2 / 3)


def test_record_outcome_missing(registry):
    # Should not raise
    registry.record_outcome("nonexistent", success=True)
