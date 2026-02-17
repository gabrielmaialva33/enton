"""Tests for ForgeEngine and ForgeTools."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from enton.skills.forge_engine import ForgeEngine
from enton.skills.forge_toolkit import ForgeTools
from enton.skills.skill_protocol import SkillMetadata


@pytest.fixture()
def mock_brain():
    brain = MagicMock()
    brain.think = AsyncMock()
    brain.register_toolkit = MagicMock()
    brain.unregister_toolkit = MagicMock(return_value=True)
    return brain


@pytest.fixture()
def forge(tmp_path, mock_brain):
    return ForgeEngine(
        brain=mock_brain,
        skills_dir=tmp_path,
        sandbox_timeout=5.0,
        max_retries=1,
    )


VALID_SPEC = {
    "name": "celsius_to_fahrenheit",
    "description": "Converte Celsius para Fahrenheit",
    "params": "celsius: str",
    "code": 'c = float(celsius)\nf = c * 9/5 + 32\nreturn f"{f:.1f}°F"',
    "test_code": (
        'result = celsius_to_fahrenheit("100")\n'
        'assert "212" in result, f"Expected 212, got {result}"'
    ),
}


# -- ForgeEngine tests --


def test_parse_json_valid(forge):
    text = json.dumps(VALID_SPEC)
    result = forge._parse_json(text)
    assert result is not None
    assert result["name"] == "celsius_to_fahrenheit"


def test_parse_json_with_fences(forge):
    text = f"```json\n{json.dumps(VALID_SPEC)}\n```"
    result = forge._parse_json(text)
    assert result is not None
    assert result["name"] == "celsius_to_fahrenheit"


def test_parse_json_invalid(forge):
    assert forge._parse_json("not json") is None
    assert forge._parse_json('{"name": "x"}') is None  # missing keys


def test_parse_json_not_dict(forge):
    assert forge._parse_json("[1, 2, 3]") is None


@pytest.mark.asyncio()
async def test_sandbox_passes_valid(forge):
    passed, output = await forge._sandbox_test(
        "celsius_to_fahrenheit",
        "celsius: str",
        'c = float(celsius)\nf = c * 9/5 + 32\nreturn f"{f:.1f}°F"',
        'result = celsius_to_fahrenheit("100")\nassert "212" in result',
    )
    assert passed is True
    assert "PASS" in output


@pytest.mark.asyncio()
async def test_sandbox_catches_error(forge):
    passed, output = await forge._sandbox_test(
        "bad_func",
        "x: str",
        "return int(x) / 0  # division by zero",
        'bad_func("5")',
    )
    assert passed is False
    assert "ZeroDivision" in output or "Error" in output


@pytest.mark.asyncio()
async def test_sandbox_catches_syntax_error(forge):
    passed, output = await forge._sandbox_test(
        "broken", "x: str", "return x +", 'broken("hi")',
    )
    assert passed is False


@pytest.mark.asyncio()
async def test_sandbox_timeout(tmp_path, mock_brain):
    forge = ForgeEngine(
        brain=mock_brain, skills_dir=tmp_path, sandbox_timeout=1.0,
    )
    passed, output = await forge._sandbox_test(
        "slow", "x: str",
        "import time; time.sleep(10); return x",
        'slow("hi")',
    )
    assert passed is False
    assert "Timeout" in output or "timeout" in output.lower()


def test_deploy(forge, tmp_path):
    path = forge._deploy(
        name="celsius_to_fahrenheit",
        description="Convert C to F",
        params="celsius: str",
        code='c = float(celsius)\nf = c * 9/5 + 32\nreturn f"{f:.1f}°F"',
    )
    assert path.exists()
    content = path.read_text()
    assert "class CelsiusToFahrenheitTools(Toolkit)" in content
    assert "SKILL_NAME" in content
    assert "SKILL_AUTHOR" in content


@pytest.mark.asyncio()
async def test_create_tool_full_pipeline(forge, mock_brain, tmp_path):
    mock_brain.think = AsyncMock(return_value=json.dumps(VALID_SPEC))
    result = await forge.create_tool("Convert Celsius to Fahrenheit")
    assert result["success"] is True
    assert result["name"] == "celsius_to_fahrenheit"
    assert (tmp_path / "celsius_to_fahrenheit.py").exists()


@pytest.mark.asyncio()
async def test_create_tool_llm_fails(forge, mock_brain):
    mock_brain.think = AsyncMock(return_value="I cannot do that")
    result = await forge.create_tool("impossible task")
    assert result["success"] is False
    assert "failed" in result["error"].lower() or "LLM" in result["error"]


@pytest.mark.asyncio()
async def test_create_tool_self_correction(forge, mock_brain, tmp_path):
    bad_spec = {**VALID_SPEC, "code": "return int(celsius) / 0"}
    good_spec = VALID_SPEC

    # First call returns bad code, second returns good
    mock_brain.think = AsyncMock(
        side_effect=[json.dumps(bad_spec), json.dumps(good_spec)],
    )
    result = await forge.create_tool("Convert C to F")
    assert result["success"] is True


def test_retire_tool(forge, tmp_path):
    path = tmp_path / "test_tool.py"
    path.write_text("# dummy")
    assert forge.retire_tool("test_tool") is True
    assert not path.exists()


def test_retire_nonexistent(forge):
    assert forge.retire_tool("nope") is False


def test_tool_stats(forge):
    forge._record_outcome("tool_a", success=True)
    forge._record_outcome("tool_a", success=True)
    forge._record_outcome("tool_a", success=False)
    stats = forge.get_tool_stats()
    assert len(stats) == 1
    assert stats[0]["name"] == "tool_a"
    assert stats[0]["success_count"] == 2
    assert stats[0]["failure_count"] == 1
    assert stats[0]["success_rate"] == pytest.approx(2 / 3)


def test_tool_stats_empty(forge):
    assert forge.get_tool_stats() == []


# -- ForgeTools tests --


@pytest.fixture()
def mock_registry():
    reg = MagicMock()
    reg.loaded_skills = {}
    return reg


@pytest.fixture()
def forge_tools(forge, mock_registry):
    return ForgeTools(forge=forge, registry=mock_registry)


def test_forge_tools_registers():
    forge = MagicMock()
    reg = MagicMock()
    ft = ForgeTools(forge=forge, registry=reg)
    assert ft.name == "forge_tools"


@pytest.mark.asyncio()
async def test_forge_tools_create(forge_tools, mock_brain, tmp_path):
    mock_brain.think = AsyncMock(return_value=json.dumps(VALID_SPEC))
    result = await forge_tools.create_tool("Convert C to F")
    assert "sucesso" in result


@pytest.mark.asyncio()
async def test_forge_tools_create_failure(forge_tools, mock_brain):
    mock_brain.think = AsyncMock(return_value="nope")
    result = await forge_tools.create_tool("impossible")
    assert "Falha" in result


def test_forge_tools_list_empty(forge_tools, mock_registry):
    mock_registry.loaded_skills = {}
    result = forge_tools.list_dynamic_tools()
    assert "Nenhuma" in result


def test_forge_tools_list_with_skills(forge_tools, mock_registry):
    mock_registry.loaded_skills = {
        "greet": SkillMetadata(
            name="greet", file_path="/tmp/g.py",
            description="Greeting", success_count=5, failure_count=1,
        ),
    }
    result = forge_tools.list_dynamic_tools()
    assert "greet" in result
    assert "83%" in result


def test_forge_tools_stats_empty(forge_tools):
    result = forge_tools.tool_stats()
    assert "Nenhuma" in result


def test_forge_tools_retire(forge_tools, tmp_path):
    (tmp_path / "test_tool.py").write_text("# dummy")
    result = forge_tools.retire_tool("test_tool")
    assert "sucesso" in result


def test_forge_tools_retire_nonexistent(forge_tools):
    result = forge_tools.retire_tool("nope")
    assert "nao encontrada" in result
