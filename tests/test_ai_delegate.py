"""Tests for AI Delegation — ClaudeCodeProvider, GeminiCliProvider, AIDelegateTools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from enton.providers.claude_code import ClaudeCodeProvider, ClaudeResult
from enton.providers.gemini_cli import GeminiCliProvider, GeminiResult
from enton.skills.ai_delegate_toolkit import AIDelegateTools

# ──────────────────────────────────────────────────────────────────────
# ClaudeCodeProvider
# ──────────────────────────────────────────────────────────────────────


class TestClaudeCodeProvider:
    def test_init_defaults(self):
        p = ClaudeCodeProvider()
        assert p._model == "sonnet"
        assert p._timeout == 120.0
        assert p._max_turns == 10

    def test_init_custom(self):
        p = ClaudeCodeProvider(model="opus", timeout=60.0, max_turns=3)
        assert p._model == "opus"
        assert p._timeout == 60.0
        assert p._max_turns == 3

    def test_id(self):
        p = ClaudeCodeProvider(model="haiku")
        assert p.id == "claude-code:haiku"

    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_available_true(self, mock_which):
        p = ClaudeCodeProvider()
        assert p.available is True
        mock_which.assert_called_once_with("claude")

    @patch("shutil.which", return_value=None)
    def test_available_false(self, mock_which):
        p = ClaudeCodeProvider()
        assert p.available is False

    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_available_cached(self, mock_which):
        p = ClaudeCodeProvider()
        _ = p.available
        _ = p.available
        mock_which.assert_called_once()

    def test_parse_output_valid_json(self):
        data = {
            "type": "result",
            "result": "A resposta aqui",
            "session_id": "abc123",
            "total_cost_usd": 0.003,
            "num_turns": 2,
            "duration_ms": 1500,
            "is_error": False,
        }
        result = ClaudeCodeProvider._parse_output(json.dumps(data))
        assert result.content == "A resposta aqui"
        assert result.session_id == "abc123"
        assert result.cost_usd == 0.003
        assert result.num_turns == 2
        assert result.duration_ms == 1500
        assert result.is_error is False

    def test_parse_output_error_json(self):
        data = {"result": "", "is_error": True}
        result = ClaudeCodeProvider._parse_output(json.dumps(data))
        assert result.is_error is True
        assert result.content == ""

    def test_parse_output_invalid_json(self):
        result = ClaudeCodeProvider._parse_output("Just plain text output")
        assert result.content == "Just plain text output"
        assert result.is_error is False

    def test_parse_output_empty(self):
        result = ClaudeCodeProvider._parse_output("")
        assert result.is_error is True
        assert result.content == ""

    @pytest.mark.asyncio()
    @patch("shutil.which", return_value=None)
    async def test_generate_not_available(self, _):
        p = ClaudeCodeProvider()
        result = await p.generate("test")
        assert result == ""

    @pytest.mark.asyncio()
    @patch("shutil.which", return_value="/usr/bin/claude")
    async def test_generate_success(self, _):
        p = ClaudeCodeProvider()
        response_json = json.dumps({"result": "Resposta do Claude"})

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(response_json.encode(), b""),
        )
        mock_proc.returncode = 0
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await p.generate("Oi Claude")

        assert result == "Resposta do Claude"

    @pytest.mark.asyncio()
    @patch("shutil.which", return_value="/usr/bin/claude")
    async def test_generate_failure(self, _):
        p = ClaudeCodeProvider()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(b"", b"Error occurred"),
        )
        mock_proc.returncode = 1
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await p.generate("test")

        assert result == ""

    @pytest.mark.asyncio()
    @patch("shutil.which", return_value="/usr/bin/claude")
    async def test_generate_timeout(self, _):
        p = ClaudeCodeProvider(timeout=0.1)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            side_effect=TimeoutError(),
        )
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await p.generate("test")

        assert result == ""
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio()
    @patch("shutil.which", return_value="/usr/bin/claude")
    async def test_generate_json_returns_full_result(self, _):
        p = ClaudeCodeProvider()
        response_json = json.dumps(
            {
                "result": "Resposta",
                "session_id": "sess1",
                "total_cost_usd": 0.01,
                "num_turns": 3,
                "duration_ms": 2000,
            }
        )

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(response_json.encode(), b""),
        )
        mock_proc.returncode = 0
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await p.generate_json("test")

        assert isinstance(result, ClaudeResult)
        assert result.content == "Resposta"
        assert result.session_id == "sess1"
        assert result.cost_usd == 0.01

    @pytest.mark.asyncio()
    @patch("shutil.which", return_value="/usr/bin/claude")
    async def test_code_task(self, _):
        p = ClaudeCodeProvider()
        response_json = json.dumps({"result": "Codigo escrito com sucesso"})

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(response_json.encode(), b""),
        )
        mock_proc.returncode = 0
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await p.code_task("Crie um hello world")

        assert "sucesso" in result

    @pytest.mark.asyncio()
    @patch("shutil.which", return_value="/usr/bin/claude")
    async def test_subprocess_args(self, _):
        """Verify correct CLI arguments are passed."""
        p = ClaudeCodeProvider(model="opus", max_turns=3)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(json.dumps({"result": "ok"}).encode(), b""),
        )
        mock_proc.returncode = 0
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await p.generate("test prompt")

        args = mock_exec.call_args[0]
        assert "/usr/bin/claude" in args
        assert "-p" in args
        assert "test prompt" in args
        assert "--output-format" in args
        assert "json" in args
        assert "--model" in args
        assert "opus" in args
        assert "--max-turns" in args
        assert "3" in args


# ──────────────────────────────────────────────────────────────────────
# GeminiCliProvider
# ──────────────────────────────────────────────────────────────────────


class TestGeminiCliProvider:
    def test_init_defaults(self):
        p = GeminiCliProvider()
        assert p._model == "gemini-2.5-flash"
        assert p._timeout == 120.0
        assert p._yolo is False

    def test_init_custom(self):
        p = GeminiCliProvider(model="gemini-2.5-pro", timeout=60.0, yolo=True)
        assert p._model == "gemini-2.5-pro"
        assert p._yolo is True

    def test_id(self):
        p = GeminiCliProvider(model="gemini-2.5-pro")
        assert p.id == "gemini-cli:gemini-2.5-pro"

    @patch("shutil.which", return_value="/usr/bin/gemini")
    def test_available_true(self, mock_which):
        p = GeminiCliProvider()
        assert p.available is True

    @patch("shutil.which", return_value=None)
    def test_available_false(self, _):
        p = GeminiCliProvider()
        assert p.available is False

    def test_parse_output_valid_json(self):
        data = {"response": "Resposta do Gemini"}
        result = GeminiCliProvider._parse_output(json.dumps(data))
        assert result.content == "Resposta do Gemini"
        assert result.is_error is False

    def test_parse_output_fallback_fields(self):
        data = {"result": "Fallback result"}
        result = GeminiCliProvider._parse_output(json.dumps(data))
        assert result.content == "Fallback result"

    def test_parse_output_plain_text(self):
        result = GeminiCliProvider._parse_output("Plain text response")
        assert result.content == "Plain text response"

    def test_parse_output_empty(self):
        result = GeminiCliProvider._parse_output("")
        assert result.is_error is True

    @pytest.mark.asyncio()
    @patch("shutil.which", return_value="/usr/bin/gemini")
    async def test_generate_success(self, _):
        p = GeminiCliProvider()
        response_json = json.dumps({"response": "Gemini response"})

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(response_json.encode(), b""),
        )
        mock_proc.returncode = 0
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await p.generate("Oi Gemini")

        assert result == "Gemini response"

    @pytest.mark.asyncio()
    @patch("shutil.which", return_value="/usr/bin/gemini")
    async def test_generate_failure(self, _):
        p = GeminiCliProvider()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(b"", b"Error"),
        )
        mock_proc.returncode = 1
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await p.generate("test")

        assert result == ""

    @pytest.mark.asyncio()
    @patch("shutil.which", return_value="/usr/bin/gemini")
    async def test_yolo_flag(self, _):
        p = GeminiCliProvider(yolo=True)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(json.dumps({"response": "ok"}).encode(), b""),
        )
        mock_proc.returncode = 0
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await p.generate("test")

        args = mock_exec.call_args[0]
        assert "--yolo" in args

    @pytest.mark.asyncio()
    @patch("shutil.which", return_value="/usr/bin/gemini")
    async def test_research(self, _):
        p = GeminiCliProvider()
        response_json = json.dumps({"response": "Deep research result"})

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(response_json.encode(), b""),
        )
        mock_proc.returncode = 0
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await p.research("quantum computing")

        assert result == "Deep research result"


# ──────────────────────────────────────────────────────────────────────
# AIDelegateTools
# ──────────────────────────────────────────────────────────────────────


class TestAIDelegateTools:
    @pytest.fixture()
    def mock_claude(self):
        p = MagicMock(spec=ClaudeCodeProvider)
        p.available = True
        p.id = "claude-code:sonnet"
        p.generate = AsyncMock(return_value="Claude says hello")
        p.code_task = AsyncMock(return_value="Code written by Claude")
        return p

    @pytest.fixture()
    def mock_gemini(self):
        p = MagicMock(spec=GeminiCliProvider)
        p.available = True
        p.id = "gemini-cli:gemini-2.5-flash"
        p.generate = AsyncMock(return_value="Gemini says hello")
        p.code_task = AsyncMock(return_value="Code written by Gemini")
        p.research = AsyncMock(return_value="Research results from Gemini")
        return p

    @pytest.fixture()
    def tools(self, mock_claude, mock_gemini):
        return AIDelegateTools(claude=mock_claude, gemini=mock_gemini)

    def test_init(self, tools):
        assert tools.name == "ai_delegate"

    @pytest.mark.asyncio()
    async def test_ask_claude(self, tools, mock_claude):
        result = await tools.ask_claude("Oi")
        assert "[Claude Code]" in result
        assert "Claude says hello" in result
        mock_claude.generate.assert_awaited_once_with("Oi")

    @pytest.mark.asyncio()
    async def test_ask_claude_with_context(self, tools, mock_claude):
        await tools.ask_claude("Explica", context="def foo(): pass")
        mock_claude.generate.assert_awaited_once()
        call_prompt = mock_claude.generate.call_args[0][0]
        assert "Explica" in call_prompt
        assert "def foo(): pass" in call_prompt

    @pytest.mark.asyncio()
    async def test_ask_claude_not_available(self, tools, mock_claude):
        mock_claude.available = False
        result = await tools.ask_claude("test")
        assert "nao esta instalado" in result

    @pytest.mark.asyncio()
    async def test_ask_claude_empty_response(self, tools, mock_claude):
        mock_claude.generate = AsyncMock(return_value="")
        result = await tools.ask_claude("test")
        assert "nao retornou" in result

    @pytest.mark.asyncio()
    async def test_ask_gemini(self, tools, mock_gemini):
        result = await tools.ask_gemini("Oi")
        assert "[Gemini CLI]" in result
        assert "Gemini says hello" in result

    @pytest.mark.asyncio()
    async def test_ask_gemini_not_available(self, tools, mock_gemini):
        mock_gemini.available = False
        result = await tools.ask_gemini("test")
        assert "nao esta instalado" in result

    @pytest.mark.asyncio()
    async def test_claude_code_task(self, tools, mock_claude):
        result = await tools.claude_code_task("Crie um hello world")
        assert "[Claude Code Task]" in result
        assert "Code written by Claude" in result
        mock_claude.code_task.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_gemini_code_task(self, tools, mock_gemini):
        result = await tools.gemini_code_task("Crie um hello world")
        assert "[Gemini Code Task]" in result
        assert "Code written by Gemini" in result

    @pytest.mark.asyncio()
    async def test_ai_consensus_both_available(self, tools, mock_claude, mock_gemini):
        result = await tools.ai_consensus("O que e Python?")
        assert "Consensus" in result
        assert "Claude Code" in result
        assert "Gemini CLI" in result
        assert "Claude says hello" in result
        assert "Gemini says hello" in result
        mock_claude.generate.assert_awaited_once()
        mock_gemini.generate.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_ai_consensus_one_available(self, tools, mock_claude, mock_gemini):
        mock_gemini.available = False
        result = await tools.ai_consensus("test")
        assert "Claude Code" in result
        assert "Gemini CLI" not in result

    @pytest.mark.asyncio()
    async def test_ai_consensus_none_available(self, tools, mock_claude, mock_gemini):
        mock_claude.available = False
        mock_gemini.available = False
        result = await tools.ai_consensus("test")
        assert "Nenhum AI CLI" in result

    @pytest.mark.asyncio()
    async def test_ai_consensus_handles_exception(self, tools, mock_claude, mock_gemini):
        mock_claude.generate = AsyncMock(side_effect=RuntimeError("broke"))
        result = await tools.ai_consensus("test")
        assert "ERRO" in result
        assert "Gemini says hello" in result

    @pytest.mark.asyncio()
    async def test_ai_research_gemini_primary(self, tools, mock_gemini):
        result = await tools.ai_research("AI trends")
        assert "[Gemini Research]" in result
        assert "Research results" in result
        mock_gemini.research.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_ai_research_claude_fallback(self, tools, mock_claude, mock_gemini):
        mock_gemini.available = False
        result = await tools.ai_research("AI trends")
        assert "[Claude Research]" in result
        mock_claude.generate.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_ai_research_none_available(self, tools, mock_claude, mock_gemini):
        mock_claude.available = False
        mock_gemini.available = False
        result = await tools.ai_research("test")
        assert "Nenhum AI CLI" in result

    @pytest.mark.asyncio()
    async def test_ai_status_both(self, tools):
        result = await tools.ai_status()
        assert "DISPONIVEL" in result
        assert "2/2" in result

    @pytest.mark.asyncio()
    async def test_ai_status_none(self, tools, mock_claude, mock_gemini):
        mock_claude.available = False
        mock_gemini.available = False
        result = await tools.ai_status()
        assert "NAO INSTALADO" in result
        assert "0/2" in result


# ──────────────────────────────────────────────────────────────────────
# Brain CLI Fallback Integration
# ──────────────────────────────────────────────────────────────────────


class TestBrainCliIntegration:
    """Test that CLI providers are initialized in brain config."""

    def test_config_defaults(self):
        from enton.core.config import Settings

        s = Settings()
        assert s.claude_code_enabled is True
        assert s.claude_code_model == "sonnet"
        assert s.claude_code_timeout == 120.0
        assert s.claude_code_max_turns == 10
        assert s.gemini_cli_enabled is True
        assert s.gemini_cli_model == "gemini-2.5-flash"
        assert s.gemini_cli_timeout == 120.0
        assert s.gemini_cli_yolo is False

    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_claude_provider_creation(self, _):
        p = ClaudeCodeProvider(model="sonnet", timeout=60.0, max_turns=5)
        assert p.available is True
        assert p.id == "claude-code:sonnet"

    @patch("shutil.which", return_value="/usr/bin/gemini")
    def test_gemini_provider_creation(self, _):
        p = GeminiCliProvider(model="gemini-2.5-flash", timeout=60.0)
        assert p.available is True
        assert p.id == "gemini-cli:gemini-2.5-flash"


# ──────────────────────────────────────────────────────────────────────
# ClaudeResult / GeminiResult dataclasses
# ──────────────────────────────────────────────────────────────────────


class TestDataclasses:
    def test_claude_result_defaults(self):
        r = ClaudeResult(content="test")
        assert r.content == "test"
        assert r.session_id == ""
        assert r.cost_usd == 0.0
        assert r.is_error is False

    def test_claude_result_frozen(self):
        r = ClaudeResult(content="test")
        with pytest.raises(AttributeError):
            r.content = "other"

    def test_gemini_result_defaults(self):
        r = GeminiResult(content="test")
        assert r.content == "test"
        assert r.is_error is False
        assert r.raw_output == ""

    def test_gemini_result_frozen(self):
        r = GeminiResult(content="test")
        with pytest.raises(AttributeError):
            r.content = "other"
