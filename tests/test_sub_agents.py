"""Tests for SubAgentOrchestrator."""
from __future__ import annotations

import asyncio

import pytest

from enton.cognition.sub_agents import (
    ROLE_CONFIGS,
    AgentResult,
    SubAgent,
    SubAgentOrchestrator,
)


class TestAgentResult:
    def test_summary(self):
        r = AgentResult(
            agent_role="coding",
            content="Here is the code...",
            confidence=0.85,
            elapsed_ms=150.0,
        )
        s = r.summary()
        assert "coding" in s
        assert "85%" in s
        assert "150ms" in s


class TestSubAgentOrchestrator:
    @pytest.fixture
    def orchestrator(self):
        from agno.models.ollama import Ollama
        model = Ollama(id="qwen2.5:14b")
        return SubAgentOrchestrator(models=[model], toolkits={})

    def test_list_agents(self, orchestrator):
        agents = orchestrator.list_agents()
        assert "vision" in agents
        assert "coding" in agents
        assert "research" in agents
        assert "system" in agents

    def test_classify_vision(self, orchestrator):
        assert orchestrator._classify_task("Descreva a cena da camera") == "vision"
        assert orchestrator._classify_task("Quem é esse rosto?") == "vision"

    def test_classify_coding(self, orchestrator):
        assert orchestrator._classify_task("Escreva um script Python") == "coding"
        assert orchestrator._classify_task("Debug este bug no codigo") == "coding"

    def test_classify_system(self, orchestrator):
        assert orchestrator._classify_task("Status da GPU") == "system"
        assert orchestrator._classify_task("Verifique o disco") == "system"

    def test_classify_research_default(self, orchestrator):
        assert orchestrator._classify_task("Qual a capital da França?") == "research"

    def test_summary(self, orchestrator):
        s = orchestrator.summary()
        assert "4 roles" in s

    def test_get_agent(self, orchestrator):
        agent = orchestrator.get_agent("coding")
        assert agent is not None
        assert agent.role == "coding"

    def test_get_unknown_agent(self, orchestrator):
        assert orchestrator.get_agent("unknown_role") is None

    def test_delegate_unknown_role(self, orchestrator):
        result = asyncio.get_event_loop().run_until_complete(
            orchestrator.delegate("nonexistent", "do something")
        )
        assert result.confidence == 0.0
        assert "nao existe" in result.content


class TestRoleConfigs:
    def test_all_roles_have_required_fields(self):
        for role, config in ROLE_CONFIGS.items():
            assert "name" in config, f"Role {role} missing 'name'"
            assert "system" in config, f"Role {role} missing 'system'"
            assert "toolkit_names" in config, f"Role {role} missing 'toolkit_names'"
            assert "description" in config, f"Role {role} missing 'description'"

    def test_system_prompts_in_portuguese(self):
        for role, config in ROLE_CONFIGS.items():
            system = config["system"]
            assert "voce" in system.lower() or "sua" in system.lower(), (
                f"Role {role} system prompt not in Portuguese"
            )


class TestSubAgent:
    def test_success_rate_default(self):
        from agno.models.ollama import Ollama
        model = Ollama(id="qwen2.5:14b")
        agent = SubAgent(role="research", models=[model])
        assert agent.success_rate == 1.0
