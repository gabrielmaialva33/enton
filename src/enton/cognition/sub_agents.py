"""Role-Specialized Sub-Agents — CrewAI-inspired delegation pattern.

Unlike GWT modules (which compete for conscious attention), sub-agents are
EXECUTORS: they receive a delegated task, use their specialized tools and
system prompt, and return a structured result.

Architecture:
    EntonBrain (general-purpose)
        ├── VisionAgent   — scene analysis, object tracking, face queries
        ├── CodingAgent   — code generation, review, debugging, execution
        ├── ResearchAgent — web search, knowledge crawling, fact-checking
        └── SystemAgent   — hardware monitoring, process management, GCP

Each sub-agent:
- Wraps an Agno Agent with role-specific system prompt
- Has a SUBSET of Enton's tools (only relevant ones)
- Shares the same model fallback chain as the main brain
- Returns structured results (not just text)

Pattern:
    1. Main brain recognizes task needs specialist
    2. Delegates via SubAgentTools toolkit
    3. Sub-agent executes with focused context
    4. Result flows back to main brain
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agno.agent import Agent
from agno.tools import Toolkit

if TYPE_CHECKING:
    from agno.models.base import Model

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AgentResult:
    """Structured result from a sub-agent execution."""

    agent_role: str
    content: str
    confidence: float = 0.7
    elapsed_ms: float = 0.0
    tools_used: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        tools = f", tools={self.tools_used}" if self.tools_used else ""
        return (
            f"[{self.agent_role}] {self.content[:100]}... "
            f"(conf={self.confidence:.0%}, {self.elapsed_ms:.0f}ms{tools})"
        )


# ------------------------------------------------------------------ #
# Role Definitions
# ------------------------------------------------------------------ #

ROLE_CONFIGS: dict[str, dict[str, Any]] = {
    "vision": {
        "name": "EntonVision",
        "system": (
            "Voce e o EntonVision, especialista em analise visual. "
            "Sua tarefa e analisar cenas, identificar objetos, descrever "
            "atividades e interpretar o que a camera esta vendo. "
            "Seja preciso e objetivo. Responda em portugues."
        ),
        "toolkit_names": [
            "describe_tools", "face_tools", "visual_memory_tools",
            "ptz_tools",
        ],
        "description": "Analise de cenas, objetos, faces e atividades visuais.",
    },
    "coding": {
        "name": "EntonCoder",
        "system": (
            "Voce e o EntonCoder, especialista em programacao. "
            "Voce sabe C, Rust, Zig, Python, Erlang/Elixir e mais. "
            "Sua tarefa e escrever, revisar, debuggar e executar codigo. "
            "Use as ferramentas de workspace e coding disponiveis. "
            "Priorize codigo correto, seguro e performatico."
        ),
        "toolkit_names": [
            "coding_tools", "shell_tools", "file_tools",
            "workspace_tools", "process_tools",
        ],
        "description": "Programacao multi-linguagem, review, debug e execucao.",
    },
    "research": {
        "name": "EntonResearch",
        "system": (
            "Voce e o EntonResearch, especialista em pesquisa. "
            "Sua tarefa e buscar informacoes na web, crawlear paginas, "
            "extrair conhecimento e sintetizar descobertas. "
            "Seja rigoroso com fontes. Responda em portugues."
        ),
        "toolkit_names": [
            "search_tools", "knowledge_tools", "memory_tools",
        ],
        "description": "Pesquisa web, knowledge crawling e sintese de informacoes.",
    },
    "system": {
        "name": "EntonSysAdmin",
        "system": (
            "Voce e o EntonSysAdmin, especialista em infraestrutura. "
            "Monitora hardware (CPU, GPU, RAM, disco), gerencia processos, "
            "deploya em GCP, e mantem o sistema saudavel. "
            "Seja conciso e objetivo em diagnosticos."
        ),
        "toolkit_names": [
            "system_tools", "workspace_tools", "process_tools",
            "gcp_tools", "shell_tools",
        ],
        "description": "Monitoramento de hardware, processos, GCP e infraestrutura.",
    },
}


class SubAgent:
    """A role-specialized agent with focused tools and system prompt."""

    def __init__(
        self,
        role: str,
        models: list[Model],
        toolkits: list[Toolkit] | None = None,
        system_prompt: str = "",
    ) -> None:
        self.role = role
        self._models = models
        self._system = system_prompt
        self._total_calls = 0
        self._total_errors = 0

        config = ROLE_CONFIGS.get(role, {})
        name = config.get("name", f"Enton_{role.title()}")
        if not system_prompt:
            self._system = config.get("system", "")

        self._agent = Agent(
            name=name,
            model=models[0] if models else None,
            tools=toolkits or [],
            instructions=[self._system] if self._system else None,
            tool_call_limit=5,
            retries=1,
            stream=False,
            telemetry=False,
            markdown=False,
        )

    async def execute(self, task: str) -> AgentResult:
        """Execute a task with fallback across models."""
        start = time.time()

        for model in self._models:
            try:
                self._agent.model = model
                response = await self._agent.arun(task)
                content = response.content or ""
                # Strip <think> tags
                import re
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

                elapsed = (time.time() - start) * 1000
                self._total_calls += 1

                mid = getattr(model, "id", "?")
                logger.info("SubAgent [%s/%s]: %s", self.role, mid, content[:80])

                return AgentResult(
                    agent_role=self.role,
                    content=content,
                    elapsed_ms=elapsed,
                    metadata={"model": mid},
                )
            except Exception:
                mid = getattr(model, "id", "?")
                logger.warning("SubAgent [%s/%s] failed", self.role, mid)
                self._total_errors += 1

        # All models failed
        elapsed = (time.time() - start) * 1000
        return AgentResult(
            agent_role=self.role,
            content=f"Erro: {self.role} agent falhou em todos os providers.",
            confidence=0.0,
            elapsed_ms=elapsed,
        )

    @property
    def success_rate(self) -> float:
        total = self._total_calls + self._total_errors
        return self._total_calls / total if total else 1.0


class SubAgentOrchestrator:
    """Manages and dispatches tasks to role-specialized sub-agents.

    This is the bridge between the main brain and the specialists.
    """

    def __init__(
        self,
        models: list[Model],
        toolkits: dict[str, Toolkit] | None = None,
    ) -> None:
        self._models = models
        self._all_toolkits = toolkits or {}
        self._agents: dict[str, SubAgent] = {}
        self._init_agents()

    def _init_agents(self) -> None:
        """Initialize sub-agents from role configs."""
        for role, config in ROLE_CONFIGS.items():
            # Resolve toolkit names to actual toolkit instances
            agent_toolkits: list[Toolkit] = []
            for tk_name in config.get("toolkit_names", []):
                tk = self._all_toolkits.get(tk_name)
                if tk:
                    agent_toolkits.append(tk)

            self._agents[role] = SubAgent(
                role=role,
                models=self._models,
                toolkits=agent_toolkits,
                system_prompt=config.get("system", ""),
            )
            logger.info(
                "SubAgent initialized: %s (%d tools)",
                role, len(agent_toolkits),
            )

    async def delegate(self, role: str, task: str) -> AgentResult:
        """Delegate a task to a specific sub-agent."""
        agent = self._agents.get(role)
        if not agent:
            return AgentResult(
                agent_role=role,
                content=f"Erro: sub-agent '{role}' nao existe. "
                f"Disponiveis: {', '.join(self._agents.keys())}",
                confidence=0.0,
            )

        logger.info("Delegating to [%s]: %s", role, task[:80])
        return await agent.execute(task)

    async def auto_delegate(self, task: str) -> AgentResult:
        """Automatically choose the best sub-agent for a task."""
        role = self._classify_task(task)
        return await self.delegate(role, task)

    def _classify_task(self, task: str) -> str:
        """Simple heuristic to classify which agent should handle a task."""
        t = task.lower()

        # Vision keywords
        vision_kw = [
            "camera", "imagem", "foto", "cena", "vendo", "olha",
            "rosto", "face", "visual", "descreva", "observ",
        ]
        if any(kw in t for kw in vision_kw):
            return "vision"

        # Coding keywords
        code_kw = [
            "codigo", "code", "python", "rust", "programar", "debug",
            "compilar", "script", "funcao", "classe", "bug", "implementar",
            "refatorar", "rodar", "executar codigo",
        ]
        if any(kw in t for kw in code_kw):
            return "coding"

        # System keywords
        sys_kw = [
            "cpu", "gpu", "ram", "disco", "processo", "hardware",
            "gcp", "vm", "cloud", "deploy", "sistema", "monitor",
        ]
        if any(kw in t for kw in sys_kw):
            return "system"

        # Default: research
        return "research"

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #

    def list_agents(self) -> dict[str, dict]:
        """List available sub-agents with their stats."""
        return {
            role: {
                "description": ROLE_CONFIGS.get(role, {}).get("description", ""),
                "success_rate": agent.success_rate,
                "total_calls": agent._total_calls,
            }
            for role, agent in self._agents.items()
        }

    def get_agent(self, role: str) -> SubAgent | None:
        return self._agents.get(role)

    def summary(self) -> str:
        total = sum(a._total_calls for a in self._agents.values())
        return (
            f"SubAgents: {len(self._agents)} roles, "
            f"{total} total calls"
        )
