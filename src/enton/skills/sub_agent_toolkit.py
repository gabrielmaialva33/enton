"""SubAgentTools — Agno Toolkit for delegating tasks to specialized sub-agents.

Exposes the SubAgentOrchestrator to the LLM so Enton can:
- Delegate tasks to the best specialist (auto or manual)
- List available sub-agents and their capabilities
- Run consensus across multiple agents
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from agno.tools import Toolkit

if TYPE_CHECKING:
    from enton.cognition.sub_agents import SubAgentOrchestrator

logger = logging.getLogger(__name__)


class SubAgentTools(Toolkit):
    """Delegacao de tarefas para sub-agentes especializados."""

    def __init__(self, orchestrator: SubAgentOrchestrator) -> None:
        super().__init__(name="sub_agent_tools")
        self._orchestrator = orchestrator
        self.register(self.delegate_task)
        self.register(self.auto_delegate)
        self.register(self.agent_consensus)
        self.register(self.list_agents)

    async def delegate_task(self, role: str, task: str) -> str:
        """Delega uma tarefa para um sub-agente especializado.

        Sub-agentes disponiveis: vision, coding, research, system.
        Cada um tem ferramentas e prompt especificos pro seu papel.

        Args:
            role: Papel do agente (vision/coding/research/system).
            task: Descricao da tarefa para o agente executar.
        """
        result = await self._orchestrator.delegate(role, task)

        parts = [f"[{result.agent_role}]"]
        if result.content:
            parts.append(result.content)
        parts.append(f"\n(confianca: {result.confidence:.0%}, tempo: {result.elapsed_ms:.0f}ms)")

        return "\n".join(parts)

    async def auto_delegate(self, task: str) -> str:
        """Delega uma tarefa automaticamente para o melhor sub-agente.

        Analisa a tarefa e escolhe o especialista mais adequado:
        - vision: cameras, imagens, cenas, rostos
        - coding: programacao, debug, scripts, compilacao
        - research: pesquisa, busca, conhecimento
        - system: hardware, processos, GCP, infraestrutura

        Args:
            task: Descricao da tarefa para classificar e executar.
        """
        result = await self._orchestrator.auto_delegate(task)

        parts = [
            f"[auto → {result.agent_role}]",
            result.content,
            f"\n(confianca: {result.confidence:.0%}, tempo: {result.elapsed_ms:.0f}ms)",
        ]

        return "\n".join(parts)

    async def agent_consensus(self, task: str) -> str:
        """Pede a MESMA tarefa para multiplos sub-agentes e compara resultados.

        Util quando precisa de uma resposta confiavel — varios especialistas
        respondem a mesma pergunta. Roda vision + research + coding em paralelo.

        Args:
            task: A tarefa para todos os agentes executarem.
        """
        roles = ["vision", "research", "coding", "system"]
        coros = [self._orchestrator.delegate(role, task) for role in roles]
        results = await asyncio.gather(*coros, return_exceptions=True)

        parts = [f"=== Consensus ({len(roles)} agentes) ===\n"]
        for role, result in zip(roles, results, strict=True):
            if isinstance(result, Exception):
                parts.append(f"[{role}] ERRO: {result}\n")
            else:
                parts.append(
                    f"[{result.agent_role}] (conf={result.confidence:.0%})\n{result.content}\n"
                )

        return "\n---\n".join(parts)

    async def list_agents(self) -> str:
        """Lista todos os sub-agentes disponiveis e suas estatisticas.

        Mostra papel, descricao, taxa de sucesso e total de chamadas.
        """
        agents = self._orchestrator.list_agents()
        if not agents:
            return "Nenhum sub-agente configurado."

        lines = [f"Sub-agentes ({len(agents)}):"]
        for role, info in agents.items():
            rate = info.get("success_rate", 1.0)
            calls = info.get("total_calls", 0)
            desc = info.get("description", "")
            lines.append(f"  [{role}] {desc} (sucesso: {rate:.0%}, calls: {calls})")

        return "\n".join(lines)
