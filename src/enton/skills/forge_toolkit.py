"""ForgeTools — Agno toolkit that lets the Brain create its own tools.

Implements LATM (LLM-as-Tool-Maker): expensive model creates tools,
cheap model uses them. The Brain can invoke these functions via
native Agno tool-calling.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agno.tools import Toolkit

if TYPE_CHECKING:
    from enton.skills.forge_engine import ForgeEngine
    from enton.skills.skill_registry import SkillRegistry

logger = logging.getLogger(__name__)


class ForgeTools(Toolkit):
    """Tools for self-improvement: create, list, and retire dynamic tools."""

    def __init__(
        self,
        forge: ForgeEngine,
        registry: SkillRegistry,
    ) -> None:
        super().__init__(name="forge_tools")
        self._forge = forge
        self._registry = registry
        self.register(self.create_tool)
        self.register(self.list_dynamic_tools)
        self.register(self.retire_tool)
        self.register(self.tool_stats)

    async def create_tool(self, task_description: str) -> str:
        """Cria uma nova ferramenta automaticamente a partir de uma descricao.

        Usa o padrao LATM (LLM-as-Tool-Maker): gera codigo Python,
        testa em sandbox isolado, e disponibiliza como ferramenta do Enton.

        Args:
            task_description: Descricao clara do que a ferramenta deve fazer.
        """
        result = await self._forge.create_tool(task_description)
        if result["success"]:
            return (
                f"Ferramenta '{result['name']}' criada com sucesso! "
                f"Descricao: {result['description']}. "
                "Ja esta disponivel para uso."
            )
        return f"Falha ao criar ferramenta: {result.get('error', 'erro desconhecido')}"

    def list_dynamic_tools(self) -> str:
        """Lista todas as ferramentas dinamicas carregadas no momento.

        Mostra nome, descricao, versao e taxa de sucesso de cada skill.
        """
        skills = self._registry.loaded_skills
        if not skills:
            return "Nenhuma ferramenta dinamica carregada."
        lines = []
        for name, meta in skills.items():
            total = meta.success_count + meta.failure_count
            rate = f"{meta.success_rate:.0%}" if total > 0 else "n/a"
            lines.append(
                f"- {name}: {meta.description} "
                f"(v{meta.version}, taxa sucesso: {rate})"
            )
        return "\n".join(lines)

    def retire_tool(self, tool_name: str) -> str:
        """Remove uma ferramenta dinamica (exclui o arquivo e descarrega).

        Args:
            tool_name: Nome da ferramenta a remover.
        """
        if self._forge.retire_tool(tool_name):
            return f"Ferramenta '{tool_name}' removida com sucesso."
        return f"Ferramenta '{tool_name}' nao encontrada."

    def tool_stats(self) -> str:
        """Mostra estatisticas de todas as ferramentas criadas pelo ToolForge.

        Inclui contagem de sucesso/falha e taxa de acerto.
        """
        stats = self._forge.get_tool_stats()
        if not stats:
            return "Nenhuma ferramenta foi criada pelo ToolForge ainda."
        lines = []
        for s in stats:
            lines.append(
                f"- {s['name']}: {s['success_count']} ok, "
                f"{s['failure_count']} falhas, "
                f"taxa: {s['success_rate']:.0%}"
            )
        return "\n".join(lines)
