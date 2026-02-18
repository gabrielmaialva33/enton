"""ExtensionTools — Agno Toolkit for managing Enton extensions at runtime.

Exposes the ExtensionRegistry to the LLM so Enton can:
- List installed extensions and their status
- Enable/disable extensions dynamically
- Install new extensions from git repos
- View extension stats and health
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agno.tools import Toolkit

if TYPE_CHECKING:
    from enton.core.extension_registry import ExtensionRegistry

logger = logging.getLogger(__name__)


class ExtensionTools(Toolkit):
    """Gerenciamento de extensoes/plugins do Enton."""

    def __init__(self, registry: ExtensionRegistry) -> None:
        super().__init__(name="extension_tools")
        self._registry = registry
        self.register(self.extension_list)
        self.register(self.extension_enable)
        self.register(self.extension_disable)
        self.register(self.extension_install)
        self.register(self.extension_stats)

    async def extension_list(self, filter_state: str = "") -> str:
        """Lista todas as extensoes instaladas no Enton.

        Mostra nome, estado (enabled/disabled/error), fonte e numero de tools.
        Use filter_state para filtrar: 'enabled', 'disabled', 'error'.

        Args:
            filter_state: Filtro opcional por estado.
        """
        from enton.core.extension_registry import ExtensionState

        state = None
        if filter_state:
            try:
                state = ExtensionState(filter_state.lower())
            except ValueError:
                return (
                    f"Estado invalido: '{filter_state}'. "
                    "Use: enabled, disabled, discovered, loaded, error"
                )

        exts = self._registry.list_extensions(state=state)
        if not exts:
            return (
                "Nenhuma extensao encontrada"
                + (f" com estado '{filter_state}'" if filter_state else "")
                + "."
            )

        lines = [f"Extensoes ({len(exts)}):"]
        for ext in exts:
            lines.append(f"  {ext.summary()}")

        return "\n".join(lines)

    async def extension_enable(self, name: str) -> str:
        """Ativa uma extensao pelo nome.

        A extensao precisa estar instalada (descoberta) para ser ativada.
        Ao ativar, as ferramentas da extensao ficam disponiveis pro brain.

        Args:
            name: Nome da extensao para ativar.
        """
        ok = self._registry.enable(name)
        if ok:
            meta = self._registry.get(name)
            tools = meta.tool_count if meta else 0
            return f"Extensao '{name}' ativada com {tools} ferramentas."
        return f"Falha ao ativar extensao '{name}'. Verifique se existe com extension_list."

    async def extension_disable(self, name: str) -> str:
        """Desativa uma extensao pelo nome.

        Remove as ferramentas da extensao do brain. Pode ser reativada depois.

        Args:
            name: Nome da extensao para desativar.
        """
        ok = self._registry.disable(name)
        if ok:
            return f"Extensao '{name}' desativada."
        return f"Falha ao desativar extensao '{name}'. Verifique se esta ativa."

    async def extension_install(self, repo_url: str, name: str = "") -> str:
        """Instala uma extensao de um repositorio git.

        Clona o repo, descobre o manifest.json e ativa automaticamente.
        O repo precisa ter um manifest.json na raiz.

        Args:
            repo_url: URL do repositorio git.
            name: Nome customizado (opcional, extrai do URL).
        """
        ok = await self._registry.install_from_git(repo_url, name=name)
        if ok:
            final_name = name or repo_url.rstrip("/").split("/")[-1]
            return f"Extensao '{final_name}' instalada e ativada."
        return f"Falha ao instalar extensao de {repo_url}."

    async def extension_stats(self) -> str:
        """Mostra estatisticas do registro de extensoes.

        Inclui total de extensoes, tools disponiveis, e status geral.
        """
        s = self._registry.stats()
        lines = [
            f"Total de extensoes: {s['total_extensions']}",
            f"Tools disponiveis: {s['total_tools']}",
            "",
            "Por estado:",
        ]
        for state, count in s["by_state"].items():
            lines.append(f"  {state}: {count}")

        lines.append("")
        lines.append("Por fonte:")
        for source, count in s["by_source"].items():
            lines.append(f"  {source}: {count}")

        return "\n".join(lines)
