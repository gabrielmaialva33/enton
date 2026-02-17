"""AIDelegateTools — Enton delegates tasks to Claude Code & Gemini CLI.

Gives Enton the ability to:
- Ask Claude Code or Gemini CLI for help
- Delegate full coding tasks (agentic mode)
- Run multi-AI consensus (ask both, compare)
- Deep research via Gemini (Google-grounded)

This is meta-AI: an AI that uses other AIs as tools.
"""
from __future__ import annotations

import asyncio
import logging

from agno.tools import Toolkit

from enton.providers.claude_code import ClaudeCodeProvider
from enton.providers.gemini_cli import GeminiCliProvider

logger = logging.getLogger(__name__)


class AIDelegateTools(Toolkit):
    """Toolkit for delegating tasks to external AI CLIs."""

    def __init__(
        self,
        claude: ClaudeCodeProvider | None = None,
        gemini: GeminiCliProvider | None = None,
    ) -> None:
        super().__init__(name="ai_delegate")
        self._claude = claude or ClaudeCodeProvider()
        self._gemini = gemini or GeminiCliProvider()

        # Register tools based on availability
        self.register(self.ask_claude)
        self.register(self.ask_gemini)
        self.register(self.claude_code_task)
        self.register(self.gemini_code_task)
        self.register(self.ai_consensus)
        self.register(self.ai_research)
        self.register(self.ai_status)

    async def ask_claude(self, prompt: str, context: str = "") -> str:
        """Delega uma pergunta ou tarefa pro Claude Code CLI.

        Use quando precisa de ajuda com codigo, analise, ou raciocinio complexo.
        Claude Code tem acesso a ferramentas de leitura/escrita de arquivos.

        Args:
            prompt: A pergunta ou tarefa para o Claude.
            context: Contexto adicional opcional (codigo, logs, etc).
        """
        if not self._claude.available:
            return (
                "Claude Code CLI nao esta instalado. "
                "Instale com: npm i -g @anthropic-ai/claude-code"
            )

        full_prompt = prompt
        if context:
            full_prompt = f"{prompt}\n\nContexto:\n{context}"

        result = await self._claude.generate(full_prompt)
        if not result:
            return "Claude Code nao retornou resposta."
        return f"[Claude Code]\n{result}"

    async def ask_gemini(self, prompt: str, context: str = "") -> str:
        """Delega uma pergunta ou tarefa pro Gemini CLI.

        Use quando precisa de pesquisa web, informacoes atualizadas,
        ou uma segunda opiniao sobre algo.

        Args:
            prompt: A pergunta ou tarefa para o Gemini.
            context: Contexto adicional opcional (codigo, logs, etc).
        """
        if not self._gemini.available:
            return "Gemini CLI nao esta instalado. Instale com: npm i -g @google/gemini-cli"

        full_prompt = prompt
        if context:
            full_prompt = f"{prompt}\n\nContexto:\n{context}"

        result = await self._gemini.generate(full_prompt)
        if not result:
            return "Gemini CLI nao retornou resposta."
        return f"[Gemini CLI]\n{result}"

    async def claude_code_task(self, task: str, directory: str = "") -> str:
        """Delega uma tarefa de CODIGO pro Claude Code (modo agentico completo).

        Claude Code vai ler arquivos, escrever codigo, rodar comandos, etc.
        Use pra tarefas complexas de programacao que precisam de multiplos passos.

        Args:
            task: Descricao da tarefa de codigo.
            directory: Diretorio de trabalho (padrao: diretorio atual).
        """
        if not self._claude.available:
            return "Claude Code CLI nao esta instalado."

        result = await self._claude.code_task(
            task,
            working_dir=directory or None,
        )
        if not result:
            return "Claude Code nao completou a tarefa."
        return f"[Claude Code Task]\n{result}"

    async def gemini_code_task(self, task: str, directory: str = "") -> str:
        """Delega uma tarefa de CODIGO pro Gemini CLI (modo agentico com --yolo).

        Gemini vai ler/escrever arquivos e rodar comandos automaticamente.
        Use pra tarefas que precisam de acesso web ou pesquisa junto.

        Args:
            task: Descricao da tarefa de codigo.
            directory: Diretorio de trabalho (padrao: diretorio atual).
        """
        if not self._gemini.available:
            return "Gemini CLI nao esta instalado."

        result = await self._gemini.code_task(
            task,
            working_dir=directory or None,
        )
        if not result:
            return "Gemini CLI nao completou a tarefa."
        return f"[Gemini Code Task]\n{result}"

    async def ai_consensus(self, question: str) -> str:
        """Pergunta a MESMA coisa pro Claude Code e Gemini, compara respostas.

        Use quando quer uma resposta mais confiavel — duas IAs independentes
        respondendo a mesma pergunta. Retorna ambas respostas pra comparacao.

        Args:
            question: A pergunta para ambas as IAs.
        """
        tasks = []
        labels = []

        if self._claude.available:
            tasks.append(self._claude.generate(question))
            labels.append("Claude Code")

        if self._gemini.available:
            tasks.append(self._gemini.generate(question))
            labels.append("Gemini CLI")

        if not tasks:
            return "Nenhum AI CLI disponivel. Instale claude e/ou gemini."

        results = await asyncio.gather(*tasks, return_exceptions=True)

        parts = []
        for label, result in zip(labels, results, strict=True):
            if isinstance(result, Exception):
                parts.append(f"[{label}] ERRO: {result}")
            elif result:
                parts.append(f"[{label}]\n{result}")
            else:
                parts.append(f"[{label}] Sem resposta.")

        header = f"=== Consensus ({len(labels)} IAs) ==="
        return f"{header}\n\n" + "\n\n---\n\n".join(parts)

    async def ai_research(self, topic: str) -> str:
        """Pesquisa profunda sobre um topico usando Gemini CLI.

        Gemini tem acesso ao Google Search integrado, ideal pra pesquisas.
        Se Gemini nao ta disponivel, usa Claude Code como fallback.

        Args:
            topic: O topico para pesquisar.
        """
        if self._gemini.available:
            result = await self._gemini.research(topic)
            if result:
                return f"[Gemini Research]\n{result}"

        if self._claude.available:
            result = await self._claude.generate(
                f"Pesquise e explique detalhadamente: {topic}",
            )
            if result:
                return f"[Claude Research]\n{result}"

        return "Nenhum AI CLI disponivel para pesquisa."

    async def ai_status(self) -> str:
        """Verifica quais AI CLIs estao disponiveis no sistema.

        Retorna o status de instalacao do Claude Code e Gemini CLI.
        """
        lines = []

        claude_ok = self._claude.available
        lines.append(
            f"Claude Code CLI: {'DISPONIVEL' if claude_ok else 'NAO INSTALADO'}"
            f" ({self._claude.id})"
        )

        gemini_ok = self._gemini.available
        lines.append(
            f"Gemini CLI: {'DISPONIVEL' if gemini_ok else 'NAO INSTALADO'}"
            f" ({self._gemini.id})"
        )

        total = sum([claude_ok, gemini_ok])
        lines.append(f"\nTotal: {total}/2 providers de AI CLI ativos.")

        return "\n".join(lines)
