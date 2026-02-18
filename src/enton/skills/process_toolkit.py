"""Process management toolkit — Enton's background task orchestrator.

Inspired by wcgw's persistent shell sessions. Enton can run long commands
in background, track their progress, retry on failure, and manage multiple
concurrent tasks without blocking conversation.
"""

from __future__ import annotations

import logging

from agno.tools import Toolkit

from enton.core.process_manager import ProcessManager, TaskStatus

logger = logging.getLogger(__name__)


class ProcessTools(Toolkit):
    """Gerenciamento de processos em background — multi-tasking do Enton."""

    def __init__(self, manager: ProcessManager, cwd: str = "") -> None:
        super().__init__(name="process_tools")
        self._pm = manager
        self._cwd = cwd
        self.register(self.task_run)
        self.register(self.task_status)
        self.register(self.task_list)
        self.register(self.task_output)
        self.register(self.task_cancel)
        self.register(self.task_summary)

    async def task_run(
        self,
        name: str,
        command: str,
        timeout: float = 300.0,
        retries: int = 1,
    ) -> str:
        """Executa um comando em background sem bloquear a conversa.

        O comando roda num processo separado. Use task_status pra checar
        e task_output pra ver o resultado quando terminar.

        Args:
            name: Nome descritivo da task (ex: 'compilar-yolo', 'treinar-modelo').
            command: Comando bash pra executar.
            timeout: Timeout em segundos (default: 300 = 5min).
            retries: Tentativas em caso de falha (default: 1 = sem retry).
        """
        task_id = await self._pm.submit(
            name=name,
            command=command,
            timeout=timeout,
            max_retries=max(0, retries - 1),
            cwd=self._cwd,
        )
        return (
            f"Task '{name}' submetida em background!\n"
            f"  ID: {task_id}\n"
            f"  Comando: {command[:100]}\n"
            f"  Timeout: {timeout}s | Retries: {retries}\n"
            f"Use task_status('{task_id}') pra acompanhar."
        )

    async def task_status(self, task_id: str) -> str:
        """Verifica o status de uma task em background.

        Args:
            task_id: ID da task (retornado por task_run).
        """
        task = self._pm.get(task_id)
        if not task:
            return f"Task '{task_id}' nao encontrada."
        return task.summary()

    async def task_list(self, filter_status: str = "") -> str:
        """Lista todas as tasks em background.

        Args:
            filter_status: Filtrar por status (running/completed/failed). Vazio = todas.
        """
        status_filter = None
        if filter_status:
            try:
                status_filter = TaskStatus(filter_status.lower())
            except ValueError:
                return (
                    f"Status invalido: '{filter_status}'. "
                    f"Use: running, completed, failed, cancelled"
                )

        tasks = self._pm.list_tasks(status=status_filter)
        if not tasks:
            suffix = f" com status '{filter_status}'" if filter_status else ""
            return f"Nenhuma task{suffix}."

        lines = [f"Tasks ({len(tasks)}):"]
        for t in tasks:
            lines.append(f"  {t.summary()}")
        lines.append(f"\n{self._pm.summary()}")
        return "\n".join(lines)

    async def task_output(self, task_id: str) -> str:
        """Mostra o output completo de uma task finalizada.

        Args:
            task_id: ID da task.
        """
        task = self._pm.get(task_id)
        if not task:
            return f"Task '{task_id}' nao encontrada."

        if not task.is_done and task.status != TaskStatus.RUNNING:
            return f"Task '{task.name}' ainda nao iniciou."

        lines = [
            f"Task: {task.name} [{task.status}]",
            f"Comando: {task.command}",
            f"Tempo: {task.elapsed:.1f}s",
        ]
        if task.exit_code is not None:
            lines.append(f"Exit code: {task.exit_code}")
        if task.output:
            # Truncate very long output
            out = task.output
            if len(out) > 5000:
                out = out[:2000] + "\n...(truncado)...\n" + out[-2000:]
            lines.append(f"\n--- STDOUT ---\n{out}")
        if task.error:
            err = task.error
            if len(err) > 2000:
                err = err[:1000] + "\n...(truncado)...\n" + err[-500:]
            lines.append(f"\n--- STDERR ---\n{err}")
        return "\n".join(lines)

    async def task_cancel(self, task_id: str) -> str:
        """Cancela uma task em execucao.

        Args:
            task_id: ID da task pra cancelar.
        """
        ok = await self._pm.cancel(task_id)
        if ok:
            return f"Task '{task_id}' cancelada."
        task = self._pm.get(task_id)
        if task and task.is_done:
            return f"Task '{task_id}' ja terminou ({task.status})."
        return f"Task '{task_id}' nao encontrada ou nao pode ser cancelada."

    async def task_summary(self) -> str:
        """Resumo rapido de todas as tasks — quantas rodando, completadas, etc.

        Args:
            (nenhum)
        """
        summary = self._pm.summary()
        active = self._pm.list_tasks(status=TaskStatus.RUNNING)
        if active:
            summary += "\n\nRodando agora:"
            for t in active:
                summary += f"\n  {t.id[:8]}: {t.name} ({t.elapsed:.0f}s)"
        return summary
