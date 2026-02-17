"""Agno Toolkit for safe shell command execution with persistent CWD."""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
import uuid

from agno.tools import Toolkit

from enton.skills._shell_state import BackgroundProcess, ShellState

logger = logging.getLogger(__name__)

# Permission levels for commands
SAFE_COMMANDS = frozenset({
    "ls", "cat", "head", "tail", "pwd", "date", "uptime", "df", "free",
    "ps", "whoami", "id", "groups", "hostname", "which", "echo", "wc",
    "du", "file", "stat", "uname", "lsb_release", "ip", "ss", "ping",
    "dig", "nslookup", "curl", "wget", "env", "printenv", "locale",
    "docker", "systemctl", "journalctl", "top", "htop", "nvtop",
    "nvidia-smi", "sensors", "lsblk", "lsusb", "lspci", "lscpu",
    "git", "python", "pip", "uv", "ollama", "ruff", "pytest",
    "find", "grep", "rg", "sort", "uniq", "cut", "tr", "awk", "sed",
    "xdg-open", "code", "flatpak", "cd", "mkdir", "touch", "cp", "mv",
    "less", "more", "tree", "realpath", "dirname", "basename",
})

ELEVATED_COMMANDS = frozenset({
    "apt", "apt-get", "dpkg", "snap", "kill", "killall", "pkill",
    "service", "mount", "umount", "chown", "chmod",
    "pip install", "uv add", "npm install",
    "crontab", "at", "nohup",
})

DANGEROUS_PATTERNS = frozenset({
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=", "fdisk",
    "shutdown", "reboot", "poweroff", "halt",
    "> /dev/sd", "wipefs", "shred",
    ":(){ :|:& };:",  # fork bomb
})

_MAX_OUTPUT = 4000
_TIMEOUT = 30.0
_CWD_MARKER = "<<<CWD>>>"


def _classify_command(command: str) -> str:
    """Classify a command's risk level.

    Returns one of: "safe", "elevated", "dangerous".
    """
    cmd_lower = command.strip().lower()

    for pattern in DANGEROUS_PATTERNS:
        if pattern in cmd_lower:
            return "dangerous"

    try:
        parts = shlex.split(command)
    except ValueError:
        return "elevated"

    base = parts[0] if parts else ""

    # sudo wrapping -- classify the inner command
    if base == "sudo" and len(parts) > 1:
        inner_cmd = " ".join(parts[1:])
        inner_base = parts[1]
        for pattern in ELEVATED_COMMANDS:
            if inner_cmd.startswith(pattern):
                return "elevated"
        if inner_base in SAFE_COMMANDS:
            return "safe"
        return "elevated"

    # Check multi-word elevated patterns before single-word safe
    full_cmd = " ".join(parts)
    for pattern in ELEVATED_COMMANDS:
        if full_cmd.startswith(pattern):
            return "elevated"

    if base in SAFE_COMMANDS:
        return "safe"
    if base in ELEVATED_COMMANDS:
        return "elevated"

    return "elevated"


class ShellTools(Toolkit):
    """Safe shell command execution with CWD tracking and background support."""

    def __init__(self, state: ShellState | None = None):
        super().__init__(name="shell_tools")
        self._state = state or ShellState()
        self.register(self.run_command)
        self.register(self.run_command_sudo)
        self.register(self.get_cwd)
        self.register(self.run_background)
        self.register(self.check_background)
        self.register(self.stop_background)

    def _wrap_command(self, command: str) -> str:
        """Wrap command with CWD tracking."""
        cwd = shlex.quote(str(self._state.cwd))
        return (
            f"cd {cwd} && {{ {command} ; }}; "
            f"__e=$?; echo '{_CWD_MARKER}'\"$(pwd)\"'{_CWD_MARKER}'; "
            f"exit $__e"
        )

    def _parse_cwd(self, output: str) -> str:
        """Extract and update CWD from output, return cleaned output."""
        pattern = re.compile(
            re.escape(_CWD_MARKER) + r"(.+?)" + re.escape(_CWD_MARKER)
        )
        match = pattern.search(output)
        if match:
            from pathlib import Path

            new_cwd = Path(match.group(1).strip())
            if new_cwd.is_dir():
                self._state.cwd = new_cwd
            output = pattern.sub("", output).rstrip("\n")
        return output

    async def run_command(self, command: str) -> str:
        """Executa um comando no terminal Linux com diretorio persistente.

        O diretorio atual persiste entre chamadas (cd funciona).
        Comandos seguros executam direto, elevados com aviso, perigosos bloqueados.

        Args:
            command: O comando shell completo a ser executado.
        """
        level = _classify_command(command)

        if level == "dangerous":
            logger.warning("BLOCKED dangerous command: %s", command)
            return (
                f"BLOQUEADO: Comando '{command}' e perigoso demais. "
                "Recuso executar isso."
            )

        if level == "elevated":
            logger.warning("Elevated command: %s", command)

        wrapped = self._wrap_command(command)

        try:
            proc = await asyncio.create_subprocess_shell(
                wrapped,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_TIMEOUT
            )

            output = stdout.decode(errors="replace").strip()
            output = self._parse_cwd(output)
            err = stderr.decode(errors="replace").strip()

            result_parts: list[str] = []
            if output:
                result_parts.append(output[:_MAX_OUTPUT])
            if err:
                result_parts.append(f"STDERR: {err[:1000]}")
            result_parts.append(f"[cwd: {self._state.cwd}] Exit code: {proc.returncode}")

            result = "\n".join(result_parts)
            logger.info(
                "Shell [%s] (%s): exit %d",
                level, command[:60], proc.returncode,
            )
            return result

        except TimeoutError:
            return f"TIMEOUT: Comando '{command}' excedeu {_TIMEOUT:.0f} segundos."
        except Exception as e:
            logger.exception("Shell execution failed: %s", command)
            return f"ERRO: {e}"

    async def run_command_sudo(self, command: str) -> str:
        """Executa um comando com sudo (acesso root). Use com cuidado.

        Args:
            command: O comando a executar como root (sem o prefixo 'sudo').
        """
        return await self.run_command(f"sudo {command}")

    async def get_cwd(self) -> str:
        """Retorna o diretorio de trabalho atual do shell."""
        return str(self._state.cwd)

    async def run_background(self, command: str) -> str:
        """Executa um comando em background (longa duracao).

        Retorna um ID para consultar status depois com check_background.

        Args:
            command: O comando a executar em background.
        """
        level = _classify_command(command)
        if level == "dangerous":
            return f"BLOQUEADO: Comando '{command}' e perigoso demais."

        bg_id = uuid.uuid4().hex[:8]
        cwd = str(self._state.cwd)

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
        )

        bp = BackgroundProcess(id=bg_id, command=command, process=proc)
        self._state.background[bg_id] = bp

        # Start reader task
        asyncio.create_task(self._read_bg_output(bp))

        logger.info("Background [%s] started: %s", bg_id, command[:60])
        return f"Background iniciado. ID: {bg_id}\nComando: {command}"

    async def _read_bg_output(self, bp: BackgroundProcess) -> None:
        """Read background process output into buffer."""
        assert bp.process.stdout is not None
        try:
            async for line in bp.process.stdout:
                bp.output.append(line.decode(errors="replace").rstrip())
        except Exception:
            pass
        finally:
            bp.done = True

    async def check_background(self, bg_id: str) -> str:
        """Verifica o status de um comando em background.

        Args:
            bg_id: O ID retornado por run_background.
        """
        bp = self._state.background.get(bg_id)
        if bp is None:
            ids = ", ".join(self._state.background.keys()) or "nenhum"
            return f"ID '{bg_id}' nao encontrado. IDs ativos: {ids}"

        status = "concluido" if bp.done else "rodando"
        lines = list(bp.output)[-30:]  # last 30 lines
        output = "\n".join(lines) if lines else "(sem output ainda)"

        ret = bp.process.returncode
        code_str = f" (exit {ret})" if ret is not None else ""

        return f"Status: {status}{code_str}\nComando: {bp.command}\n---\n{output}"

    async def stop_background(self, bg_id: str) -> str:
        """Para um comando em background.

        Args:
            bg_id: O ID retornado por run_background.
        """
        bp = self._state.background.get(bg_id)
        if bp is None:
            return f"ID '{bg_id}' nao encontrado."

        if not bp.done:
            try:
                bp.process.terminate()
                await asyncio.wait_for(bp.process.wait(), timeout=5.0)
            except TimeoutError:
                bp.process.kill()

        del self._state.background[bg_id]
        return f"Background '{bg_id}' parado e removido."
