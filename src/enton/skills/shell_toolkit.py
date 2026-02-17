"""Agno Toolkit for safe shell command execution."""

from __future__ import annotations

import asyncio
import logging
import shlex

from agno.tools import Toolkit

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
    "xdg-open", "code", "flatpak",
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

_MAX_OUTPUT = 2000
_TIMEOUT = 30.0


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
    """Safe shell command execution with risk classification."""

    def __init__(self):
        super().__init__(name="shell_tools")
        self.register(self.run_command)
        self.register(self.run_command_sudo)

    async def run_command(self, command: str) -> str:
        """Executa um comando no terminal do sistema Linux.

        Comandos seguros (ls, cat, ps, docker, git, nvidia-smi, etc) executam direto.
        Comandos elevados (apt, kill, mount) executam com aviso.
        Comandos perigosos (rm -rf /, mkfs, shutdown) sao bloqueados.

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

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_TIMEOUT
            )

            output = stdout.decode(errors="replace").strip()
            err = stderr.decode(errors="replace").strip()

            result_parts: list[str] = []
            if output:
                result_parts.append(output[:_MAX_OUTPUT])
            if err:
                result_parts.append(f"STDERR: {err[:1000]}")
            result_parts.append(f"Exit code: {proc.returncode}")

            result = "\n".join(result_parts)
            logger.info(
                "Shell [%s] (%s): %s -> exit %d",
                level, command[:60], result[:80], proc.returncode,
            )
            return result

        except TimeoutError:
            return f"TIMEOUT: Comando '{command}' excedeu {_TIMEOUT:.0f} segundos."
        except Exception as e:
            logger.exception("Shell execution failed: %s", command)
            return f"ERRO: {e}"

    async def run_command_sudo(self, command: str) -> str:
        """Executa um comando com sudo (acesso root). Use com cuidado.

        O comando e executado com prefixo sudo. As mesmas regras de seguranca
        se aplicam: comandos perigosos continuam bloqueados.

        Args:
            command: O comando a executar como root (sem o prefixo 'sudo').
        """
        return await self.run_command(f"sudo {command}")
