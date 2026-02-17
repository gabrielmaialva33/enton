"""System monitoring toolkit — CPU, memory, disk, battery, GPU, processes."""

from __future__ import annotations

import datetime
import logging
import platform
import subprocess

import psutil
from agno.tools import Toolkit

logger = logging.getLogger(__name__)


class SystemTools(Toolkit):
    """Tools for querying host system health and status."""

    def __init__(self) -> None:
        super().__init__(name="system_tools")
        self.register(self.get_system_stats)
        self.register(self.get_time)
        self.register(self.list_processes)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def get_system_stats(self) -> str:
        """Retorna estatisticas atuais do sistema (CPU, Memoria, Disco, Bateria, GPU).

        Args:
            (nenhum)
        """
        uname = platform.uname()
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        stats = [
            f"OS: {uname.system} {uname.release} ({uname.machine})",
            f"CPU: {cpu}%",
            f"Memory: {mem.percent}% ({mem.used // (1024**3)}GB/{mem.total // (1024**3)}GB)",
            f"Disk: {disk.percent}% ({disk.used // (1024**3)}GB/{disk.total // (1024**3)}GB)",
        ]

        # Battery (laptops)
        if hasattr(psutil, "sensors_battery"):
            battery = psutil.sensors_battery()
            if battery:
                plug = "(Charging)" if battery.power_plugged else "(Discharging)"
                stats.append(f"Battery: {battery.percent}% {plug}")

        # GPU via nvidia-smi
        gpu_info = self._get_gpu_info()
        if gpu_info:
            stats.append(gpu_info)

        return "\n".join(stats)

    def get_time(self) -> str:
        """Retorna a data e hora atual no formato ISO.

        Args:
            (nenhum)
        """
        return datetime.datetime.now().isoformat()

    def list_processes(self, limit: int = 5) -> str:
        """Lista os processos que mais consomem CPU.

        Args:
            limit: Numero de processos a listar (default: 5).
        """
        procs: list[dict] = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                procs.append(p.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        procs.sort(key=lambda x: x.get("cpu_percent", 0) or 0, reverse=True)

        lines: list[str] = []
        for p in procs[:limit]:
            pid = p.get("pid", "?")
            name = p.get("name", "unknown")
            cpu = p.get("cpu_percent", 0) or 0
            mem = p.get("memory_percent", 0) or 0
            lines.append(f"PID {pid}: {name} — CPU {cpu:.1f}%, MEM {mem:.1f}%")

        return "\n".join(lines) if lines else "Nenhum processo encontrado."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_gpu_info() -> str | None:
        """Try to read GPU utilization via nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                if len(parts) >= 4:
                    return (
                        f"GPU: {parts[0]} — {parts[1]}% util, "
                        f"{parts[2]}MB / {parts[3]}MB VRAM"
                    )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None
