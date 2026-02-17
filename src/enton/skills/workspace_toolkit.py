"""Unified workspace toolkit — Enton's domain over his personal space.

Gives Enton awareness of his workspace on the external HD, hardware
profile, project management, and disk intelligence. The workspace is
Enton's "home" — separate from the user's files but with full access
to the system when needed.

Architecture inspired by OpenHands workspace abstraction + wcgw MCP.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from agno.tools import Toolkit

from enton.core.hardware import HardwareProfile, detect_hardware

logger = logging.getLogger(__name__)


class WorkspaceTools(Toolkit):
    """Enton's workspace awareness — disk, hardware, projects, resources."""

    def __init__(self, workspace: Path, hardware: HardwareProfile | None = None) -> None:
        super().__init__(name="workspace_tools")
        self._workspace = workspace
        self._hardware = hardware or detect_hardware(str(workspace))
        self.register(self.workspace_info)
        self.register(self.workspace_list)
        self.register(self.hardware_status)
        self.register(self.hardware_gpu)
        self.register(self.hardware_full)
        self.register(self.project_create)
        self.register(self.project_list)
        self.register(self.disk_usage)

    def _refresh_hardware(self) -> None:
        """Refresh hardware stats (CPU/RAM/GPU are dynamic)."""
        self._hardware = detect_hardware(str(self._workspace))

    async def workspace_info(self) -> str:
        """Mostra info do workspace do Enton — onde eu vivo e trabalho.

        Retorna path, espaco livre, subdiretorios, e status do HD.

        Args:
            (nenhum)
        """
        ws = self._workspace
        try:
            usage = shutil.disk_usage(ws)
            free_gb = usage.free / (1 << 30)
            total_gb = usage.total / (1 << 30)
        except OSError:
            free_gb = total_gb = 0.0

        subdirs = sorted(
            d.name for d in ws.iterdir() if d.is_dir()
        ) if ws.exists() else []

        lines = [
            f"Workspace: {ws}",
            f"Disco: {free_gb:.0f}GB livre / {total_gb:.0f}GB total",
            f"Subdirs: {', '.join(subdirs) or 'nenhum'}",
        ]

        # Count files in each subdir
        for sd in subdirs:
            path = ws / sd
            count = sum(1 for _ in path.rglob("*") if _.is_file())
            if count:
                lines.append(f"  {sd}/: {count} arquivos")

        return "\n".join(lines)

    async def workspace_list(self, subdir: str = "", pattern: str = "*") -> str:
        """Lista arquivos no workspace ou num subdiretorio.

        Args:
            subdir: Subdiretorio (code, projects, downloads, tmp). Vazio = raiz.
            pattern: Glob pattern pra filtrar (default: *).
        """
        path = self._workspace / subdir if subdir else self._workspace
        if not path.exists():
            return f"Diretorio '{path}' nao existe."

        files = sorted(path.glob(pattern))[:50]
        if not files:
            return f"Nenhum arquivo em {path} com pattern '{pattern}'."

        lines = [f"Arquivos em {path} ({len(files)}):"]
        for f in files:
            if f.is_dir():
                lines.append(f"  [DIR] {f.name}/")
            else:
                size = f.stat().st_size
                if size > 1 << 20:
                    size_str = f"{size / (1 << 20):.1f}MB"
                elif size > 1 << 10:
                    size_str = f"{size / (1 << 10):.0f}KB"
                else:
                    size_str = f"{size}B"
                lines.append(f"  {f.name} ({size_str})")
        return "\n".join(lines)

    async def hardware_status(self) -> str:
        """Resumo rapido do hardware — CPU, RAM, GPU, disco.

        Atualiza stats em tempo real. Use pra monitorar performance.

        Args:
            (nenhum)
        """
        self._refresh_hardware()
        return self._hardware.summary()

    async def hardware_gpu(self) -> str:
        """Info detalhada da(s) GPU(s) — VRAM, temperatura, CUDA, potencia.

        Args:
            (nenhum)
        """
        self._refresh_hardware()
        if not self._hardware.gpus:
            return "Nenhuma GPU NVIDIA detectada."

        lines = []
        for g in self._hardware.gpus:
            lines.extend([
                f"GPU {g.index}: {g.name}",
                f"  VRAM: {g.vram_used_mb}MB / {g.vram_total_mb}MB "
                f"({g.vram_free_mb}MB livre)",
                f"  Utilizacao: {g.utilization_pct}%",
                f"  Temperatura: {g.temperature_c}C",
                f"  Potencia: {g.power_draw_w:.0f}W / {g.power_limit_w:.0f}W",
                f"  Driver: {g.driver_version}",
                f"  CUDA: {g.cuda_version}",
                f"  Compute Capability: {g.compute_capability}",
            ])
        return "\n".join(lines)

    async def hardware_full(self) -> str:
        """Perfil completo do hardware — tudo que eu sei sobre meu PC.

        CPU, RAM, GPU, discos, rede, uptime, OS. Use pra decisoes sobre
        onde executar workloads (local vs cloud).

        Args:
            (nenhum)
        """
        self._refresh_hardware()
        hw = self._hardware
        lines = [
            "=== HARDWARE PROFILE ===",
            f"Host: {hw.hostname}",
            f"OS: {hw.os_name} {hw.os_version}",
            f"Kernel: {hw.kernel}",
            f"Arch: {hw.cpu_arch}",
            f"Uptime: {hw.uptime_hours:.1f}h",
            "",
            f"CPU: {hw.cpu_model}",
            f"  Cores: {hw.cpu_cores_physical}p / {hw.cpu_cores_logical}l",
            f"  Freq: {hw.cpu_freq_current_mhz:.0f} / {hw.cpu_freq_max_mhz:.0f} MHz",
            f"  Usage: {hw.cpu_percent:.1f}%",
            "",
            f"RAM: {hw.ram_used_gb:.1f}GB / {hw.ram_total_gb:.1f}GB "
            f"({hw.ram_percent:.0f}%)",
            f"  Available: {hw.ram_available_gb:.1f}GB",
        ]

        if hw.gpus:
            lines.append("")
            for g in hw.gpus:
                lines.extend([
                    f"GPU {g.index}: {g.name}",
                    f"  VRAM: {g.vram_used_mb}/{g.vram_total_mb}MB "
                    f"| Util: {g.utilization_pct}%"
                    f" | Temp: {g.temperature_c}C"
                    f" | Power: {g.power_draw_w:.0f}/{g.power_limit_w:.0f}W",
                    f"  CUDA {g.cuda_version} | CC {g.compute_capability}"
                    f" | Driver {g.driver_version}",
                ])

        if hw.disks:
            lines.append("")
            for d in hw.disks:
                lines.append(
                    f"Disk [{d.fstype}] {d.mount}: "
                    f"{d.free_gb:.0f}GB free / {d.total_gb:.0f}GB "
                    f"({d.percent:.0f}%)"
                )

        if hw.ip_addresses:
            lines.append("")
            for iface, ip in hw.ip_addresses.items():
                lines.append(f"Net {iface}: {ip}")

        lines.extend([
            "",
            f"Workspace: {hw.workspace_path}",
            f"  Free: {hw.workspace_free_gb:.0f}GB",
        ])

        return "\n".join(lines)

    async def project_create(self, name: str, description: str = "") -> str:
        """Cria um novo projeto no workspace do Enton.

        Cada projeto fica em projects/<name>/ com README opcional.

        Args:
            name: Nome do projeto (sem espacos — use hifens).
            description: Descricao curta do projeto.
        """
        safe_name = name.lower().replace(" ", "-")
        project_dir = self._workspace / "projects" / safe_name
        if project_dir.exists():
            return f"Projeto '{safe_name}' ja existe em {project_dir}"

        project_dir.mkdir(parents=True, exist_ok=True)
        if description:
            (project_dir / "README.md").write_text(
                f"# {name}\n\n{description}\n",
                encoding="utf-8",
            )
        return f"Projeto '{safe_name}' criado em {project_dir}"

    async def project_list(self) -> str:
        """Lista os projetos do Enton no workspace.

        Args:
            (nenhum)
        """
        projects_dir = self._workspace / "projects"
        if not projects_dir.exists():
            return "Nenhum projeto ainda."

        dirs = sorted(d for d in projects_dir.iterdir() if d.is_dir())
        if not dirs:
            return "Nenhum projeto ainda."

        lines = [f"Projetos ({len(dirs)}):"]
        for d in dirs:
            files = sum(1 for _ in d.rglob("*") if _.is_file())
            readme = d / "README.md"
            desc = ""
            if readme.exists():
                first_lines = readme.read_text(encoding="utf-8").splitlines()
                for line in first_lines[1:]:
                    if line.strip():
                        desc = f" — {line.strip()[:60]}"
                        break
            lines.append(f"  {d.name}/ ({files} files){desc}")
        return "\n".join(lines)

    async def disk_usage(self) -> str:
        """Mostra uso de disco de todos os volumes montados.

        Util pra saber quanto espaco tem no HD externo, SSD, etc.

        Args:
            (nenhum)
        """
        self._refresh_hardware()
        if not self._hardware.disks:
            return "Nenhum disco detectado."

        lines = ["Discos montados:"]
        for d in self._hardware.disks:
            bar_len = 20
            filled = int(d.percent / 100 * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            lines.append(
                f"  {d.mount} [{bar}] "
                f"{d.free_gb:.0f}GB livre / {d.total_gb:.0f}GB "
                f"({d.percent:.0f}%) [{d.fstype}]"
            )
        return "\n".join(lines)
