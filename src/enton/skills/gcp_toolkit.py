"""GCP remote compute toolkit — Enton's cloud muscle.

Enton can spin up Spot VMs, run code remotely, deploy to Cloud Run,
and monitor GCP resources. Uses gcloud CLI (already authenticated).

GCP Project: project-42631060-20da-4fdf-814
APIs: Compute Engine, Cloud Run, Vertex AI, Notebooks, Cloud Build
"""
from __future__ import annotations

import asyncio
import json
import logging
import shlex
import shutil

from agno.tools import Toolkit

logger = logging.getLogger(__name__)

# Default VM configs — cheap Spot instances for heavy compute
_VM_PRESETS: dict[str, dict[str, str]] = {
    "micro": {
        "machine_type": "e2-micro",
        "disk_size": "10",
        "desc": "Teste rapido (2 vCPU, 1GB RAM) — ~$0.003/hr",
    },
    "dev": {
        "machine_type": "e2-standard-4",
        "disk_size": "50",
        "desc": "Dev geral (4 vCPU, 16GB RAM) — ~$0.04/hr spot",
    },
    "gpu-t4": {
        "machine_type": "n1-standard-4",
        "disk_size": "100",
        "accelerator": "type=nvidia-tesla-t4,count=1",
        "desc": "GPU T4 (4 vCPU, 15GB RAM, 16GB VRAM) — ~$0.11/hr spot",
    },
    "gpu-l4": {
        "machine_type": "g2-standard-4",
        "disk_size": "100",
        "accelerator": "type=nvidia-l4,count=1",
        "desc": "GPU L4 (4 vCPU, 16GB RAM, 24GB VRAM) — ~$0.22/hr spot",
    },
    "heavy": {
        "machine_type": "e2-standard-16",
        "disk_size": "200",
        "desc": "CPU heavy (16 vCPU, 64GB RAM) — ~$0.16/hr spot",
    },
}


class GcpTools(Toolkit):
    """Controle do Google Cloud Platform — VMs, Cloud Run, monitoramento."""

    def __init__(self, project: str = "", region: str = "us-central1") -> None:
        super().__init__(name="gcp_tools")
        self._project = project
        self._region = region
        self._gcloud = shutil.which("gcloud") or "gcloud"
        self.register(self.gcp_status)
        self.register(self.gcp_vm_presets)
        self.register(self.gcp_vm_create)
        self.register(self.gcp_vm_list)
        self.register(self.gcp_vm_ssh)
        self.register(self.gcp_vm_delete)
        self.register(self.gcp_vm_start)
        self.register(self.gcp_vm_stop)
        self.register(self.gcp_run_code)
        self.register(self.gcp_billing)

    async def _exec(
        self, *args: str, timeout: float = 60.0,
    ) -> tuple[str, str, int]:
        """Execute gcloud command."""
        cmd = [self._gcloud]
        if self._project:
            cmd += ["--project", self._project]
        cmd += list(args)
        cmd += ["--format=json", "--quiet"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
        return (
            stdout.decode(errors="replace").strip(),
            stderr.decode(errors="replace").strip(),
            proc.returncode or 0,
        )

    async def _exec_raw(
        self, cmd: str, timeout: float = 60.0,
    ) -> tuple[str, str, int]:
        """Execute raw shell command (for SSH, etc.)."""
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
        return (
            stdout.decode(errors="replace").strip(),
            stderr.decode(errors="replace").strip(),
            proc.returncode or 0,
        )

    async def gcp_status(self) -> str:
        """Mostra status do GCP — projeto, conta, APIs ativas, e cota.

        Verifica se gcloud ta autenticado e pronto pra usar.

        Args:
            (nenhum)
        """
        if not shutil.which("gcloud"):
            return "gcloud CLI nao encontrado. Instale: https://cloud.google.com/sdk"

        out, err, rc = await self._exec("config", "list")
        if rc != 0:
            return f"Erro ao verificar gcloud: {err}"

        try:
            config = json.loads(out)
        except json.JSONDecodeError:
            config = {}

        core = config.get("core", {})
        compute = config.get("compute", {})

        lines = [
            "=== GCP Status ===",
            f"Projeto: {core.get('project', 'N/A')}",
            f"Conta: {core.get('account', 'N/A')}",
            f"Regiao: {compute.get('region', self._region)}",
            f"Zona: {compute.get('zone', 'auto')}",
        ]

        # Check active VMs
        vout, _, vrc = await self._exec(
            "compute", "instances", "list",
            f"--project={self._project}" if self._project else "",
        )
        if vrc == 0:
            try:
                vms = json.loads(vout) if vout else []
                running = [v for v in vms if v.get("status") == "RUNNING"]
                lines.append(f"VMs: {len(vms)} total, {len(running)} rodando")
            except json.JSONDecodeError:
                lines.append("VMs: erro ao listar")

        return "\n".join(lines)

    async def gcp_vm_presets(self) -> str:
        """Lista os presets de VM disponiveis pra criar.

        Cada preset tem tipo de maquina, disco, e custo estimado.

        Args:
            (nenhum)
        """
        lines = ["Presets de VM disponiveis:"]
        for name, spec in _VM_PRESETS.items():
            gpu = f" + {spec['accelerator']}" if "accelerator" in spec else ""
            lines.append(
                f"  {name}: {spec['machine_type']}{gpu} "
                f"(disk: {spec['disk_size']}GB) — {spec['desc']}",
            )
        return "\n".join(lines)

    async def gcp_vm_create(
        self,
        name: str,
        preset: str = "dev",
        zone: str = "us-central1-a",
        spot: bool = True,
        startup_script: str = "",
    ) -> str:
        """Cria uma VM no GCP (Spot por default = barato).

        Args:
            name: Nome da VM (ex: enton-worker-1).
            preset: Preset de config (micro, dev, gpu-t4, gpu-l4, heavy).
            zone: Zona GCP (default: us-central1-a).
            spot: Usar Spot/Preemptible pra economizar (default: True).
            startup_script: Script bash pra rodar no boot (opcional).
        """
        if preset not in _VM_PRESETS:
            return f"Preset '{preset}' invalido. Use: {', '.join(_VM_PRESETS)}"

        spec = _VM_PRESETS[preset]
        args = [
            "compute", "instances", "create", name,
            f"--zone={zone}",
            f"--machine-type={spec['machine_type']}",
            f"--boot-disk-size={spec['disk_size']}GB",
            "--boot-disk-type=pd-balanced",
            "--image-family=ubuntu-2404-lts-amd64",
            "--image-project=ubuntu-os-cloud",
            "--scopes=cloud-platform",
        ]

        if spot:
            args.append("--provisioning-model=SPOT")
            args.append("--instance-termination-action=STOP")

        if "accelerator" in spec:
            args.append(f"--accelerator={spec['accelerator']}")
            args.append("--maintenance-policy=TERMINATE")

        if startup_script:
            args.append(f"--metadata=startup-script={startup_script}")

        out, err, rc = await self._exec(*args, timeout=120.0)
        if rc != 0:
            return f"Erro ao criar VM: {err}"

        try:
            vms = json.loads(out) if out else []
            if vms:
                vm = vms[0] if isinstance(vms, list) else vms
                ext_ip = ""
                for iface in vm.get("networkInterfaces", []):
                    for ac in iface.get("accessConfigs", []):
                        if ac.get("natIP"):
                            ext_ip = ac["natIP"]
                return (
                    f"VM '{name}' criada com sucesso!\n"
                    f"  Tipo: {spec['machine_type']}\n"
                    f"  Zona: {zone}\n"
                    f"  Spot: {'Sim' if spot else 'Nao'}\n"
                    f"  IP: {ext_ip or 'alocando...'}\n"
                    f"  SSH: gcloud compute ssh {name} --zone={zone}"
                )
        except json.JSONDecodeError:
            pass
        return f"VM '{name}' criada. Output: {out[:200]}"

    async def gcp_vm_list(self) -> str:
        """Lista todas as VMs do projeto GCP.

        Args:
            (nenhum)
        """
        out, err, rc = await self._exec("compute", "instances", "list")
        if rc != 0:
            return f"Erro: {err}"

        try:
            vms = json.loads(out) if out else []
        except json.JSONDecodeError:
            return f"Erro parsing: {out[:200]}"

        if not vms:
            return "Nenhuma VM ativa no projeto."

        lines = [f"VMs ({len(vms)}):"]
        for vm in vms:
            name = vm.get("name", "?")
            status = vm.get("status", "?")
            mtype = vm.get("machineType", "").rsplit("/", 1)[-1]
            zone = vm.get("zone", "").rsplit("/", 1)[-1]
            ext_ip = ""
            for iface in vm.get("networkInterfaces", []):
                for ac in iface.get("accessConfigs", []):
                    if ac.get("natIP"):
                        ext_ip = ac["natIP"]
            icon = "🟢" if status == "RUNNING" else "🔴"
            lines.append(
                f"  {icon} {name} [{status}] {mtype} @ {zone}"
                + (f" (IP: {ext_ip})" if ext_ip else ""),
            )
        return "\n".join(lines)

    async def gcp_vm_ssh(
        self, name: str, command: str, zone: str = "us-central1-a",
    ) -> str:
        """Executa um comando via SSH numa VM do GCP.

        Args:
            name: Nome da VM.
            command: Comando bash pra executar remotamente.
            zone: Zona da VM (default: us-central1-a).
        """
        q_cmd = shlex.quote(command)
        project_flag = f"--project={self._project}" if self._project else ""
        ssh_cmd = (
            f"gcloud compute ssh {shlex.quote(name)} "
            f"--zone={zone} {project_flag} "
            f"--command={q_cmd} -- -o StrictHostKeyChecking=no"
        )
        out, err, rc = await self._exec_raw(ssh_cmd, timeout=120.0)
        result = out
        if err:
            # SSH prints warnings to stderr, filter noise
            useful_err = "\n".join(
                line for line in err.splitlines()
                if "WARNING" not in line and "banner" not in line.lower()
            )
            if useful_err:
                result += f"\nSTDERR: {useful_err}"
        if rc != 0:
            result += f"\n(exit code: {rc})"
        return result or "(sem output)"

    async def gcp_vm_delete(
        self, name: str, zone: str = "us-central1-a",
    ) -> str:
        """Deleta uma VM do GCP (cuidado: irreversivel!).

        Args:
            name: Nome da VM pra deletar.
            zone: Zona da VM (default: us-central1-a).
        """
        _out, err, rc = await self._exec(
            "compute", "instances", "delete", name,
            f"--zone={zone}",
        )
        if rc != 0:
            return f"Erro ao deletar VM: {err}"
        return f"VM '{name}' deletada com sucesso."

    async def gcp_vm_start(
        self, name: str, zone: str = "us-central1-a",
    ) -> str:
        """Liga uma VM parada.

        Args:
            name: Nome da VM.
            zone: Zona da VM (default: us-central1-a).
        """
        _, err, rc = await self._exec(
            "compute", "instances", "start", name,
            f"--zone={zone}",
        )
        if rc != 0:
            return f"Erro: {err}"
        return f"VM '{name}' iniciada."

    async def gcp_vm_stop(
        self, name: str, zone: str = "us-central1-a",
    ) -> str:
        """Desliga uma VM (economiza custo quando nao esta usando).

        Args:
            name: Nome da VM.
            zone: Zona da VM (default: us-central1-a).
        """
        _, err, rc = await self._exec(
            "compute", "instances", "stop", name,
            f"--zone={zone}",
        )
        if rc != 0:
            return f"Erro: {err}"
        return f"VM '{name}' parada."

    async def gcp_run_code(
        self,
        code: str,
        language: str = "python",
        vm_name: str = "",
        zone: str = "us-central1-a",
    ) -> str:
        """Executa codigo remotamente numa VM do GCP.

        Se nenhuma VM especificada, tenta usar uma VM ja existente
        ou sugere criar uma.

        Args:
            code: Codigo-fonte pra executar.
            language: Linguagem (python, bash). Default: python.
            vm_name: Nome da VM (vazio = auto-detectar).
            zone: Zona (default: us-central1-a).
        """
        # Find a running VM if none specified
        if not vm_name:
            out, _, rc = await self._exec("compute", "instances", "list")
            if rc == 0:
                try:
                    vms = json.loads(out) if out else []
                    running = [
                        v for v in vms if v.get("status") == "RUNNING"
                    ]
                    if running:
                        vm_name = running[0]["name"]
                        zone = running[0].get("zone", "").rsplit("/", 1)[-1]
                except json.JSONDecodeError:
                    pass

            if not vm_name:
                return (
                    "Nenhuma VM rodando. Crie uma primeiro com gcp_vm_create.\n"
                    "Ex: gcp_vm_create(name='enton-worker', preset='dev')"
                )

        # Build remote command
        if language == "python":
            escaped = code.replace("'", "'\"'\"'")
            remote_cmd = f"python3 -c '{escaped}'"
        elif language == "bash":
            escaped = code.replace("'", "'\"'\"'")
            remote_cmd = f"bash -c '{escaped}'"
        else:
            return f"Linguagem '{language}' nao suportada pra execucao remota."

        return await self.gcp_vm_ssh(vm_name, remote_cmd, zone)

    async def gcp_billing(self) -> str:
        """Mostra estimativa de custo/billing do projeto GCP.

        Args:
            (nenhum)
        """
        # gcloud billing doesn't have great CLI support,
        # use budgets API or compute instances cost estimate
        out, err, rc = await self._exec(
            "compute", "instances", "list",
        )
        if rc != 0:
            return f"Erro: {err}"

        try:
            vms = json.loads(out) if out else []
        except json.JSONDecodeError:
            vms = []

        running = [v for v in vms if v.get("status") == "RUNNING"]
        stopped = [v for v in vms if v.get("status") != "RUNNING"]

        lines = [
            "=== GCP Billing Estimate ===",
            f"Projeto: {self._project}",
            f"VMs rodando: {len(running)} (gerando custo)",
            f"VMs paradas: {len(stopped)} (disco cobra, compute nao)",
        ]

        # Estimate hourly cost based on machine types
        cost_map = {
            "e2-micro": 0.008, "e2-standard-4": 0.134,
            "e2-standard-16": 0.538, "n1-standard-4": 0.190,
            "g2-standard-4": 0.560,
        }
        total_hourly = 0.0
        for vm in running:
            mtype = vm.get("machineType", "").rsplit("/", 1)[-1]
            scheduling = vm.get("scheduling", {})
            is_spot = scheduling.get("provisioningModel") == "SPOT"
            base_cost = cost_map.get(mtype, 0.10)
            cost = base_cost * 0.3 if is_spot else base_cost  # Spot ~70% off
            total_hourly += cost
            lines.append(
                f"  {vm['name']}: ~${cost:.3f}/hr "
                f"({'SPOT' if is_spot else 'on-demand'})",
            )

        if total_hourly > 0:
            lines.append(f"\nTotal estimado: ~${total_hourly:.3f}/hr "
                         f"(~${total_hourly * 24:.2f}/dia)")
        else:
            lines.append("\nSem VMs rodando — custo zero de compute.")

        lines.append(
            "\nDica: Use Spot VMs e desligue quando nao precisar!"
        )
        return "\n".join(lines)
