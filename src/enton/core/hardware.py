"""Hardware awareness — Enton knows exactly what power he has.

Detects GPU (NVIDIA), CPU, RAM, disk (local + external HD), network,
CUDA capabilities, and running services. Updates periodically to give
Enton real-time awareness of his computational resources.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field

import psutil

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Single GPU device info."""

    index: int = 0
    name: str = "unknown"
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    vram_free_mb: int = 0
    utilization_pct: int = 0
    temperature_c: int = 0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0
    driver_version: str = ""
    cuda_version: str = ""
    compute_capability: str = ""


@dataclass
class DiskInfo:
    """Disk mount info."""

    mount: str = ""
    device: str = ""
    total_gb: float = 0.0
    used_gb: float = 0.0
    free_gb: float = 0.0
    percent: float = 0.0
    fstype: str = ""


@dataclass
class HardwareProfile:
    """Full hardware profile — Enton's self-awareness of his power."""

    # CPU
    cpu_model: str = ""
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    cpu_freq_max_mhz: float = 0.0
    cpu_freq_current_mhz: float = 0.0
    cpu_percent: float = 0.0
    cpu_arch: str = ""

    # RAM
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_percent: float = 0.0

    # GPU(s)
    gpus: list[GPUInfo] = field(default_factory=list)

    # Disks
    disks: list[DiskInfo] = field(default_factory=list)

    # System
    hostname: str = ""
    os_name: str = ""
    os_version: str = ""
    kernel: str = ""
    uptime_hours: float = 0.0

    # Network
    ip_addresses: dict[str, str] = field(default_factory=dict)

    # Workspace
    workspace_path: str = ""
    workspace_free_gb: float = 0.0

    def summary(self) -> str:
        """One-line power summary for Enton's consciousness."""
        gpu_str = ""
        if self.gpus:
            g = self.gpus[0]
            gpu_str = (
                f" | GPU: {g.name} {g.vram_total_mb}MB "
                f"({g.utilization_pct}% util, {g.temperature_c}C)"
            )
        return (
            f"CPU: {self.cpu_model} {self.cpu_cores_physical}c/{self.cpu_cores_logical}t "
            f"@ {self.cpu_freq_max_mhz:.0f}MHz ({self.cpu_percent:.0f}%)"
            f" | RAM: {self.ram_used_gb:.1f}/{self.ram_total_gb:.1f}GB ({self.ram_percent:.0f}%)"
            f"{gpu_str}"
            f" | Workspace: {self.workspace_free_gb:.0f}GB free"
        )

    def to_dict(self) -> dict:
        """Serializable dict for LLM context injection."""
        return {
            "cpu": {
                "model": self.cpu_model,
                "cores": f"{self.cpu_cores_physical}p/{self.cpu_cores_logical}l",
                "freq_mhz": round(self.cpu_freq_max_mhz),
                "usage_pct": round(self.cpu_percent),
            },
            "ram": {
                "total_gb": round(self.ram_total_gb, 1),
                "available_gb": round(self.ram_available_gb, 1),
                "usage_pct": round(self.ram_percent),
            },
            "gpu": [
                {
                    "name": g.name,
                    "vram_mb": g.vram_total_mb,
                    "vram_used_mb": g.vram_used_mb,
                    "util_pct": g.utilization_pct,
                    "temp_c": g.temperature_c,
                    "cuda": g.cuda_version,
                }
                for g in self.gpus
            ],
            "disks": [
                {
                    "mount": d.mount,
                    "free_gb": round(d.free_gb, 1),
                    "total_gb": round(d.total_gb, 1),
                }
                for d in self.disks
            ],
            "workspace_free_gb": round(self.workspace_free_gb, 1),
            "os": f"{self.os_name} {self.os_version}",
            "kernel": self.kernel,
            "uptime_h": round(self.uptime_hours, 1),
        }


def detect_hardware(workspace_path: str = "") -> HardwareProfile:
    """Detect all hardware — called on boot and periodically."""
    hw = HardwareProfile()

    # --- CPU ---
    hw.cpu_arch = platform.machine()
    hw.cpu_cores_physical = psutil.cpu_count(logical=False) or 1
    hw.cpu_cores_logical = psutil.cpu_count(logical=True) or 1
    freq = psutil.cpu_freq()
    if freq:
        hw.cpu_freq_max_mhz = freq.max or freq.current
        hw.cpu_freq_current_mhz = freq.current
    hw.cpu_percent = psutil.cpu_percent(interval=0.1)
    hw.cpu_model = _get_cpu_model()

    # --- RAM ---
    mem = psutil.virtual_memory()
    hw.ram_total_gb = mem.total / (1 << 30)
    hw.ram_available_gb = mem.available / (1 << 30)
    hw.ram_used_gb = mem.used / (1 << 30)
    hw.ram_percent = mem.percent

    # --- GPU ---
    hw.gpus = _detect_gpus()

    # --- Disks ---
    hw.disks = _detect_disks()

    # --- System ---
    uname = platform.uname()
    hw.hostname = uname.node
    hw.os_name = uname.system
    hw.os_version = platform.version()
    hw.kernel = uname.release
    hw.uptime_hours = (time.time() - psutil.boot_time()) / 3600

    # --- Network ---
    hw.ip_addresses = _detect_ips()

    # --- Workspace ---
    if workspace_path:
        hw.workspace_path = workspace_path
        try:
            usage = shutil.disk_usage(workspace_path)
            hw.workspace_free_gb = usage.free / (1 << 30)
        except OSError:
            pass

    return hw


def _get_cpu_model() -> str:
    """Extract CPU model name from /proc/cpuinfo or platform."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _detect_gpus() -> list[GPUInfo]:
    """Detect NVIDIA GPUs via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu="
                "index,name,memory.total,memory.used,memory.free,"
                "utilization.gpu,temperature.gpu,power.draw,power.limit,"
                "driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 10:
                continue
            gpu = GPUInfo(
                index=int(parts[0]),
                name=parts[1],
                vram_total_mb=int(float(parts[2])),
                vram_used_mb=int(float(parts[3])),
                vram_free_mb=int(float(parts[4])),
                utilization_pct=int(float(parts[5])),
                temperature_c=int(float(parts[6])),
                power_draw_w=float(parts[7]),
                power_limit_w=float(parts[8]),
                driver_version=parts[9],
            )
            gpus.append(gpu)

        # CUDA version from nvidia-smi header
        if gpus:
            cuda = _get_cuda_version()
            cc = _get_compute_capability(gpus[0].name)
            for g in gpus:
                g.cuda_version = cuda
                g.compute_capability = cc

        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def _get_cuda_version() -> str:
    """Get CUDA version from nvidia-smi."""
    try:
        # Try nvcc for precise CUDA version
        nvcc = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if nvcc.returncode == 0:
            for line in nvcc.stdout.splitlines():
                if "release" in line.lower():
                    parts = line.split("release")
                    if len(parts) > 1:
                        return parts[1].strip().split(",")[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def _get_compute_capability(gpu_name: str) -> str:
    """Known compute capabilities for common GPUs."""
    cc_map = {
        "4090": "8.9",
        "4080": "8.9",
        "4070": "8.9",
        "4060": "8.9",
        "3090": "8.6",
        "3080": "8.6",
        "3070": "8.6",
        "3060": "8.6",
        "A100": "8.0",
        "H100": "9.0",
        "L40": "8.9",
        "V100": "7.0",
        "T4": "7.5",
        "A10": "8.6",
    }
    for key, cc in cc_map.items():
        if key in gpu_name:
            return cc
    return ""


def _detect_disks() -> list[DiskInfo]:
    """Detect mounted disks with usage info."""
    disks = []
    seen_devices = set()
    for part in psutil.disk_partitions(all=False):
        if part.device in seen_devices:
            continue
        # Skip squashfs snaps — noise
        if part.fstype == "squashfs":
            continue
        seen_devices.add(part.device)
        try:
            usage = psutil.disk_usage(part.mountpoint)
            disks.append(
                DiskInfo(
                    mount=part.mountpoint,
                    device=part.device,
                    total_gb=usage.total / (1 << 30),
                    used_gb=usage.used / (1 << 30),
                    free_gb=usage.free / (1 << 30),
                    percent=usage.percent,
                    fstype=part.fstype,
                )
            )
        except (PermissionError, OSError):
            continue
    return disks


def _detect_ips() -> dict[str, str]:
    """Detect network interface IPs."""
    ips = {}
    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == 2 and not addr.address.startswith("127."):  # AF_INET
                ips[iface] = addr.address
    return ips
