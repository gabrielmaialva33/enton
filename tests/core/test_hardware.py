"""Tests for hardware detection dataclasses and helpers."""

from __future__ import annotations

from enton.core.hardware import DiskInfo, GPUInfo, HardwareProfile


class TestGPUInfo:
    def test_defaults(self):
        gpu = GPUInfo()
        assert gpu.index == 0
        assert gpu.name == "unknown"
        assert gpu.vram_total_mb == 0

    def test_custom_values(self):
        gpu = GPUInfo(
            index=0,
            name="RTX 4090",
            vram_total_mb=24576,
            vram_used_mb=8000,
            vram_free_mb=16576,
            utilization_pct=45,
            temperature_c=72,
        )
        assert gpu.name == "RTX 4090"
        assert gpu.vram_total_mb == 24576
        assert gpu.temperature_c == 72


class TestDiskInfo:
    def test_defaults(self):
        disk = DiskInfo()
        assert disk.mount == ""
        assert disk.total_gb == 0.0

    def test_custom(self):
        disk = DiskInfo(
            mount="/",
            device="/dev/sda1",
            total_gb=500.0,
            used_gb=200.0,
            free_gb=300.0,
            percent=40.0,
            fstype="ext4",
        )
        assert disk.mount == "/"
        assert disk.free_gb == 300.0


class TestHardwareProfile:
    def test_defaults(self):
        hw = HardwareProfile()
        assert hw.cpu_model == ""
        assert hw.gpus == []
        assert hw.disks == []

    def test_summary_without_gpu(self):
        hw = HardwareProfile(
            cpu_model="i9-13900K",
            cpu_cores_physical=24,
            cpu_cores_logical=32,
            cpu_freq_max_mhz=5800.0,
            cpu_percent=25.0,
            ram_total_gb=30.0,
            ram_used_gb=15.0,
            ram_percent=50.0,
            workspace_free_gb=500.0,
        )
        s = hw.summary()
        assert "i9-13900K" in s
        assert "24c/32t" in s
        assert "500" in s

    def test_summary_with_gpu(self):
        gpu = GPUInfo(name="RTX 4090", vram_total_mb=24576, utilization_pct=50, temperature_c=65)
        hw = HardwareProfile(
            cpu_model="i9",
            cpu_cores_physical=24,
            cpu_cores_logical=32,
            cpu_freq_max_mhz=5800.0,
            cpu_percent=10.0,
            ram_total_gb=30.0,
            ram_used_gb=10.0,
            ram_percent=33.0,
            gpus=[gpu],
            workspace_free_gb=100.0,
        )
        s = hw.summary()
        assert "RTX 4090" in s
        assert "65C" in s

    def test_to_dict_structure(self):
        gpu = GPUInfo(name="RTX 4090", vram_total_mb=24576, cuda_version="12.4")
        disk = DiskInfo(mount="/", free_gb=100.0, total_gb=500.0)
        hw = HardwareProfile(
            cpu_model="i9",
            cpu_cores_physical=24,
            cpu_cores_logical=32,
            cpu_freq_max_mhz=5800.0,
            cpu_percent=10.0,
            ram_total_gb=30.0,
            ram_available_gb=20.0,
            ram_percent=33.0,
            gpus=[gpu],
            disks=[disk],
            os_name="Linux",
            os_version="6.17",
            kernel="6.17.0",
            uptime_hours=5.5,
            workspace_free_gb=200.0,
        )
        d = hw.to_dict()
        assert d["cpu"]["model"] == "i9"
        assert d["cpu"]["cores"] == "24p/32l"
        assert d["ram"]["total_gb"] == 30.0
        assert len(d["gpu"]) == 1
        assert d["gpu"][0]["name"] == "RTX 4090"
        assert len(d["disks"]) == 1
        assert d["workspace_free_gb"] == 200.0
        assert "Linux" in d["os"]

    def test_to_dict_empty(self):
        hw = HardwareProfile()
        d = hw.to_dict()
        assert d["gpu"] == []
        assert d["disks"] == []
