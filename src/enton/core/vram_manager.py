"""VRAMManager — dynamic CUDA model hot-swapping with LRU eviction.

Manages multiple PyTorch models sharing a single GPU.  Models are
loaded to CPU on first use and promoted to CUDA on demand.  When VRAM
budget is exceeded the least-recently-used non-critical model is
evicted back to CPU (or fully unloaded).

Replaces the naive cuda_lock.py with a proper resource manager.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class ModelPriority(IntEnum):
    LOW = auto()        # only if room (SigLIP, optional models)
    NORMAL = auto()     # on-demand (Whisper, Kokoro, Qwen-VL)
    CRITICAL = auto()   # always in VRAM (YOLO during active modes)


@dataclass(slots=True)
class ModelSlot:
    """Tracks a model's lifecycle and location."""

    name: str
    loader: Callable[[], Any]
    vram_mb: int
    priority: ModelPriority = ModelPriority.NORMAL
    model: Any = field(default=None, repr=False)
    on_device: bool = False
    last_used: float = 0.0
    use_count: int = 0

    def load(self) -> Any:
        if self.model is None:
            self.model = self.loader()
            logger.info("Model '%s' loaded to CPU", self.name)
        return self.model

    def to_cuda(self) -> None:
        if self.model is not None and not self.on_device:
            self.model = self.model.cuda()
            self.on_device = True
            self.last_used = time.time()
            logger.info("Model '%s' -> CUDA (%dMB)", self.name, self.vram_mb)

    def to_cpu(self) -> None:
        if self.model is not None and self.on_device:
            self.model = self.model.cpu()
            self.on_device = False
            _empty_cache()
            logger.info("Model '%s' -> CPU", self.name)

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            self.on_device = False
            gc.collect()
            _empty_cache()
            logger.info("Model '%s' unloaded", self.name)


def _empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _get_vram_free_mb() -> int:
    """Return free VRAM in MB (or 0 if unavailable)."""
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return int(free / (1024 * 1024))
    except Exception:
        pass
    return 0


class VRAMManager:
    """Manages CUDA memory across multiple models with LRU eviction."""

    def __init__(self, budget_mb: int = 6000) -> None:
        self._slots: dict[str, ModelSlot] = {}
        self._budget_mb = budget_mb
        self._lock = asyncio.Lock()

    # -- registration --

    def register(self, slot: ModelSlot) -> None:
        self._slots[slot.name] = slot

    def register_model(
        self,
        name: str,
        loader: Callable[[], Any],
        vram_mb: int,
        priority: ModelPriority = ModelPriority.NORMAL,
    ) -> None:
        self._slots[name] = ModelSlot(
            name=name, loader=loader, vram_mb=vram_mb, priority=priority,
        )

    # -- queries --

    @property
    def used_mb(self) -> int:
        return sum(s.vram_mb for s in self._slots.values() if s.on_device)

    @property
    def free_mb(self) -> int:
        return self._budget_mb - self.used_mb

    @property
    def hw_free_mb(self) -> int:
        return _get_vram_free_mb()

    # -- core ops --

    async def acquire(self, name: str) -> Any:
        """Get model on CUDA, evicting others if needed. Returns model."""
        async with self._lock:
            slot = self._slots.get(name)
            if slot is None:
                raise KeyError(f"Unknown model: {name}")

            slot.load()

            if slot.on_device:
                slot.last_used = time.time()
                slot.use_count += 1
                return slot.model

            # evict until we fit
            while self.free_mb < slot.vram_mb:
                victim = self._pick_eviction(exclude=name)
                if victim is None:
                    raise RuntimeError(
                        f"Cannot fit '{name}' ({slot.vram_mb}MB): "
                        f"used={self.used_mb}MB budget={self._budget_mb}MB"
                    )
                victim.to_cpu()

            slot.to_cuda()
            slot.use_count += 1
            return slot.model

    async def release(self, name: str) -> None:
        """Hint that model is idle (stays in VRAM if room)."""
        slot = self._slots.get(name)
        if slot:
            slot.last_used = time.time()

    def set_priority(self, name: str, priority: ModelPriority) -> None:
        """Change a model's priority (e.g. based on awareness level)."""
        slot = self._slots.get(name)
        if slot:
            slot.priority = priority

    # -- eviction --

    def _pick_eviction(self, exclude: str) -> ModelSlot | None:
        """LRU eviction among on-device models, lowest priority first."""
        candidates = [
            s for s in self._slots.values()
            if s.on_device and s.name != exclude
        ]
        if not candidates:
            return None
        # sort: lowest priority first, then oldest last_used
        candidates.sort(key=lambda s: (s.priority, s.last_used))
        # skip CRITICAL unless nothing else
        non_crit = [c for c in candidates if c.priority != ModelPriority.CRITICAL]
        return non_crit[0] if non_crit else candidates[0]

    # -- bulk ops --

    def evict_all(self, *, keep_critical: bool = True) -> None:
        """Move all models to CPU (e.g. before entering DORMANT)."""
        for slot in self._slots.values():
            if keep_critical and slot.priority == ModelPriority.CRITICAL:
                continue
            if slot.on_device:
                slot.to_cpu()

    def unload_all(self) -> None:
        """Completely unload all models."""
        for slot in self._slots.values():
            slot.unload()

    # -- status --

    def status(self) -> str:
        lines = [f"VRAM: {self.used_mb}/{self._budget_mb}MB (hw free: {self.hw_free_mb}MB)"]
        for s in sorted(self._slots.values(), key=lambda x: -x.priority):
            loc = "CUDA" if s.on_device else ("CPU" if s.model else "off")
            lines.append(
                f"  {s.name}: {loc} ({s.vram_mb}MB, "
                f"P{s.priority.name}, used {s.use_count}x)"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "budget_mb": self._budget_mb,
            "used_mb": self.used_mb,
            "models": {
                s.name: {
                    "on_device": s.on_device,
                    "priority": s.priority.name,
                    "vram_mb": s.vram_mb,
                    "use_count": s.use_count,
                }
                for s in self._slots.values()
            },
        }
