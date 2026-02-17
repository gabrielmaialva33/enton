"""Shared state between ShellTools and FileTools."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio


@dataclass
class BackgroundProcess:
    """Tracks a background shell process."""

    id: str
    command: str
    process: asyncio.subprocess.Process
    output: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    done: bool = False


@dataclass
class ShellState:
    """Persistent shell state shared across toolkits."""

    cwd: Path = field(default_factory=Path.cwd)
    background: dict[str, BackgroundProcess] = field(default_factory=dict)

    def resolve_path(self, path: str) -> Path:
        """Resolve a path relative to current working directory."""
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.cwd / p
        return p.resolve()
