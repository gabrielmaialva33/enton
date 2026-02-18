"""Protocol and metadata for dynamic skills."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SkillMetadata:
    """Metadata for a loaded dynamic skill."""

    name: str
    file_path: str
    description: str = ""
    author: str = "enton"
    version: str = "1.0"
    success_count: int = 0
    failure_count: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 1.0
