from dataclasses import dataclass, field
from typing import Any, Optional
import time

@dataclass
class BroadcastMessage:
    """
    Representa um conteúdo que ganhou acesso ao Global Workspace.
    É transmitido para todos os módulos no início de cada ciclo.
    """
    content: Any
    source: str
    saliency: float  # 0.0 a 1.0 (Quão importante/surpreendente é)
    modality: str    # "vision", "audio", "inner_speech", "emotion", etc.
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.source}:{self.modality}] (saliency={self.saliency:.2f}) {str(self.content)[:50]}"
