import logging
import time
from typing import Optional, Any
from enton.core.gwt.module import CognitiveModule
from enton.core.gwt.message import BroadcastMessage
from enton.cognition.prediction import PredictionEngine

logger = logging.getLogger(__name__)

class PerceptionModule(CognitiveModule):
    """
    Módulo responsável pela percepção visual e cálculo de surpresa.
    Atua como 'Sensory Cortex'.
    """
    
    def __init__(self, prediction_engine: PredictionEngine):
        super().__init__(name="perception")
        self.engine = prediction_engine
        self._current_surprise: float = 0.5  # Neutral default

    def update_state(self, state: Any) -> float:
        """
        Atualiza o estado sensorial e calcula a surpresa.
        Chamado pelo loop principal antes do tick do GWT.
        """
        # Delegamos ao motor de predição
        self._current_surprise = self.engine.tick(state)
        return self._current_surprise

    def run_step(self, context: Optional[BroadcastMessage]) -> Optional[BroadcastMessage]:
        # Saliência é baseada no desvio da neutralidade (0.5).
        # Tanto surpresa muito alta (CAOS) quanto muito baixa (TÉDIO) são salientes.
        # 0.0 -> |0.0 - 0.5| = 0.5 * 2 = 1.0 Saliência
        # 1.0 -> |1.0 - 0.5| = 0.5 * 2 = 1.0 Saliência
        # 0.5 -> |0.5 - 0.5| = 0.0 * 2 = 0.0 Saliência
        
        dist_from_neutral = abs(self._current_surprise - 0.5)
        saliency = dist_from_neutral * 1.8 # Boost factor
        saliency = min(1.0, max(0.0, saliency))
        
        # Filtro de ruído: Só reporta se tiver alguma relevância
        if saliency > 0.2:
            content_type = "High Novelty" if self._current_surprise > 0.5 else "High Predictability"
            return BroadcastMessage(
                content=f"Visual: {content_type} ({self._current_surprise:.2f})",
                source=self.name,
                saliency=saliency,
                modality="vision",
                metadata={"surprise": self._current_surprise}
            )
        
        return None
