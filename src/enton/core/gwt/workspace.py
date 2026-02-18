import logging

from enton.core.gwt.message import BroadcastMessage
from enton.core.gwt.module import CognitiveModule

logger = logging.getLogger(__name__)


class GlobalWorkspace:
    """
    O 'Palco' da consciência. Gerencia a competição entre módulos
    e faz o broadcast do vencedor.
    """

    def __init__(self):
        self.modules: list[CognitiveModule] = []
        self.current_conscious_content: BroadcastMessage | None = None
        self.history: list[BroadcastMessage] = []
        self.step_counter: int = 0

    def register_module(self, module: CognitiveModule) -> None:
        self.modules.append(module)
        logger.info(f"Module registered in GWT: {module.name}")

    def tick(self) -> BroadcastMessage | None:
        """
        Executa um ciclo cognitivo completo.
        1. Envia contexto atual para todos os módulos.
        2. Coleta candidatos.
        3. Seleciona vencedor (Winner-Take-All baseado em saliência).
        4. Atualiza contexto.
        """
        self.step_counter += 1
        candidates: list[BroadcastMessage] = []

        # 1. Parallel execution (synchronous for now, but conceptual parallel)
        for module in self.modules:
            try:
                candidate = module.run_step(self.current_conscious_content)
                if candidate:
                    candidates.append(candidate)
            except Exception as e:
                logger.error(f"Error in module {module.name}: {e}", exc_info=True)

        if not candidates:
            # Silence... nothing happened.
            # We naturally decay the current thought? Or keep it?
            # For now, let's keep it but mark as 'stale' or just return None (unconscious moment)
            return None

        # 2. Competition: Winner-Take-All
        # TODO: Add noise/probabilistic selection logic (Softmax)
        winner = max(candidates, key=lambda msg: msg.saliency)

        # 3. Update Global Workspace
        self.current_conscious_content = winner
        self.history.append(winner)  # Short-term history

        # Limit history size
        if len(self.history) > 100:
            self.history.pop(0)

        logger.debug(f"GWT Tick {self.step_counter}: Winner -> {winner}")
        return winner
