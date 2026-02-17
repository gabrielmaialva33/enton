from abc import ABC, abstractmethod

from enton.core.gwt.message import BroadcastMessage


class CognitiveModule(ABC):
    """
    Classe base para todos os sub-processos inconscientes do Enton.
    Eles processam o contexto global e propõem novos conteúdos.
    """
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run_step(self, context: BroadcastMessage | None) -> BroadcastMessage | None:
        """
        Executa um passo de processamento.
        
        Args:
            context: O conteúdo que estava no Global Workspace no passo anterior.
                     Pode ser None no primeiro passo.
                     
        Returns:
            Um BroadcastMessage candidato a entrar no Workspace, ou None se não tiver nada a dizer.
        """
        pass
