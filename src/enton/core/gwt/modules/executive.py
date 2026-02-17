from typing import Optional
from enton.core.gwt.module import CognitiveModule
from enton.core.gwt.message import BroadcastMessage
from enton.cognition.metacognition import MetaCognitiveEngine
from enton.skills.skill_registry import SkillRegistry

class ExecutiveModule(CognitiveModule):
    """
    Módulo executivo (Metacognição).
    Monitora o Workspace e decide ações de alto nível (Tédio, Curiosidade).
    Agora ciente das skills disponíveis dinamicamente via Registry.
    """

    def __init__(self, metacognition_engine: MetaCognitiveEngine, skill_registry: SkillRegistry = None):
        super().__init__(name="executive")
        self.engine = metacognition_engine
        self.skill_registry = skill_registry

    def run_step(self, context: Optional[BroadcastMessage]) -> Optional[BroadcastMessage]:
        # 1. Analisa contexto (Input Consciente)
        surprise_score = 0.5 # Default neutral
        
        if context and context.source == "perception" and context.modality == "vision":
             # Se for algo visual/perceptivo, extrai a surpresa (metadata)
             surprise_score = context.metadata.get("surprise", 0.5)

        # 2. Atualiza estado interno (Tédio)
        action_intent = self.engine.tick(surprise_score)
        
        # 3. Se o motor metacognitivo decidiu agir (ex: Tédio > Threshold)
        if action_intent == "study_github":
             # Verifica skills dinamicamente
             available = self.skill_registry.list_skills() if self.skill_registry else []
             
             if "github_learner" in available:
                 topic = self.engine.get_next_topic()
                 return BroadcastMessage(
                     content=f"use_tool:github_learner:Estude sobre {topic}",
                     source=self.name,
                     saliency=1.0, 
                     modality="intention",
                     metadata={"topic": topic}
                 )
             else:
                 # Fallback para tarefa genérica se github não estiver disponível
                 topic = self.engine.get_next_topic()
                 return BroadcastMessage(
                     content=f"agentic_task:Pesquise sobre {topic}",
                     source=self.name,
                     saliency=1.0,
                     modality="intention",
                     metadata={"instruction": f"Pesquise sobre {topic}"}
                 )
        
        # Se tédio estiver alto mas não disparou ação ainda, pode emitir um "feeling"
        if self.engine.boredom_level > 0.5:
             return BroadcastMessage(
                 content="feeling_bored",
                 source=self.name,
                 saliency=self.engine.boredom_level,
                 modality="emotion"
             )
             
        return None
