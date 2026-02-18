import asyncio
import logging

from enton.core.gwt.message import BroadcastMessage
from enton.core.gwt.module import CognitiveModule
from enton.skills.github_learner import GitHubLearner

logger = logging.getLogger(__name__)


class GitHubModule(CognitiveModule):
    """
    Módulo de habilidade (Skill).
    Reage a intenções de estudo e executa aprendizado.
    """

    def __init__(self, learner: GitHubLearner):
        super().__init__(name="github_skill")
        self.learner = learner
        self.is_busy = False
        self._pending_result: str | None = None

    def run_step(self, context: BroadcastMessage | None) -> BroadcastMessage | None:
        # 1. Se tem resultado pendente de uma tarefa anterior, entrega agora
        if self._pending_result:
            content = self._pending_result
            self._pending_result = None
            return BroadcastMessage(
                content=f"Study Result: {content[:100]}...",  # Resumo p/ broadcast
                source=self.name,
                saliency=1.0,  # Resultados de ações são muito salientes!
                modality="memory_recall",
                metadata={"full_text": content},
            )

        # 2. Se está ocupado, silêncio
        if self.is_busy:
            return None

        # 3. Verifica se há uma ordem para este módulo
        if (
            context
            and context.modality == "intention"
            and context.content.startswith("study_github:")
        ):
            topic = context.metadata.get("topic")
            if topic:
                self.is_busy = True
                asyncio.create_task(self._perform_study(topic))

                return BroadcastMessage(
                    content=f"Starting study on {topic}",
                    source=self.name,
                    saliency=0.8,
                    modality="inner_speech",
                )

        return None

    async def _perform_study(self, topic: str):
        try:
            logger.info(f"GitHubModule: Starting study on {topic}")
            result = await asyncio.to_thread(self.learner.study_github_topic, topic)
            self._pending_result = result
        except Exception as e:
            logger.error(f"GitHubModule error: {e}")
            self._pending_result = f"Error studying {topic}: {e}"
        finally:
            self.is_busy = False
