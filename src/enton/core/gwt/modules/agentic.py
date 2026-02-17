import asyncio
import logging
from typing import Any

from enton.cognition.brain import EntonBrain
from enton.core.gwt.message import BroadcastMessage
from enton.core.gwt.module import CognitiveModule

logger = logging.getLogger(__name__)

class AgenticModule(CognitiveModule):
    """
    Agência Universal.
    Escuta intenções do Workspace e usa o EntonBrain (e seus toolkits)
    para realizar ações no mundo (digital ou físico).
    """

    def __init__(self, brain: EntonBrain):
        super().__init__(name="agentic_module")
        self.brain = brain
        self.is_busy = False
        self._pending_result: str | None = None
        self._action_memory: dict[str, Any] = {}

    def run_step(self, context: BroadcastMessage | None) -> BroadcastMessage | None:
        # 1. Entrega resultados pendentes (Feedback Loop)
        if self._pending_result:
            content = self._pending_result
            self._pending_result = None
            return BroadcastMessage(
                content=f"Action Result: {content[:200]}...",
                source=self.name,
                saliency=1.0,
                modality="memory_recall",
                metadata={"full_text": content, "type": "action_result"}
            )

        # 2. Se está ocupado, não aceita novas tarefas
        if self.is_busy:
            return None

        # 3. Escuta intenções de ação (Action Intentions)
        # Formato esperado da intenção: "use_tool:<tool_name>:<instruction>"
        # Ou intenções de alto nível: "perform_task:<task_description>"
        if context and context.modality == "intention":
            
            # Caso 1: Uso explícito de ferramenta
            if context.content.startswith("use_tool:"):
                parts = context.content.split(":", 2)
                if len(parts) == 3:
                    tool_name, instruction = parts[1], parts[2]
                    self.is_busy = True
                    asyncio.create_task(self._execute_tool(tool_name, instruction))
                    
                    return BroadcastMessage(
                        content=f"Executing tool {tool_name}...",
                        source=self.name,
                        saliency=0.9,
                        modality="inner_speech"
                    )

            # Caso 2: Tarefa genérica (Agentic execution)
            elif context.content.startswith("agentic_task:"):
                instruction = context.metadata.get("instruction") or context.content.split(":", 1)[1]
                self.is_busy = True
                asyncio.create_task(self._execute_agentic_task(instruction))
                
                return BroadcastMessage(
                    content=f"Starting agentic task: {instruction[:50]}...",
                    source=self.name,
                    saliency=0.9,
                    modality="inner_speech"
                )

        return None

    async def _execute_tool(self, tool_name: str, instruction: str):
        """Executa uma ferramenta específica via Brain."""
        try:
            logger.info(f"AgenticModule: Using tool {tool_name} with instruction: {instruction}")
            
            # Aqui usamos o brain.think, mas forçando o contexto da ferramenta se possível
            # Como o brain já tem toolkits registrados, pedimos pra ele usar.
            from enton.cognition.prompts import AGENTIC_TOOL_PROMPT
            prompt = AGENTIC_TOOL_PROMPT.format(
                tool_name=tool_name, instruction=instruction,
            )
            result = await self.brain.think(prompt)
            
            self._pending_result = f"Tool {tool_name} output: {result}"
        except Exception as e:
            logger.error(f"AgenticModule tool error: {e}")
            self._pending_result = f"Error using {tool_name}: {e}"
        finally:
            self.is_busy = False

    async def _execute_agentic_task(self, instruction: str):
        """Deixa o Brain decidir quais ferramentas usar para uma tarefa."""
        try:
            logger.info(f"AgenticModule: Executing task: {instruction}")
            result = await self.brain.think(instruction)
            self._pending_result = f"Task result: {result}"
        except Exception as e:
            logger.error(f"AgenticModule task error: {e}")
            self._pending_result = f"Error executing task: {e}"
        finally:
            self.is_busy = False
