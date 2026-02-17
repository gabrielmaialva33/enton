"""n8n Toolkit — Digital Hands for Enton via Automation Workflows."""

from __future__ import annotations

import logging
from typing import Any

from agno.tools import Toolkit

from enton.core.config import settings

logger = logging.getLogger(__name__)


class N8nTools(Toolkit):
    """Tools for triggering n8n automation workflows."""

    def __init__(self) -> None:
        super().__init__(name="n8n_tools")
        self.register(self.trigger_automation)

    def trigger_automation(
        self,
        workflow_id: str,
        data: dict[str, Any] | None = None,
    ) -> str:
        """Dispara um workflow de automacao no n8n.

        Use isso para realizar acoes no mundo digital integrado (criar cards no Trello,
        enviar emails, atualizar planilha, notificar no Slack, etc).

        O `workflow_id` deve corresponder ao ID ou nome do webhook configurado no n8n.
        Ex: "criar_card_trello", "notificar_slack", "salvar_notion".

        Args:
            workflow_id: O identificador do webhook do workflow (parte final da URL).
            data: Dicionario de dados para enviar ao workflow (ex: {"titulo": "...", "desc": "..."}).
        """
        try:
            import httpx
        except ImportError:
            return "Erro: httpx nao instalado."

        if not settings.n8n_webhook_base:
            return "Erro: N8N_WEBHOOK_BASE nao configurado no .env."

        # Construct webhook URL
        # Assumes N8N_WEBHOOK_BASE is something like "https://n8n.meudominio.com/webhook"
        # and we append the ID. Or if it ends with slash, we just append ID.
        base = settings.n8n_webhook_base.rstrip("/")
        url = f"{base}/{workflow_id}"

        payload = data or {}
        
        try:
            with httpx.Client(timeout=10.0) as client:
                # n8n webhooks usually accept POST with JSON body
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                
                # Check if response has content
                if resp.status_code == 200:
                    try:
                        return f"Workflow '{workflow_id}' disparado com sucesso. Resposta: {resp.json()}"
                    except Exception:
                        return f"Workflow '{workflow_id}' disparado com sucesso."
                else:
                     return f"Workflow '{workflow_id}' disparado. Status: {resp.status_code}"

        except httpx.HTTPStatusError as e:
            logger.error("n8n webhook failed: %s", e)
            return f"Erro ao disparar workflow (Status {e.response.status_code}): {e.response.text}"
        except Exception as e:
            logger.error("n8n webhook failed: %s", e)
            return f"Erro de conexao com n8n: {e}"
