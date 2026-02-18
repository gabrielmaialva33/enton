"""HomeAssistantTools — Agno toolkit para controle de dispositivos IoT via Home Assistant.

Enton pode controlar luzes, switches, media players, e qualquer entidade
registrada no Home Assistant. Comunica via REST API oficial do HA.

Requer: Home Assistant rodando com Long-Lived Access Token configurado.
Config: hass_url, hass_token, hass_enabled em Settings (.env).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import httpx
from agno.tools import Toolkit

if TYPE_CHECKING:
    from enton.core.config import Settings

logger = logging.getLogger(__name__)

# Presets de cena — cores e brilho pra diferentes situacoes
SCENE_PRESETS: dict[str, dict[str, Any]] = {
    "coding": {"brightness": 128, "rgb_color": [128, 0, 255]},  # roxo
    "relax": {"brightness": 80, "rgb_color": [255, 147, 41]},  # quente
    "sleep": {"brightness": 10, "rgb_color": [255, 100, 50]},  # vermelho fraco
    "movie": {"brightness": 30, "rgb_color": [0, 0, 128]},  # azul escuro
}

_TIMEOUT = 10.0


class HomeAssistantTools(Toolkit):
    """Controle de dispositivos IoT via Home Assistant REST API."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(name="home_assistant_tools")
        self._base_url = settings.hass_url.rstrip("/") if settings.hass_url else ""
        self._token = settings.hass_token
        self._enabled = settings.hass_enabled
        self.register(self.ha_get_state)
        self.register(self.ha_toggle)
        self.register(self.ha_turn_on)
        self.register(self.ha_turn_off)
        self.register(self.ha_set_scene)
        self.register(self.ha_play_media)
        self.register(self.ha_list_entities)
        self.register(self.ha_call_service)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _check_enabled(self) -> str | None:
        """Retorna mensagem de erro se HA nao esta configurado, None se OK."""
        if not self._enabled:
            return "Home Assistant desabilitado. Configure hass_enabled=true no .env."
        if not self._base_url:
            return "URL do Home Assistant nao configurada. Defina hass_url no .env."
        if not self._token:
            return "Token do Home Assistant nao configurado. Defina hass_token no .env."
        return None

    async def _get(self, path: str) -> tuple[dict | list | None, str | None]:
        """GET request para a API do Home Assistant."""
        err = self._check_enabled()
        if err:
            return None, err
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    f"{self._base_url}/api{path}",
                    headers=self._headers(),
                )
                if resp.status_code == 401:
                    return None, "Token do HA invalido ou expirado (401)."
                if resp.status_code == 404:
                    return None, f"Entidade nao encontrada: {path} (404)."
                resp.raise_for_status()
                return resp.json(), None
        except httpx.TimeoutException:
            return None, f"Timeout conectando ao HA ({_TIMEOUT}s)."
        except httpx.ConnectError:
            return None, f"Nao consegui conectar ao HA em {self._base_url}."
        except httpx.HTTPStatusError as e:
            return None, f"Erro HTTP do HA: {e.response.status_code}."
        except Exception as e:
            return None, f"Erro inesperado no HA: {e}"

    async def _post(
        self, path: str, payload: dict | None = None
    ) -> tuple[dict | list | None, str | None]:
        """POST request para a API do Home Assistant."""
        err = self._check_enabled()
        if err:
            return None, err
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    f"{self._base_url}/api{path}",
                    headers=self._headers(),
                    json=payload or {},
                )
                if resp.status_code == 401:
                    return None, "Token do HA invalido ou expirado (401)."
                if resp.status_code == 404:
                    return None, f"Servico nao encontrado: {path} (404)."
                resp.raise_for_status()
                return resp.json(), None
        except httpx.TimeoutException:
            return None, f"Timeout conectando ao HA ({_TIMEOUT}s)."
        except httpx.ConnectError:
            return None, f"Nao consegui conectar ao HA em {self._base_url}."
        except httpx.HTTPStatusError as e:
            return None, f"Erro HTTP do HA: {e.response.status_code}."
        except Exception as e:
            return None, f"Erro inesperado no HA: {e}"

    async def ha_get_state(self, entity_id: str) -> str:
        """Consulta o estado atual de uma entidade no Home Assistant.

        Retorna estado, atributos e ultima atualizacao da entidade.

        Args:
            entity_id: ID da entidade (ex: light.sala, switch.tv, sensor.temperatura).
        """
        data, err = await self._get(f"/states/{entity_id}")
        if err:
            return err
        if not isinstance(data, dict):
            return "Resposta inesperada do HA."
        state = data.get("state", "unknown")
        attrs = data.get("attributes", {})
        friendly = attrs.get("friendly_name", entity_id)
        updated = data.get("last_updated", "?")
        lines = [
            f"{friendly} ({entity_id})",
            f"  Estado: {state}",
            f"  Atualizado: {updated}",
        ]
        # Mostrar atributos relevantes
        for key in ("brightness", "rgb_color", "temperature", "unit_of_measurement"):
            if key in attrs:
                lines.append(f"  {key}: {attrs[key]}")
        return "\n".join(lines)

    async def ha_toggle(self, entity_id: str) -> str:
        """Alterna o estado de uma entidade (liga/desliga).

        Funciona com qualquer entidade que suporte toggle:
        luzes, switches, fans, media_players, etc.

        Args:
            entity_id: ID da entidade (ex: light.sala, switch.ventilador).
        """
        payload = {"entity_id": entity_id}
        data, err = await self._post("/services/homeassistant/toggle", payload)
        if err:
            return err
        return f"Toggle executado em {entity_id}."

    async def ha_turn_on(
        self,
        entity_id: str,
        brightness: int = 255,
        rgb_color: str = "",
    ) -> str:
        """Liga uma luz com brilho e cor opcionais.

        Args:
            entity_id: ID da luz (ex: light.sala, light.quarto).
            brightness: Brilho de 0 a 255 (default: 255 = maximo).
            rgb_color: Cor RGB como "R,G,B" (ex: "255,0,128"). Vazio = sem mudanca.
        """
        service_data: dict[str, Any] = {"entity_id": entity_id}
        if 0 <= brightness <= 255:
            service_data["brightness"] = brightness
        if rgb_color:
            try:
                parts = [int(c.strip()) for c in rgb_color.split(",")]
                if len(parts) == 3:
                    service_data["rgb_color"] = parts
            except ValueError:
                return f"Cor RGB invalida: '{rgb_color}'. Use formato 'R,G,B' (ex: '255,0,128')."
        data, err = await self._post("/services/light/turn_on", service_data)
        if err:
            return err
        color_info = f", cor={rgb_color}" if rgb_color else ""
        return f"Luz {entity_id} ligada (brilho={brightness}{color_info})."

    async def ha_turn_off(self, entity_id: str) -> str:
        """Desliga uma luz ou dispositivo.

        Args:
            entity_id: ID da entidade (ex: light.sala, switch.tv).
        """
        payload = {"entity_id": entity_id}
        data, err = await self._post("/services/light/turn_off", payload)
        if err:
            return err
        return f"Dispositivo {entity_id} desligado."

    async def ha_set_scene(self, scene_name: str) -> str:
        """Aplica um preset de cena nas luzes.

        Cenas disponiveis: coding (roxo), relax (quente), sleep (vermelho fraco),
        movie (azul escuro). Aplica em todas as luzes do grupo 'all'.

        Args:
            scene_name: Nome da cena (coding, relax, sleep, movie).
        """
        preset = SCENE_PRESETS.get(scene_name.lower())
        if not preset:
            available = ", ".join(SCENE_PRESETS.keys())
            return f"Cena '{scene_name}' nao existe. Disponiveis: {available}."
        service_data: dict[str, Any] = {
            "entity_id": "all",
            "brightness": preset["brightness"],
            "rgb_color": preset["rgb_color"],
        }
        data, err = await self._post("/services/light/turn_on", service_data)
        if err:
            return err
        rgb = preset["rgb_color"]
        return (
            f"Cena '{scene_name}' aplicada! "
            f"Brilho={preset['brightness']}, "
            f"Cor=RGB({rgb[0]},{rgb[1]},{rgb[2]})."
        )

    async def ha_play_media(self, entity_id: str, media_url: str) -> str:
        """Toca uma midia (musica, video, stream) num media player do HA.

        Args:
            entity_id: ID do media player (ex: media_player.sala, media_player.tv).
            media_url: URL da midia pra tocar (stream, arquivo, etc).
        """
        payload = {
            "entity_id": entity_id,
            "media_content_id": media_url,
            "media_content_type": "music",
        }
        data, err = await self._post("/services/media_player/play_media", payload)
        if err:
            return err
        return f"Tocando midia em {entity_id}: {media_url}"

    async def ha_list_entities(self, domain: str = "") -> str:
        """Lista entidades registradas no Home Assistant.

        Pode filtrar por dominio (light, switch, sensor, media_player, etc).
        Sem filtro, lista todas as entidades.

        Args:
            domain: Dominio pra filtrar (ex: light, switch, sensor). Vazio = todas.
        """
        data, err = await self._get("/states")
        if err:
            return err
        if not isinstance(data, list):
            return "Resposta inesperada do HA."
        entities = data
        if domain:
            entities = [e for e in entities if e.get("entity_id", "").startswith(f"{domain}.")]
        if not entities:
            filt = f" no dominio '{domain}'" if domain else ""
            return f"Nenhuma entidade encontrada{filt}."
        lines = [f"Entidades ({len(entities)}):"]
        for e in entities[:50]:  # limitar a 50 pra nao explodir
            eid = e.get("entity_id", "?")
            state = e.get("state", "?")
            name = e.get("attributes", {}).get("friendly_name", eid)
            lines.append(f"  {name} ({eid}): {state}")
        if len(entities) > 50:
            lines.append(f"  ... e mais {len(entities) - 50} entidades.")
        return "\n".join(lines)

    async def ha_call_service(
        self,
        domain: str,
        service: str,
        entity_id: str,
        data: str = "{}",
    ) -> str:
        """Chama qualquer servico do Home Assistant (generico).

        Use pra servicos que nao tem uma tool dedicada. Aceita qualquer
        dominio, servico, e dados extras em JSON.

        Args:
            domain: Dominio do servico (ex: light, switch, climate, automation).
            service: Nome do servico (ex: turn_on, turn_off, set_temperature).
            entity_id: ID da entidade alvo (ex: climate.sala).
            data: Dados extras em JSON (ex: '{"temperature": 22}').
        """
        try:
            extra = json.loads(data) if data else {}
        except json.JSONDecodeError:
            return f"JSON invalido nos dados extras: '{data}'."
        if not isinstance(extra, dict):
            return "Dados extras devem ser um objeto JSON (dict)."
        payload: dict[str, Any] = {"entity_id": entity_id, **extra}
        result, err = await self._post(f"/services/{domain}/{service}", payload)
        if err:
            return err
        return f"Servico {domain}/{service} executado em {entity_id}."
