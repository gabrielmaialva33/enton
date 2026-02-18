"""Tests for HomeAssistantTools (mocked HTTP, no real HA instance)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from enton.skills.hass_toolkit import SCENE_PRESETS, HomeAssistantTools


@pytest.fixture()
def hass_settings():
    """Minimal Settings mock for Home Assistant."""
    s = MagicMock()
    s.hass_url = "http://ha.local:8123"
    s.hass_token = "test-token-abc123"
    s.hass_enabled = True
    return s


@pytest.fixture()
def hass_disabled_settings():
    """Settings with HA disabled."""
    s = MagicMock()
    s.hass_url = "http://ha.local:8123"
    s.hass_token = "test-token-abc123"
    s.hass_enabled = False
    return s


@pytest.fixture()
def hass_no_url_settings():
    """Settings with empty HA URL."""
    s = MagicMock()
    s.hass_url = ""
    s.hass_token = "test-token-abc123"
    s.hass_enabled = True
    return s


@pytest.fixture()
def hass_no_token_settings():
    """Settings with empty HA token."""
    s = MagicMock()
    s.hass_url = "http://ha.local:8123"
    s.hass_token = ""
    s.hass_enabled = True
    return s


@pytest.fixture()
def tools(hass_settings):
    return HomeAssistantTools(hass_settings)


# --- Instantiation ---


def test_registers_toolkit(hass_settings):
    t = HomeAssistantTools(hass_settings)
    assert t.name == "home_assistant_tools"


def test_base_url_strips_trailing_slash():
    s = MagicMock()
    s.hass_url = "http://ha.local:8123/"
    s.hass_token = "tok"
    s.hass_enabled = True
    t = HomeAssistantTools(s)
    assert t._base_url == "http://ha.local:8123"


# --- Disabled / misconfigured ---


async def test_disabled_returns_error(hass_disabled_settings):
    t = HomeAssistantTools(hass_disabled_settings)
    result = await t.ha_get_state("light.sala")
    assert "desabilitado" in result.lower()


async def test_no_url_returns_error(hass_no_url_settings):
    t = HomeAssistantTools(hass_no_url_settings)
    result = await t.ha_get_state("light.sala")
    assert "url" in result.lower()


async def test_no_token_returns_error(hass_no_token_settings):
    t = HomeAssistantTools(hass_no_token_settings)
    result = await t.ha_get_state("light.sala")
    assert "token" in result.lower()


# --- ha_get_state ---


async def test_get_state_success(tools):
    mock_response = httpx.Response(
        200,
        json={
            "entity_id": "light.sala",
            "state": "on",
            "attributes": {
                "friendly_name": "Luz da Sala",
                "brightness": 200,
                "rgb_color": [255, 255, 255],
            },
            "last_updated": "2026-02-17T10:00:00Z",
        },
        request=httpx.Request("GET", "http://ha.local:8123/api/states/light.sala"),
    )
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
        result = await tools.ha_get_state("light.sala")
    assert "Luz da Sala" in result
    assert "on" in result
    assert "brightness" in result


async def test_get_state_404(tools):
    mock_response = httpx.Response(
        404,
        json={"message": "Entity not found"},
        request=httpx.Request("GET", "http://ha.local:8123/api/states/light.nope"),
    )
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
        result = await tools.ha_get_state("light.nope")
    assert "404" in result


async def test_get_state_401(tools):
    mock_response = httpx.Response(
        401,
        json={"message": "Unauthorized"},
        request=httpx.Request("GET", "http://ha.local:8123/api/states/light.sala"),
    )
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
        result = await tools.ha_get_state("light.sala")
    assert "401" in result


async def test_get_state_timeout(tools):
    with patch(
        "httpx.AsyncClient.get",
        new_callable=AsyncMock,
        side_effect=httpx.TimeoutException("timeout"),
    ):
        result = await tools.ha_get_state("light.sala")
    assert "timeout" in result.lower()


async def test_get_state_connection_error(tools):
    with patch(
        "httpx.AsyncClient.get",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("refused"),
    ):
        result = await tools.ha_get_state("light.sala")
    assert "conectar" in result.lower()


# --- ha_toggle ---


async def test_toggle_success(tools):
    mock_response = httpx.Response(
        200,
        json=[{"entity_id": "light.sala", "state": "off"}],
        request=httpx.Request("POST", "http://ha.local:8123/api/services/homeassistant/toggle"),
    )
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        result = await tools.ha_toggle("light.sala")
    assert "toggle" in result.lower()
    assert "light.sala" in result


# --- ha_turn_on ---


async def test_turn_on_default(tools):
    mock_response = httpx.Response(
        200,
        json=[{"entity_id": "light.sala", "state": "on"}],
        request=httpx.Request("POST", "http://ha.local:8123/api/services/light/turn_on"),
    )
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        result = await tools.ha_turn_on("light.sala")
    assert "ligada" in result.lower()
    assert "brilho=255" in result


async def test_turn_on_with_color(tools):
    mock_response = httpx.Response(
        200,
        json=[{"entity_id": "light.sala", "state": "on"}],
        request=httpx.Request("POST", "http://ha.local:8123/api/services/light/turn_on"),
    )
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        result = await tools.ha_turn_on("light.sala", brightness=128, rgb_color="255,0,128")
    assert "brilho=128" in result
    assert "cor=255,0,128" in result


async def test_turn_on_invalid_color(tools):
    result = await tools.ha_turn_on("light.sala", rgb_color="not,a,color")
    assert "invalida" in result.lower()


# --- ha_turn_off ---


async def test_turn_off_success(tools):
    mock_response = httpx.Response(
        200,
        json=[{"entity_id": "light.sala", "state": "off"}],
        request=httpx.Request("POST", "http://ha.local:8123/api/services/light/turn_off"),
    )
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        result = await tools.ha_turn_off("light.sala")
    assert "desligado" in result.lower()


# --- ha_set_scene ---


async def test_set_scene_coding(tools):
    mock_response = httpx.Response(
        200,
        json=[],
        request=httpx.Request("POST", "http://ha.local:8123/api/services/light/turn_on"),
    )
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        result = await tools.ha_set_scene("coding")
    assert "coding" in result.lower()
    assert "128" in result  # brightness


async def test_set_scene_all_presets_exist():
    """Verify all scene presets have required keys."""
    for name, preset in SCENE_PRESETS.items():
        assert "brightness" in preset, f"Missing brightness in scene '{name}'"
        assert "rgb_color" in preset, f"Missing rgb_color in scene '{name}'"
        assert isinstance(preset["rgb_color"], list), f"rgb_color not list in '{name}'"
        assert len(preset["rgb_color"]) == 3, f"rgb_color not 3 values in '{name}'"


async def test_set_scene_invalid(tools):
    result = await tools.ha_set_scene("rave")
    assert "nao existe" in result.lower()
    assert "coding" in result  # should list available


# --- ha_play_media ---


async def test_play_media_success(tools):
    mock_response = httpx.Response(
        200,
        json=[],
        request=httpx.Request("POST", "http://ha.local:8123/api/services/media_player/play_media"),
    )
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        result = await tools.ha_play_media(
            "media_player.sala", "http://stream.example.com/radio.mp3"
        )
    assert "tocando" in result.lower()
    assert "media_player.sala" in result


# --- ha_list_entities ---


async def test_list_entities_all(tools):
    mock_data = [
        {
            "entity_id": "light.sala",
            "state": "on",
            "attributes": {"friendly_name": "Luz da Sala"},
        },
        {
            "entity_id": "switch.tv",
            "state": "off",
            "attributes": {"friendly_name": "TV"},
        },
        {
            "entity_id": "sensor.temp",
            "state": "22.5",
            "attributes": {"friendly_name": "Temperatura"},
        },
    ]
    mock_response = httpx.Response(
        200,
        json=mock_data,
        request=httpx.Request("GET", "http://ha.local:8123/api/states"),
    )
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
        result = await tools.ha_list_entities()
    assert "Entidades (3)" in result
    assert "Luz da Sala" in result
    assert "TV" in result


async def test_list_entities_filtered(tools):
    mock_data = [
        {
            "entity_id": "light.sala",
            "state": "on",
            "attributes": {"friendly_name": "Luz da Sala"},
        },
        {
            "entity_id": "light.quarto",
            "state": "off",
            "attributes": {"friendly_name": "Luz do Quarto"},
        },
        {
            "entity_id": "switch.tv",
            "state": "off",
            "attributes": {"friendly_name": "TV"},
        },
    ]
    mock_response = httpx.Response(
        200,
        json=mock_data,
        request=httpx.Request("GET", "http://ha.local:8123/api/states"),
    )
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
        result = await tools.ha_list_entities(domain="light")
    assert "Entidades (2)" in result
    assert "TV" not in result


async def test_list_entities_empty_domain(tools):
    mock_response = httpx.Response(
        200,
        json=[
            {
                "entity_id": "switch.tv",
                "state": "off",
                "attributes": {"friendly_name": "TV"},
            },
        ],
        request=httpx.Request("GET", "http://ha.local:8123/api/states"),
    )
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
        result = await tools.ha_list_entities(domain="light")
    assert "nenhuma" in result.lower()


# --- ha_call_service ---


async def test_call_service_success(tools):
    mock_response = httpx.Response(
        200,
        json=[],
        request=httpx.Request("POST", "http://ha.local:8123/api/services/climate/set_temperature"),
    )
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        result = await tools.ha_call_service(
            domain="climate",
            service="set_temperature",
            entity_id="climate.sala",
            data='{"temperature": 22}',
        )
    assert "climate/set_temperature" in result
    assert "climate.sala" in result


async def test_call_service_invalid_json(tools):
    result = await tools.ha_call_service(
        domain="light",
        service="turn_on",
        entity_id="light.sala",
        data="not-json",
    )
    assert "json invalido" in result.lower()


async def test_call_service_non_dict_json(tools):
    result = await tools.ha_call_service(
        domain="light",
        service="turn_on",
        entity_id="light.sala",
        data="[1,2,3]",
    )
    assert "objeto json" in result.lower()


# --- Headers ---


def test_headers_contain_bearer_token(tools):
    headers = tools._headers()
    assert headers["Authorization"] == "Bearer test-token-abc123"
    assert headers["Content-Type"] == "application/json"


# --- Scene presets structure ---


def test_scene_presets_keys():
    assert "coding" in SCENE_PRESETS
    assert "relax" in SCENE_PRESETS
    assert "sleep" in SCENE_PRESETS
    assert "movie" in SCENE_PRESETS
