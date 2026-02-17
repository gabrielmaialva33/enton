import pytest
from enton.core.config import settings
from enton.skills.screenpipe_toolkit import ScreenpipeTools
from enton.skills.n8n_toolkit import N8nTools

def test_config_defaults():
    assert settings.screenpipe_url == "http://localhost:3030"
    assert settings.n8n_webhook_base == ""

def test_screenpipe_init():
    tools = ScreenpipeTools()
    assert tools.name == "screenpipe_tools"
    # agno Toolkit stores tools in self.functions (OrderedDict)
    assert len(tools.functions) >= 2
    assert "search_screen" in tools.functions
    assert "get_recent_activity" in tools.functions

def test_n8n_init():
    tools = N8nTools()
    assert tools.name == "n8n_tools"
    assert len(tools.functions) >= 1
    assert "trigger_automation" in tools.functions
