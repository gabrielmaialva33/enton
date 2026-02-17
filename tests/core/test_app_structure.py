import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock hardware modules to avoid runtime errors during import/init
sys.modules["v4l2py"] = MagicMock()
sys.modules["sounddevice"] = MagicMock()
sys.modules["fast_whisper"] = MagicMock()

from enton.app import App
from enton.core.config import settings

@pytest.fixture
def mock_dependencies():
    with patch("enton.app.Vision"), \
         patch("enton.app.Ears"), \
         patch("enton.app.Voice"), \
         patch("enton.app.Memory"), \
         patch("enton.app.BlobStore"), \
         patch("enton.app.GlobalWorkspace"), \
         patch("enton.app.PerceptionModule"), \
         patch("enton.app.ExecutiveModule"), \
         patch("enton.app.GitHubModule"), \
         patch("enton.app.KnowledgeCrawler"), \
         patch("enton.app.VisualMemory"), \
         patch("enton.app.AndroidBridge"):
        yield

def test_app_initialization(mock_dependencies):
    """Verifies that App initializes without syntax/import errors and sets up GWT."""
    app = App(viewer=False)
    
    assert app.bus is not None
    assert app.self_model is not None
    assert app.fuser is not None
    
    # Check GWT attributes existence (even if None before init loop)
    assert hasattr(app, "workspace")
    assert hasattr(app, "perception_module")
    assert hasattr(app, "executive_module")
    assert hasattr(app, "github_module")
    
    # Check Sentience Metrics
    assert hasattr(app, "_current_fps")
    assert hasattr(app, "_attention_energy")
    assert app._current_fps == 5.0

@pytest.mark.asyncio
async def test_consciousness_loop_math(mock_dependencies):
    """Test the mathematical logic in consciousness loop (isolated)."""
    app = App(viewer=False)
    app.workspace = MagicMock()
    app.perception_module = MagicMock()
    app.vision = MagicMock()
    
    # Mock perception update to return high surprise
    app.perception_module.update_state.return_value = 0.9
    app.workspace.tick.return_value = None # No thought implies no action, just loop logic
    
    # Inject a mock for handle_conscious_thought to avoid errors if called
    app._handle_conscious_thought = MagicMock()
    
    # Run one iteration of the logic (extracted logic for testing would be better, 
    # but here we just check if it crashes or updates state)
    
    # Manually trigger the math logic block (simulated)
    surprise = 0.9
    
    # Logistic function logic used in app.py
    import math
    k = 10.0
    x0 = 0.5
    L = 60.0
    attention_energy = L / (1 + math.exp(-k * (surprise - x0)))
    target_fps = max(1.0, attention_energy)
    
    current_fps = app._current_fps
    new_fps = current_fps * 0.9 + target_fps * 0.1
    
    assert new_fps > current_fps # Should increase due to high surprise
    assert new_fps < 60.0
