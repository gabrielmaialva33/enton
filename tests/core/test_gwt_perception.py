import pytest
from unittest.mock import MagicMock
from enton.core.gwt.modules.perception import PerceptionModule
from enton.cognition.prediction import PredictionEngine, WorldState

@pytest.fixture
def mock_prediction_engine():
    engine = MagicMock(spec=PredictionEngine)
    # Mock tick behavior if needed, but we mostly care about _current_surprise updates
    return engine

def test_perception_saliency_high_surprise(mock_prediction_engine):
    """Test that high surprise leads to high positive saliency (Novelty)."""
    module = PerceptionModule(mock_prediction_engine)
    
    # Simulate high surprise (0.9) from prediction engine
    mock_prediction_engine.tick.return_value = 0.9
    state = WorldState(timestamp=0, user_present=True)
    
    surprise = module.update_state(state)
    assert surprise == 0.9
    
    # Run step to generate broadcast
    msg = module.run_step()
    
    assert msg is not None
    assert msg.source == "perception"
    assert "High Novelty" in msg.content
    assert msg.saliency > 0.5 # Should be very salient
    assert msg.metadata["surprise"] == 0.9

def test_perception_saliency_low_surprise(mock_prediction_engine):
    """Test that low surprise (boredom) also leads to high saliency (Predictability)."""
    module = PerceptionModule(mock_prediction_engine)
    
    # Simulate low surprise (0.1)
    mock_prediction_engine.tick.return_value = 0.1
    state = WorldState(timestamp=0, user_present=True)
    
    surprise = module.update_state(state)
    assert surprise == 0.1
    
    msg = module.run_step()
    
    assert msg is not None
    assert "High Predictability" in msg.content
    assert msg.saliency > 0.5 # Deviation from 0.5 is 0.4, x1.8 = 0.72
    
def test_perception_saliency_neutral_surprise(mock_prediction_engine):
    """Test that neutral surprise leads to low saliency (message might be suppressed)."""
    module = PerceptionModule(mock_prediction_engine)
    
    # Simulate neutral surprise (0.5)
    mock_prediction_engine.tick.return_value = 0.5
    state = WorldState(timestamp=0, user_present=True)
    
    module.update_state(state)
    msg = module.run_step()
    
    # 0.5 -> 0 dist -> 0 saliency. Should be suppressed if threshold > 0
    # Threshold in code is 0.2
    assert msg is None
