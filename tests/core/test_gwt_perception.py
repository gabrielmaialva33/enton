"""Tests for PerceptionModule saliency calculation."""

from unittest.mock import MagicMock

import pytest

from enton.cognition.prediction import PredictionEngine, WorldState
from enton.core.gwt.modules.perception import PerceptionModule


@pytest.fixture
def mock_prediction_engine():
    engine = MagicMock(spec=PredictionEngine)
    return engine


def test_perception_saliency_high_surprise(mock_prediction_engine):
    """Test that high surprise leads to high positive saliency (Novelty)."""
    module = PerceptionModule(mock_prediction_engine)

    mock_prediction_engine.tick.return_value = 0.9
    state = WorldState(timestamp=0, user_present=True)

    surprise = module.update_state(state)
    assert surprise == 0.9

    msg = module.run_step(context=None)

    assert msg is not None
    assert msg.source == "perception"
    assert "High Novelty" in msg.content
    assert msg.saliency > 0.5
    assert msg.metadata["surprise"] == 0.9


def test_perception_saliency_low_surprise(mock_prediction_engine):
    """Test that low surprise (boredom) leads to high saliency (Predictability)."""
    module = PerceptionModule(mock_prediction_engine)

    mock_prediction_engine.tick.return_value = 0.1
    state = WorldState(timestamp=0, user_present=True)

    surprise = module.update_state(state)
    assert surprise == 0.1

    msg = module.run_step(context=None)

    assert msg is not None
    assert "High Predictability" in msg.content
    assert msg.saliency > 0.5


def test_perception_saliency_neutral_surprise(mock_prediction_engine):
    """Test that neutral surprise leads to low saliency (suppressed)."""
    module = PerceptionModule(mock_prediction_engine)

    mock_prediction_engine.tick.return_value = 0.5
    state = WorldState(timestamp=0, user_present=True)

    module.update_state(state)
    msg = module.run_step(context=None)

    # 0.5 -> 0 dist -> 0 saliency -> suppressed (threshold > 0.2)
    assert msg is None
