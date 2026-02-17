import time
import pytest
from pathlib import Path
from enton.cognition.prediction import PredictionEngine, WorldState, WorldModel

def test_world_model_learning(tmp_path):
    """Test that WorldModel learns patterns over repeated exposure."""
    model_file = tmp_path / "world_model.json"
    
    # Initialize engine pointing to tmp file
    # We need to ensure the WorldModel inside uses this path
    # Since PredictionEngine creates WorldModel internally without args,
    # we patch it after creation or rely on dependency injection if we refactored it.
    # In current implementation, PredictionEngine() -> WorldModel().
    # We can set _path manually.
    
    engine = PredictionEngine()
    engine.model._path = model_file
    
    # 1. Cold Start
    ts = 1696240800.0  # Mon 10AM
    state = WorldState(timestamp=ts, user_present=True)
    
    surprise_start = engine.tick(state)
    # Cold start: high uncertainty (1.0) -> low surprise logic (0.1)
    assert surprise_start < 0.3
    
    # 2. Training (20 iters)
    for _ in range(20):
        engine.tick(state)
        
    # 3. High Confidence Prediction
    prediction = engine.model.predict(ts)
    assert prediction["p_present"] > 0.8
    assert prediction["uncertainty"] < 0.5
    
    # 4. Anomaly (Absent when expected Present)
    state_anomaly = WorldState(timestamp=ts, user_present=False)
    surprise_anomaly = engine.tick(state_anomaly)
    
    # Should be high surprise (> 0.5)
    assert surprise_anomaly > 0.5
    assert surprise_anomaly > surprise_start

def test_persistence(tmp_path):
    """Test saving and loading."""
    model_file = tmp_path / "world_model.json"
    engine = PredictionEngine()
    engine.model._path = model_file
    
    ts = 1700000000.0
    engine.tick(WorldState(timestamp=ts, user_present=True))
    engine.shutdown()
    
    assert model_file.exists()
    
    # Reload
    engine2 = PredictionEngine()
    engine2.model._path = model_file
    engine2.model._load()
    
    # Check internal stats
    from datetime import datetime
    key = datetime.fromtimestamp(ts).strftime("%a-%H")
    assert engine2.model._stats[key]["total"] == 1
