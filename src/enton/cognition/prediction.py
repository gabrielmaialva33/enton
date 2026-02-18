"""Prediction Engine — Predictive Coding & Surprise Minimization.

Core component for Enton's sentience. It predicts the future state of the world
(primarily user presence and activity levels) based on historical patterns.

Concepts:
- WorldModel: A lightweight statistical model (frequency table) that learns
  routines (e.g., "User is usually active on Mondays at 10 AM").
- Surprise: The deviation between Expectation (WorldModel) and Reality (Sensors).
  High Surprise -> Alertness, Curiosity, Investigation.
  Low Surprise -> Boredom, Optimization, Internal Tasks (Dream/Study).
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PREDICTION_FILE = Path.home() / ".enton" / "memory" / "world_model.json"


@dataclass
class WorldState:
    """Snapshot of the perceivable world state."""

    timestamp: float = field(default_factory=time.time)
    user_present: bool = False
    activity_level: str = "low"  # low, medium, high
    location: str = "unknown"

    @property
    def hour_key(self) -> str:
        """Returns 'Weekday-Hour' key e.g., 'Mon-14'."""
        dt = datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%a-%H")


class WorldModel:
    """Statistical model of the world (User habits & environment patterns).

    Uses a simple frequency table mapped by (Weekday, Hour) to predict
    probability of user presence and activity.
    """


from enton.core.config import settings

# ...


class WorldModel:
    """Statistical model of the world (User habits & environment patterns)."""

    def __init__(self, persistence_path: Path | None = None) -> None:
        if persistence_path is None:
            persistence_path = Path(settings.memory_root) / "world_model.json"

        self._path = persistence_path
        # Key: "Mon-14", Value: {total: 10, present: 8, activity_high: 2...}
        self._stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "total": 0,
                "present": 0,
                "activity_low": 0,
                "activity_medium": 0,
                "activity_high": 0,
            }
        )
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self._stats[k] = v
            except Exception as e:
                logger.error("Failed to load WorldModel: %s", e)

    def save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._stats, f)
        except Exception as e:
            logger.error("Failed to save WorldModel: %s", e)

    def predict(self, timestamp: float) -> dict[str, float]:
        """Return probabilities for user presence and activity at timestamp."""
        dt = datetime.fromtimestamp(timestamp)
        key = dt.strftime("%a-%H")

        stats = self._stats.get(key)
        if not stats or stats["total"] < 5:
            # Not enough data (cold start) -> Assume uncertainty
            return {"p_present": 0.5, "uncertainty": 1.0}

        total = stats["total"]
        p_present = stats["present"] / total

        # Calculate most likely activity
        p_high = stats["activity_high"] / total
        p_med = stats["activity_medium"] / total
        p_low = stats["activity_low"] / total

        return {
            "p_present": p_present,
            "p_activity_high": p_high,
            "p_activity_medium": p_med,
            "p_activity_low": p_low,
            "uncertainty": 1.0 / math.log(total + 2),  # decay uncertainty as samples grow
        }

    def learn(self, state: WorldState) -> None:
        """Update statistics with new observation."""
        key = state.hour_key

        self._stats[key]["total"] += 1
        if state.user_present:
            self._stats[key]["present"] += 1

        self._stats[key][f"activity_{state.activity_level}"] += 1
        # ...


# ...


class PredictionEngine:
    """Main engine for Predictive Coding."""

    def __init__(self) -> None:
        self.model = WorldModel()
        self._last_save = time.time()
        self._current_surprise = 0.0

    @property
    def surprise_score(self) -> float:
        """Current surprise level (0.0 to 1.0)."""
        return self._current_surprise

    def tick(self, current_state: WorldState) -> float:
        """Process a new world state, update model, and return surprise score."""
        # 1. Predict (Expectation)
        prediction = self.model.predict(current_state.timestamp)

        # 2. Compare (Surprise Calculation)
        self._current_surprise = self._calculate_surprise(prediction, current_state)

        # 3. Learn (Update Model)
        # We always learn, but maybe with different weights?
        # For simple frequentist model, we just add the sample.
        self.model.learn(current_state)

        # Periodic save (every 5 mins)
        if time.time() - self._last_save > 300:
            self.model.save()
            self._last_save = time.time()

        return self._current_surprise

    def _calculate_surprise(self, pred: dict[str, float], state: WorldState) -> float:
        """Calculate surprise (KL-divergence-ish) between prediction and reality.

        Surprise is high when:
        - We were 90% sure User is absent, but User is present.
        - We were 90% sure User is present, but User is absent.
        """
        uncertainty = pred.get("uncertainty", 1.0)
        if uncertainty > 0.8:
            # If we don't know anything, nothing is surprising.
            # "I expected nothing, and I'm still not disappointed."
            return 0.1

        p_present = pred.get("p_present", 0.5)

        # Surprise regarding presence
        # binary cross entropy style error
        if state.user_present:
            # Surprising if p_present was low
            surprise_presence = 1.0 - p_present
        else:
            # Surprising if p_present was high
            surprise_presence = p_present

        # Surprise regarding activity (simplified)
        # If we didn't calculate activity probs (e.g. absent), skip
        surprise_activity = 0.0
        if state.user_present:
            p_act = pred.get(f"p_activity_{state.activity_level}", 0.33)
            surprise_activity = 1.0 - p_act

        # Weighted avg
        total_surprise = (0.7 * surprise_presence) + (0.3 * surprise_activity)

        return max(0.0, min(1.0, total_surprise))

    def shutdown(self) -> None:
        self.model.save()
