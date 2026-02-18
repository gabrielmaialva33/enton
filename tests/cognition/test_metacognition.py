import time

from enton.cognition.metacognition import MetaCognitiveEngine


def test_metacognition_initialization():
    engine = MetaCognitiveEngine()
    assert engine.boredom_level == 0.0
    assert len(engine.curiosity_queue) == 0


def test_boredom_accumulation():
    engine = MetaCognitiveEngine()

    # Simulate time jump
    engine._last_tick = time.time() - 100.0

    # Tick with low surprise (should increase boredom)
    action = engine.tick(surprise_score=0.1)

    # 0.0 + (0.01 * 100) = 1.0 (clamped)
    assert engine.boredom_level >= 0.99

    # Should trigger action
    assert action == "study_github"


def test_boredom_reduction_on_surprise():
    engine = MetaCognitiveEngine()
    engine.boredom_level = 0.8

    # Tick with high surprise
    engine.tick(surprise_score=0.9)

    # Should drop by 0.5 -> 0.3
    assert engine.boredom_level <= 0.30001


def test_curiosity_queue_management():
    engine = MetaCognitiveEngine()
    engine.add_curiosity("rust lang", "user")

    assert len(engine.curiosity_queue) == 1
    topic = engine.get_next_topic()
    assert topic == "rust lang"
    assert len(engine.curiosity_queue) == 0

    # Default fallback
    fallback_topic = engine.get_next_topic()
    assert fallback_topic in engine.default_interests
