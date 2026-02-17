import pytest
import asyncio
from unittest.mock import MagicMock
from enton.core.gwt.workspace import GlobalWorkspace
from enton.core.gwt.modules.executive import ExecutiveModule
from enton.core.gwt.modules.github import GitHubModule
from enton.core.gwt.message import BroadcastMessage
from enton.cognition.metacognition import MetaCognitiveEngine
from enton.skills.github_learner import GitHubLearner

@pytest.mark.asyncio
async def test_gwt_cycle_boredom_trigger():
    # 1. Setup
    workspace = GlobalWorkspace()
    
    # Mock Metacognition (Executive)
    meta_engine = MetaCognitiveEngine()
    meta_engine.boredom_level = 0.9 # High boredom
    meta_engine.get_next_topic = MagicMock(return_value="rust_lang")
    
    executive = ExecutiveModule(meta_engine)
    
    # Mock GitHub Learner
    learner_skill = MagicMock(spec=GitHubLearner)
    learner_property = MagicMock() # Create a separate mock for the property
    type(learner_skill).name = learner_property # Assign property to class attribute
    
    github_module = GitHubModule(learner_skill)
    
    workspace.register_module(executive)
    workspace.register_module(github_module)
    
    # 2. Run Step 1: Executive detects boredom and broadcasts intention
    # Simulate a previous perception message context
    workspace.current_conscious_content = BroadcastMessage(
        content="nothing_happening", source="perception", saliency=0.1, modality="vision"
    )
    
    thought_1 = workspace.tick()
    
    assert thought_1 is not None
    assert thought_1.source == "executive"
    assert thought_1.modality == "intention"
    assert "study_github:rust_lang" in thought_1.content
    assert thought_1.saliency == 1.0
    
    # 3. Run Step 2: GitHub module picks up intention and starts working
    # Since run_step spawns a task, we need to wait a bit or check internal state
    thought_2 = workspace.tick()
    
    # GitHub module should reply "Starting study..."
    assert thought_2 is not None
    assert thought_2.source == "github_skill"
    assert "Starting study" in thought_2.content
    assert thought_2.modality == "inner_speech"
    
    # 4. Run Step 3: Job is running in background, module is busy -> No output
    thought_3 = workspace.tick()
    assert thought_3 is None # Silence while working
    
    # 5. Simulate async job completion
    # We cheat by manually setting the pending result since we mocked the actual study call
    github_module._pending_result = "Learned amazing things about Rust"
    github_module.is_busy = False # Manually free it up
    
    # 6. Run Step 4: Result delivery
    thought_4 = workspace.tick()
    assert thought_4 is not None
    assert "Study Result" in thought_4.content
    assert thought_4.modality == "memory_recall"
    assert thought_4.saliency == 1.0

if __name__ == "__main__":
    msg = BroadcastMessage(content="test", source="me", saliency=0.5, modality="text")
    print(msg)
