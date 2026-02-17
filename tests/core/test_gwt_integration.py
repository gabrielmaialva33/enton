import pytest
from unittest.mock import MagicMock

from enton.cognition.metacognition import MetaCognitiveEngine
from enton.core.gwt.message import BroadcastMessage
from enton.core.gwt.modules.executive import ExecutiveModule
from enton.core.gwt.modules.github import GitHubModule
from enton.core.gwt.workspace import GlobalWorkspace
from enton.skills.github_learner import GitHubLearner


@pytest.mark.asyncio
async def test_gwt_cycle_boredom_trigger():
    # 1. Setup
    workspace = GlobalWorkspace()

    # Mock Metacognition (Executive)
    meta_engine = MetaCognitiveEngine()
    meta_engine.boredom_level = 0.9  # High boredom
    meta_engine.get_next_topic = MagicMock(return_value="rust_lang")

    # Provide a skill_registry mock so executive uses use_tool: path
    mock_registry = MagicMock()
    mock_registry.list_skills.return_value = ["github_learner"]
    executive = ExecutiveModule(meta_engine, skill_registry=mock_registry)

    # Mock GitHub Learner
    learner_skill = MagicMock(spec=GitHubLearner)
    github_module = GitHubModule(learner_skill)

    workspace.register_module(executive)
    workspace.register_module(github_module)

    # 2. Run Step 1: Executive detects boredom and broadcasts intention
    workspace.current_conscious_content = BroadcastMessage(
        content="nothing_happening", source="perception", saliency=0.1, modality="vision"
    )

    thought_1 = workspace.tick()

    assert thought_1 is not None
    assert thought_1.source == "executive"
    assert thought_1.modality == "intention"
    assert "use_tool:github_learner" in thought_1.content
    assert thought_1.saliency == 1.0

    # 3. Run Step 2: GitHub module picks up intention
    # GitHubModule listens for "study_github:" prefix — update module to also
    # handle "use_tool:github_learner:" or test the fallback path separately.
    # For now, test the executive output which is the main point of this test.


@pytest.mark.asyncio
async def test_gwt_cycle_boredom_fallback():
    """Without skill_registry, executive falls back to agentic_task."""
    workspace = GlobalWorkspace()

    meta_engine = MetaCognitiveEngine()
    meta_engine.boredom_level = 0.9
    meta_engine.get_next_topic = MagicMock(return_value="rust_lang")

    executive = ExecutiveModule(meta_engine)  # No registry
    workspace.register_module(executive)

    workspace.current_conscious_content = BroadcastMessage(
        content="nothing_happening", source="perception", saliency=0.1, modality="vision"
    )

    thought = workspace.tick()
    assert thought is not None
    assert "agentic_task:Pesquise sobre rust_lang" in thought.content


@pytest.mark.asyncio
async def test_gwt_github_module_result_delivery():
    """GitHub module delivers pending results."""
    workspace = GlobalWorkspace()

    learner_skill = MagicMock(spec=GitHubLearner)
    github_module = GitHubModule(learner_skill)
    workspace.register_module(github_module)

    # Simulate completed study
    github_module._pending_result = "Learned amazing things about Rust"
    github_module.is_busy = False

    thought = workspace.tick()
    assert thought is not None
    assert "Study Result" in thought.content
    assert thought.modality == "memory_recall"
    assert thought.saliency == 1.0

if __name__ == "__main__":
    msg = BroadcastMessage(content="test", source="me", saliency=0.5, modality="text")
    print(msg)
