import logging
import json
from unittest.mock import AsyncMock

# Configure logging
logging.basicConfig(level=logging.INFO)

# 1. Test Registry and Skills
print("--- Testing Registry & Skills ---")
from enton.core.tools import registry
import enton.skills.system_skill
import enton.skills.search_skill

print(f"Registered tools: {list(registry.get_all_tools().keys())}")

schemas = registry.schemas
print(f"Schemas generated: {len(schemas)}")
print(json.dumps(schemas, indent=2))

assert "get_system_stats" in registry.get_all_tools()
assert "search_web" in registry.get_all_tools()

# 2. Test Brain Integration
print("\n--- Testing Brain.think_agent ---")
from enton.cognition.brain import Brain
from enton.core.config import Settings, Provider

async def test_brain_agent():
    settings = Settings()
    settings.brain_provider = Provider.LOCAL
    
    brain = Brain(settings)
    mock_provider = AsyncMock()
    brain._providers[Provider.LOCAL] = mock_provider
    
    # Mock response calling a tool
    mock_provider.generate_with_tools.side_effect = [
        {"content": "", "tool_calls": [{"name": "get_time", "arguments": {}}]},
        {"content": "São 10:00", "tool_calls": []}
    ]
    
    response = await brain.think_agent("Que horas são?", system="Sys")
    
    print(f"Final Agent Response: {response}")
    assert "São 10:00" in response
    assert mock_provider.generate_with_tools.call_count == 2
    
    # Verify first call had tools
    call_args = mock_provider.generate_with_tools.call_args_list[0]
    # args: (prompt, tools=..., system=..., history=...)
    # verify tools were passed
    assert len(call_args.kwargs['tools']) > 0
    print("Brain agent test passed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_brain_agent())
