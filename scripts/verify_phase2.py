import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO)

from enton.cognition.brain import Brain
from enton.core.config import Settings, Provider

async def test_brain_loop():
    print("Testing Brain think_with_tools loop...")
    
    # Mock settings
    settings = Settings()
    settings.brain_provider = Provider.LOCAL
    settings.brain_max_turns = 3
    
    # Mock provider
    mock_provider = AsyncMock()
    
    # Simula: 
    # 1. Retorna tool call "get_weather"
    # 2. Retorna resposta final baseada no tool result
    mock_provider.generate_with_tools.side_effect = [
        {"content": "", "tool_calls": [{"name": "get_weather", "arguments": {"city": "São Paulo"}}]},
        {"content": "O clima em São Paulo é ensolarado.", "tool_calls": []}
    ]
    
    brain = Brain(settings)
    # Inject mock provider
    brain._providers[Provider.LOCAL] = mock_provider
    
    # Mock tool function
    async def get_weather(city: str):
        print(f"Executing tool get_weather for {city}")
        return "Ensolarado, 25C"
    
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }
    }]
    
    tool_functions = {"get_weather": get_weather}
    
    response = await brain.think_with_tools("Como está o tempo em SP?", tools, tool_functions)
    
    print(f"Final response: {response}")
    
    assert "ensolarado" in response.lower()
    assert mock_provider.generate_with_tools.call_count == 2
    print("Test passed!")

if __name__ == "__main__":
    asyncio.run(test_brain_loop())
