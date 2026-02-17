import asyncio
import logging
import os
import sys
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO)

# Mock sounddevice and torch for dry run in limited env if needed, 
# but let's try to load real modules if available to test imports.
# We might fail on device access if no audio device, so we mock sd.InputStream

sys.path.insert(0, os.path.abspath("src"))

try:
    import sounddevice
    sounddevice.InputStream = MagicMock()
    sounddevice.rec = MagicMock()
    print("Mocked sounddevice")
except ImportError:
    pass

from enton.app import App
from enton.core.config import settings

async def main():
    print("Initializing App...")
    # Disable heavy models for smoke test if needed, or keep them to test loading
    settings.stt_provider = "local"
    
    app = App()
    
    print("App initialized successfully.")
    print(f"Self Model: {app.self_model.introspect()}")
    
    # Check skills
    if hasattr(app, 'greet_skill') and app.greet_skill:
        print("Greet Skill: OK")
    else:
        print("Greet Skill: MISSING")

    if hasattr(app, 'react_skill') and app.react_skill:
        print("React Skill: OK")
    else:
        print("React Skill: MISSING")
        
    if hasattr(app, 'fuser') and app.fuser:
        print("Fuser: OK")
    else:
        print("Fuser: MISSING")
    assert app.greet_skill is not None
    assert app.react_skill is not None
    assert app.fuser is not None
    
    print("Smoke test passed!")

if __name__ == "__main__":
    asyncio.run(main())
