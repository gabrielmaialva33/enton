# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Enton is an autonomous AI robot assistant with vision, voice, personality, and consciousness. Built on **Agno 2.x** framework with a 7-provider fallback chain. Python 3.12+, async-first, CUDA-accelerated.

**Language**: All code comments, docstrings, and AI prompts are in **PT-BR**. The coding sub-agent prompt is intentionally in English (LLMs code better in EN). User communication should be in informal Brazilian Portuguese (SP dialect).

## Commands

```bash
# Run the app
uv run enton --viewer              # with vision HUD
uv run enton --webcam --viewer     # webcam mode

# Tests
uv run pytest tests/               # all tests
uv run pytest tests/test_brain.py  # single file
uv run pytest tests/test_brain.py::test_clean_removes_think -v  # single test
uv run pytest tests/ -k "test_mood" -v    # pattern match
uv run pytest tests/ -x --tb=short        # stop on first failure

# Lint
uv run ruff check src/ tests/             # check
uv run ruff check --fix src/ tests/       # auto-fix
uv run ruff format src/ tests/            # format

# Quick import check
uv run python -c "from enton.skills.my_toolkit import MyTools; print('OK')"
```

## Architecture

### Core Flow

```
__main__.py → App.__init__() → App.run()
                  │                  └─ 15+ background loops via TaskGroup
                  ├─ EventBus (async pub/sub)
                  ├─ EntonBrain (Agno Agent + fallback chain)
                  │    └─ ErrorLoopBack (retry with error context)
                  ├─ 30+ Agno Toolkits (tools exposed to LLM)
                  ├─ GlobalWorkspace (GWT consciousness cycle)
                  ├─ SubAgentOrchestrator (vision/coding/research/system)
                  ├─ ExtensionRegistry (plugin system)
                  ├─ ChannelManager (Telegram/Discord/WebSocket/Voice)
                  └─ Perception (Vision/Ears/Sounds)
```

### Provider Fallback Chain (brain.py)

Tier 1 (Agno models, full tool calling): Ollama → NVIDIA NIM (x4 keys) → HuggingFace → Groq → OpenRouter → AIMLAPI → Google Gemini

Tier 2 (CLI subprocess, text-only): Claude Code CLI → Gemini CLI

Each provider is retried with error context (ErrorLoopBack) before falling back to the next.

### Event System (core/events.py)

```python
bus.on(DetectionEvent, handler)   # register
await bus.emit(DetectionEvent())  # dispatch
```

Event types: Detection, Activity, Emotion, Transcription, Speech, Face, Sound, SceneChange, System, ChannelMessage, Skill.

### Adding a Toolkit

```python
from agno.tools import Toolkit

class MyTools(Toolkit):
    def __init__(self):
        super().__init__(name="my_tools")
        self.register(self.my_tool)

    async def my_tool(self, query: str, n: int = 5) -> str:
        """Descrição da tool (mostrada ao agent).

        Args:
            query: O que buscar.
            n: Número de resultados.
        """
        return f"Resultado: {query}"
```

Then register in `app.py`: add to `toolkits` list or use `brain.register_toolkit(MyTools(), "_my_tools")`.

### Adding a GWT Module

Extend `CognitiveModule`, implement `run_step(context) -> BroadcastMessage | None`. Register via `workspace.register_module(module)`. Modules compete by saliency (winner-take-all).

### Circular Dependency Pattern

DescribeTools and KnowledgeCrawler need brain, but brain needs them as toolkits:
```python
describe_tools = DescribeTools(self.vision)  # brain=None initially
self.brain = EntonBrain(..., toolkits=[describe_tools, ...])
describe_tools._brain = self.brain  # inject after init
```

## Key Gotchas

**Qdrant API**: Use `.query_points(query=embedding)` not `.search()`. Results are in `response.points`.

**Agno Agent in tests**: Cannot use `MagicMock()` as model — use `Ollama(id="model")` or mock at `brain.arun()` level.

**Paths with spaces**: Always `shlex.quote()` for subprocess args. The workspace HD path contains "Memory Dump".

**asyncio_mode = "auto"**: Tests auto-detect async, but `@pytest.mark.asyncio()` still works explicitly.

**NVIDIA API key rotation**: `nvidia_api_keys` is comma-separated; each key becomes a separate model in the fallback chain for natural rate-limit distribution.

**Config singleton**: `from enton.core.config import settings` — reads from `.env` file. Never hardcode settings.

## Ruff Config

Line length: 100. Ignored rules: `PLR0913` (too many args), `PLR2004` (magic numbers), `RUF012` (mutable class defaults), `PLC0415` (conditional imports), `RUF006` (fire-and-forget tasks), `TC002` (conditional 3rd-party imports), `PLW1510` (subprocess check).

## Test Conventions

- `conftest.py` isolates tests from real `~/.enton` via monkeypatch
- Qdrant mocks: mock `.query_points()` returning object with `.points` list
- Async tests: just write `async def test_...()` (auto-detected)
- Markers: `@pytest.mark.slow`, `@pytest.mark.hardware`
