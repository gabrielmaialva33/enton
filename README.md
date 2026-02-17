# 🤖 Enton

> **Autonomous AI Robot Assistant with Vision, Voice, and Genuine Personality**

**Enton** is an advanced AI agent framework designed to power autonomous robots. It combines state-of-the-art perception models, cognitive planning, and expressive interaction capabilities to create a digital being that feels alive.

Built with performance and modularity in mind, Enton integrates vision, hearing, and speech into a cohesive cognitive loop.

## ✨ Key Features

- **🧠 Advanced Cognition**: Powered by a sophisticated `Brain` module that handles memory, planning, and persona consistency.
- **👀 Computer Vision**: Real-time object detection (YOLO), face recognition (InsightFace), and visual understanding (Qwen-VL).
- **🗣️ Natural Interaction**:
  - **Hearing**: High-fidelity speech-to-text with `Faster-Whisper` and `Silero VAD`.
  - **Speaking**: Expressive TTS using `Kokoro` and cloud providers (Google/NVIDIA).
- **🎭 Dynamic Persona**: Maintains emotional state and personality traits that evolve during interactions.
- **⚡ High Performance**: Optimized for CUDA acceleration on NVIDIA GPUs (RTX 3090/4090 recommended).
- **🛠️ Modular Providers**: Plug-and-play support for Ollama, OpenAI, Google Gemini, and local LLMs.

## 🏗️ Architecture

Enton creates a **Cognitive Loop** that cycles through:

1.  **Perception**: Ingesting video/audio streams and extracting semantic meaning.
2.  **Cognition**: Processing inputs against memory and current goals to decide actions.
3.  **Action**: Executing physical movements (PTZ), speaking, or running tools.

## 🚀 Quick Start

Ensure you have Python 3.12+ and CUDA drivers installed.

```bash
# Install dependencies with uv (recommended)
uv sync

# Run the autonomous agent
uv run enton
```

## 📦 Tech Stack

- **Core**: Python 3.12, FastAPI, Pydantic
- **AI/ML**: PyTorch, Ultralytics YOLO, InsightFace, Hugging Face Transformers
- **Audio**: SoundDevice, Faster-Whisper, Kokoro TTS
- **Storage**: Qdrant (Vector DB), Redis (State)

## 📜 License

MIT - Gabriel Maia ([@gabrielmaialva33](https://github.com/gabrielmaialva33))

<p align="center">
  <strong>✨ Bringing digital entities to life ✨</strong>
</p>
