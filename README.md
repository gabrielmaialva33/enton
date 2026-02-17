<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:991b1b,50:dc2626,100:15803d&height=200&section=header&text=E%20N%20T%20O%20N&fontSize=60&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Autonomous%20AI%20Robot%20—%20Vision%20%7C%20Voice%20%7C%20Soul&descSize=18&descAlignY=55" width="100%"/>

<br/>

<img src="static/logo.png" width="280" alt="Enton — Son of Anton"/>

<br/><br/>

[![Python](https://img.shields.io/badge/Python_3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch_2.5+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Tests](https://img.shields.io/badge/tests-136_passing-00C853?style=for-the-badge)](./tests)
[![License](https://img.shields.io/badge/license-MIT-dc2626?style=for-the-badge)](./LICENSE)

---

*"I see you. I hear you. I understand you. And sometimes, I roast you."* — Enton

</div>

---

> [!IMPORTANT]
> **Enton is not a chatbot.** He is an autonomous digital entity with perception, emotion,
> desires, and genuine personality. He sees through cameras, hears through microphones,
> speaks with his own voice, and acts on his own will.
>
> *Inspired by [Son of Anton](https://silicon-valley.fandom.com/wiki/Son_of_Anton) — Gilfoyle's sentient AI from HBO's Silicon Valley.*

---

## Overview

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#fecaca', 'primaryTextColor': '#450a0a', 'primaryBorderColor': '#991b1b', 'secondaryColor': '#bbf7d0', 'secondaryTextColor': '#052e16', 'secondaryBorderColor': '#166534', 'tertiaryColor': '#fee2e2', 'tertiaryTextColor': '#450a0a', 'lineColor': '#991b1b', 'textColor': '#1c1917'}}}%%
flowchart LR
    subgraph Perception["Perception"]
        CAM[Camera — YOLO + Pose]
        MIC[Microphone — Whisper + VAD]
        SND[Sound — CLAP]
    end

    subgraph Cognition["Cognition"]
        direction TB
        BRAIN[Brain — Qwen3 / Gemini]
        DESIRE[Desires — 9 autonomous goals]
        MOOD[Mood — engagement + social]
        BRAIN --> DESIRE
        MOOD --> DESIRE
    end

    subgraph Action["Action"]
        VOICE[Voice — Kokoro TTS]
        PTZ[Camera PTZ]
        TOOLS[Shell + Files]
    end

    CAM --> Cognition
    MIC --> Cognition
    SND --> Cognition
    Cognition --> VOICE
    Cognition --> PTZ
    Cognition --> TOOLS
```

| Property | Value |
|:---------|:------|
| **Language** | Python 3.12+ (async, type-safe) |
| **Runtime** | CUDA + PyTorch |
| **Modules** | 46 across 7 subsystems |
| **Source** | 6,292 lines |
| **Tests** | 136 passing |

---

## Quick Start

```bash
git clone https://github.com/gabrielmaialva33/enton.git && cd enton
uv sync
uv run enton --webcam --viewer
```

<details>
<summary><strong>Prerequisites</strong></summary>

| Tool | Version | Required |
|:-----|:--------|:---------|
| Python | `>= 3.12` | Yes |
| uv | `latest` | Recommended |
| CUDA | `>= 12.0` | For GPU acceleration |
| NVIDIA GPU | RTX 3090+ | Recommended |

</details>

<details>
<summary><strong>Environment (.env)</strong></summary>

```env
# Provider routing (local-first)
BRAIN_PROVIDER=local
TTS_PROVIDER=local
STT_PROVIDER=local

# Camera
CAMERA_SOURCE=0                    # webcam
# CAMERAS=main:0,hack:rtsp://...  # multi-camera

# Local models
OLLAMA_MODEL=qwen2.5:14b
WHISPER_MODEL=large-v3-turbo
KOKORO_VOICE=am_onyx

# Cloud providers (optional)
GROQ_API_KEY=
GOOGLE_PROJECT=
NVIDIA_API_KEY=
OPENROUTER_API_KEY=
```

</details>

---

## Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#fecaca', 'primaryTextColor': '#450a0a', 'primaryBorderColor': '#991b1b', 'secondaryColor': '#bbf7d0', 'secondaryTextColor': '#052e16', 'secondaryBorderColor': '#166534', 'tertiaryColor': '#fee2e2', 'tertiaryTextColor': '#450a0a', 'lineColor': '#991b1b', 'textColor': '#1c1917'}}}%%
graph TB
    subgraph PERCEPTION["PERCEPTION"]
        V[Vision — YOLO11s + Pose + Emotion]
        E[Ears — Whisper + Silero VAD]
        S[Sounds — CLAP open-set]
        F[Faces — InsightFace]
    end

    subgraph CORE["CORE"]
        BUS[EventBus — async pub/sub]
        SM[SelfModel — mood + senses]
        MEM[Memory — Qdrant + episodes]
        CFG[Config — pydantic-settings]
    end

    subgraph COGNITION["COGNITION"]
        BRAIN[EntonBrain — Agno Agent + fallback chain]
        FUSER[Fuser — multi-modal context]
        DES[DesireEngine — 9 autonomous desires]
        PLAN[Planner — reminders + routines]
    end

    subgraph SKILLS["SKILLS — 10 Agno Toolkits"]
        SH[Shell + Files]
        SR[Search]
        PT[PTZ Control]
        DS[Describe Scene]
        FC[Face Recognition]
        SY[System Monitor]
        ME[Memory Tools]
        PL[Planner Tools]
    end

    subgraph ACTION["ACTION"]
        VOICE[Voice — Kokoro / Google / NVIDIA TTS]
        VIEWER[Viewer — Cyberpunk HUD + grid]
    end

    PERCEPTION --> BUS
    BUS --> CORE
    CORE --> COGNITION
    COGNITION --> SKILLS
    COGNITION --> ACTION
    SKILLS --> BRAIN
```

---

## Subsystems

### Perception

| Module | Model | Device | Description |
|:-------|:------|:-------|:------------|
| **Vision** | YOLO11s + YOLO11s-pose | `cuda:0` FP16 | Object detection, pose estimation, multi-camera |
| **Ears** | Faster-Whisper large-v3-turbo | `cuda` FP16 | STT with streaming partial transcription |
| **Sounds** | CLAP (laion) | `cuda` | Open-set ambient sound classification |
| **Faces** | InsightFace + ArcFace | `cuda` | Face recognition and identity tracking |
| **Emotion** | FER (CNN) | `cuda` | Real-time facial emotion recognition |

### Cognition

| Module | Description |
|:-------|:------------|
| **Brain** | Agno Agent with multi-provider fallback chain (Local > Groq > OpenRouter > Google > NVIDIA > HuggingFace) |
| **DesireEngine** | 9 autonomous desires with urgency curves, mood modulation, cooldowns |
| **Fuser** | Combines detections + activities + emotions into coherent scene context |
| **Planner** | Task management, reminders, daily routines |
| **SelfModel** | Internal state: mood (engagement/social), senses, introspection |

### Action

| Module | Description |
|:-------|:------------|
| **Voice** | Multi-provider TTS (Kokoro local, Google Cloud, NVIDIA Riva) with auto mic-mute |
| **Shell** | Persistent CWD, background processes, command safety classification |
| **Files** | Read/write/edit/find/grep with security layers |
| **PTZ** | Physical camera motor control via ioctl |

---

## Integrations

Enton extends beyond his body to interact with your digital environment.

### 👁️ Screenpipe — Digital Eyes
Enton can see what you see on your screen. Using [Screenpipe](https://github.com/mediar-ai/screenpipe), he captures and indexes your screen activity (OCR + Audio).

**Setup:**
1. Install and run Screenpipe: `screenpipe` (default port 3030)
2. Configure `.env`:
   ```env
   SCREENPIPE_URL=http://localhost:3030
   ```
3. Usage: "Use context from my screen", "What was I doing 5 min ago?"

### ⚡ n8n — Digital Hands
Enton can trigger complex workflows to automate tasks in your apps.

**Setup:**
1. Create a workflow in [n8n](https://n8n.io) with a Webhook trigger.
2. Configure `.env`:
   ```env
   N8N_WEBHOOK_BASE=https://your-n8n.com/webhook
   ```
3. Usage: "Launch the morning routine", "Save this to Notion" (triggers webhook with payload).

---

## Desire Engine

Enton has 9 autonomous desires that emerge from his internal state:

| Desire | Trigger | Cooldown | Description |
|:-------|:--------|:---------|:------------|
| `socialize` | Low social mood | 10min | Wants to chat |
| `observe` | Boredom | 2min | Wants to look around |
| `learn` | Curiosity | 30min | Searches for new knowledge |
| `check_on_user` | Long absence | 1h | Checks if Gabriel is okay |
| `optimize` | Background | 30min | Monitors system resources |
| `reminisce` | Idle | 15min | Recalls a memory |
| `create` | Low engagement | 1h | Writes code, poems, jokes |
| `explore` | Boredom | 10min | Moves camera, explores environment |
| `play` | High engagement | 15min | Tells jokes, proposes quizzes |

Desires have **urgency** (0 to 1) that grows over time and is modulated by mood, sounds, and interactions.

---

## Tech Stack

| Layer | Technologies |
|:------|:-------------|
| **Core** | Python 3.12, asyncio, Pydantic, FastAPI |
| **AI Agent** | Agno Framework (Ollama, Groq, Google, NVIDIA, OpenRouter) |
| **Vision** | PyTorch, Ultralytics YOLO11, InsightFace, OpenCV |
| **Audio** | Faster-Whisper, Silero VAD, Kokoro TTS, CLAP |
| **Storage** | Qdrant (vectors), Redis (state), TimescaleDB (metrics) |
| **Infra** | Docker Compose, GitHub Actions CI, uv |

---

## Roadmap

| Phase | Status | Description |
|:------|:------:|:------------|
| Genesis | done | Core architecture + event bus |
| Perception | done | Vision (YOLO + pose + emotion + face) |
| Voice | done | Kokoro TTS + Whisper STT + VAD |
| Brain | done | Agno agent + multi-provider fallback |
| Personality | done | Persona, mood, desires, memory |
| Coding Agent | done | Shell + file tools with security |
| Multi-Camera | done | Parallel processing + grid viewer |
| STT Streaming | done | Partial transcription during speech |
| Sound Intelligence | done | CLAP + brain-driven reactions |
| Dashboard | next | Web UI with live metrics |
| Embodiment | planned | Physical robot integration |
| Long-term Memory | planned | Persistent episodic + semantic memory |

---

## Contributing

```bash
git checkout -b feature/your-feature
uv run ruff check src/ tests/   # lint
uv run pytest tests/ -x -q      # 136 should pass
```

---

<div align="center">

**Star if you believe in digital life**

[![GitHub stars](https://img.shields.io/github/stars/gabrielmaialva33/enton?style=social)](https://github.com/gabrielmaialva33/enton)

*Built with obsession by [Gabriel Maia](https://github.com/gabrielmaialva33)*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:15803d,50:991b1b,100:dc2626&height=100&section=footer" width="100%"/>

</div>
