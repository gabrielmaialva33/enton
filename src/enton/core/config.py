from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(StrEnum):
    LOCAL = "local"
    NVIDIA = "nvidia"
    HUGGINGFACE = "huggingface"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    AIMLAPI = "aimlapi"
    GOOGLE = "google"
    QWEN3 = "qwen3"
    EDGE = "edge"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Camera — use "0" for /dev/video0, or full RTSP URL
    camera_source: str = "0"
    camera_ip: str = "localhost"
    camera_rtsp_port: int = 554
    camera_rtsp_path: str = "/onvif1"

    # Multi-camera: comma-separated "id:source" pairs
    # e.g. "main:0,hack:rtsp://192.168.18.23:554/video0_unicast"
    cameras: str = ""

    # Provider routing (local-first)
    brain_provider: Provider = Provider.LOCAL
    tts_provider: Provider = Provider.QWEN3
    stt_provider: Provider = Provider.LOCAL

    # Google Cloud
    google_project: str = ""
    google_location: str = "us-central1"
    google_brain_model: str = "gemini-2.0-flash"
    google_vision_model: str = "gemini-2.0-flash"

    # NVIDIA NIM (round-robin: comma-separated keys, 40 RPM each)
    nvidia_api_keys: str = ""  # comma-separated for round-robin
    nvidia_api_key: str = ""  # legacy single key (STT/TTS)
    nvidia_nim_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    nvidia_nim_vision_model: str = "meta/llama-3.2-90b-vision-instruct"
    nvidia_tts_voice: str = "English-US.Male-1"
    nvidia_stt_model: str = "parakeet-1.1b-rnnt-multilingual-asr"

    # Groq (free tier, fast inference)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # OpenRouter (free tier, multi-provider router)
    openrouter_api_key: str = ""
    openrouter_model: str = "qwen/qwen3-235b-a22b:free"
    openrouter_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct:free"

    # AIML API
    aimlapi_api_key: str = ""
    aimlapi_model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

    # HuggingFace (Pro account — serverless inference API)
    huggingface_token: str = ""
    huggingface_model: str = "Qwen/Qwen2.5-72B-Instruct"
    huggingface_vision_model: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # Local fallback
    ollama_model: str = "qwen2.5:14b"
    whisper_model: str = "large-v3-turbo"
    kokoro_lang: str = "p"  # pt-BR
    kokoro_voice: str = "pm_alex"  # pt-BR male voice

    # Qwen3-TTS (primary — local GPU with voice design)
    qwen3_tts_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    qwen3_tts_voice_instruct: str = (
        "A deep, authoritative male voice with a robotic, futuristic tone. "
        "Speaks Portuguese with confidence and clarity, like an AI assistant "
        "from a sci-fi movie. Slightly metallic timbre."
    )
    qwen3_tts_device: str = "cuda:0"

    # Edge-TTS (cloud fallback — free Microsoft Neural TTS)
    edge_tts_voice: str = "pt-BR-AntonioNeural"
    
    # Brain
    brain_timeout: float = 30.0
    brain_max_turns: int = 5

    # Vision
    yolo_model: str = "models/yolo11s.pt"
    yolo_confidence: float = 0.35
    yolo_device: str = "cuda:0"
    yolo_pose_model: str = "models/yolo11s-pose.pt"
    yolo_pose_confidence: float = 0.35
    yolo_pose_device: str = "cuda:0"  # separate GPU for pose if available

    # Audio
    sample_rate: int = 16000
    audio_channels: int = 1

    # Dashboard
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8080

    # Behavior
    reaction_cooldown: float = 5.0
    idle_timeout: float = 30.0
    memory_size: int = 20
    memory_root: str = str(Path.home() / ".enton" / "memory")
    scene_describe_interval: float = 30.0

    # Local VLM
    ollama_vlm_model: str = "qwen2.5-vl:7b"
    vlm_transformers_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Visual Memory (v0.3.0)
    siglip_model: str = "ViT-B-16-SigLIP"
    siglip_pretrained: str = "webli"
    frames_dir: str = str(Path.home() / ".enton" / "frames")
    commonsense_collection: str = "enton_commonsense"

    # BlobStore (v0.5.0) — external HD binary memory
    # Using fallback as default if not provided via env
    blob_store_root: str = str(Path.home() / ".enton" / "blobs")
    blob_store_fallback: str = str(Path.home() / ".enton" / "blobs")

    # Workspace — Enton's personal sandbox (v0.7.0)
    workspace_root: str = str(Path.home() / ".enton" / "workspace")

    # Self-Evolution (v0.4.0)
    skills_dir: str = str(Path.home() / ".enton" / "skills")
    forge_sandbox_timeout: float = 10.0
    forge_max_retries: int = 1

    # CLI AI Providers (subprocess fallback)
    claude_code_enabled: bool = True
    claude_code_model: str = "sonnet"
    claude_code_timeout: float = 120.0
    claude_code_max_turns: int = 10
    gemini_cli_enabled: bool = True
    gemini_cli_model: str = "gemini-2.5-flash"
    gemini_cli_timeout: float = 120.0
    gemini_cli_yolo: bool = False

    # Android Phone Control (v0.6.0)
    phone_adb_path: str = ""  # auto-detect: PATH then ~/Android/Sdk/platform-tools/adb
    phone_serial: str = ""  # device serial (empty = first connected device)
    phone_enabled: bool = True
    phone_wifi_host: str = ""  # phone WiFi/Tailscale IP for wireless ADB
    phone_wifi_port: int = 5555  # ADB TCP port (default 5555)

    # Integrations (Screenpipe + n8n)
    screenpipe_url: str = "http://localhost:3030"
    n8n_webhook_base: str = ""  # e.g. https://n8n.example.com/webhook

    # Infrastructure
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379"
    timescale_dsn: str = "postgresql://enton:enton@localhost:5432/enton"
    mem0_enabled: bool = True

    # Metrics
    metrics_interval: float = 10.0

    @property
    def camera_url(self) -> str | int:
        if self.camera_source.isdigit():
            return int(self.camera_source)
        if self.camera_source.startswith("rtsp://"):
            return self.camera_source
        return f"rtsp://{self.camera_ip}:{self.camera_rtsp_port}{self.camera_rtsp_path}"

    @property
    def camera_sources(self) -> dict[str, str | int]:
        """Parse multi-camera config into {id: source} dict."""
        if self.cameras:
            result: dict[str, str | int] = {}
            for entry in self.cameras.split(","):
                entry = entry.strip()
                if ":" in entry:
                    cam_id, source = entry.split(":", 1)
                    result[cam_id.strip()] = (
                        int(source) if source.strip().isdigit() else source.strip()
                    )
            if result:
                return result
        # Fallback: single camera from camera_source
        return {"main": self.camera_url}

    @property
    def yolo_model_path(self) -> Path:
        return self._resolve_engine(self.yolo_model)

    @property
    def yolo_pose_model_path(self) -> Path:
        return self._resolve_engine(self.yolo_pose_model)

    @staticmethod
    def _resolve_engine(model: str) -> Path:
        """Return .engine path if it exists, otherwise the original .pt path."""
        pt = Path(model)
        engine = pt.with_suffix(".engine")
        if engine.exists():
            return engine
        return pt


settings: Settings = Settings()
