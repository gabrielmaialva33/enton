from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(StrEnum):
    GOOGLE = "google"
    NVIDIA = "nvidia"
    LOCAL = "local"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Camera — use "0" for /dev/video0, or full RTSP URL
    camera_source: str = "0"
    camera_ip: str = "192.168.1.23"
    camera_rtsp_port: int = 554
    camera_rtsp_path: str = "/onvif1"

    # Provider routing
    brain_provider: Provider = Provider.GOOGLE
    tts_provider: Provider = Provider.GOOGLE
    stt_provider: Provider = Provider.GOOGLE

    # Google Cloud
    google_project: str = ""
    google_location: str = "us-central1"
    google_brain_model: str = "gemini-2.0-flash"
    google_vision_model: str = "gemini-2.0-flash"

    # NVIDIA
    nvidia_api_key: str = ""
    nvidia_tts_voice: str = "English-US.Male-1"
    nvidia_stt_model: str = "parakeet-1.1b-rnnt-multilingual-asr"

    # HuggingFace
    huggingface_token: str = ""

    # Local fallback
    ollama_model: str = "qwen3:4b"
    whisper_model: str = "large-v3-turbo"
    kokoro_lang: str = "p"  # pt-BR
    kokoro_voice: str = "af_heart"

    # Vision
    yolo_model: str = "models/yolo11x.pt"
    yolo_confidence: float = 0.15
    yolo_device: str = "cuda:0"
    yolo_pose_model: str = "models/yolo11x-pose.pt"
    yolo_pose_confidence: float = 0.2
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

    @property
    def camera_url(self) -> str | int:
        if self.camera_source.isdigit():
            return int(self.camera_source)
        if self.camera_source.startswith("rtsp://"):
            return self.camera_source
        return f"rtsp://{self.camera_ip}:{self.camera_rtsp_port}{self.camera_rtsp_path}"

    @property
    def yolo_model_path(self) -> Path:
        return Path(self.yolo_model)


settings: Settings = Settings()
