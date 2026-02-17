"""MediaTools — Agno toolkit for media control and downloads.

Enton can download videos/audio (yt-dlp), control media playback (playerctl),
adjust system volume (pactl), play audio files (mpv/ffplay).

Requires: yt-dlp, playerctl, pactl, mpv, ffmpeg (all pre-installed).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from agno.tools import Toolkit

logger = logging.getLogger(__name__)


class MediaTools(Toolkit):
    """Tools for media download, playback, and audio control."""

    def __init__(self, workspace: Path | None = None) -> None:
        super().__init__(name="media_tools")
        self._downloads = (workspace or Path.home() / ".enton" / "workspace") / "downloads"
        self._downloads.mkdir(parents=True, exist_ok=True)

        self.register(self.download_video)
        self.register(self.download_audio)
        self.register(self.media_info)
        self.register(self.play_media)
        self.register(self.player_control)
        self.register(self.volume_get)
        self.register(self.volume_set)
        self.register(self.list_audio_sinks)
        self.register(self.tts_speak)

    def download_video(self, url: str, quality: str = "best") -> str:
        """Download video from YouTube or any supported site (yt-dlp).

        Args:
            url: Video URL (YouTube, Twitter, Reddit, etc.)
            quality: Quality preset ("best", "720", "480", "audio_only")
        """
        fmt = {
            "best": "bestvideo+bestaudio/best",
            "720": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "480": "bestvideo[height<=480]+bestaudio/best[height<=480]",
            "audio_only": "bestaudio/best",
        }.get(quality, quality)

        try:
            result = subprocess.run(
                [
                    "yt-dlp", "-f", fmt,
                    "--merge-output-format", "mp4",
                    "-o", str(self._downloads / "%(title)s.%(ext)s"),
                    "--no-playlist",
                    "--restrict-filenames",
                    url,
                ],
                timeout=300,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                # Find downloaded file
                lines = result.stdout.split("\n")
                for line in reversed(lines):
                    if "Destination:" in line or "has already been downloaded" in line:
                        return f"Download OK: {line.strip()}"
                    if "[Merger]" in line:
                        return f"Download OK: {line.strip()}"
                return f"Download completo em {self._downloads}"
            return f"Erro yt-dlp: {result.stderr[:300]}"
        except subprocess.TimeoutExpired:
            return "Timeout (5min)"
        except Exception as e:
            return f"Erro: {e}"

    def download_audio(self, url: str, audio_format: str = "mp3") -> str:
        """Download audio only from URL (yt-dlp).

        Args:
            url: Video/audio URL
            audio_format: Output format ("mp3", "wav", "opus", "flac")
        """
        try:
            result = subprocess.run(
                [
                    "yt-dlp", "-x",
                    "--audio-format", audio_format,
                    "-o", str(self._downloads / "%(title)s.%(ext)s"),
                    "--no-playlist",
                    "--restrict-filenames",
                    url,
                ],
                timeout=300,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return f"Audio baixado em {self._downloads}"
            return f"Erro: {result.stderr[:300]}"
        except Exception as e:
            return f"Erro: {e}"

    def media_info(self, url_or_path: str) -> str:
        """Get media info from URL or local file (title, duration, formats).

        Args:
            url_or_path: URL or local file path
        """
        if url_or_path.startswith(("http://", "https://")):
            try:
                result = subprocess.run(
                    ["yt-dlp", "--dump-json", "--no-download", url_or_path],
                    timeout=30, capture_output=True, text=True,
                )
                if result.returncode == 0:
                    import json
                    info = json.loads(result.stdout)
                    return (
                        f"Titulo: {info.get('title', '?')}\n"
                        f"Duracao: {info.get('duration_string', '?')}\n"
                        f"Canal: {info.get('channel', '?')}\n"
                        f"Views: {info.get('view_count', '?')}\n"
                        f"Upload: {info.get('upload_date', '?')}"
                    )
                return f"Erro: {result.stderr[:200]}"
            except Exception as e:
                return f"Erro: {e}"
        else:
            try:
                result = subprocess.run(
                    [
                        "ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", url_or_path,
                    ],
                    timeout=10, capture_output=True, text=True,
                )
                if result.returncode == 0:
                    import json
                    info = json.loads(result.stdout).get("format", {})
                    duration = float(info.get("duration", 0))
                    mins, secs = divmod(int(duration), 60)
                    return (
                        f"Arquivo: {info.get('filename', '?')}\n"
                        f"Formato: {info.get('format_long_name', '?')}\n"
                        f"Duracao: {mins}m{secs}s\n"
                        f"Tamanho: {int(info.get('size', 0)) / 1024:.0f} KB"
                    )
                return f"Erro ffprobe: {result.stderr[:200]}"
            except Exception as e:
                return f"Erro: {e}"

    def play_media(self, path: str) -> str:
        """Play a media file with mpv (background, non-blocking).

        Args:
            path: Path to media file (audio or video)
        """
        try:
            subprocess.Popen(
                ["mpv", "--no-terminal", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return f"Reproduzindo: {path}"
        except Exception as e:
            return f"Erro: {e}"

    def player_control(self, action: str) -> str:
        """Control media player (play/pause/next/previous/stop).

        Args:
            action: One of "play", "pause", "play-pause", "next", "previous", "stop"
        """
        try:
            result = subprocess.run(
                ["playerctl", action],
                timeout=3, capture_output=True, text=True,
            )
            if result.returncode == 0:
                # Get current status
                status = subprocess.run(
                    ["playerctl", "status"],
                    timeout=3, capture_output=True, text=True,
                ).stdout.strip()
                metadata = subprocess.run(
                    ["playerctl", "metadata", "--format", "{{artist}} - {{title}}"],
                    timeout=3, capture_output=True, text=True,
                ).stdout.strip()
                return f"Player: {status} | {metadata}"
            return f"Erro: {result.stderr.strip()}"
        except Exception as e:
            return f"Erro playerctl: {e}"

    def volume_get(self) -> str:
        """Get current system volume level."""
        try:
            result = subprocess.run(
                ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                timeout=3, capture_output=True, text=True,
            )
            mute = subprocess.run(
                ["pactl", "get-sink-mute", "@DEFAULT_SINK@"],
                timeout=3, capture_output=True, text=True,
            ).stdout.strip()
            return f"Volume: {result.stdout.strip()}\n{mute}"
        except Exception as e:
            return f"Erro: {e}"

    def volume_set(self, level: str) -> str:
        """Set system volume level.

        Args:
            level: Volume level ("50%", "+10%", "-10%", "mute", "unmute", "toggle-mute")
        """
        try:
            if level in ("mute", "unmute"):
                val = "1" if level == "mute" else "0"
                subprocess.run(
                    ["pactl", "set-sink-mute", "@DEFAULT_SINK@", val],
                    timeout=3, check=True,
                )
                return f"Volume: {level}"
            elif level == "toggle-mute":
                subprocess.run(
                    ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"],
                    timeout=3, check=True,
                )
                return "Volume: mute toggled"
            else:
                subprocess.run(
                    ["pactl", "set-sink-volume", "@DEFAULT_SINK@", level],
                    timeout=3, check=True,
                )
                return f"Volume ajustado para {level}"
        except Exception as e:
            return f"Erro: {e}"

    def list_audio_sinks(self) -> str:
        """List available audio output devices (sinks)."""
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sinks"],
                timeout=5, capture_output=True, text=True,
            )
            return result.stdout.strip() or "Nenhum sink encontrado."
        except Exception as e:
            return f"Erro: {e}"

    def tts_speak(self, text: str, lang: str = "pt-BR") -> str:
        """Speak text using espeak-ng (offline TTS, fast).

        Args:
            text: Text to speak
            lang: Language code
        """
        try:
            subprocess.Popen(
                ["espeak-ng", "-v", lang, "-s", "150", text],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return f"Falando: {text[:50]}..."
        except Exception as e:
            return f"Erro espeak: {e}"
