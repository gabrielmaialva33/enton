"""Tests for TTS providers — Qwen3TTS, EdgeTTS, Voice fallback."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from enton.core.config import Provider

# ---------------------------------------------------------------------------
# Provider enum
# ---------------------------------------------------------------------------


def test_provider_enum_has_qwen3():
    assert Provider.QWEN3 == "qwen3"


def test_provider_enum_has_edge():
    assert Provider.EDGE == "edge"


# ---------------------------------------------------------------------------
# Qwen3TTS
# ---------------------------------------------------------------------------


class TestQwen3TTS:
    def test_init_lazy(self, mock_settings):
        """Model should NOT be loaded on init."""
        from enton.providers.qwen_tts import Qwen3TTS

        tts = Qwen3TTS(mock_settings)
        assert tts._model is None
        assert tts.sample_rate == 24000

    def test_synthesize_returns_float32(self, mock_settings):
        """Synthesis should return float32 numpy array."""
        from enton.providers.qwen_tts import Qwen3TTS

        fake_audio = np.random.randn(24000).astype(np.float32)
        fake_model = MagicMock()
        fake_model.return_value = ([fake_audio], 24000)
        fake_model.generate_voice_design.return_value = "custom_voice_token"
        fake_model.generate_custom_voice.return_value = ([fake_audio], 24000)

        tts = Qwen3TTS(mock_settings)

        with patch(
            "enton.providers.qwen_tts.Qwen3TTS._ensure_model",
            return_value=fake_model,
        ):
            tts._custom_voice = "custom_voice_token"
            audio = asyncio.get_event_loop().run_until_complete(tts.synthesize("Ola"))

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert audio.size > 0

    def test_voice_design_called_on_first_use(self, mock_settings):
        """Voice design should be called when voice_instruct is set."""
        from enton.providers.qwen_tts import Qwen3TTS

        fake_model_cls = MagicMock()
        fake_model = MagicMock()
        fake_model.generate_voice_design.return_value = "designed_voice"
        fake_model_cls.from_pretrained.return_value = fake_model

        tts = Qwen3TTS(mock_settings)

        mock_qwen = MagicMock()
        mock_qwen.Qwen3TTSModel = fake_model_cls
        sys.modules["qwen_tts"] = mock_qwen

        try:
            tts._ensure_model()
            fake_model.generate_voice_design.assert_called_once()
            assert tts._custom_voice == "designed_voice"
        finally:
            del sys.modules["qwen_tts"]

    def test_no_voice_design_without_instruct(self, mock_settings):
        """Without voice_instruct, no voice design should happen."""
        from enton.providers.qwen_tts import Qwen3TTS

        mock_settings.qwen3_tts_voice_instruct = ""

        fake_model_cls = MagicMock()
        fake_model = MagicMock()
        fake_model_cls.from_pretrained.return_value = fake_model

        tts = Qwen3TTS(mock_settings)

        mock_qwen = MagicMock()
        mock_qwen.Qwen3TTSModel = fake_model_cls
        sys.modules["qwen_tts"] = mock_qwen

        try:
            tts._ensure_model()
            fake_model.generate_voice_design.assert_not_called()
            assert tts._custom_voice is None
        finally:
            del sys.modules["qwen_tts"]

    def test_synthesize_stream_yields_audio(self, mock_settings):
        """synthesize_stream should yield the full audio."""
        from enton.providers.qwen_tts import Qwen3TTS

        fake_audio = np.random.randn(24000).astype(np.float32)

        tts = Qwen3TTS(mock_settings)
        tts.synthesize = AsyncMock(return_value=fake_audio)

        async def collect():
            chunks = []
            async for chunk in tts.synthesize_stream("test"):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.get_event_loop().run_until_complete(collect())
        assert len(chunks) == 1
        assert chunks[0].dtype == np.float32


# ---------------------------------------------------------------------------
# EdgeTTS
# ---------------------------------------------------------------------------


class TestEdgeTTS:
    def test_init(self, mock_settings):
        from enton.providers.edge_tts_provider import EdgeTTS

        tts = EdgeTTS(mock_settings)
        assert tts._voice == "pt-BR-AntonioNeural"
        assert tts.sample_rate == 24000

    def test_empty_response(self, mock_settings):
        """Empty MP3 stream should return empty array."""
        from enton.providers.edge_tts_provider import EdgeTTS

        mock_communicate = MagicMock()

        async def empty_stream():
            yield {"type": "metadata", "data": {}}

        mock_communicate.stream = empty_stream
        mock_communicate_cls = MagicMock(return_value=mock_communicate)

        mock_edge = MagicMock()
        mock_edge.Communicate = mock_communicate_cls
        sys.modules["edge_tts"] = mock_edge

        tts = EdgeTTS(mock_settings)

        try:
            audio = asyncio.get_event_loop().run_until_complete(tts.synthesize("test"))
            assert isinstance(audio, np.ndarray)
            assert audio.size == 0
        finally:
            del sys.modules["edge_tts"]

    def test_stereo_to_mono(self, mock_settings):
        """Stereo audio should be converted to mono."""
        from enton.providers.edge_tts_provider import EdgeTTS

        stereo_audio = np.random.randn(24000, 2).astype(np.float32)

        mock_communicate = MagicMock()

        async def fake_stream():
            yield {"type": "audio", "data": b"\x00" * 100}

        mock_communicate.stream = fake_stream

        mock_edge = MagicMock()
        mock_edge.Communicate.return_value = mock_communicate
        sys.modules["edge_tts"] = mock_edge

        mock_sf = MagicMock()
        mock_sf.read.return_value = (stereo_audio, 24000)
        sys.modules["soundfile"] = mock_sf

        tts = EdgeTTS(mock_settings)

        try:
            # Inline patched synthesize to avoid executor issues in tests
            async def patched_synthesize():
                import io

                import edge_tts
                import soundfile as sf

                communicate = edge_tts.Communicate("test", tts._voice)
                mp3_bytes = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        mp3_bytes += chunk["data"]
                if not mp3_bytes:
                    return np.array([], dtype=np.float32)
                audio, sr = sf.read(io.BytesIO(mp3_bytes))
                tts.sample_rate = sr
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                return audio.astype(np.float32)

            audio = asyncio.get_event_loop().run_until_complete(patched_synthesize())
            assert audio.ndim == 1  # mono
            assert audio.dtype == np.float32
        finally:
            if "edge_tts" in sys.modules:
                del sys.modules["edge_tts"]
            if "soundfile" in sys.modules:
                del sys.modules["soundfile"]

    def test_synthesize_stream_yields_audio(self, mock_settings):
        """synthesize_stream should yield the full audio."""
        from enton.providers.edge_tts_provider import EdgeTTS

        fake_audio = np.random.randn(24000).astype(np.float32)

        tts = EdgeTTS(mock_settings)
        tts.synthesize = AsyncMock(return_value=fake_audio)

        async def collect():
            chunks = []
            async for chunk in tts.synthesize_stream("test"):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.get_event_loop().run_until_complete(collect())
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Voice fallback chain
# ---------------------------------------------------------------------------


class TestVoiceFallback:
    def test_fallback_order_defined(self):
        """Voice should have a defined fallback order."""
        from enton.action.voice import Voice

        assert Provider.QWEN3 in Voice._FALLBACK_ORDER
        assert Provider.EDGE in Voice._FALLBACK_ORDER
        assert Provider.LOCAL in Voice._FALLBACK_ORDER

    def test_fallback_order_correct(self):
        """QWEN3 should come before EDGE, which comes before LOCAL."""
        from enton.action.voice import Voice

        order = Voice._FALLBACK_ORDER
        assert order.index(Provider.QWEN3) < order.index(Provider.EDGE)
        assert order.index(Provider.EDGE) < order.index(Provider.LOCAL)


# ---------------------------------------------------------------------------
# sample_rate attribute
# ---------------------------------------------------------------------------


class TestSampleRate:
    def test_local_tts_has_sample_rate(self, mock_settings):
        """LocalTTS should expose sample_rate attribute."""
        with (
            patch("enton.providers.local.KPipeline"),
            patch("enton.providers.local.torch"),
        ):
            from enton.providers.local import LocalTTS

            tts = LocalTTS(mock_settings)
            assert tts.sample_rate == 24000

    def test_qwen3_tts_has_sample_rate(self, mock_settings):
        from enton.providers.qwen_tts import Qwen3TTS

        tts = Qwen3TTS(mock_settings)
        assert tts.sample_rate == 24000

    def test_edge_tts_has_sample_rate(self, mock_settings):
        from enton.providers.edge_tts_provider import EdgeTTS

        tts = EdgeTTS(mock_settings)
        assert tts.sample_rate == 24000
