"""Tests for Global Workspace Theory (GWT) implementation."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from enton.core.gwt.message import BroadcastMessage
from enton.core.gwt.module import CognitiveModule
from enton.core.gwt.workspace import GlobalWorkspace

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BroadcastMessage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBroadcastMessage:
    def test_creation(self):
        msg = BroadcastMessage(
            content="hello", source="test", saliency=0.8, modality="text",
        )
        assert msg.content == "hello"
        assert msg.source == "test"
        assert msg.saliency == 0.8
        assert msg.modality == "text"
        assert msg.timestamp > 0
        assert msg.metadata == {}

    def test_str_representation(self):
        msg = BroadcastMessage(
            content="test content", source="vision", saliency=0.75, modality="image",
        )
        s = str(msg)
        assert "vision" in s
        assert "image" in s
        assert "0.75" in s

    def test_long_content_truncated_in_str(self):
        msg = BroadcastMessage(
            content="x" * 200, source="test", saliency=0.5, modality="text",
        )
        s = str(msg)
        assert len(s) < 200  # truncated

    def test_metadata(self):
        msg = BroadcastMessage(
            content="test", source="s", saliency=0.5, modality="m",
            metadata={"key": "value"},
        )
        assert msg.metadata["key"] == "value"

    def test_timestamp_auto_set(self):
        before = time.time()
        msg = BroadcastMessage(
            content="t", source="s", saliency=0.5, modality="m",
        )
        after = time.time()
        assert before <= msg.timestamp <= after


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CognitiveModule (abstract base)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DummyModule(CognitiveModule):
    """Concrete implementation for testing."""

    def __init__(self, name: str, response: BroadcastMessage | None = None):
        super().__init__(name)
        self.response = response
        self.received_context = None

    def run_step(self, context: BroadcastMessage | None) -> BroadcastMessage | None:
        self.received_context = context
        return self.response


class TestCognitiveModule:
    def test_init(self):
        mod = DummyModule("test_mod")
        assert mod.name == "test_mod"

    def test_run_step_returns_none(self):
        mod = DummyModule("test")
        assert mod.run_step(None) is None

    def test_run_step_returns_message(self):
        msg = BroadcastMessage(content="hi", source="test", saliency=0.5, modality="t")
        mod = DummyModule("test", response=msg)
        result = mod.run_step(None)
        assert result is msg

    def test_receives_context(self):
        ctx = BroadcastMessage(content="ctx", source="src", saliency=0.5, modality="m")
        mod = DummyModule("test")
        mod.run_step(ctx)
        assert mod.received_context is ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GlobalWorkspace
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGlobalWorkspace:
    def test_init(self):
        gw = GlobalWorkspace()
        assert gw.modules == []
        assert gw.current_conscious_content is None
        assert gw.history == []
        assert gw.step_counter == 0

    def test_register_module(self):
        gw = GlobalWorkspace()
        mod = DummyModule("test")
        gw.register_module(mod)
        assert len(gw.modules) == 1
        assert gw.modules[0] is mod

    def test_tick_no_modules(self):
        gw = GlobalWorkspace()
        result = gw.tick()
        assert result is None
        assert gw.step_counter == 1

    def test_tick_no_candidates(self):
        gw = GlobalWorkspace()
        gw.register_module(DummyModule("silent"))
        result = gw.tick()
        assert result is None

    def test_tick_single_candidate(self):
        gw = GlobalWorkspace()
        msg = BroadcastMessage(
            content="thought", source="mod1", saliency=0.7, modality="inner_speech",
        )
        gw.register_module(DummyModule("mod1", response=msg))
        result = gw.tick()
        assert result is msg
        assert gw.current_conscious_content is msg
        assert len(gw.history) == 1

    def test_tick_winner_take_all(self):
        gw = GlobalWorkspace()
        low = BroadcastMessage(content="low", source="s", saliency=0.2, modality="m")
        high = BroadcastMessage(content="high", source="s", saliency=0.9, modality="m")
        gw.register_module(DummyModule("low", response=low))
        gw.register_module(DummyModule("high", response=high))
        result = gw.tick()
        assert result is high
        assert gw.current_conscious_content is high

    def test_tick_passes_context_to_modules(self):
        gw = GlobalWorkspace()
        msg = BroadcastMessage(content="first", source="s", saliency=0.5, modality="m")
        mod = DummyModule("mod1", response=msg)
        gw.register_module(mod)

        gw.tick()  # first tick — no context yet
        assert mod.received_context is None

        gw.tick()  # second tick — should pass previous winner
        assert mod.received_context is msg

    def test_step_counter_increments(self):
        gw = GlobalWorkspace()
        gw.register_module(DummyModule("mod"))
        gw.tick()
        gw.tick()
        gw.tick()
        assert gw.step_counter == 3

    def test_history_limit(self):
        gw = GlobalWorkspace()
        msg = BroadcastMessage(content="t", source="s", saliency=0.5, modality="m")
        gw.register_module(DummyModule("mod", response=msg))
        for _ in range(150):
            gw.tick()
        assert len(gw.history) <= 100

    def test_module_error_doesnt_crash(self):
        """Module that raises exception should not crash the workspace."""
        class ErrorModule(CognitiveModule):
            def __init__(self):
                super().__init__("error")

            def run_step(self, context):
                raise RuntimeError("boom")

        gw = GlobalWorkspace()
        gw.register_module(ErrorModule())
        msg = BroadcastMessage(content="ok", source="s", saliency=0.5, modality="m")
        gw.register_module(DummyModule("safe", response=msg))

        result = gw.tick()  # should not raise
        assert result is msg  # safe module still wins

    def test_multiple_ticks_with_competition(self):
        gw = GlobalWorkspace()
        msg1 = BroadcastMessage(content="a", source="s", saliency=0.3, modality="m")
        msg2 = BroadcastMessage(content="b", source="s", saliency=0.8, modality="m")
        gw.register_module(DummyModule("m1", response=msg1))
        gw.register_module(DummyModule("m2", response=msg2))

        for _ in range(10):
            result = gw.tick()
            assert result is msg2  # always wins due to higher saliency


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ExecutiveModule
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestExecutiveModule:
    def test_init(self):
        from enton.cognition.metacognition import MetaCognitiveEngine
        from enton.core.gwt.modules.executive import ExecutiveModule

        engine = MetaCognitiveEngine()
        mod = ExecutiveModule(engine)
        assert mod.name == "executive"
        assert mod.engine is engine

    def test_run_step_no_action(self):
        from enton.cognition.metacognition import MetaCognitiveEngine
        from enton.core.gwt.modules.executive import ExecutiveModule

        engine = MetaCognitiveEngine()
        mod = ExecutiveModule(engine)
        result = mod.run_step(None)
        # Boredom starts at 0, so no action expected
        assert result is None

    def test_run_step_with_high_boredom(self):
        from enton.cognition.metacognition import MetaCognitiveEngine
        from enton.core.gwt.modules.executive import ExecutiveModule

        engine = MetaCognitiveEngine()
        engine.boredom_level = 0.7  # above threshold
        mod = ExecutiveModule(engine)
        result = mod.run_step(None)
        # Should emit feeling_bored
        assert result is not None
        assert result.modality == "emotion"

    def test_with_skill_registry(self):
        from enton.cognition.metacognition import MetaCognitiveEngine
        from enton.core.gwt.modules.executive import ExecutiveModule

        engine = MetaCognitiveEngine()
        registry = MagicMock()
        registry.list_skills.return_value = ["github_learner"]
        mod = ExecutiveModule(engine, skill_registry=registry)
        assert mod.skill_registry is registry


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ShellState
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestShellState:
    def test_init(self):
        from enton.skills._shell_state import ShellState
        state = ShellState()
        assert state.cwd.exists()
        assert state.background == {}

    def test_resolve_absolute_path(self):
        from enton.skills._shell_state import ShellState
        state = ShellState()
        p = state.resolve_path("/tmp/test")
        assert str(p) == "/tmp/test"

    def test_resolve_relative_path(self):
        from pathlib import Path

        from enton.skills._shell_state import ShellState
        state = ShellState(cwd=Path("/tmp"))
        p = state.resolve_path("subdir/file.txt")
        assert str(p) == "/tmp/subdir/file.txt"

    def test_resolve_home_path(self):
        from enton.skills._shell_state import ShellState
        state = ShellState()
        p = state.resolve_path("~/test")
        assert "~" not in str(p)  # expanded


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Provider protocols
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestProviderProtocols:
    def test_stt_protocol(self):
        from enton.providers.base import STTProvider
        assert hasattr(STTProvider, "transcribe")
        assert hasattr(STTProvider, "stream")

    def test_tts_protocol(self):
        from enton.providers.base import TTSProvider
        assert hasattr(TTSProvider, "synthesize")
        assert hasattr(TTSProvider, "synthesize_stream")

    def test_stt_is_runtime_checkable(self):
        import numpy as np

        from enton.providers.base import STTProvider

        class FakeSTT:
            async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
                return "hello"
            async def stream(self):
                yield ""

        assert isinstance(FakeSTT(), STTProvider)

    def test_tts_is_runtime_checkable(self):
        import numpy as np

        from enton.providers.base import TTSProvider

        class FakeTTS:
            sample_rate = 16000
            async def synthesize(self, text: str) -> np.ndarray:
                return np.zeros(100)
            async def synthesize_stream(self, text: str):
                yield np.zeros(100)

        assert isinstance(FakeTTS(), TTSProvider)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ChannelMessage & BaseChannel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestChannelBase:
    def test_message_type_enum(self):
        from enton.channels.base import MessageType
        assert MessageType.TEXT == "text"
        assert MessageType.IMAGE == "image"
        assert MessageType.AUDIO == "audio"
        assert MessageType.VIDEO == "video"
        assert MessageType.FILE == "file"
        assert MessageType.COMMAND == "command"
        assert MessageType.REACTION == "reaction"

    def test_channel_message_creation(self):
        from enton.channels.base import ChannelMessage, MessageType
        msg = ChannelMessage(
            channel="telegram",
            sender_id="123",
            sender_name="Gabriel",
            text="Opa mano",
        )
        assert msg.channel == "telegram"
        assert msg.sender_id == "123"
        assert msg.sender_name == "Gabriel"
        assert msg.text == "Opa mano"
        assert msg.message_type == MessageType.TEXT
        assert msg.media is None
        assert not msg.has_media

    def test_channel_message_with_media(self):
        from enton.channels.base import ChannelMessage
        msg = ChannelMessage(
            channel="discord",
            sender_id="456",
            sender_name="Test",
            media=b"image_data",
        )
        assert msg.has_media
        assert msg.media == b"image_data"

    def test_channel_message_with_media_url(self):
        from enton.channels.base import ChannelMessage
        msg = ChannelMessage(
            channel="web",
            sender_id="789",
            sender_name="User",
            media_url="https://example.com/img.png",
        )
        assert msg.has_media

    def test_channel_message_timestamp(self):
        from enton.channels.base import ChannelMessage
        before = time.time()
        msg = ChannelMessage(
            channel="test", sender_id="1", sender_name="t",
        )
        after = time.time()
        assert before <= msg.timestamp <= after

    def test_base_channel_is_abstract(self):
        from enton.channels.base import BaseChannel
        with pytest.raises(TypeError):
            BaseChannel(bus=MagicMock())  # can't instantiate abstract


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CudaLock
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCudaLock:
    def test_thread_lock_exists(self):
        import threading

        from enton.core.cuda_lock import cuda_thread_lock
        assert isinstance(cuda_thread_lock, type(threading.Lock()))

    def test_async_lock_exists(self):
        import asyncio

        from enton.core.cuda_lock import cuda_async_lock
        assert isinstance(cuda_async_lock, asyncio.Lock)

    def test_thread_lock_acquire_release(self):
        from enton.core.cuda_lock import cuda_thread_lock
        acquired = cuda_thread_lock.acquire(timeout=1)
        assert acquired
        cuda_thread_lock.release()
