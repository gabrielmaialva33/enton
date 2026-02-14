from __future__ import annotations

import asyncio
import threading

# Global lock to serialize CUDA operations across threads.
# Prevents heap corruption from concurrent PyTorch CUDA access
# (YOLO inference + Kokoro TTS synthesis).
cuda_thread_lock = threading.Lock()
cuda_async_lock = asyncio.Lock()
