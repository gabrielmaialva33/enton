from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np  # noqa: TC002 — used at runtime

from enton.events import DetectionEvent, EventBus, SystemEvent

if TYPE_CHECKING:
    from enton.config import Settings

logger = logging.getLogger(__name__)


class Vision:
    def __init__(self, settings: Settings, bus: EventBus) -> None:
        self._settings = settings
        self._bus = bus
        self._model = None
        self._cap: cv2.VideoCapture | None = None
        self._last_frame: np.ndarray | None = None
        self._last_detections: list[DetectionEvent] = []
        self._fps: float = 0.0

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self._settings.yolo_model_path))
            self._model.to(self._settings.yolo_device)
            logger.info(
                "YOLO loaded: %s on %s", self._settings.yolo_model, self._settings.yolo_device
            )
        return self._model

    def _ensure_camera(self) -> cv2.VideoCapture:
        if self._cap is None or not self._cap.isOpened():
            source = self._settings.camera_url
            if isinstance(source, int):
                self._cap = cv2.VideoCapture(source)
            else:
                self._cap = cv2.VideoCapture(
                    source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000]
                )
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self._cap.isOpened():
                logger.info("Camera connected: %s", source)
            else:
                logger.error("Camera failed: %s", source)
        return self._cap

    @property
    def last_frame(self) -> np.ndarray | None:
        return self._last_frame

    @property
    def last_detections(self) -> list[DetectionEvent]:
        return self._last_detections

    @property
    def fps(self) -> float:
        return self._fps

    def get_frame_jpeg(self) -> bytes | None:
        if self._last_frame is None:
            return None
        _, buf = cv2.imencode(".jpg", self._last_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        frame_count = 0
        t_start = time.monotonic()
        was_connected = False

        while True:
            cap = self._ensure_camera()
            if not cap.isOpened():
                if was_connected:
                    self._bus.emit_nowait(SystemEvent(kind="camera_lost"))
                    was_connected = False
                await asyncio.sleep(10.0)
                self._cap = None
                continue

            if not was_connected:
                self._bus.emit_nowait(SystemEvent(kind="camera_connected"))
                was_connected = True

            try:
                model = self._ensure_model()
            except Exception:
                logger.exception("YOLO model load failed, retrying in 30s")
                await asyncio.sleep(30.0)
                continue

            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret:
                logger.warning("Frame read failed, reconnecting...")
                self._cap = None
                await asyncio.sleep(1.0)
                continue

            self._last_frame = frame
            frame_count += 1

            conf = self._settings.yolo_confidence

            def _predict(f=frame, c=conf, m=model):
                return m.predict(f, conf=c, verbose=False)

            results = await loop.run_in_executor(None, _predict)

            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = r.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    det = DetectionEvent(
                        label=label,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        frame_shape=(frame.shape[0], frame.shape[1]),
                    )
                    detections.append(det)

            self._last_detections = detections

            for det in detections:
                self._bus.emit_nowait(det)

            elapsed = time.monotonic() - t_start
            if elapsed >= 1.0:
                self._fps = frame_count / elapsed
                frame_count = 0
                t_start = time.monotonic()

            await asyncio.sleep(0.01)
