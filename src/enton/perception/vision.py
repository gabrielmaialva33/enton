from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np  # noqa: TC002 — used at runtime

from enton.core.events import (
    ActivityEvent,
    DetectionEvent,
    EmotionEvent,
    EventBus,
    FaceEvent,
    SystemEvent,
)
from enton.perception.activity import classify as classify_activity
from enton.perception.emotion import EmotionRecognizer

if TYPE_CHECKING:
    from enton.core.config import Settings

logger = logging.getLogger(__name__)


class CameraFeed:
    """Single camera capture + per-camera state."""

    __slots__ = (
        "id", "source", "cap", "last_frame",
        "last_detections", "last_activities", "last_emotions", "last_faces",
        "fps", "_frame_count", "_t_start", "_was_connected",
    )

    def __init__(self, cam_id: str, source: str | int) -> None:
        self.id = cam_id
        self.source = source
        self.cap: cv2.VideoCapture | None = None
        self.last_frame: np.ndarray | None = None
        self.last_detections: list[DetectionEvent] = []
        self.last_activities: list[ActivityEvent] = []
        self.last_emotions: list[EmotionEvent] = []
        self.last_faces: list[FaceEvent] = []
        self.fps: float = 0.0
        self._frame_count = 0
        self._t_start = time.monotonic()
        self._was_connected = False

    def ensure_capture(self) -> cv2.VideoCapture:
        if self.cap is None or not self.cap.isOpened():
            if isinstance(self.source, int):
                self.cap = cv2.VideoCapture(self.source)
            else:
                self.cap = cv2.VideoCapture(
                    self.source, cv2.CAP_FFMPEG,
                    [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000],
                )
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.cap.isOpened():
                logger.info("Camera [%s] connected: %s", self.id, self.source)
            else:
                logger.error("Camera [%s] failed: %s", self.id, self.source)
        return self.cap

    def update_fps(self) -> None:
        self._frame_count += 1
        elapsed = time.monotonic() - self._t_start
        if elapsed >= 1.0:
            self.fps = self._frame_count / elapsed
            self._frame_count = 0
            self._t_start = time.monotonic()


class Vision:
    def __init__(self, settings: Settings, bus: EventBus) -> None:
        self._settings = settings
        self._bus = bus
        self._det_model = None
        self._pose_model = None
        self._emotion_recognizer = EmotionRecognizer(
            device=settings.yolo_device, interval_frames=5,
        )
        self._face_recognizer = None
        self._face_interval = 10

        # Multi-camera setup
        self._cameras: dict[str, CameraFeed] = {}
        for cam_id, source in settings.camera_sources.items():
            self._cameras[cam_id] = CameraFeed(cam_id, source)
        self._active_camera = next(iter(self._cameras), "main")

    def _ensure_det_model(self):
        if self._det_model is None:
            from ultralytics import YOLO

            self._det_model = YOLO(str(self._settings.yolo_model_path))
            self._det_model.to(self._settings.yolo_device)
            logger.info(
                "YOLO detect loaded: %s on %s",
                self._settings.yolo_model, self._settings.yolo_device,
            )
        return self._det_model

    def _ensure_pose_model(self):
        if self._pose_model is None:
            from ultralytics import YOLO

            self._pose_model = YOLO(self._settings.yolo_pose_model)
            self._pose_model.to(self._settings.yolo_pose_device)
            logger.info(
                "YOLO pose loaded: %s on %s",
                self._settings.yolo_pose_model, self._settings.yolo_device,
            )
        return self._pose_model

    # Convenience properties for active camera (backward compat)

    @property
    def last_frame(self) -> np.ndarray | None:
        cam = self._cameras.get(self._active_camera)
        return cam.last_frame if cam else None

    @property
    def last_detections(self) -> list[DetectionEvent]:
        cam = self._cameras.get(self._active_camera)
        return cam.last_detections if cam else []

    @property
    def last_activities(self) -> list[ActivityEvent]:
        cam = self._cameras.get(self._active_camera)
        return cam.last_activities if cam else []

    @property
    def last_emotions(self) -> list[EmotionEvent]:
        cam = self._cameras.get(self._active_camera)
        return cam.last_emotions if cam else []

    @property
    def last_faces(self) -> list[FaceEvent]:
        cam = self._cameras.get(self._active_camera)
        return cam.last_faces if cam else []

    @property
    def face_recognizer(self):
        if self._face_recognizer is None:
            try:
                from enton.perception.faces import FaceRecognizer

                self._face_recognizer = FaceRecognizer(
                    device=self._settings.yolo_device,
                )
                logger.info("FaceRecognizer loaded")
            except Exception:
                logger.warning("FaceRecognizer unavailable")
        return self._face_recognizer

    @property
    def fps(self) -> float:
        cam = self._cameras.get(self._active_camera)
        return cam.fps if cam else 0.0

    @property
    def cameras(self) -> dict[str, CameraFeed]:
        return self._cameras

    @property
    def active_camera_id(self) -> str:
        return self._active_camera

    def switch_camera(self, cam_id: str) -> bool:
        if cam_id in self._cameras:
            self._active_camera = cam_id
            return True
        return False

    def get_frame_jpeg(self, camera_id: str | None = None) -> bytes | None:
        cam_id = camera_id or self._active_camera
        cam = self._cameras.get(cam_id)
        if cam is None or cam.last_frame is None:
            return None
        _, buf = cv2.imencode(".jpg", cam.last_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()

    async def run(self) -> None:
        if len(self._cameras) == 1:
            # Single camera — direct loop (no TaskGroup overhead)
            cam = next(iter(self._cameras.values()))
            await self._camera_loop(cam)
        else:
            # Multi-camera — parallel processing
            async with asyncio.TaskGroup() as tg:
                for cam in self._cameras.values():
                    tg.create_task(
                        self._camera_loop(cam),
                        name=f"cam_{cam.id}",
                    )

    async def _camera_loop(self, cam: CameraFeed) -> None:
        """Process frames from a single camera."""
        loop = asyncio.get_running_loop()
        frame_count = 0

        while True:
            cap = cam.ensure_capture()
            if not cap.isOpened():
                if cam._was_connected:
                    self._bus.emit_nowait(
                        SystemEvent(kind="camera_lost", detail=cam.id)
                    )
                    cam._was_connected = False
                await asyncio.sleep(10.0)
                cam.cap = None
                continue

            if not cam._was_connected:
                self._bus.emit_nowait(
                    SystemEvent(kind="camera_connected", detail=cam.id)
                )
                cam._was_connected = True

            try:
                det_model = self._ensure_det_model()
                pose_model = self._ensure_pose_model()
            except Exception:
                logger.exception("YOLO model load failed, retrying in 30s")
                await asyncio.sleep(30.0)
                continue

            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret:
                logger.warning("Frame read failed [%s], reconnecting...", cam.id)
                cam.cap = None
                await asyncio.sleep(1.0)
                continue

            cam.last_frame = frame
            frame_count += 1
            cam.update_fps()

            det_conf = self._settings.yolo_confidence
            pose_conf = self._settings.yolo_pose_confidence

            def _predict(f=frame, dc=det_conf, pc=pose_conf, dm=det_model, pm=pose_model):
                det_r = dm.predict(f, conf=dc, half=True, verbose=False)
                pose_r = pm.predict(f, conf=pc, half=True, verbose=False)
                return det_r, pose_r

            det_results, pose_results = await loop.run_in_executor(None, _predict)

            # --- object detections ---
            detections = []
            for r in det_results:
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
                        camera_id=cam.id,
                    )
                    detections.append(det)
            cam.last_detections = detections

            # --- activity recognition ---
            activities = []
            for r in pose_results:
                if r.keypoints is not None and len(r.keypoints) > 0:
                    for i, kpts in enumerate(r.keypoints.data):
                        activity_label, color = classify_activity(kpts)
                        act = ActivityEvent(
                            person_index=i,
                            activity=activity_label,
                            color=color,
                            camera_id=cam.id,
                        )
                        activities.append(act)
            cam.last_activities = activities

            # --- emotion recognition ---
            emotions = []
            kpts_list = []
            for r in pose_results:
                if r.keypoints is not None and len(r.keypoints) > 0:
                    kpts_list.extend(r.keypoints.data)
            if kpts_list:
                face_emotions = await loop.run_in_executor(
                    None, self._emotion_recognizer.classify, frame, kpts_list,
                )
                for i, fe in enumerate(face_emotions):
                    emo = EmotionEvent(
                        person_index=i,
                        emotion=fe.label,
                        emotion_en=fe.label_en,
                        score=fe.score,
                        color=fe.color,
                        bbox=fe.bbox,
                        camera_id=cam.id,
                    )
                    emotions.append(emo)
            cam.last_emotions = emotions

            # --- face recognition (every N frames, only if persons detected) ---
            faces = []
            has_person = any(d.label == "person" for d in detections)
            if has_person and frame_count % self._face_interval == 0:
                fr = self.face_recognizer
                if fr is not None:
                    face_results = await loop.run_in_executor(
                        None, fr.identify, frame,
                    )
                    for f_res in face_results:
                        faces.append(
                            FaceEvent(
                                identity=f_res.identity,
                                confidence=f_res.confidence,
                                bbox=f_res.bbox,
                                camera_id=cam.id,
                            )
                        )
            cam.last_faces = faces

            # --- emit events ---
            for det in detections:
                self._bus.emit_nowait(det)
            for act in activities:
                self._bus.emit_nowait(act)
            for emo in emotions:
                self._bus.emit_nowait(emo)
            for face in faces:
                self._bus.emit_nowait(face)

            await asyncio.sleep(0.01)
