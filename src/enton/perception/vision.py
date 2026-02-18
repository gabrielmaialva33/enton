from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from enton.core.events import (
    ActivityEvent,
    DetectionEvent,
    EmotionEvent,
    EventBus,
    FaceEvent,
    SystemEvent,
)
from enton.perception.activity import classify as classify_activity

if TYPE_CHECKING:
    from enton.core.config import Settings

logger = logging.getLogger(__name__)


class CameraFeed:
    """Single camera capture + per-camera state."""

    __slots__ = (
        "_frame_count",
        "_t_start",
        "_was_connected",
        "cap",
        "fps",
        "id",
        "last_activities",
        "last_detections",
        "last_emotions",
        "last_faces",
        "last_frame",
        "source",
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
                    self.source,
                    cv2.CAP_FFMPEG,
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
        self._target_fps = 30.0  # Default target FPS
        self._det_model = None
        self._pose_model = None

        # Initialize sub-modules
        from enton.perception.emotion import EmotionRecognizer
        from enton.perception.faces import FaceRecognizer

        self._emotion_recognizer = EmotionRecognizer(device=settings.yolo_device, interval_frames=5)
        self.face_recognizer = FaceRecognizer(device=settings.yolo_device)
        self._face_interval = 30  # Run face rec every 30 frames

        # Initialize cameras
        self.cameras: dict[str, CameraFeed] = {}
        self.active_camera_id = "main"

        # Main camera
        src = settings.camera_source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self.cameras["main"] = CameraFeed("main", src)

    def set_target_fps(self, fps: float) -> None:
        """Dynamically adjust target FPS for all cameras."""
        self._target_fps = max(0.1, min(60.0, fps))

    def _ensure_det_model(self):
        if self._det_model is None:
            from ultralytics import YOLO

            self._det_model = YOLO(self._settings.yolo_model)
            self._det_model.to(self._settings.yolo_device)
            logger.info("YOLO detection model loaded: %s", self._settings.yolo_model)
        return self._det_model

    def _ensure_pose_model(self):
        if self._pose_model is None:
            from ultralytics import YOLO

            self._pose_model = YOLO(self._settings.yolo_pose_model)
            self._pose_model.to(self._settings.yolo_pose_device)
            logger.info("YOLO pose model loaded: %s", self._settings.yolo_pose_model)
        return self._pose_model

    def get_frame_jpeg(self) -> bytes | None:
        """Get current frame from active camera as JPEG bytes."""
        cam = self.cameras.get(self.active_camera_id)
        if cam is None or cam.last_frame is None:
            return None
        success, buffer = cv2.imencode(".jpg", cam.last_frame)
        if not success:
            return None
        return buffer.tobytes()

    def switch_camera(self, cam_id: str) -> None:
        if cam_id in self.cameras:
            self.active_camera_id = cam_id
            logger.info("Switched active camera to [%s]", cam_id)
        else:
            logger.warning("Camera [%s] not found", cam_id)

    @property
    def last_detections(self) -> list[DetectionEvent]:
        """Aggregate detections from all cameras."""
        all_dets = []
        for cam in self.cameras.values():
            all_dets.extend(cam.last_detections)
        return all_dets

    @property
    def last_activities(self) -> list[ActivityEvent]:
        all_acts = []
        for cam in self.cameras.values():
            all_acts.extend(cam.last_activities)
        return all_acts

    @property
    def last_emotions(self) -> list[EmotionEvent]:
        all_emos = []
        for cam in self.cameras.values():
            all_emos.extend(cam.last_emotions)
        return all_emos

    async def run(self) -> None:
        """Main vision loop."""
        logger.info("Vision system starting (Target FPS: %.1f)", self._target_fps)

        # Start loops for all cameras
        async with asyncio.TaskGroup() as tg:
            for cam in self.cameras.values():
                tg.create_task(self._camera_loop(cam))
                logger.info("Started camera loop for [%s]", cam.id)

    async def _camera_loop(self, cam: CameraFeed) -> None:
        """Process frames from a single camera."""
        loop = asyncio.get_running_loop()
        frame_count = 0

        while True:
            loop_start = time.monotonic()

            cap = cam.ensure_capture()
            if not cap.isOpened():
                if cam._was_connected:
                    self._bus.emit_nowait(SystemEvent(kind="camera_lost", detail=cam.id))
                    cam._was_connected = False
                await asyncio.sleep(10.0)
                cam.cap = None
                continue

            if not cam._was_connected:
                self._bus.emit_nowait(SystemEvent(kind="camera_connected", detail=cam.id))
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
                    None,
                    self._emotion_recognizer.classify,
                    frame,
                    kpts_list,
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
                        None,
                        fr.identify,
                        frame,
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

            # --- Dynamic FPS Sleep ---
            elapsed = time.monotonic() - loop_start
            target_delay = 1.0 / self._target_fps
            sleep_time = max(0.001, target_delay - elapsed)
            await asyncio.sleep(sleep_time)
