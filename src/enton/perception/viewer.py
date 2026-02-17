from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from enton.perception.vision import Vision, CameraFeed

logger = logging.getLogger(__name__)


class Viewer:
    """Manages the visual interface (HUD) for Enton's perception system."""

    def __init__(self, vision: Vision, thoughts: deque[str]) -> None:
        self.vision = vision
        self.thoughts = thoughts
        self.running: bool = False
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_sm = cv2.FONT_HERSHEY_PLAIN
        self._scan_y = 0
        self._fullscreen = False
        self._grid_mode = len(vision.cameras) > 1

    async def run(self) -> None:
        """Live vision window — cv2-only HUD with multi-camera grid support."""
        cv2.namedWindow("Enton Vision", cv2.WINDOW_NORMAL)
        cam_ids = list(self.vision.cameras.keys())
        logger.info(
            "Viewer opened (%d camera%s)",
            len(cam_ids), "s" if len(cam_ids) != 1 else "",
        )
        self.running = True

        try:
            while self.running:
                # Update camera list dynamically if needed, though usually static
                cam_ids = list(self.vision.cameras.keys())
                
                if self._grid_mode and len(cam_ids) > 1:
                    annotated = self._build_grid(cam_ids)
                else:
                    active_cam = self.vision.cameras.get(
                        self.vision.active_camera_id, 
                        next(iter(self.vision.cameras.values()), None)
                    )
                    if active_cam:
                        annotated = self._annotate_camera(active_cam)
                    else:
                        annotated = None

                if annotated is None:
                    await asyncio.sleep(0.1)
                    continue

                fh, fw = annotated.shape[:2]

                # Scan line effect
                self._scan_y = (self._scan_y + 3) % fh
                cv2.line(annotated, (0, self._scan_y), (fw, self._scan_y), (0, 255, 120), 1)

                # Thoughts panel (bottom left)
                if self.thoughts:
                    y_base = fh - 30
                    # Display last 6 thoughts
                    visible_thoughts = list(self.thoughts)[-6:]
                    for thought in reversed(visible_thoughts):
                        pt = (12, y_base)
                        # Shadow
                        cv2.putText(annotated, thought, pt, self._font, 0.45, (0, 0, 0), 3)
                        # Text
                        cv2.putText(annotated, thought, pt, self._font, 0.45, (0, 255, 200), 1)
                        y_base -= 20

                cv2.imshow("Enton Vision", annotated)
                
                # Handle key inputs
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.running = False
                    break
                elif key == ord("f"):
                    self._fullscreen = not self._fullscreen
                    prop = cv2.WINDOW_FULLSCREEN if self._fullscreen else cv2.WINDOW_NORMAL
                    cv2.setWindowProperty("Enton Vision", cv2.WND_PROP_FULLSCREEN, prop)
                elif key == ord("g"):
                    self._grid_mode = not self._grid_mode
                elif key == ord("c"):
                    # Cycle active camera
                    if cam_ids:
                        try:
                            curr_idx = cam_ids.index(self.vision.active_camera_id)
                            next_idx = (curr_idx + 1) % len(cam_ids)
                            nxt = cam_ids[next_idx]
                            self.vision.switch_camera(nxt)
                        except ValueError:
                            # Current camera ID not found (maybe disconnected), reset to 0
                            if cam_ids:
                                self.vision.switch_camera(cam_ids[0])
                elif ord("1") <= key <= ord("9"):
                    idx = key - ord("1")
                    if idx < len(cam_ids):
                        self.vision.switch_camera(cam_ids[idx])

                await asyncio.sleep(0.01)

        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            logger.info("Viewer closed")

    def _annotate_camera(self, cam: CameraFeed) -> np.ndarray | None:
        """Draw HUD overlay on a single camera frame."""
        frame = cam.last_frame
        if frame is None:
            return None

        # Create a writable copy
        annotated = frame.copy()
        
        # Detection overlays
        for det in cam.last_detections:
            if det.bbox:
                color = (0, 255, 120) if det.label == "person" else (255, 160, 0)
                if det.label in ("cat", "dog"):
                    color = (0, 200, 255)
                
                x1, y1, x2, y2 = det.bbox
                bw, bh = x2 - x1, y2 - y1
                
                # Draw corner brackets instead of full box for cleaner look
                c = max(15, min(bw, bh) // 5)
                cv2.line(annotated, (x1, y1), (x1 + c, y1), color, 2)
                cv2.line(annotated, (x1, y1), (x1, y1 + c), color, 2)
                cv2.line(annotated, (x2, y1), (x2 - c, y1), color, 2)
                cv2.line(annotated, (x2, y1), (x2, y1 + c), color, 2)
                cv2.line(annotated, (x1, y2), (x1 + c, y2), color, 2)
                cv2.line(annotated, (x1, y2), (x1, y2 - c), color, 2)
                cv2.line(annotated, (x2, y2), (x2 - c, y2), color, 2)
                cv2.line(annotated, (x2, y2), (x2, y2 - c), color, 2)
                
                lbl = f"{det.label} {det.confidence:.0%}"
                pt = (x1, y1 - 6)
                cv2.putText(annotated, lbl, pt, self._font_sm, 1.0, (0, 0, 0), 3)
                cv2.putText(annotated, lbl, pt, self._font_sm, 1.0, color, 1)

        # Emotion labels
        for emo in cam.last_emotions:
            if emo.bbox and emo.bbox != (0, 0, 0, 0):
                x1, _, x2, y2 = emo.bbox
                lbl = f"{emo.emotion} {emo.score:.0%}"
                cx = (x1 + x2) // 2
                pt = (cx - 40, y2 + 16)
                cv2.putText(annotated, lbl, pt, self._font_sm, 1.0, (0, 0, 0), 3)
                cv2.putText(annotated, lbl, pt, self._font_sm, 1.0, emo.color, 1)

        # HUD panel (top left)
        n_persons = sum(1 for d in cam.last_detections if d.label == "person")
        n_obj = len(cam.last_detections) - n_persons
        
        # Semi-transparent background for HUD
        overlay = annotated.copy()
        cv2.rectangle(overlay, (8, 8), (260, 80), (10, 12, 18), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        cv2.rectangle(annotated, (8, 8), (260, 80), (0, 255, 120), 1)
        
        title = f"ENTON [{cam.id}]" if len(self.vision.cameras) > 1 else "ENTON"
        cv2.putText(annotated, title, (16, 34), self._font, 0.6, (0, 255, 120), 2)
        
        fps_txt = f"{cam.fps:.0f} fps"
        cv2.putText(annotated, fps_txt, (200, 34), self._font_sm, 1.0, (80, 90, 100), 1)
        
        if n_persons:
            status = f"{n_persons} pessoa{'s' if n_persons != 1 else ''}"
        else:
            status = "scanning..."
        if n_obj > 0:
            status += f"  {n_obj} obj"
            
        cv2.putText(annotated, status, (16, 58), self._font, 0.45, (0, 210, 230), 1)
        
        # Activity log in HUD
        for i, act in enumerate(cam.last_activities[:2]):
            pt = (16, 74 + i * 14)
            cv2.putText(annotated, act.activity, pt, self._font_sm, 0.9, act.color, 1)

        return annotated

    def _build_grid(self, cam_ids: list[str]) -> np.ndarray | None:
        """Build grid view from multiple cameras."""
        n = len(cam_ids)
        if n == 0:
            return None
            
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        # Target tile size
        tile_w, tile_h = 640, 480
        grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

        has_frame = False
        for idx, cam_id in enumerate(cam_ids):
            cam = self.vision.cameras.get(cam_id)
            if cam is None:
                continue
                
            annotated = self._annotate_camera(cam)
            if annotated is None:
                continue
                
            has_frame = True
            tile = cv2.resize(annotated, (tile_w, tile_h))
            r, c = divmod(idx, cols)
            y0, x0 = r * tile_h, c * tile_w
            grid[y0:y0 + tile_h, x0:x0 + tile_w] = tile

        return grid if has_frame else None
