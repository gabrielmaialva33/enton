"""Enton Vision — Cyberpunk AI Camera.

YOLO11x detection + pose + activity recognition + neon glow HUD.

Usage:
    python scripts/live_yolo.py          # câmera IP (RTSP)
    python scripts/live_yolo.py --webcam  # webcam local

Controls:
    q — quit
    f — toggle fullscreen
"""
import os, sys, time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from enton.activity import NOSE, classify as classify_activity, _visible as visible
from enton.overlay import Overlay

RTSP_URL = "rtsp://192.168.18.23:554/video0_unicast"
DET_MODEL = "yolo11x.pt"
POSE_MODEL = "yolo11x-pose.pt"
DET_CONF = 0.15
POSE_CONF = 0.2
IMGSZ = 640

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

use_webcam = "--webcam" in sys.argv
source = 0 if use_webcam else RTSP_URL

print("Carregando modelos na GPU (FP16)...")
det_model = YOLO(DET_MODEL)
det_model.to("cuda:0")
pose_model = YOLO(POSE_MODEL)
pose_model.to("cuda:0")

print(f"Conectando: {'webcam' if use_webcam else RTSP_URL}")
cap = cv2.VideoCapture(source)
if use_webcam:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
else:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not cap.isOpened():
    print("Falha ao abrir câmera!")
    sys.exit(1)

cv2.namedWindow("Enton Vision", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Enton Vision", w, h)
print(f"Enton Vision Cyberpunk | {w}x{h} FP16 CUDA | 'q'=sair 'f'=fullscreen")

hud = Overlay(font_size=18)

fps_t = time.time()
fps_count = 0
fps = 0.0
fullscreen = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame perdido, reconectando...")
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(source)
        if not use_webcam:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        continue

    det_r = det_model.predict(frame, conf=DET_CONF, imgsz=IMGSZ, half=True, verbose=False)
    pose_r = pose_model.predict(frame, conf=POSE_CONF, imgsz=IMGSZ, half=True, verbose=False)

    # Use raw frame (no ultralytics plot — we draw our own)
    annotated = frame.copy()

    # Scan line effect
    annotated = hud.draw_scan_line(annotated)

    # Detection boxes — targeting brackets + confidence badges
    boxes = det_r[0].boxes
    names = det_r[0].names
    det_summary: dict[str, int] = {}
    for box in boxes:
        cls_id = int(box.cls[0])
        name = names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        det_summary[name] = det_summary.get(name, 0) + 1

        # Color by class
        if name == "person":
            bcolor = (0, 255, 120)
        elif name in ("cat", "dog"):
            bcolor = (0, 200, 255)
        else:
            bcolor = (255, 160, 0)

        annotated = hud.draw_target_brackets(annotated, (x1, y1, x2, y2), bcolor)
        annotated = hud.draw_confidence_badge(annotated, name, conf, (x1, y1, x2, y2))

    # Pose — neon glow skeleton + activity labels
    activities_hud: list[tuple[str, tuple[int, int, int]]] = []
    n_persons = 0
    if pose_r[0].keypoints is not None and len(pose_r[0].keypoints) > 0:
        kpts_data = pose_r[0].keypoints.data
        n_persons = len(kpts_data)
        for kpts in kpts_data:
            activity, color = classify_activity(kpts)
            activities_hud.append((activity, color))

            # Glow skeleton
            annotated = hud.draw_glow_skeleton(annotated, kpts, SKELETON, color, visible)

            # Activity label
            if visible(kpts, NOSE):
                nx, ny = int(kpts[NOSE][0]), int(kpts[NOSE][1])
                annotated = hud.draw_activity_label(annotated, activity, (nx, ny), color)

    # FPS
    fps_count += 1
    now = time.time()
    if now - fps_t >= 1.0:
        fps = fps_count / (now - fps_t)
        fps_count = 0
        fps_t = now

    # HUD panel with graphs
    annotated = hud.draw_hud(
        annotated,
        fps=fps,
        n_objects=len(boxes),
        n_persons=n_persons,
        detections=det_summary,
        activities=activities_hud,
    )

    cv2.imshow("Enton Vision", annotated)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("f"):
        fullscreen = not fullscreen
        prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty("Enton Vision", cv2.WND_PROP_FULLSCREEN, prop)

cap.release()
cv2.destroyAllWindows()
