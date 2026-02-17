"""Live camera feed with YOLO detection + pose + activity recognition.

Standalone demo script — imports activity classifier from enton module.

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

# allow importing enton modules from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from enton.activity import NOSE, classify as classify_activity, _visible as visible

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
print(f"YOLO11x detect+pose | {w}x{h} FP16 | 'q'=sair 'f'=fullscreen")

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

    annotated = det_r[0].plot(line_width=2, font_size=0.5)

    # pose overlay + activity
    n_persons = 0
    if pose_r[0].keypoints is not None and len(pose_r[0].keypoints) > 0:
        kpts_data = pose_r[0].keypoints.data
        n_persons = len(kpts_data)
        for kpts in kpts_data:
            activity, color = classify_activity(kpts)

            for a, b in SKELETON:
                if visible(kpts, a) and visible(kpts, b):
                    pa = (int(kpts[a][0]), int(kpts[a][1]))
                    pb = (int(kpts[b][0]), int(kpts[b][1]))
                    cv2.line(annotated, pa, pb, color, 2)

            for ki in range(17):
                if visible(kpts, ki):
                    px, py = int(kpts[ki][0]), int(kpts[ki][1])
                    cv2.circle(annotated, (px, py), 4, (255, 255, 255), -1)
                    cv2.circle(annotated, (px, py), 3, color, -1)

            if visible(kpts, NOSE):
                nx, ny = int(kpts[NOSE][0]), int(kpts[NOSE][1])
                cv2.putText(annotated, activity, (nx - 60, ny - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(annotated, activity, (nx - 60, ny - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # detection summary
    boxes = det_r[0].boxes
    names = det_r[0].names
    det_summary = {}
    for cls_id in boxes.cls.int().tolist():
        name = names[cls_id]
        det_summary[name] = det_summary.get(name, 0) + 1

    # FPS
    fps_count += 1
    now = time.time()
    if now - fps_t >= 1.0:
        fps = fps_count / (now - fps_t)
        fps_count = 0
        fps_t = now

    # HUD
    hud_h = 70 + len(det_summary) * 26
    overlay = annotated.copy()
    cv2.rectangle(overlay, (5, 5), (380, hud_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
    cv2.putText(annotated, f"YOLO11x detect+pose | FPS: {fps:.1f}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(annotated, f"{len(boxes)} objs | {n_persons} pessoa(s)",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    for i, (name, count) in enumerate(sorted(det_summary.items(), key=lambda x: -x[1])):
        cv2.putText(annotated, f"  {name}: {count}",
                    (10, 82 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

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
