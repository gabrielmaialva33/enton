"""Activity recognition from COCO 17-keypoint pose data."""
from __future__ import annotations

import math

import numpy as np

# COCO keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

MIN_CONFIDENCE = 0.3


def _angle(a: tuple, b: tuple, c: tuple) -> float:
    """Angle at point *b* formed by segments a-b and b-c, in degrees."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos, -1, 1)))


def _xy(kpts, idx: int) -> tuple[float, float]:
    return float(kpts[idx][0]), float(kpts[idx][1])


def _visible(kpts, idx: int) -> bool:
    return float(kpts[idx][2]) > MIN_CONFIDENCE


def classify(kpts) -> tuple[str, tuple[int, int, int]]:
    """Classify a person's activity from 17 COCO keypoints.

    Parameters
    ----------
    kpts : array-like, shape (17, 3)
        Each row is ``(x, y, confidence)``.

    Returns
    -------
    activity : str
        Human-readable activity label (pt-BR).
    color : tuple[int, int, int]
        BGR colour for visualisation.
    """
    has_upper = all(_visible(kpts, i) for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP))
    has_legs = all(_visible(kpts, i) for i in (L_KNEE, R_KNEE, L_ANKLE, R_ANKLE))
    has_arms = all(_visible(kpts, i) for i in (L_ELBOW, R_ELBOW, L_WRIST, R_WRIST))

    if not has_upper:
        return "?", (128, 128, 128)

    ls, rs = _xy(kpts, L_SHOULDER), _xy(kpts, R_SHOULDER)
    lh, rh = _xy(kpts, L_HIP), _xy(kpts, R_HIP)
    mid_shoulder = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    mid_hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    shoulder_width = math.dist(ls, rs)

    # Torso angle relative to vertical (0° = upright)
    torso_angle = abs(
        math.degrees(
            math.atan2(mid_hip[0] - mid_shoulder[0], mid_hip[1] - mid_shoulder[1])
        )
    )

    # --- lying down ---
    if torso_angle > 60:
        return "Deitado", (255, 100, 100)

    tags: list[tuple[str, tuple[int, int, int]]] = []

    # --- arms up / waving ---
    if has_arms:
        lw, rw = _xy(kpts, L_WRIST), _xy(kpts, R_WRIST)
        both_up = lw[1] < ls[1] and rw[1] < rs[1]
        one_up = lw[1] < ls[1] or rw[1] < rs[1]
        if both_up:
            tags.append(("Maos pra cima!", (0, 255, 255)))
        elif one_up:
            tags.append(("Acenando", (0, 200, 255)))

    # --- hand near face (phone / touching face) ---
    if has_arms and _visible(kpts, NOSE):
        nose = _xy(kpts, NOSE)
        lw, rw = _xy(kpts, L_WRIST), _xy(kpts, R_WRIST)
        if math.dist(lw, nose) < shoulder_width * 0.7 or math.dist(rw, nose) < shoulder_width * 0.7:
            tags.append(("No celular", (255, 150, 0)))

    # --- sitting vs standing ---
    if has_legs:
        lk, rk = _xy(kpts, L_KNEE), _xy(kpts, R_KNEE)
        la, ra = _xy(kpts, L_ANKLE), _xy(kpts, R_ANKLE)
        avg_knee = (_angle(lh, lk, la) + _angle(rh, rk, ra)) / 2
        if avg_knee < 120:
            tags.append(("Sentado", (255, 200, 0)))
        else:
            tags.append(("Em pe", (0, 255, 0)))

    # --- arms crossed ---
    if has_arms and shoulder_width > 0:
        lw, rw = _xy(kpts, L_WRIST), _xy(kpts, R_WRIST)
        if lw[0] > mid_shoulder[0] and rw[0] < mid_shoulder[0]:
            if abs(lw[1] - rw[1]) < shoulder_width * 0.5:
                tags.append(("Bracos cruzados", (200, 100, 255)))

    if tags:
        return " | ".join(t[0] for t in tags), tags[0][1]
    return "Parado", (200, 200, 200)
