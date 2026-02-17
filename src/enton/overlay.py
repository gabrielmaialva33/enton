"""Cyberpunk HUD overlay — PIL TrueType + neon glow + sparkline graphs."""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Font paths (JetBrains Mono Nerd Font)
_FONT_DIR = Path.home() / ".local/share/fonts"
_FONT_REGULAR = _FONT_DIR / "JetBrainsMonoNerdFont-Regular.ttf"
_FONT_BOLD = _FONT_DIR / "JetBrainsMonoNerdFont-Bold.ttf"
_FONT_MEDIUM = _FONT_DIR / "JetBrainsMonoNerdFont-Medium.ttf"
_FALLBACK = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"

# Neon color palette
NEON_GREEN = (0, 255, 120)
NEON_CYAN = (0, 230, 255)
NEON_MAGENTA = (255, 0, 200)
NEON_ORANGE = (255, 160, 0)
NEON_RED = (255, 50, 50)


def _load_font(path: Path | str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.truetype(_FALLBACK, size)


def _composite(frame: np.ndarray, overlay_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    """Alpha-composite an RGBA numpy array onto a BGR frame at (x, y)."""
    h, w = frame.shape[:2]
    oh, ow = overlay_rgba.shape[:2]
    rh = min(oh, h - y)
    rw = min(ow, w - x)
    if rh <= 0 or rw <= 0 or x < 0 or y < 0:
        return frame
    r, g, b, a = cv2.split(overlay_rgba[:rh, :rw])
    bgr = cv2.merge([b, g, r])
    alpha = a.astype(np.float32) / 255.0
    roi = frame[y:y + rh, x:x + rw]
    alpha3 = alpha[:, :, np.newaxis]
    blended = bgr.astype(np.float32) * alpha3 + roi.astype(np.float32) * (1 - alpha3)
    frame[y:y + rh, x:x + rw] = blended.astype(np.uint8)
    return frame


class Overlay:
    """Cyberpunk-style HUD with neon glow, sparkline graphs, and smart analysis look."""

    def __init__(self, font_size: int = 18) -> None:
        self._size = font_size
        self._font = _load_font(_FONT_REGULAR, font_size)
        self._font_bold = _load_font(_FONT_BOLD, font_size)
        self._font_medium = _load_font(_FONT_MEDIUM, int(font_size * 0.85))
        self._font_big = _load_font(_FONT_BOLD, int(font_size * 1.5))
        self._font_small = _load_font(_FONT_REGULAR, int(font_size * 0.7))
        self._boot = time.time()
        # sparkline history
        self._fps_history: deque[float] = deque(maxlen=60)
        self._det_history: deque[int] = deque(maxlen=60)
        self._scan_y = 0

    # ---- neon glow on skeleton ----

    def draw_glow_skeleton(
        self,
        frame: np.ndarray,
        kpts,
        skeleton: list[tuple[int, int]],
        color: tuple[int, int, int],
        visible_fn,
    ) -> np.ndarray:
        """Draw skeleton with neon glow effect."""
        # Glow pass — thick blurred lines
        glow = np.zeros_like(frame)
        for a, b in skeleton:
            if visible_fn(kpts, a) and visible_fn(kpts, b):
                pa = (int(kpts[a][0]), int(kpts[a][1]))
                pb = (int(kpts[b][0]), int(kpts[b][1]))
                cv2.line(glow, pa, pb, color, 8, cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (21, 21), 0)
        cv2.addWeighted(frame, 1.0, glow, 0.6, 0, frame)

        # Sharp pass — thin crisp lines
        for a, b in skeleton:
            if visible_fn(kpts, a) and visible_fn(kpts, b):
                pa = (int(kpts[a][0]), int(kpts[a][1]))
                pb = (int(kpts[b][0]), int(kpts[b][1]))
                cv2.line(frame, pa, pb, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)

        # Glow keypoints
        for ki in range(17):
            if visible_fn(kpts, ki):
                px, py = int(kpts[ki][0]), int(kpts[ki][1])
                cv2.circle(frame, (px, py), 8, color, 2, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)

        return frame

    # ---- targeting brackets on persons ----

    def draw_target_brackets(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int], color: tuple[int, int, int],
    ) -> np.ndarray:
        """Draw sci-fi targeting corner brackets around a bounding box."""
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        corner = max(15, min(bw, bh) // 5)
        t = 2

        # top-left
        cv2.line(frame, (x1, y1), (x1 + corner, y1), color, t, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner), color, t, cv2.LINE_AA)
        # top-right
        cv2.line(frame, (x2, y1), (x2 - corner, y1), color, t, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + corner), color, t, cv2.LINE_AA)
        # bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner, y2), color, t, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - corner), color, t, cv2.LINE_AA)
        # bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner, y2), color, t, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner), color, t, cv2.LINE_AA)

        return frame

    # ---- scan line effect ----

    def draw_scan_line(self, frame: np.ndarray) -> np.ndarray:
        """Animated horizontal scan line sweeping down the frame."""
        h, w = frame.shape[:2]
        self._scan_y = (self._scan_y + 3) % h
        y = self._scan_y
        overlay = np.zeros((20, w, 3), dtype=np.uint8)
        for i in range(20):
            alpha = 1.0 - (abs(i - 10) / 10.0)
            overlay[i, :] = [int(120 * alpha), int(255 * alpha), 0]
        y_start = max(0, y - 10)
        y_end = min(h, y + 10)
        olen = y_end - y_start
        if olen > 0:
            cv2.addWeighted(
                frame[y_start:y_end], 0.85,
                overlay[:olen, :w], 0.15, 0,
                frame[y_start:y_end],
            )
        return frame

    # ---- sparkline graph ----

    def _draw_sparkline(
        self,
        draw: ImageDraw.Draw,
        data: deque,
        x: int, y: int, w: int, h: int,
        color: tuple[int, int, int, int],
        label: str,
        value_fmt: str,
    ) -> int:
        """Draw a mini sparkline graph. Returns y offset after drawing."""
        if not data:
            return y

        # Label
        draw.text((x, y), label, font=self._font_small, fill=(150, 160, 170, 200))
        val = data[-1] if data else 0
        val_str = value_fmt.format(val)
        draw.text((x + w - 50, y), val_str, font=self._font_small, fill=color)
        y += int(self._size * 0.65)

        # Graph background
        draw.rounded_rectangle([(x, y), (x + w, y + h)], radius=4, fill=(30, 30, 40, 150))

        # Plot line
        max_val = max(data) if data else 1
        if max_val == 0:
            max_val = 1
        points = []
        n = len(data)
        for i, v in enumerate(data):
            px = x + int(i / max(n - 1, 1) * w)
            py = y + h - int(v / max_val * (h - 4)) - 2
            points.append((px, py))

        if len(points) >= 2:
            # Fill area under curve
            fill_points = list(points) + [(points[-1][0], y + h), (points[0][0], y + h)]
            fill_color = (color[0], color[1], color[2], 40)
            draw.polygon(fill_points, fill=fill_color)
            # Line
            draw.line(points, fill=color, width=2)

        return y + h + 6

    # ---- main HUD ----

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        n_objects: int,
        n_persons: int,
        detections: dict[str, int],
        activities: list[tuple[str, tuple[int, int, int]]],
    ) -> np.ndarray:
        """Draw the full cyberpunk HUD panel."""
        fh, fw = frame.shape[:2]
        self._fps_history.append(fps)
        self._det_history.append(n_objects)

        uptime = time.time() - self._boot
        m, s = divmod(int(uptime), 60)

        # Panel dimensions
        padding = 14
        line_h = self._size + 6
        big_line_h = int(self._size * 1.5) + 8
        graph_h = 40
        panel_w = 420

        # Calculate content height
        content_h = padding
        content_h += big_line_h  # title
        content_h += line_h  # status line
        content_h += line_h  # uptime
        content_h += 8  # spacer
        content_h += int(self._size * 0.65) + graph_h + 6  # fps graph
        content_h += int(self._size * 0.65) + graph_h + 6  # det graph
        if detections:
            content_h += 8 + min(len(detections), 6) * line_h
        if activities:
            content_h += 8 + min(len(activities), 4) * line_h
        content_h += padding

        # Create RGBA panel
        panel = Image.new("RGBA", (panel_w, content_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)

        # Background with gradient-ish effect
        draw.rounded_rectangle(
            [(0, 0), (panel_w - 1, content_h - 1)],
            radius=14,
            fill=(10, 12, 18, 200),
            outline=(0, 255, 120, 120),
            width=2,
        )
        # Inner glow line at top
        draw.line([(14, 2), (panel_w - 14, 2)], fill=(0, 255, 120, 60), width=1)

        y = padding

        # Title
        draw.text((padding + 1, y + 1), "ENTON VISION", font=self._font_big, fill=(0, 0, 0, 180))
        draw.text((padding, y), "ENTON VISION", font=self._font_big, fill=(0, 255, 120, 240))
        y += big_line_h

        # Status
        status = f" {n_objects} objs   {n_persons} pessoa(s)"
        draw.text((padding, y), status, font=self._font_bold, fill=(0, 230, 255, 220))
        y += line_h

        # Uptime + timestamp
        ts = time.strftime("%H:%M:%S")
        draw.text((padding, y), f" {m:02d}:{s:02d} uptime  |  {ts}",
                  font=self._font_small, fill=(100, 110, 120, 180))
        y += line_h + 8

        # FPS sparkline
        gw = panel_w - padding * 2
        y = self._draw_sparkline(draw, self._fps_history, padding, y, gw, graph_h,
                                 (0, 255, 120, 220), "FPS", "{:.0f}")

        # Detection sparkline
        y = self._draw_sparkline(draw, self._det_history, padding, y, gw, graph_h,
                                 (0, 200, 255, 220), "DETECTIONS", "{:.0f}")

        # Detection list
        if detections:
            y += 2
            for name, count in sorted(detections.items(), key=lambda x: -x[1])[:6]:
                bar_w = min(int(count / max(detections.values()) * (gw - 120)), gw - 120)
                draw.rounded_rectangle(
                    [(padding + 100, y + 4), (padding + 100 + bar_w, y + line_h - 4)],
                    radius=3, fill=(0, 255, 120, 50),
                )
                draw.text((padding + 4, y), f"{name}", font=self._font_medium, fill=(180, 200, 220, 200))
                draw.text((padding + 100 + bar_w + 6, y + 2), f"{count}",
                          font=self._font_small, fill=(0, 255, 120, 200))
                y += line_h

        # Activities
        if activities:
            y += 2
            for act_label, (b, g, r) in activities[:4]:
                draw.text((padding + 4, y), f" {act_label}",
                          font=self._font, fill=(r, g, b, 230))
                y += line_h

        panel_np = np.array(panel)
        frame = _composite(frame, panel_np, 10, 10)
        return frame

    # ---- activity label ----

    def draw_activity_label(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int],
    ) -> np.ndarray:
        """Draw a floating neon activity label above a person's head."""
        x, y = position
        fh, fw = frame.shape[:2]

        bbox = self._font_bold.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = 10
        label_w = tw + pad * 2 + 4
        label_h = th + pad * 2

        lx = max(0, min(x - label_w // 2, fw - label_w))
        ly = max(0, y - label_h - 15)

        b, g, r = color

        # Glow behind label
        glow_img = Image.new("RGBA", (label_w + 20, label_h + 20), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow_img)
        gd.rounded_rectangle([(10, 10), (label_w + 9, label_h + 9)], radius=10,
                             fill=(r, g, b, 80))
        glow_img = glow_img.filter(ImageFilter.GaussianBlur(8))
        glow_np = np.array(glow_img)
        frame = _composite(frame, glow_np, lx - 10, ly - 10)

        # Label
        label_img = Image.new("RGBA", (label_w, label_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(label_img)
        draw.rounded_rectangle(
            [(0, 0), (label_w - 1, label_h - 1)],
            radius=10,
            fill=(r // 6, g // 6, b // 6, 210),
            outline=(r, g, b, 220),
            width=2,
        )
        # Accent line
        draw.line([(6, 2), (label_w - 6, 2)], fill=(r, g, b, 100), width=1)
        draw.text((pad + 2, pad // 2 + 1), text, font=self._font_bold, fill=(0, 0, 0, 150))
        draw.text((pad + 1, pad // 2), text, font=self._font_bold, fill=(r, g, b, 255))

        label_np = np.array(label_img)
        frame = _composite(frame, label_np, lx, ly)
        return frame

    # ---- confidence badge on objects ----

    def draw_confidence_badge(
        self,
        frame: np.ndarray,
        label: str,
        conf: float,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Small confidence indicator on detected objects."""
        x1, y1, _, _ = bbox
        text = f"{label} {conf:.0%}"
        tbbox = self._font_small.getbbox(text)
        tw = tbbox[2] - tbbox[0]
        bw = tw + 12
        bh = int(self._size * 0.75) + 6

        # Color based on confidence
        if conf > 0.7:
            c = NEON_GREEN
        elif conf > 0.4:
            c = NEON_CYAN
        else:
            c = NEON_ORANGE

        badge = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        bd = ImageDraw.Draw(badge)
        bd.rounded_rectangle([(0, 0), (bw - 1, bh - 1)], radius=5,
                             fill=(10, 10, 15, 180), outline=(*c, 150), width=1)
        bd.text((6, 1), text, font=self._font_small, fill=(*c, 230))

        badge_np = np.array(badge)
        frame = _composite(frame, badge_np, max(0, x1), max(0, y1 - bh - 2))
        return frame
