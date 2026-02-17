"""High-quality HUD overlay using PIL TrueType rendering on OpenCV frames."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Font paths (JetBrains Mono Nerd Font)
_FONT_DIR = Path.home() / ".local/share/fonts"
_FONT_REGULAR = _FONT_DIR / "JetBrainsMonoNerdFont-Regular.ttf"
_FONT_BOLD = _FONT_DIR / "JetBrainsMonoNerdFont-Bold.ttf"
_FONT_MEDIUM = _FONT_DIR / "JetBrainsMonoNerdFont-Medium.ttf"

# Fallback
_FALLBACK = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"


def _load_font(path: Path | str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.truetype(_FALLBACK, size)


class Overlay:
    """Renders beautiful anti-aliased text onto OpenCV frames via PIL."""

    def __init__(self, font_size: int = 18) -> None:
        self._size = font_size
        self._font = _load_font(_FONT_REGULAR, font_size)
        self._font_bold = _load_font(_FONT_BOLD, font_size)
        self._font_medium = _load_font(_FONT_MEDIUM, int(font_size * 0.85))
        self._font_big = _load_font(_FONT_BOLD, int(font_size * 1.4))

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        n_objects: int,
        n_persons: int,
        detections: dict[str, int],
        activities: list[tuple[str, tuple[int, int, int]]],
    ) -> np.ndarray:
        """Draw the full HUD panel on the top-left of the frame."""
        h, w = frame.shape[:2]

        # Build HUD content
        lines: list[tuple[str, ImageFont.FreeTypeFont, tuple[int, int, int, int]]] = []
        lines.append((f"ENTON VISION", self._font_big, (0, 255, 120, 230)))
        lines.append((f"FPS {fps:.0f}  |  {n_objects} objs  |  {n_persons} pessoa(s)", self._font_bold, (0, 230, 200, 220)))

        if detections:
            lines.append(("", self._font_medium, (0, 0, 0, 0)))  # spacer
            for name, count in sorted(detections.items(), key=lambda x: -x[1])[:8]:
                lines.append((f"  {name}: {count}", self._font_medium, (180, 200, 220, 200)))

        if activities:
            lines.append(("", self._font_medium, (0, 0, 0, 0)))  # spacer
            for act_label, (b, g, r) in activities[:4]:
                lines.append((f"  {act_label}", self._font, (r, g, b, 230)))

        # Calculate panel size
        padding = 14
        line_height = self._size + 6
        big_line_height = int(self._size * 1.4) + 8
        panel_h = padding * 2
        for text, font, _ in lines:
            if not text:
                panel_h += 6
            elif font == self._font_big:
                panel_h += big_line_height
            else:
                panel_h += line_height

        panel_w = 400

        # Create RGBA overlay with PIL
        overlay_img = Image.new("RGBA", (panel_w, panel_h), (15, 15, 20, 180))
        draw = ImageDraw.Draw(overlay_img)

        # Rounded rectangle border
        draw.rounded_rectangle(
            [(0, 0), (panel_w - 1, panel_h - 1)],
            radius=12,
            fill=(15, 15, 20, 180),
            outline=(0, 255, 120, 100),
            width=2,
        )

        # Draw lines
        y = padding
        for text, font, color in lines:
            if not text:
                y += 6
                continue
            # Shadow
            draw.text((padding + 1, y + 1), text, font=font, fill=(0, 0, 0, 160))
            # Main text
            draw.text((padding, y), text, font=font, fill=color)
            if font == self._font_big:
                y += big_line_height
            else:
                y += line_height

        # Composite PIL overlay onto OpenCV frame
        overlay_np = np.array(overlay_img)
        # RGBA -> split
        r, g, b, a = cv2.split(overlay_np)
        overlay_bgr = cv2.merge([b, g, r])
        alpha = a.astype(np.float32) / 255.0

        # Region of interest
        x_off, y_off = 10, 10
        roi_h = min(panel_h, h - y_off)
        roi_w = min(panel_w, w - x_off)
        roi = frame[y_off:y_off + roi_h, x_off:x_off + roi_w]
        alpha_roi = alpha[:roi_h, :roi_w, np.newaxis]
        blended = (overlay_bgr[:roi_h, :roi_w].astype(np.float32) * alpha_roi +
                   roi.astype(np.float32) * (1 - alpha_roi))
        frame[y_off:y_off + roi_h, x_off:x_off + roi_w] = blended.astype(np.uint8)

        return frame

    def draw_activity_label(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int],
    ) -> np.ndarray:
        """Draw an activity label above a person's head with nice rendering."""
        x, y = position
        h, w = frame.shape[:2]

        # Measure text
        bbox = self._font_bold.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        pad = 8
        label_w = tw + pad * 2
        label_h = th + pad * 2

        # Center above point
        lx = max(0, x - label_w // 2)
        ly = max(0, y - label_h - 8)

        # Create label image
        label_img = Image.new("RGBA", (label_w, label_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(label_img)

        b, g, r = color
        draw.rounded_rectangle(
            [(0, 0), (label_w - 1, label_h - 1)],
            radius=8,
            fill=(r // 4, g // 4, b // 4, 200),
            outline=(r, g, b, 200),
            width=2,
        )
        draw.text((pad, pad // 2), text, font=self._font_bold, fill=(r, g, b, 255))

        # Composite
        label_np = np.array(label_img)
        lr, lg, lb, la = cv2.split(label_np)
        label_bgr = cv2.merge([lb, lg, lr])
        alpha = la.astype(np.float32) / 255.0

        roi_h = min(label_h, h - ly)
        roi_w = min(label_w, w - lx)
        if roi_h <= 0 or roi_w <= 0 or lx < 0 or ly < 0:
            return frame

        roi = frame[ly:ly + roi_h, lx:lx + roi_w]
        alpha_roi = alpha[:roi_h, :roi_w, np.newaxis]
        blended = (label_bgr[:roi_h, :roi_w].astype(np.float32) * alpha_roi +
                   roi.astype(np.float32) * (1 - alpha_roi))
        frame[ly:ly + roi_h, lx:lx + roi_w] = blended.astype(np.uint8)

        return frame
