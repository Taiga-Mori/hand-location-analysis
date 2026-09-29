import colorsys
import hashlib
import math
from typing import Any

import cv2
import numpy as np

HAND_COLORS = {"right": (60, 60, 255), "left": (255, 170, 30)}
HAND_SHORT = {"right": "R", "left": "L"}


class FrameAnnotator:
    """Draw the nine-sector grid and wrist positions on video frames."""

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        scale = max(frame_height, 360) / 720.0
        self.line_thickness = max(1, int(round(2 * scale)))
        self.wrist_radius = max(4, int(round(7 * scale)))
        self.font_scale = 0.55 * scale
        self.font_thickness = max(1, int(round(1.5 * scale)))

    def id_to_color(self, track_id: Any) -> tuple[int, int, int]:
        digest = hashlib.sha1(str(track_id).encode("utf-8")).digest()
        hue_seed = int.from_bytes(digest[:8], "big") / float(1 << 64)
        hue = (hue_seed * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        return int(b * 255), int(g * 255), int(r * 255)

    def draw_person(self, frame: np.ndarray, row: dict[str, Any], orientation_mirrored: bool = False) -> np.ndarray:
        color = self.id_to_color(row["track_id"])
        grid = self._grid(row)
        if grid is not None:
            self._draw_grid(frame, grid, color, row, orientation_mirrored)
        self._draw_track_label(frame, row, color)
        for hand in ("right", "left"):
            self._draw_wrist(frame, row, hand)
        return frame

    def _grid(self, row: dict[str, Any]) -> tuple[float, float, float, float] | None:
        values = [row.get(key) for key in ("grid_x_right", "grid_x_left", "grid_y_top", "grid_y_bottom")]
        if any(value is None or not math.isfinite(float(value)) for value in values):
            return None
        x_right, x_left, y_top, y_bottom = (float(value) for value in values)
        return min(x_right, x_left), max(x_right, x_left), y_top, y_bottom

    def _draw_grid(
        self,
        frame: np.ndarray,
        grid: tuple[float, float, float, float],
        color: tuple[int, int, int],
        row: dict[str, Any],
        mirrored: bool,
    ) -> None:
        low_x, high_x, y_top, y_bottom = grid
        width = max(high_x - low_x, 1.0)
        height = max(y_bottom - y_top, 1.0)
        # Lines extend over the person's box and at least one torso size beyond the center rectangle.
        span_x1 = min(float(row["x1"]), low_x - width)
        span_x2 = max(float(row["x2"]), high_x + width)
        span_y1 = min(float(row["y1"]), y_top - height)
        span_y2 = max(float(row["y2"]), y_bottom + height)
        span_x1, span_x2 = max(0.0, span_x1), min(self.frame_width - 1.0, span_x2)
        span_y1, span_y2 = max(0.0, span_y1), min(self.frame_height - 1.0, span_y2)

        overlay = frame.copy()
        cv2.rectangle(overlay, self._pt(low_x, y_top), self._pt(high_x, y_bottom), color, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0.0, dst=frame)
        dashed = row.get("hip_estimated") in (True, 1, "True")
        for x in (low_x, high_x):
            cv2.line(frame, self._pt(x, span_y1), self._pt(x, span_y2), color, self.line_thickness, cv2.LINE_AA)
        cv2.line(frame, self._pt(span_x1, y_top), self._pt(span_x2, y_top), color, self.line_thickness, cv2.LINE_AA)
        if dashed:
            self._dashed_hline(frame, span_x1, span_x2, y_bottom, color)
        else:
            cv2.line(frame, self._pt(span_x1, y_bottom), self._pt(span_x2, y_bottom), color, self.line_thickness, cv2.LINE_AA)

        column_centers = [(span_x1 + low_x) / 2.0, (low_x + high_x) / 2.0, (high_x + span_x2) / 2.0]
        row_centers = [(span_y1 + y_top) / 2.0, (y_top + y_bottom) / 2.0, (y_bottom + span_y2) / 2.0]
        for row_index, cy in enumerate(row_centers):
            for image_col, cx in enumerate(column_centers):
                person_col = 2 - image_col if mirrored else image_col
                self._draw_centered_text(frame, str(row_index * 3 + person_col + 1), cx, cy, color)

    def _dashed_hline(self, frame: np.ndarray, x1: float, x2: float, y: float, color: tuple[int, int, int]) -> None:
        dash = max(6, self.line_thickness * 5)
        x = x1
        while x < x2:
            cv2.line(frame, self._pt(x, y), self._pt(min(x + dash, x2), y), color, self.line_thickness, cv2.LINE_AA)
            x += dash * 2

    def _draw_wrist(self, frame: np.ndarray, row: dict[str, Any], hand: str) -> None:
        x = row.get(f"{hand}_wrist_x")
        y = row.get(f"{hand}_wrist_y")
        if x is None or y is None or not math.isfinite(float(x)) or not math.isfinite(float(y)):
            return
        color = HAND_COLORS[hand]
        center = self._pt(float(x), float(y))
        cv2.circle(frame, center, self.wrist_radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, center, self.wrist_radius, color, -1, cv2.LINE_AA)
        location = row.get(f"{hand}_location")
        text = HAND_SHORT[hand] if location is None or _is_nan(location) else f"{HAND_SHORT[hand]}{int(location)}"
        # Right-hand labels sit left of the wrist and left-hand labels right of it, so they never overlap.
        offset = self.wrist_radius + 3
        if hand == "right":
            self._draw_label(frame, text, center[0] - offset, center[1] - self.wrist_radius, color, anchor="right")
        else:
            self._draw_label(frame, text, center[0] + offset, center[1] - self.wrist_radius, color)

    def _draw_track_label(self, frame: np.ndarray, row: dict[str, Any], color: tuple[int, int, int]) -> None:
        parts = [f"ID {row['track_id']}"]
        for hand in ("right", "left"):
            location = row.get(f"{hand}_location")
            parts.append(f"{HAND_SHORT[hand]}:{'-' if location is None or _is_nan(location) else int(location)}")
        x = int(max(0, float(row["x1"])))
        y = int(max(0, float(row["y1"])))
        self._draw_label(frame, " ".join(parts), x, y, color)

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
        anchor: str = "left",
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, self.font_scale, self.font_thickness)
        pad = 3
        if anchor == "right":
            x = x - text_width - 2 * pad
        x = int(min(max(0, x), self.frame_width - text_width - 2 * pad))
        y1 = int(y - text_height - baseline - 2 * pad)
        if y1 < 0:
            y1 = 0
        y2 = y1 + text_height + baseline + 2 * pad
        cv2.rectangle(frame, (x, y1), (x + text_width + 2 * pad, y2), color, -1)
        cv2.putText(frame, text, (x + pad, y2 - baseline - pad), font, self.font_scale, (255, 255, 255), self.font_thickness, cv2.LINE_AA)

    def _draw_centered_text(self, frame: np.ndarray, text: str, cx: float, cy: float, color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(text, font, self.font_scale, self.font_thickness)
        origin = (int(cx - text_width / 2), int(cy + text_height / 2))
        cv2.putText(frame, text, origin, font, self.font_scale, (255, 255, 255), self.font_thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, origin, font, self.font_scale, color, self.font_thickness, cv2.LINE_AA)

    def _pt(self, x: float, y: float) -> tuple[int, int]:
        return int(round(x)), int(round(y))


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True
