import colorsys
import hashlib
from typing import Any

import cv2
import numpy as np

HAND_SHORT = {"right": "R", "left": "L"}
# Each wrist fills its sector once, blended over what is already there, so overlapping fills get darker
# (both wrists of a person in one sector: 1 - 0.85^2 = 28%) and different persons' colors mix.
FILL_ALPHA = 0.15
LINE_ALPHA = 0.65


class FrameAnnotator:
    """Draw each person's nine-sector grid with its outer box, sector numbers, wrists, and track id, all
    semi-transparent.

    Everything of one track uses one color. Only the sectors that contain a wrist are filled, once per wrist,
    so overlapping fills darken and mix colors. When neither hand of a person is valid in a frame, only the
    track id is drawn.
    """

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        scale = max(frame_height, 360) / 720.0
        self.line_thickness = max(1, int(round(2 * scale)))
        self.wrist_radius = max(4, int(round(8 * scale)))
        self.font_scale = 0.6 * scale
        self.font_thickness = max(1, int(round(1.5 * scale)))

    def id_to_color(self, track_id: Any) -> tuple[int, int, int]:
        digest = hashlib.sha1(str(track_id).encode("utf-8")).digest()
        hue_seed = int.from_bytes(digest[:8], "big") / float(1 << 64)
        hue = (hue_seed * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        return int(b * 255), int(g * 255), int(r * 255)

    def draw(self, frame: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
        people = []
        for row in rows:
            hands = [hand for hand in HAND_SHORT if bool(row.get(f"{hand}_valid"))]
            mirrored = bool(hands) and self.mirrored(row)
            people.append((row, self.id_to_color(row["track_id"]), hands, mirrored, self._grid_edges(row) if hands else None))

        for row, color, hands, mirrored, edges in people:
            for hand in hands:
                self._fill_sector(frame, edges, int(row[f"{hand}_location"]), mirrored, color)

        # Copied after the fill is blended in, so undrawn pixels stay unchanged by the second blend.
        line_layer = frame.copy()
        for row, color, hands, mirrored, edges in people:
            if hands:
                self._draw_grid(line_layer, edges, mirrored, color)
                for hand in hands:
                    self._draw_wrist(line_layer, row, hand, color)
            self._draw_track_id(line_layer, row, color, edges if edges is not None else self._label_edges(row))
        cv2.addWeighted(line_layer, LINE_ALPHA, frame, 1 - LINE_ALPHA, 0.0, dst=frame)
        return frame

    @staticmethod
    def mirrored(row: dict[str, Any]) -> bool:
        """True when the 1/4/7 column is on the image right (its line is right of the 3/6/9 line)."""

        return float(row["grid_x_first"]) > float(row["grid_x_second"])

    def _label_edges(self, row: dict[str, Any]) -> tuple[list[float], list[float]] | None:
        """Grid edges for placing the track id when no grid is drawn (None if the grid cannot be computed)."""

        values = [row.get(key) for key in ("grid_x_first", "grid_x_second", "grid_y_top", "grid_y_bottom")]
        try:
            if all(np.isfinite(float(value)) for value in values):
                return self._grid_edges(row)
        except (TypeError, ValueError):
            pass
        return None

    def _grid_edges(self, row: dict[str, Any]) -> tuple[list[float], list[float]]:
        """Column and row edges in image coordinates: [outer, line, line, outer] for x and for y."""

        low_x, high_x = sorted([float(row["grid_x_first"]), float(row["grid_x_second"])])
        y_top, y_bottom = float(row["grid_y_top"]), float(row["grid_y_bottom"])
        width, height = max(high_x - low_x, 1.0), max(y_bottom - y_top, 1.0)
        # Lines extend over the person's box and at least one torso size beyond the center rectangle.
        x1 = max(0.0, min(float(row["x1"]), low_x - width))
        x2 = min(self.frame_width - 1.0, max(float(row["x2"]), high_x + width))
        y1 = max(0.0, min(float(row["y1"]), y_top - height))
        y2 = min(self.frame_height - 1.0, max(float(row["y2"]), y_bottom + height))
        return [x1, low_x, high_x, x2], [y1, y_top, y_bottom, y2]

    def _fill_sector(self, frame: np.ndarray, edges, location: int, mirrored: bool, color) -> None:
        """Blend the sector rectangle toward `color` by FILL_ALPHA, on top of earlier fills."""

        xs, ys = edges
        row_index, person_col = divmod(location - 1, 3)
        image_col = 2 - person_col if mirrored else person_col
        x1, y1 = self._pt(xs[image_col], ys[row_index])
        x2, y2 = self._pt(xs[image_col + 1], ys[row_index + 1])
        region = frame[max(0, y1) : max(0, y2) + 1, max(0, x1) : max(0, x2) + 1]
        if region.size:
            region[:] = (region * (1 - FILL_ALPHA) + np.array(color, dtype=np.float32) * FILL_ALPHA).astype(np.uint8)

    def _draw_grid(self, layer, edges, mirrored: bool, color) -> None:
        xs, ys = edges
        # Outer box around the whole nine-sector area, for reference.
        cv2.rectangle(layer, self._pt(xs[0], ys[0]), self._pt(xs[3], ys[3]), color, self.line_thickness, cv2.LINE_AA)
        for x in xs[1:3]:
            cv2.line(layer, self._pt(x, ys[0]), self._pt(x, ys[3]), color, self.line_thickness, cv2.LINE_AA)
        for y in ys[1:3]:
            cv2.line(layer, self._pt(xs[0], y), self._pt(xs[3], y), color, self.line_thickness, cv2.LINE_AA)
        for row_index in range(3):
            for image_col in range(3):
                person_col = 2 - image_col if mirrored else image_col
                cx = (xs[image_col] + xs[image_col + 1]) / 2
                cy = (ys[row_index] + ys[row_index + 1]) / 2
                self._draw_centered_text(layer, str(row_index * 3 + person_col + 1), cx, cy, color)

    def _draw_wrist(self, layer, row: dict[str, Any], hand: str, color) -> None:
        center = self._pt(float(row[f"{hand}_wrist_x"]), float(row[f"{hand}_wrist_y"]))
        cv2.circle(layer, center, self.wrist_radius, color, -1, cv2.LINE_AA)
        cv2.circle(layer, center, self.wrist_radius, (255, 255, 255), max(1, self.line_thickness // 2), cv2.LINE_AA)
        # Right-hand labels sit left of the wrist and left-hand labels right of it, so they never overlap.
        text = HAND_SHORT[hand]
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness)
        offset = self.wrist_radius + 4
        x = center[0] - offset - text_width if hand == "right" else center[0] + offset
        self._draw_text(layer, text, x, center[1] + text_height // 2, color)

    def _draw_track_id(self, layer, row: dict[str, Any], color, edges) -> None:
        """Label just outside the top-left corner of the outer box (the person's box when there is no grid)."""

        text = f"ID {row['track_id']}"
        scale = self.font_scale * 1.2
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, self.font_thickness)
        left, top = (edges[0][0], edges[1][0]) if edges is not None else (float(row["x1"]), float(row["y1"]))
        pad = 4
        box_height = text_height + baseline + 2 * pad
        x = int(min(max(0, left), self.frame_width - text_width - 2 * pad))
        y_bottom = int(top) - max(1, self.line_thickness // 2)
        if y_bottom - box_height < 0:  # no room above the box: keep the label inside the frame
            y_bottom = box_height
        cv2.rectangle(layer, (x, y_bottom - box_height), (x + text_width + 2 * pad, y_bottom), color, -1)
        cv2.putText(layer, text, (x + pad, y_bottom - baseline - pad), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), self.font_thickness, cv2.LINE_AA)

    def _draw_text(self, layer, text: str, x: float, y: float, color) -> None:
        origin = (int(x), int(y))
        cv2.putText(layer, text, origin, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), self.font_thickness + 2, cv2.LINE_AA)
        cv2.putText(layer, text, origin, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.font_thickness, cv2.LINE_AA)

    def _draw_centered_text(self, layer, text: str, cx: float, cy: float, color) -> None:
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness)
        self._draw_text(layer, text, cx - text_width / 2, cy + text_height / 2, color)

    def _pt(self, x: float, y: float) -> tuple[int, int]:
        return int(round(x)), int(round(y))

