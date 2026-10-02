"""Nine-sector hand location coding after Choi et al. (2014), Fig. 3.

A central rectangle and the two vertical and two horizontal lines through it split the space around the
person into nine sectors:

    1 2 3      1/2/3: above the top line, 7/8/9: below the bottom line
    4 5 6      1/4/7: beyond the "first" vertical line, 3/6/9: beyond the "second" one
    7 8 9

frontal mode (camera facing the person):
    vertical lines at the right and left shoulders, horizontal lines at the shoulders and the hips.
    1/4/7 is the person's right side (with orientation "frontal", the image left).
    Normalized wrist x: right-shoulder line = 0, left-shoulder line = 1; y: shoulder line = 0, hip line = 1.

side mode (camera at the person's side, person sitting upright):
    vertical lines at the body (mean of the shoulder and hip x) and at the knees; horizontal lines at the
    shoulders and at the thighs (mean of the hip and knee y). 1/4/7 is in front of the person (beyond the
    knees), 3/6/9 behind the body. The front is the side of the body line the knees are on, decided once per
    person. Normalized wrist x: body line = 0, knee line = 1 (forward is positive); y: shoulder line = 0,
    thigh line = 1.

Shoulders, hips, and knees are the midpoints of the left and right keypoints. A hand's location is the
sector containing its wrist; points exactly on a line belong to the central column / row.
"""

import numpy as np
import pandas as pd

from .constants import HAND_WRIST_KEYPOINTS, HANDS, SEGMENT_COLUMNS
from .temporal import run_lengths

GRID_KEYPOINTS = {
    "frontal": ["right_shoulder", "left_shoulder", "right_hip", "left_hip"],
    "side": ["right_shoulder", "left_shoulder", "right_hip", "left_hip", "right_knee", "left_knee"],
}


def _midpoint(frames: pd.DataFrame, joint: str, axis: str) -> np.ndarray:
    return (frames[f"right_{joint}_{axis}"].to_numpy(dtype=float) + frames[f"left_{joint}_{axis}"].to_numpy(dtype=float)) / 2


class HandLocationClassifier:
    """Compute the per-frame 3x3 grid of every person and assign each wrist to a sector."""

    def classify(self, frames: pd.DataFrame, conf_thresh: float, orientation: str = "frontal", mode: str = "frontal") -> pd.DataFrame:
        """Add the grid (`grid_x_first`, `grid_x_second`, `grid_y_top`, `grid_y_bottom`, `grid_valid`) and, per hand,
        `<hand>_valid`, `<hand>_x_norm`, `<hand>_y_norm`, `<hand>_location`.

        A hand is valid in a frame when its wrist and every grid keypoint of the mode reach `conf_thresh`
        (0 keeps every frame) and the grid is well formed.
        """

        if frames.empty:
            return frames
        frames = frames.copy()
        if mode == "side":
            first, second, top, bottom, norm_zero, norm_one = self._side_grid(frames)
        else:
            first, second, top, bottom, norm_zero, norm_one = self._frontal_grid(frames, orientation)
        frames["grid_x_first"], frames["grid_x_second"] = first, second
        frames["grid_y_top"], frames["grid_y_bottom"] = top, bottom

        grid_valid = (np.abs(second - first) > 1e-6) & (bottom > top)
        for name in GRID_KEYPOINTS[mode]:
            grid_valid &= frames[f"{name}_conf"].to_numpy(dtype=float) >= conf_thresh
        frames["grid_valid"] = grid_valid

        for hand in HANDS:
            wrist = HAND_WRIST_KEYPOINTS[hand]
            valid = grid_valid & (frames[f"{wrist}_conf"].to_numpy(dtype=float) >= conf_thresh)
            wrist_x = frames[f"{wrist}_x"].to_numpy(dtype=float)
            wrist_y = frames[f"{wrist}_y"].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_norm = (wrist_x - norm_zero) / (norm_one - norm_zero)
                y_norm = (wrist_y - top) / (bottom - top)
            location = self.sector(wrist_x, wrist_y, first, second, top, bottom)
            frames[f"{hand}_valid"] = valid
            frames[f"{hand}_x_norm"] = np.where(valid, x_norm, np.nan)
            frames[f"{hand}_y_norm"] = np.where(valid, y_norm, np.nan)
            frames[f"{hand}_location"] = pd.array(np.where(valid, location, np.nan), dtype="Int64")
        return frames

    def _frontal_grid(self, frames: pd.DataFrame, orientation: str):
        x_right = frames["right_shoulder_x"].to_numpy(dtype=float)
        x_left = frames["left_shoulder_x"].to_numpy(dtype=float)
        if orientation == "auto":
            first, second = x_right, x_left  # the person's right side is 1/4/7 whichever way they face
        else:
            first, second = np.minimum(x_right, x_left), np.maximum(x_right, x_left)  # image left is 1/4/7
        top = _midpoint(frames, "shoulder", "y")
        bottom = _midpoint(frames, "hip", "y")
        return first, second, top, bottom, x_right, x_left

    def _side_grid(self, frames: pd.DataFrame):
        body = (_midpoint(frames, "shoulder", "x") + _midpoint(frames, "hip", "x")) / 2
        knee = _midpoint(frames, "knee", "x")
        top = _midpoint(frames, "shoulder", "y")
        bottom = (_midpoint(frames, "hip", "y") + _midpoint(frames, "knee", "y")) / 2
        # The knee line is the front line; flip it to the other side of the body in frames where the knees
        # appear behind the body, so the front stays where the person faces over the whole video.
        offset = knee - body
        forward = pd.Series(np.sign(offset)).groupby(frames["track_id"].to_numpy()).transform("median").to_numpy()
        forward = np.where(forward == 0, 1.0, np.sign(forward))
        front = body + forward * np.abs(offset)
        return front, body, top, bottom, body, front

    @staticmethod
    def sector(
        wrist_x: np.ndarray,
        wrist_y: np.ndarray,
        first: np.ndarray,
        second: np.ndarray,
        top: np.ndarray,
        bottom: np.ndarray,
    ) -> np.ndarray:
        """Sector numbers 1-9 (NaN when an input is missing). `first` is the vertical line on the 1/4/7 side,
        `second` the one on the 3/6/9 side, in image x."""

        with np.errstate(divide="ignore", invalid="ignore"):
            u = (wrist_x - first) / (second - first)
        col = np.where(u < 0, 0, np.where(u > 1, 2, 1))
        row = np.where(wrist_y < top, 0, np.where(wrist_y > bottom, 2, 1))
        sector = (row * 3 + col + 1).astype(float)
        missing = np.isnan(u) | np.isnan(wrist_y) | np.isnan(top) | np.isnan(bottom)
        sector[missing] = np.nan
        return sector


class SegmentBuilder:
    """Turn per-frame sector labels into time segments; excluded frames end a segment."""

    def build(self, frames: pd.DataFrame, fps: float) -> pd.DataFrame:
        if frames.empty:
            return pd.DataFrame(columns=SEGMENT_COLUMNS)
        segments: list[dict] = []
        for track_id, group in frames.groupby("track_id", sort=True):
            group = group.sort_values("frame_idx")
            start_frame = int(group["frame_idx"].iloc[0])
            for hand in HANDS:
                labels = group[f"{hand}_location"].astype("float").to_numpy()
                for start, length, value in run_lengths(labels):
                    if np.isnan(value):
                        continue
                    segments.append(
                        {
                            "track_id": int(track_id),
                            "hand": hand,
                            "startTime": round((start_frame + start) / fps, 3),
                            "endTime": round((start_frame + start + length) / fps, 3),
                            "location": int(value),
                        }
                    )
        segment_df = pd.DataFrame(segments, columns=SEGMENT_COLUMNS)
        if not segment_df.empty:
            segment_df = segment_df.sort_values(["track_id", "hand", "startTime"], ascending=[True, False, True])
        return segment_df.reset_index(drop=True)
