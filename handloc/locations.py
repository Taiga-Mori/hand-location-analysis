"""Nine-sector hand location coding after Choi et al. (2014), Fig. 3.

The central sector (5) is the torso rectangle spanning both shoulders horizontally and running from
the shoulder line down to the hip line. The two vertical and two horizontal lines through that
rectangle split the space around the person into nine sectors, numbered from the person's own
viewpoint:

    1 2 3      1/4/7: person's right side, 3/6/9: person's left side
    4 5 6      1/2/3: above the shoulders,  7/8/9: below the hips
    7 8 9

A hand's location is the sector that contains its wrist keypoint.
"""

import heapq

import numpy as np
import pandas as pd

from .constants import HAND_WRIST_KEYPOINTS, HANDS, SEGMENT_COLUMNS
from .temporal import run_lengths

MIN_SHOULDER_WIDTH_PX = 2.0


class HandLocationClassifier:
    """Compute the per-frame 3x3 grid for every person and assign each wrist to a sector."""

    def classify(self, frames: pd.DataFrame, hip_ratio: float, orientation: str) -> pd.DataFrame:
        if frames.empty:
            return frames
        frames = frames.copy()
        frames = self._add_grid(frames, hip_ratio)
        mirrored = self._mirrored(frames, orientation)
        for hand in HANDS:
            wrist = HAND_WRIST_KEYPOINTS[hand]
            frames[f"{hand}_location_raw"] = self.sector(
                frames[f"{wrist}_x"].to_numpy(dtype=float),
                frames[f"{wrist}_y"].to_numpy(dtype=float),
                frames["grid_x_right"].to_numpy(dtype=float),
                frames["grid_x_left"].to_numpy(dtype=float),
                frames["grid_y_top"].to_numpy(dtype=float),
                frames["grid_y_bottom"].to_numpy(dtype=float),
                mirrored,
            )
        return frames

    @staticmethod
    def sector(
        wrist_x: np.ndarray,
        wrist_y: np.ndarray,
        x_right: np.ndarray,
        x_left: np.ndarray,
        y_top: np.ndarray,
        y_bottom: np.ndarray,
        mirrored: np.ndarray | bool = False,
    ) -> np.ndarray:
        """Return sector numbers 1-9 (NaN when the wrist or grid is missing).

        x_right / x_left are the image x coordinates of the grid lines on the person's right / left
        side. Points exactly on a line belong to the central column / row.
        """

        low_x = np.minimum(x_right, x_left)
        high_x = np.maximum(x_right, x_left)
        # Image column: 0 = image left, 1 = center, 2 = image right.
        image_col = np.where(wrist_x < low_x, 0, np.where(wrist_x > high_x, 2, 1))
        # Without mirroring the person's right side is the image left (camera facing the person).
        person_col = np.where(mirrored, 2 - image_col, image_col)
        row = np.where(wrist_y < y_top, 0, np.where(wrist_y > y_bottom, 2, 1))
        sector = (row * 3 + person_col + 1).astype(float)
        missing = np.isnan(wrist_x) | np.isnan(wrist_y) | np.isnan(low_x) | np.isnan(high_x) | np.isnan(y_top) | np.isnan(y_bottom)
        sector[missing] = np.nan
        return sector

    def _add_grid(self, frames: pd.DataFrame, hip_ratio: float) -> pd.DataFrame:
        right_x = frames["right_shoulder_x"].to_numpy(dtype=float)
        left_x = frames["left_shoulder_x"].to_numpy(dtype=float)
        both_shoulders = ~np.isnan(right_x) & ~np.isnan(left_x)
        shoulder_y = (frames["right_shoulder_y"].to_numpy(dtype=float) + frames["left_shoulder_y"].to_numpy(dtype=float)) / 2.0
        shoulder_width = np.abs(left_x - right_x)
        shoulder_width = np.where(shoulder_width >= MIN_SHOULDER_WIDTH_PX, shoulder_width, np.nan)

        right_hip_y = frames["right_hip_y"].to_numpy(dtype=float)
        left_hip_y = frames["left_hip_y"].to_numpy(dtype=float)
        hip_y = np.where(
            np.isnan(right_hip_y),
            left_hip_y,
            np.where(np.isnan(left_hip_y), right_hip_y, (right_hip_y + left_hip_y) / 2.0),
        )
        hip_y = np.where(hip_y > shoulder_y, hip_y, np.nan)

        # When hips are hidden (e.g. behind a table), estimate the hip line from the shoulder width
        # using the torso proportion observed for the same person, or hip_ratio as a fallback.
        observed_ratio = (hip_y - shoulder_y) / shoulder_width
        ratio_by_track = (
            pd.Series(observed_ratio).groupby(frames["track_id"].to_numpy()).median()
        )
        track_ratio = frames["track_id"].map(ratio_by_track).to_numpy(dtype=float)
        track_ratio = np.where(np.isnan(track_ratio), hip_ratio, track_ratio)
        estimated_hip_y = shoulder_y + track_ratio * shoulder_width
        hip_estimated = np.isnan(hip_y) & ~np.isnan(estimated_hip_y)

        frames["grid_x_right"] = np.where(both_shoulders, right_x, np.nan)
        frames["grid_x_left"] = np.where(both_shoulders, left_x, np.nan)
        frames["grid_y_top"] = shoulder_y
        frames["grid_y_bottom"] = np.where(np.isnan(hip_y), estimated_hip_y, hip_y)
        frames["hip_estimated"] = hip_estimated
        return frames

    def _mirrored(self, frames: pd.DataFrame, orientation: str) -> np.ndarray:
        if orientation == "auto":
            # The person faces away from the camera when their right shoulder is on the image right.
            return (frames["grid_x_right"] > frames["grid_x_left"]).to_numpy()
        return np.zeros(len(frames), dtype=bool)


class SegmentBuilder:
    """Turn per-frame sector labels into time segments."""

    def build(self, frames: pd.DataFrame, fps: float, min_segment_seconds: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (frames with final `<hand>_location` columns, segment table)."""

        if frames.empty:
            return frames, pd.DataFrame(columns=SEGMENT_COLUMNS)
        frames = frames.copy()
        min_frames = int(round(min_segment_seconds * fps))
        segments: list[dict] = []
        for hand in HANDS:
            frames[f"{hand}_location"] = np.nan
        for track_id, group in frames.groupby("track_id", sort=True):
            group = group.sort_values("frame_idx")
            frame_idx = group["frame_idx"].to_numpy(dtype=int)
            for hand in HANDS:
                labels = self._dense_labels(frame_idx, group[f"{hand}_location_raw"].to_numpy(dtype=float))
                merged = merge_short_runs(labels, min_frames)
                start_frame = int(frame_idx[0])
                frames.loc[group.index, f"{hand}_location"] = merged[frame_idx - start_frame]
                for start, length, value in run_lengths(merged):
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
            segment_df = segment_df.reset_index(drop=True)
        frames["right_location"] = frames["right_location"].astype("Int64")
        frames["left_location"] = frames["left_location"].astype("Int64")
        return frames, segment_df

    @staticmethod
    def _dense_labels(frame_idx: np.ndarray, labels: np.ndarray) -> np.ndarray:
        dense = np.full(int(frame_idx[-1] - frame_idx[0] + 1), np.nan)
        dense[frame_idx - frame_idx[0]] = labels
        return dense


def merge_short_runs(labels: np.ndarray, min_frames: int) -> np.ndarray:
    """Absorb labelled runs shorter than min_frames into neighbouring labelled runs.

    Runs are processed from the shortest one. A short run between two runs of the same label is
    merged into them; otherwise it joins the longer adjacent labelled run. A short run isolated
    between missing frames is kept as is.
    """

    labels = np.asarray(labels, dtype=float).copy()
    if min_frames <= 1 or len(labels) == 0:
        return labels
    runs = run_lengths(labels)
    starts = [run[0] for run in runs]
    lengths = [run[1] for run in runs]
    values = [float(run[2]) for run in runs]
    prev = list(range(-1, len(runs) - 1))
    nxt = list(range(1, len(runs) + 1))
    nxt[-1] = -1
    alive = [True] * len(runs)
    heap = [(lengths[i], i) for i in range(len(runs)) if not np.isnan(values[i]) and lengths[i] < min_frames]
    heapq.heapify(heap)

    def labelled(index: int) -> bool:
        return index >= 0 and not np.isnan(values[index])

    def absorb(keep: int, drop: int) -> None:
        # Merge run `drop` into its adjacent run `keep`.
        if starts[drop] < starts[keep]:
            starts[keep] = starts[drop]
            prev[keep] = prev[drop]
            if prev[drop] >= 0:
                nxt[prev[drop]] = keep
        else:
            nxt[keep] = nxt[drop]
            if nxt[drop] >= 0:
                prev[nxt[drop]] = keep
        lengths[keep] += lengths[drop]
        alive[drop] = False

    while heap:
        length, index = heapq.heappop(heap)
        if not alive[index] or length != lengths[index] or length >= min_frames:
            continue
        before, after = prev[index], nxt[index]
        if not labelled(before) and not labelled(after):
            continue
        if labelled(before) and labelled(after):
            if values[before] == values[after] or lengths[before] >= lengths[after]:
                target = before
            else:
                target = after
        else:
            target = before if labelled(before) else after
        values[index] = values[target]
        absorb(target, index)
        # Coalesce with the run on the other side when it now carries the same label.
        for neighbour in (prev[target], nxt[target]):
            if labelled(neighbour) and values[neighbour] == values[target]:
                absorb(target, neighbour)
        if lengths[target] < min_frames:
            heapq.heappush(heap, (lengths[target], target))

    for index in range(len(runs)):
        if alive[index]:
            labels[starts[index] : starts[index] + lengths[index]] = values[index]
    return labels
