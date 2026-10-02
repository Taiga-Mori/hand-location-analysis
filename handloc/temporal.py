import numpy as np
import pandas as pd

from .constants import BBOX_COLUMNS, KEYPOINT_COLUMNS


def run_lengths(values: np.ndarray) -> list[tuple[int, int, object]]:
    """Return (start, length, value) runs. NaN / None values form their own runs."""

    runs: list[tuple[int, int, object]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or not _same(values[index], values[start]):
            runs.append((start, index - start, values[start]))
            start = index
    return runs


def _same(a, b) -> bool:
    if _is_missing(a) and _is_missing(b):
        return True
    return a == b


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def fill_series(frame_idx: np.ndarray, values: np.ndarray, total_frames: int) -> np.ndarray:
    """Values for every frame 0..total_frames-1: linear between samples, held constant before/after them."""

    order = np.argsort(frame_idx)
    return np.interp(np.arange(total_frames), frame_idx[order], values[order])


class PoseInterpolator:
    """Fill every track over the whole video, whatever the gap length or the keypoint confidence."""

    def interpolate(self, poses: pd.DataFrame, total_frames: int) -> pd.DataFrame:
        """One row per frame and track with bbox, keypoint x / y / conf, and whether YOLO saw the track."""

        if poses.empty:
            return pd.DataFrame()
        tracks: list[pd.DataFrame] = []
        for track_id, group in poses.groupby("track_id", sort=True):
            group = group.sort_values(["frame_idx", "conf"], ascending=[True, False]).drop_duplicates("frame_idx")
            frame_idx = group["frame_idx"].to_numpy(dtype=float)
            output = pd.DataFrame({"frame_idx": np.arange(total_frames), "track_id": int(track_id)})
            output["detected"] = output["frame_idx"].isin(group["frame_idx"])
            for column in [*BBOX_COLUMNS, *KEYPOINT_COLUMNS]:
                output[column] = fill_series(frame_idx, group[column].to_numpy(dtype=float), total_frames)
            tracks.append(output)
        return pd.concat(tracks, ignore_index=True).sort_values(["frame_idx", "track_id"]).reset_index(drop=True)
