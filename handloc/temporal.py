import numpy as np
import pandas as pd

from .constants import BBOX_COLUMNS, KEYPOINT_NAMES


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


def interpolate_short_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill interior NaN runs no longer than max_gap; longer gaps stay NaN."""

    values = np.asarray(values, dtype=float)
    missing = np.isnan(values)
    if not missing.any() or missing.all():
        return values.copy()
    indices = np.arange(len(values))
    filled = np.interp(indices, indices[~missing], values[~missing])
    keep_missing = np.zeros(len(values), dtype=bool)
    for start, length, is_missing in run_lengths(missing):
        interior = start > 0 and start + length < len(values)
        if is_missing and (not interior or length > max_gap):
            keep_missing[start : start + length] = True
    filled[keep_missing] = np.nan
    return filled


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average that never fills NaN positions."""

    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    smoothed = pd.Series(values).rolling(window=window, min_periods=1, center=True).mean().to_numpy()
    smoothed[np.isnan(values)] = np.nan
    return smoothed


class PoseSmoother:
    """Densify each track to every frame, fill short gaps, and smooth keypoints."""

    def smooth(
        self,
        poses: pd.DataFrame,
        fps: float,
        stride: int,
        keypoint_conf_thresh: float,
        max_gap_seconds: float,
        min_track_seconds: float,
        window: int,
    ) -> pd.DataFrame:
        if poses.empty:
            return pd.DataFrame()

        max_gap_frames = int(round(max_gap_seconds * fps)) + stride - 1
        min_track_frames = int(round(min_track_seconds * fps))
        tracks: list[pd.DataFrame] = []
        for track_id, group in poses.groupby("track_id", sort=True):
            group = group.sort_values(["frame_idx", "conf"], ascending=[True, False]).drop_duplicates("frame_idx")
            if len(group) * stride < min_track_frames:
                continue
            frame_range = np.arange(int(group["frame_idx"].min()), int(group["frame_idx"].max()) + 1)
            dense = group.set_index("frame_idx").reindex(frame_range)
            output = pd.DataFrame({"frame_idx": frame_range, "track_id": int(track_id)})
            output["time"] = output["frame_idx"] / fps
            output["detected"] = dense["conf"].notna().to_numpy()

            for column in BBOX_COLUMNS:
                output[column] = smooth_series(interpolate_short_gaps(dense[column].to_numpy(), max_gap_frames), window)
            for name in KEYPOINT_NAMES:
                conf = dense[f"{name}_conf"].to_numpy(dtype=float)
                reliable = conf >= keypoint_conf_thresh
                for axis in ("x", "y"):
                    raw = dense[f"{name}_{axis}"].to_numpy(dtype=float)
                    raw = np.where(reliable, raw, np.nan)
                    output[f"{name}_{axis}"] = smooth_series(interpolate_short_gaps(raw, max_gap_frames), window)
                output[f"{name}_conf"] = conf
            output = output[output[BBOX_COLUMNS].notna().all(axis=1)]
            tracks.append(output)

        if not tracks:
            return pd.DataFrame()
        return pd.concat(tracks, ignore_index=True).sort_values(["frame_idx", "track_id"]).reset_index(drop=True)
