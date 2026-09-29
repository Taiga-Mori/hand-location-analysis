import numpy as np
import pandas as pd

from handloc.constants import KEYPOINT_NAMES
from handloc.locations import HandLocationClassifier, SegmentBuilder, merge_short_runs
from handloc.temporal import interpolate_short_gaps

NAN = np.nan


def test_sector_numbering_frontal():
    # Person's right shoulder at image x=100, left shoulder at x=200; shoulders y=100, hips y=300.
    xs = np.array([50, 150, 250, 50, 150, 250, 50, 150, 250], dtype=float)
    ys = np.array([50, 50, 50, 200, 200, 200, 400, 400, 400], dtype=float)
    ones = np.ones_like(xs)
    sectors = HandLocationClassifier.sector(xs, ys, 100 * ones, 200 * ones, 100 * ones, 300 * ones)
    assert sectors.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_sector_mirrored_and_boundaries():
    sectors = HandLocationClassifier.sector(
        np.array([50.0, 100.0, 200.0, NAN]),
        np.array([200.0, 100.0, 300.0, 200.0]),
        np.array([200.0, 100.0, 100.0, 100.0]),
        np.array([100.0, 200.0, 200.0, 200.0]),
        np.full(4, 100.0),
        np.full(4, 300.0),
        np.array([True, False, False, False]),
    )
    # Mirrored: image-left of the torso is the person's left side (6). Lines belong to the center.
    assert sectors[:3].tolist() == [6, 5, 5]
    assert np.isnan(sectors[3])


def test_merge_short_runs():
    labels = np.array([5, 5, 5, 5, 4, 5, 5, 5, 6, 6, 6, 6, NAN, NAN, 2, NAN], dtype=float)
    merged = merge_short_runs(labels, 3)
    assert merged[:12].tolist() == [5] * 8 + [6] * 4
    assert np.isnan(merged[12]) and np.isnan(merged[13])
    assert merged[14] == 2  # isolated short run is kept


def test_merge_short_runs_joins_longer_neighbour():
    labels = np.array([4, 4, 4, 4, 4, 1, 6, 6, 6], dtype=float)
    assert merge_short_runs(labels, 2).tolist() == [4] * 6 + [6] * 3


def test_interpolate_short_gaps():
    values = np.array([0, NAN, 2, NAN, NAN, NAN, 6, NAN], dtype=float)
    filled = interpolate_short_gaps(values, 2)
    assert filled[1] == 1
    assert np.isnan(filled[3:6]).all()
    assert np.isnan(filled[7])


def _frames(n: int, wrist_x: list[float], hips_visible: bool) -> pd.DataFrame:
    data = {"frame_idx": np.arange(n), "track_id": 1, "x1": 0.0, "y1": 0.0, "x2": 400.0, "y2": 500.0}
    for name in KEYPOINT_NAMES:
        data[f"{name}_x"] = NAN
        data[f"{name}_y"] = NAN
        data[f"{name}_conf"] = 0.0
    frame = pd.DataFrame(data)
    frame["right_shoulder_x"], frame["left_shoulder_x"] = 100.0, 200.0
    frame["right_shoulder_y"] = frame["left_shoulder_y"] = 100.0
    if hips_visible:
        frame["right_hip_y"] = frame["left_hip_y"] = 250.0
    frame["right_wrist_x"] = wrist_x
    frame["right_wrist_y"] = 200.0
    return frame


def test_hip_fallback_uses_ratio_and_segments():
    frames = _frames(6, [50, 50, 150, 150, 150, 250], hips_visible=False)
    classified = HandLocationClassifier().classify(frames, hip_ratio=1.5, orientation="frontal")
    assert classified["grid_y_bottom"].iloc[0] == 250.0
    assert classified["hip_estimated"].all()
    _, segments = SegmentBuilder().build(classified, fps=10.0, min_segment_seconds=0.0)
    right = segments[segments["hand"] == "right"]
    assert right[["startTime", "endTime", "location"]].values.tolist() == [[0.0, 0.2, 4], [0.2, 0.5, 5], [0.5, 0.6, 6]]
    assert segments[segments["hand"] == "left"].empty
