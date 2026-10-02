import numpy as np
import pandas as pd

from handloc.constants import KEYPOINT_NAMES, POSE_COLUMNS
from handloc.locations import HandLocationClassifier, SegmentBuilder
from handloc.temporal import PoseInterpolator, fill_series

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
        np.array([50.0, 100.0, 200.0]),
        np.array([200.0, 100.0, 300.0]),
        np.array([200.0, 100.0, 100.0]),
        np.array([100.0, 200.0, 200.0]),
        np.full(3, 100.0),
        np.full(3, 300.0),
    )
    # First line (1/4/7 side) on the image right: image-left of the rectangle is 6. Lines belong to the center.
    assert sectors.tolist() == [6, 5, 5]


def test_fill_series_interpolates_and_holds_ends():
    filled = fill_series(np.array([2.0, 6.0]), np.array([10.0, 30.0]), 9)
    assert filled.tolist() == [10, 10, 10, 15, 20, 25, 30, 30, 30]


def _pose(frame_idx: int, wrist_x: float, conf: float = 0.9) -> dict:
    row = {"frame_idx": frame_idx, "track_id": 1, "conf": 0.9, "x1": 0.0, "y1": 0.0, "x2": 400.0, "y2": 500.0}
    for name in KEYPOINT_NAMES:
        row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_conf"] = 0.0, 0.0, 0.9
    row.update(right_shoulder_x=100.0, left_shoulder_x=200.0, right_shoulder_y=100.0, left_shoulder_y=100.0)
    row.update(right_hip_x=110.0, left_hip_x=190.0, right_hip_y=300.0, left_hip_y=300.0)
    row.update(right_wrist_x=wrist_x, right_wrist_y=200.0, right_wrist_conf=conf, left_wrist_x=150.0, left_wrist_y=200.0)
    return row


def test_interpolate_classify_and_threshold():
    # Detected at frames 1, 3, 5 of 7; the wrist at frame 3 has low confidence.
    poses = pd.DataFrame([_pose(1, 50.0), _pose(3, 150.0, conf=0.1), _pose(5, 250.0)], columns=POSE_COLUMNS)
    frames = PoseInterpolator().interpolate(poses, total_frames=7)
    assert frames["frame_idx"].tolist() == list(range(7))
    assert frames["detected"].tolist() == [False, True, False, True, False, True, False]
    # Held before the first and after the last detection, linear in between, confidence included.
    assert frames["right_wrist_x"].tolist() == [50, 50, 100, 150, 200, 250, 250]
    assert np.allclose(frames["right_wrist_conf"], [0.9, 0.9, 0.5, 0.1, 0.5, 0.9, 0.9])

    every = HandLocationClassifier().classify(frames, conf_thresh=0.0, orientation="frontal")
    assert every["right_location"].tolist() == [4, 4, 5, 5, 5, 6, 6]
    assert np.allclose(every["right_x_norm"], [-0.5, -0.5, 0.0, 0.5, 1.0, 1.5, 1.5])
    assert np.allclose(every["right_y_norm"], 0.5)

    strict = HandLocationClassifier().classify(frames, conf_thresh=0.6, orientation="frontal")
    assert strict["right_valid"].tolist() == [True, True, False, False, False, True, True]
    assert strict["left_valid"].all()
    segments = SegmentBuilder().build(strict, fps=10.0)
    right = segments[segments["hand"] == "right"]
    assert right[["startTime", "endTime", "location"]].values.tolist() == [[0.0, 0.2, 4], [0.5, 0.7, 6]]


def test_grid_keypoint_below_threshold_excludes_both_hands():
    low_hip = _pose(0, 150.0)
    low_hip["left_hip_conf"] = 0.2
    frames = PoseInterpolator().interpolate(pd.DataFrame([low_hip], columns=POSE_COLUMNS), total_frames=2)
    classified = HandLocationClassifier().classify(frames, conf_thresh=0.5, orientation="frontal")
    assert not classified["right_valid"].any() and not classified["left_valid"].any()
    assert SegmentBuilder().build(classified, fps=10.0).empty


def test_fill_stacks_per_wrist():
    from handloc.annotate import FILL_ALPHA, FrameAnnotator

    annotator = FrameAnnotator(400, 400)
    edges = ([0.0, 100.0, 200.0, 300.0], [0.0, 100.0, 200.0, 300.0])
    once = np.zeros((400, 400, 3), np.uint8)
    annotator._fill_sector(once, edges, 5, False, (200, 200, 200))
    twice = once.copy()
    annotator._fill_sector(twice, edges, 5, False, (200, 200, 200))
    assert once[150, 150, 0] == int(200 * FILL_ALPHA)
    assert twice[150, 150, 0] > once[150, 150, 0]
    assert once[50, 50, 0] == 0  # other sectors untouched


def _side_pose(frame_idx: int, wrist_x: float, wrist_y: float, facing_left: bool) -> dict:
    """Seated person in profile: shoulders x=500, hips x=520 (body line 510), knees 200 px in front."""

    row = _pose(frame_idx, 0.0)
    sign = -1 if facing_left else 1
    for side in ("right", "left"):
        row.update({f"{side}_shoulder_x": 500.0, f"{side}_shoulder_y": 100.0, f"{side}_hip_x": 520.0, f"{side}_hip_y": 300.0})
        row.update({f"{side}_knee_x": 510.0 + sign * 200.0, f"{side}_knee_y": 340.0})
    row.update(right_wrist_x=wrist_x, right_wrist_y=wrist_y)
    return row


def test_side_mode_grid_and_front_columns():
    # Facing image left: knees at x=310, body line at 510, thigh line at (300 + 340) / 2 = 320.
    poses = pd.DataFrame(
        [
            _side_pose(0, 250.0, 200.0, facing_left=True),  # beyond the knees -> front column (4)
            _side_pose(1, 400.0, 200.0, facing_left=True),  # between knees and body -> 5
            _side_pose(2, 600.0, 50.0, facing_left=True),  # behind the body, above the shoulders -> 3
        ],
        columns=POSE_COLUMNS,
    )
    frames = PoseInterpolator().interpolate(poses, total_frames=3)
    classified = HandLocationClassifier().classify(frames, conf_thresh=0.0, mode="side")
    assert classified["right_location"].tolist() == [4, 5, 3]
    assert np.allclose(classified["grid_y_bottom"], 320.0)
    # x: body line = 0, knee line = 1 (forward positive); y: shoulder line = 0, thigh line = 1.
    assert np.allclose(classified["right_x_norm"], [1.3, 0.55, -0.45])
    assert np.allclose(classified["right_y_norm"], [100 / 220, 100 / 220, -50 / 220])

    facing_right = pd.DataFrame([_side_pose(0, 770.0, 200.0, facing_left=False)], columns=POSE_COLUMNS)
    classified = HandLocationClassifier().classify(PoseInterpolator().interpolate(facing_right, 1), 0.0, mode="side")
    assert classified["right_location"].tolist() == [4]  # front is still 1/4/7
