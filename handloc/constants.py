KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

HANDS = ["right", "left"]
HAND_WRIST_KEYPOINTS = {"right": "right_wrist", "left": "left_wrist"}

POSE_MODELS = ["yolo26n-pose", "yolo26s-pose", "yolo26m-pose", "yolo26l-pose", "yolo26x-pose"]
DEFAULT_POSE_MODEL = "yolo26x-pose"

# "frontal": the camera faces the person, so the person's right side is on the image left.
# "auto": decide the person's right side per frame from the shoulder keypoint order.
# "frontal": camera facing the person (shoulder/hip grid). "side": camera at the person's side (body/knee grid).
MODES = ["frontal", "side"]
DEFAULT_MODE = "frontal"

ORIENTATION_MODES = ["frontal", "auto"]
DEFAULT_ORIENTATION = "frontal"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".webm"}

BBOX_COLUMNS = ["x1", "y1", "x2", "y2"]
KEYPOINT_COLUMNS = [f"{name}_{axis}" for name in KEYPOINT_NAMES for axis in ("x", "y", "conf")]
POSE_COLUMNS = ["frame_idx", "track_id", "conf", *BBOX_COLUMNS, *KEYPOINT_COLUMNS]
FRAME_COLUMNS = ["frame_idx", "track_id", "hand", "x", "y"]
SEGMENT_COLUMNS = ["track_id", "hand", "startTime", "endTime", "location"]
