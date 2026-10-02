from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AppPaths:
    """Resolved filesystem paths used by the pipeline."""

    working_dir: Path
    app_dir: Path
    botsort_template_path: Path
    botsort_runtime_path: Path
    ffmpeg_path: Path

    def pose_model_path(self, backend: str) -> Path:
        return self.app_dir / f"{backend}.pt"


@dataclass
class PipelineConfig:
    """Runtime settings for one pipeline run."""

    media_path: Path
    output_dir: Path
    device: str
    pose_model: str
    person_det_thresh: float
    keypoint_conf_thresh: float
    person_target_fps: float
    tracker_updates: dict[str, Any]
    mode: str
    orientation: str
    make_video: bool
    reuse_cached_poses: bool


@dataclass
class MediaContext:
    """Derived media metadata and output paths for one run."""

    media_path: Path
    output_dir: Path
    poses_path: Path
    frames_path: Path
    locations_path: Path
    video_path: Path
    summary_path: Path
    fps: float
    total_frames: int
    width: int
    height: int
    person_stride: int
