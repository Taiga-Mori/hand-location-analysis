import shutil
from pathlib import Path
from typing import Any

import cv2
import yaml

from .constants import MODES, ORIENTATION_MODES, POSE_MODELS, VIDEO_EXTENSIONS
from .types import AppPaths, MediaContext, PipelineConfig


class DeviceManager:
    """Select the inference device and expose supported options."""

    def __init__(self) -> None:
        import torch

        self.device_options: list[str] = []
        if torch.cuda.is_available():
            self.default_device = "cuda:0"
            self.device_options.extend([f"cuda:{idx}" for idx in range(int(torch.cuda.device_count()))])
        elif torch.backends.mps.is_available():
            self.default_device = "mps"
            self.device_options.append("mps")
        else:
            self.default_device = "cpu"
        self.device_options.append("cpu")

    def resolve(self, requested_device: str | None) -> str:
        device = requested_device or self.default_device
        if device == "cuda":
            device = "cuda:0"
        if device not in self.device_options:
            raise ValueError(f"Unsupported device '{device}'. Available devices: {', '.join(self.device_options)}")
        return device


class ConfigManager:
    """Normalize config inputs and prepare runtime files."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths

    def build_config(
        self,
        input_path: Path | str,
        output_dir: Path | str | None,
        device: str,
        pose_model: str,
        person_det_thresh: float,
        keypoint_conf_thresh: float,
        person_target_fps: float | None,
        tracker_updates: dict[str, Any] | None,
        mode: str,
        orientation: str,
        make_video: bool,
        reuse_cached_poses: bool,
    ) -> PipelineConfig:
        media_path = Path(input_path).expanduser()
        if not media_path.exists():
            raise FileNotFoundError(f"Input file not found: {media_path}")
        if media_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported input type for {media_path}. Supported: {', '.join(sorted(VIDEO_EXTENSIONS))}")
        if pose_model not in POSE_MODELS:
            raise ValueError(f"pose_model must be one of {POSE_MODELS}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if orientation not in ORIENTATION_MODES:
            raise ValueError(f"orientation must be one of {ORIENTATION_MODES}")
        if not 0.0 <= person_det_thresh <= 1.0:
            raise ValueError("person_det_thresh must be between 0.0 and 1.0")
        if not 0.0 <= keypoint_conf_thresh <= 1.0:
            raise ValueError("keypoint_conf_thresh must be between 0.0 and 1.0")
        if person_target_fps is not None and person_target_fps < 0:
            raise ValueError("person_target_fps must be greater than or equal to 0")

        return PipelineConfig(
            media_path=media_path,
            output_dir=Path(output_dir).expanduser() if output_dir else self.default_output_dir(media_path),
            device=device,
            pose_model=pose_model,
            person_det_thresh=float(person_det_thresh),
            keypoint_conf_thresh=float(keypoint_conf_thresh),
            person_target_fps=float(person_target_fps or 0.0),
            tracker_updates=dict(tracker_updates or {}),
            mode=mode,
            orientation=orientation,
            make_video=bool(make_video),
            reuse_cached_poses=bool(reuse_cached_poses),
        )

    @staticmethod
    def default_output_dir(media_path: Path) -> Path:
        return media_path.parent / media_path.stem

    def load_tracker_defaults(self) -> dict[str, Any]:
        with self.paths.botsort_template_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def prepare_tracker_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        shutil.copy(self.paths.botsort_template_path, self.paths.botsort_runtime_path)
        config = self.load_tracker_defaults()
        if updates:
            config.update(updates)
            with self.paths.botsort_runtime_path.open("w", encoding="utf-8") as file:
                yaml.safe_dump(config, file, sort_keys=False)
        return config

    def build_media_context(self, config: PipelineConfig) -> MediaContext:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(config.media_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {config.media_path}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        # Tracks are filled over the whole video, so count the frames that can actually be decoded instead
        # of trusting CAP_PROP_FRAME_COUNT, which can be wrong (e.g. 251 reported, 237 readable).
        total_frames = self.count_readable_frames(config.media_path)
        if fps <= 0 or total_frames <= 0:
            raise RuntimeError(f"Invalid video metadata for {config.media_path}")

        person_target_fps = config.person_target_fps or fps
        person_stride = max(1, round(fps / person_target_fps))
        output_dir = config.output_dir
        return MediaContext(
            media_path=config.media_path,
            output_dir=output_dir,
            poses_path=output_dir / "poses.csv",
            frames_path=output_dir / "frames.csv",
            locations_path=output_dir / "locations.csv",
            video_path=output_dir / "locations.mp4",
            summary_path=output_dir / "summary.json",
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            person_stride=person_stride,
        )

    @staticmethod
    def count_readable_frames(media_path: Path) -> int:
        capture = cv2.VideoCapture(str(media_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {media_path}")
        total = 0
        try:
            while capture.grab():
                total += 1
        finally:
            capture.release()
        return total
