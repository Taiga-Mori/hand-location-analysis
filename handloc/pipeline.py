import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .config import ConfigManager, DeviceManager
from .constants import DEFAULT_ORIENTATION, DEFAULT_POSE_MODEL, HANDS
from .exporters import AnnotationExporter
from .locations import HandLocationClassifier, SegmentBuilder
from .models import ModelManager
from .paths import PathManager
from .progress import finish_progress, format_elapsed, reset_progress_timers, update_progress
from .temporal import PoseSmoother
from .tracking import PoseTracker
from .types import MediaContext, PipelineConfig


class HandLocationAnalyzer:
    """Facade over the hand location annotation pipeline."""

    def __init__(self) -> None:
        self.paths = PathManager().get()
        self.device_manager = DeviceManager()
        self.device_options = self.device_manager.device_options
        self.config_manager = ConfigManager(self.paths)
        self.model_manager = ModelManager(self.paths)
        self.pose_tracker = PoseTracker(self.model_manager, self.paths)
        self.smoother = PoseSmoother()
        self.classifier = HandLocationClassifier()
        self.segment_builder = SegmentBuilder()
        self.exporter = AnnotationExporter(self.paths)
        self.config: PipelineConfig | None = None
        self.context: MediaContext | None = None
        self.tracker_config: dict[str, Any] = {}

    def preprocess(
        self,
        input_path: Path | str,
        output_dir: Path | str | None = None,
        device: str | None = None,
        pose_model: str = DEFAULT_POSE_MODEL,
        person_det_thresh: float = 0.5,
        keypoint_conf_thresh: float = 0.5,
        person_target_fps: float | None = None,
        tracker_updates: dict[str, Any] | None = None,
        smoothing_window: int = 5,
        max_gap_seconds: float = 0.5,
        min_track_seconds: float = 1.0,
        min_segment_seconds: float = 0.2,
        hip_ratio: float = 1.3,
        orientation: str = DEFAULT_ORIENTATION,
        make_video: bool = True,
        reuse_cached_poses: bool = True,
    ) -> None:
        self.config = self.config_manager.build_config(
            input_path=input_path,
            output_dir=output_dir,
            device=self.device_manager.resolve(device),
            pose_model=pose_model,
            person_det_thresh=person_det_thresh,
            keypoint_conf_thresh=keypoint_conf_thresh,
            person_target_fps=person_target_fps,
            tracker_updates=tracker_updates,
            smoothing_window=smoothing_window,
            max_gap_seconds=max_gap_seconds,
            min_track_seconds=min_track_seconds,
            min_segment_seconds=min_segment_seconds,
            hip_ratio=hip_ratio,
            orientation=orientation,
            make_video=make_video,
            reuse_cached_poses=reuse_cached_poses,
        )
        self.tracker_config = self.config_manager.prepare_tracker_config(self.config.tracker_updates)
        self.context = self.config_manager.build_media_context(self.config)

    def det_poses(self, progress_bar=None) -> pd.DataFrame:
        config, context = self._require_config(), self._require_context()
        if config.reuse_cached_poses:
            cached = self.pose_tracker.load_cached(context, config.pose_model, config.person_det_thresh, self.tracker_config)
            if cached is not None:
                self._notify_skip(progress_bar, "Skipping person tracking: reusing cached poses.csv.")
                return cached
        return self.pose_tracker.detect(
            context=context,
            device=config.device,
            backend=config.pose_model,
            det_thresh=config.person_det_thresh,
            tracker_config=self.tracker_config,
            progress_bar=progress_bar,
        )

    def classify(self, poses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (per-frame table, segment table) and write both CSV files."""

        config, context = self._require_config(), self._require_context()
        frames = self.smoother.smooth(
            poses,
            fps=context.fps,
            stride=context.person_stride,
            keypoint_conf_thresh=config.keypoint_conf_thresh,
            max_gap_seconds=config.max_gap_seconds,
            min_track_seconds=config.min_track_seconds,
            window=config.smoothing_window,
        )
        frames = self.classifier.classify(frames, config.hip_ratio, config.orientation)
        frames, segments = self.segment_builder.build(frames, context.fps, config.min_segment_seconds)
        self.exporter.write_frames_csv(frames, context)
        self.exporter.write_segments_csv(segments, context)
        return frames, segments

    def make_video(self, frames: pd.DataFrame, progress_bar=None) -> Path:
        return self.exporter.make_video(frames, self._require_context(), self._require_config().orientation, progress_bar)

    def run_all(self, progress_bar=None) -> dict[str, Any]:
        reset_progress_timers()
        phase_seconds: dict[str, float] = {}

        def _timed(label: str, func: Callable[[], Any]) -> Any:
            start = time.monotonic()
            result = func()
            phase_seconds[label] = time.monotonic() - start
            return result

        poses = _timed("Person tracking", lambda: self.det_poses(progress_bar=progress_bar))
        update_progress(progress_bar, 0, 1, "Classifying hand locations...")
        frames, segments = _timed("Hand location classification", lambda: self.classify(poses))
        update_progress(progress_bar, 1, 1, "Classifying hand locations...")
        video_path = None
        if self._require_config().make_video:
            video_path = _timed("Video rendering", lambda: self.make_video(frames, progress_bar=progress_bar))
        elapsed_seconds = finish_progress(progress_bar)
        summary = self._build_summary(frames, segments, phase_seconds, elapsed_seconds)
        context = self._require_context()
        return {
            "poses": poses,
            "frames": frames,
            "segments": segments,
            "segments_path": context.segments_path,
            "frames_path": context.frames_path,
            "poses_path": context.poses_path,
            "video_path": video_path,
            "elapsed_seconds": elapsed_seconds,
            "summary": summary,
        }

    def _build_summary(
        self,
        frames: pd.DataFrame,
        segments: pd.DataFrame,
        phase_seconds: dict[str, float],
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        context, config = self._require_context(), self._require_config()
        persons: list[dict[str, Any]] = []
        if not frames.empty:
            for track_id, group in frames.groupby("track_id", sort=True):
                person: dict[str, Any] = {
                    "track_id": int(track_id),
                    "start_time": round(float(group["frame_idx"].min()) / context.fps, 3),
                    "end_time": round(float(group["frame_idx"].max() + 1) / context.fps, 3),
                    "hip_estimated_ratio": round(float(group["hip_estimated"].mean()), 3),
                }
                for hand in HANDS:
                    hand_segments = segments[(segments["track_id"] == track_id) & (segments["hand"] == hand)]
                    durations = (hand_segments["endTime"] - hand_segments["startTime"]).groupby(hand_segments["location"]).sum()
                    person[f"{hand}_seconds_by_location"] = {str(int(k)): round(float(v), 3) for k, v in durations.items()}
                persons.append(person)
        summary = {
            "input": str(context.media_path),
            "fps": context.fps,
            "total_frames": context.total_frames,
            "pose_model": config.pose_model,
            "device": config.device,
            "total_elapsed_seconds": round(elapsed_seconds, 2),
            "phase_seconds": {label: round(seconds, 2) for label, seconds in phase_seconds.items()},
            "person_count": len(persons),
            "segment_count": int(len(segments)),
            "persons": persons,
            "settings": {
                "person_det_thresh": config.person_det_thresh,
                "keypoint_conf_thresh": config.keypoint_conf_thresh,
                "person_stride": context.person_stride,
                "smoothing_window": config.smoothing_window,
                "max_gap_seconds": config.max_gap_seconds,
                "min_track_seconds": config.min_track_seconds,
                "min_segment_seconds": config.min_segment_seconds,
                "hip_ratio": config.hip_ratio,
                "orientation": config.orientation,
                "tracker": self.tracker_config,
            },
        }
        context.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        self._print_summary(summary)
        return summary

    def _print_summary(self, summary: dict[str, Any]) -> None:
        print("\n" + " Pipeline Summary ".center(72, "="), flush=True)
        print(f"  Total elapsed: {format_elapsed(summary['total_elapsed_seconds'])}", flush=True)
        for label, seconds in summary["phase_seconds"].items():
            print(f"    - {label}: {format_elapsed(seconds)}", flush=True)
        print(f"  Tracked persons: {summary['person_count']}, segments: {summary['segment_count']}", flush=True)
        for person in summary["persons"]:
            print(f"  - track {person['track_id']} ({person['start_time']:.1f}s-{person['end_time']:.1f}s)", flush=True)
            for hand in HANDS:
                by_location = person[f"{hand}_seconds_by_location"]
                text = ", ".join(f"{loc}: {sec:.1f}s" for loc, sec in sorted(by_location.items())) or "no data"
                print(f"      {hand:>5}: {text}", flush=True)
        print("=" * 72, flush=True)

    def _require_config(self) -> PipelineConfig:
        if self.config is None:
            raise RuntimeError("Call preprocess() before running this step.")
        return self.config

    def _require_context(self) -> MediaContext:
        if self.context is None:
            raise RuntimeError("Call preprocess() before running this step.")
        return self.context

    def _notify_skip(self, progress_bar, message: str) -> None:
        if progress_bar is not None and hasattr(progress_bar, "progress"):
            progress_bar.progress(0.0, text=message)
        print(message, flush=True)
