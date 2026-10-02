import json
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

from .config import ConfigManager, DeviceManager
from .constants import DEFAULT_MODE, DEFAULT_ORIENTATION, DEFAULT_POSE_MODEL, HANDS
from .exporters import AnnotationExporter
from .locations import HandLocationClassifier, SegmentBuilder
from .models import ModelManager
from .paths import PathManager
from .progress import finish_progress, format_elapsed, reset_progress_timers, update_progress
from .temporal import PoseInterpolator
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
        self.interpolator = PoseInterpolator()
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
        keypoint_conf_thresh: float = 0.0,
        person_target_fps: float | None = None,
        tracker_updates: dict[str, Any] | None = None,
        mode: str = DEFAULT_MODE,
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
            mode=mode,
            orientation=orientation,
            make_video=make_video,
            reuse_cached_poses=reuse_cached_poses,
        )
        self.tracker_config = self.config_manager.prepare_tracker_config(self.config.tracker_updates)
        self.context = self.config_manager.build_media_context(self.config)

    def yolo_meta(self) -> dict[str, Any]:
        config, context = self._require_config(), self._require_context()
        return self.pose_tracker.meta(context, config.pose_model, config.person_det_thresh, self.tracker_config)

    def det_poses(self, progress_bar=None) -> pd.DataFrame:
        """Raw YOLO poses: reused from poses.csv when summary.json records the same YOLO settings."""

        config, context = self._require_config(), self._require_context()
        if config.reuse_cached_poses:
            cached = self.pose_tracker.load_cached(context, config.pose_model, config.person_det_thresh, self.tracker_config)
            if cached is not None:
                self._notify_skip(progress_bar, "Skipping person tracking: reusing poses.csv (same YOLO settings in summary.json).")
                return cached
        poses = self.pose_tracker.detect(
            context=context,
            device=config.device,
            backend=config.pose_model,
            det_thresh=config.person_det_thresh,
            tracker_config=self.tracker_config,
            progress_bar=progress_bar,
        )
        # Record the YOLO settings right away, so a later failure does not force YOLO to run again.
        self._write_summary({"yolo": self.yolo_meta()})
        return poses

    def list_tracks(self, poses: pd.DataFrame) -> pd.DataFrame:
        """One row per unique tracked person: how long and where YOLO saw them."""

        columns = ["track_id", "detected_frames", "first_time", "last_time", "center_x", "center_y", "best_frame_idx"]
        if poses.empty:
            return pd.DataFrame(columns=columns)
        fps = self._require_context().fps
        rows = []
        for track_id, group in poses.groupby("track_id", sort=True):
            best = group.loc[group["conf"].idxmax()]
            rows.append(
                {
                    "track_id": int(track_id),
                    "detected_frames": int(group["frame_idx"].nunique()),
                    "first_time": round(float(group["frame_idx"].min()) / fps, 2),
                    "last_time": round(float(group["frame_idx"].max() + 1) / fps, 2),
                    "center_x": round(float(((group["x1"] + group["x2"]) / 2).median()), 1),
                    "center_y": round(float(((group["y1"] + group["y2"]) / 2).median()), 1),
                    "best_frame_idx": int(best["frame_idx"]),
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def track_thumbnails(self, poses: pd.DataFrame, tracks: pd.DataFrame, height: int = 240) -> dict[int, np.ndarray]:
        """RGB crop of each track from the frame where its detection is most confident."""

        context = self._require_context()
        capture = cv2.VideoCapture(str(context.media_path))
        thumbnails: dict[int, np.ndarray] = {}
        try:
            for track in tracks.itertuples():
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(track.best_frame_idx))
                ok, frame = capture.read()
                if not ok:
                    continue
                box = poses[(poses["track_id"] == track.track_id) & (poses["frame_idx"] == track.best_frame_idx)].iloc[0]
                x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
                pad_x, pad_y = (x2 - x1) * 0.1, (y2 - y1) * 0.1
                x1, y1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
                x2, y2 = min(frame.shape[1], int(x2 + pad_x)), min(frame.shape[0], int(y2 + pad_y))
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                scale = height / crop.shape[0]
                crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), height))
                thumbnails[int(track.track_id)] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()
        return thumbnails

    def classify(self, poses: pd.DataFrame, track_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Interpolate the target person over the whole video, then drop frames below the confidence threshold.

        Returns (per-frame table, location segments) and writes frames.csv and locations.csv.
        """

        config, context = self._require_config(), self._require_context()
        poses = poses[poses["track_id"] == track_id]
        frames = self.interpolator.interpolate(poses, context.total_frames)
        frames = self.classifier.classify(frames, config.keypoint_conf_thresh, config.orientation, config.mode)
        segments = self.segment_builder.build(frames, context.fps)
        self.exporter.write_frames_csv(frames, context)
        self.exporter.write_locations_csv(segments, context)
        return frames, segments

    def make_video(self, frames: pd.DataFrame, progress_bar=None) -> Path:
        return self.exporter.make_video(frames, self._require_context(), progress_bar)

    def run_all(self, progress_bar=None, track_id: int | None = None) -> dict[str, Any]:
        """YOLO, then annotation of one person (`track_id=None` works only when exactly one person is tracked)."""

        reset_progress_timers()
        start = time.monotonic()
        poses = self.det_poses(progress_bar=progress_bar)
        return self.annotate(poses, track_id, progress_bar=progress_bar, tracking_seconds=time.monotonic() - start)

    @staticmethod
    def resolve_track_id(poses: pd.DataFrame, track_id: int | None) -> int:
        """The single target person. Without an explicit id, the only tracked person is used."""

        available = sorted(int(t) for t in poses["track_id"].unique()) if not poses.empty else []
        if not available:
            raise ValueError("No persons were tracked in this video.")
        if track_id is None:
            if len(available) > 1:
                raise ValueError(f"{len(available)} persons were tracked {available}; choose one target person by its track id.")
            return available[0]
        if int(track_id) not in available:
            raise ValueError(f"Unknown track id {track_id}. Tracked persons: {available}")
        return int(track_id)

    def annotate(
        self,
        poses: pd.DataFrame,
        track_id: int | None = None,
        progress_bar=None,
        tracking_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Everything after YOLO for the one target person: frames.csv, locations.csv, heatmaps, video, summary.json."""

        if tracking_seconds is None:  # called on its own (e.g. after choosing persons in the GUI)
            reset_progress_timers()
        track_id = self.resolve_track_id(poses, track_id)
        phase_seconds: dict[str, float] = {} if tracking_seconds is None else {"Person tracking": tracking_seconds}

        def _timed(label: str, func: Callable[[], Any]) -> Any:
            start = time.monotonic()
            result = func()
            phase_seconds[label] = time.monotonic() - start
            return result

        update_progress(progress_bar, 0, 1, "Classifying hand locations...")
        frames, segments = _timed("Hand location classification", lambda: self.classify(poses, track_id))
        update_progress(progress_bar, 1, 1, "Classifying hand locations...")
        heatmap_paths = _timed(
            "Heatmaps", lambda: self.exporter.make_heatmaps(frames, self._require_context())
        )
        video_path = None
        if self._require_config().make_video:
            video_path = _timed("Video rendering", lambda: self.make_video(frames, progress_bar=progress_bar))
        elapsed_seconds = finish_progress(progress_bar)
        summary = self._build_summary(frames, segments, phase_seconds, elapsed_seconds, track_id, poses)
        context = self._require_context()
        return {
            "poses": poses,
            "frames": frames,
            "segments": segments,
            "poses_path": context.poses_path,
            "frames_path": context.frames_path,
            "locations_path": context.locations_path,
            "video_path": video_path,
            "heatmap_paths": heatmap_paths,
            "track_id": track_id,
            "summary_path": context.summary_path,
            "elapsed_seconds": elapsed_seconds,
            "summary": summary,
        }

    def _build_summary(
        self,
        frames: pd.DataFrame,
        segments: pd.DataFrame,
        phase_seconds: dict[str, float],
        elapsed_seconds: float,
        track_id: int,
        poses: pd.DataFrame,
    ) -> dict[str, Any]:
        context, config = self._require_context(), self._require_config()
        persons: list[dict[str, Any]] = []
        if not frames.empty:
            for track_id, group in frames.groupby("track_id", sort=True):
                person: dict[str, Any] = {
                    "track_id": int(track_id),
                    "detected_frames": int(group["detected"].sum()),
                }
                for hand in HANDS:
                    hand_segments = segments[(segments["track_id"] == track_id) & (segments["hand"] == hand)]
                    durations = (hand_segments["endTime"] - hand_segments["startTime"]).groupby(hand_segments["location"]).sum()
                    person[f"{hand}_valid_frames"] = int(group[f"{hand}_valid"].sum())
                    person[f"{hand}_seconds_by_location"] = {str(int(k)): round(float(v), 3) for k, v in durations.items()}
                persons.append(person)
        summary = {
            "input": str(context.media_path),
            "fps": context.fps,
            "total_frames": context.total_frames,
            "device": config.device,
            "total_elapsed_seconds": round(elapsed_seconds, 2),
            "phase_seconds": {label: round(seconds, 2) for label, seconds in phase_seconds.items()},
            "tracked_track_ids": sorted(int(t) for t in poses["track_id"].unique()) if not poses.empty else [],
            "selected_track_id": track_id,
            "person_count": len(persons),
            "segment_count": int(len(segments)),
            "persons": persons,
            "settings": {
                "keypoint_conf_thresh": config.keypoint_conf_thresh,
                "mode": config.mode,
                "orientation": config.orientation if config.mode == "frontal" else None,
            },
            # Settings poses.csv depends on; compared on the next run to decide whether YOLO must run again.
            "yolo": self.yolo_meta(),
        }
        self._write_summary(summary)
        self._print_summary(summary)
        return summary

    def _write_summary(self, summary: dict[str, Any]) -> None:
        self._require_context().summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def _print_summary(self, summary: dict[str, Any]) -> None:
        print("\n" + " Pipeline Summary ".center(72, "="), flush=True)
        print(f"  Total elapsed: {format_elapsed(summary['total_elapsed_seconds'])}", flush=True)
        for label, seconds in summary["phase_seconds"].items():
            print(f"    - {label}: {format_elapsed(seconds)}", flush=True)
        print(f"  Tracked persons: {summary['person_count']}, segments: {summary['segment_count']}", flush=True)
        for person in summary["persons"]:
            print(f"  - track {person['track_id']} (detected in {person['detected_frames']}/{summary['total_frames']} frames)", flush=True)
            for hand in HANDS:
                by_location = person[f"{hand}_seconds_by_location"]
                text = ", ".join(f"{loc}: {sec:.1f}s" for loc, sec in sorted(by_location.items())) or "no data"
                print(f"      {hand:>5}: {text} ({person[f'{hand}_valid_frames']} valid frames)", flush=True)
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
