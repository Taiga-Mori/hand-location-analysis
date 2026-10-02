import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .constants import KEYPOINT_NAMES, POSE_COLUMNS
from .models import ModelManager
from .progress import update_progress
from .types import AppPaths, MediaContext

POSE_CACHE_VERSION = 3


class PoseTracker:
    """Track persons with a YOLO pose model and keep their raw keypoints."""

    def __init__(self, models: ModelManager, paths: AppPaths) -> None:
        self.models = models
        self.paths = paths

    def detect(
        self,
        context: MediaContext,
        device: str,
        backend: str,
        det_thresh: float,
        tracker_config: dict[str, Any],
        progress_bar=None,
    ) -> pd.DataFrame:
        model = self.models.load(backend)
        rows: list[dict[str, Any]] = []
        expected_steps = max(1, math.ceil(context.total_frames / context.person_stride))
        update_interval = max(1, expected_steps // 200)
        label = f"Tracking persons ({backend})..."
        update_progress(progress_bar, 0, expected_steps, label)
        results = model.track(
            source=str(context.media_path),
            stream=True,
            verbose=False,
            tracker=str(self.paths.botsort_runtime_path),
            vid_stride=context.person_stride,
            device=self._yolo_device(device),
        )
        step = 0
        for step, result in enumerate(results, start=1):
            frame_idx = min((step - 1) * context.person_stride, context.total_frames - 1)
            rows.extend(self._rows_from_result(frame_idx, result, det_thresh))
            if step == expected_steps or step % update_interval == 0:
                update_progress(progress_bar, step, expected_steps, label)
        update_progress(progress_bar, expected_steps, expected_steps, label)
        if step != expected_steps:
            print(f"Tracking coverage: expected {expected_steps} frame results, got {step}.", flush=True)

        poses = pd.DataFrame(rows, columns=POSE_COLUMNS)
        poses.to_csv(context.poses_path, index=False)
        return poses

    def load_cached(
        self,
        context: MediaContext,
        backend: str,
        det_thresh: float,
        tracker_config: dict[str, Any],
    ) -> pd.DataFrame | None:
        """Reuse poses.csv when the "yolo" section of summary.json matches the current settings."""

        if not context.poses_path.exists() or not context.summary_path.exists():
            return None
        try:
            recorded = json.loads(context.summary_path.read_text(encoding="utf-8")).get("yolo")
            cached = pd.read_csv(context.poses_path)
        except Exception:
            return None
        if recorded != self.meta(context, backend, det_thresh, tracker_config):
            return None
        if list(cached.columns) != POSE_COLUMNS:
            return None
        return cached

    def _rows_from_result(self, frame_idx: int, result, det_thresh: float) -> list[dict[str, Any]]:
        boxes = result.boxes
        if boxes is None or boxes.id is None or result.keypoints is None:
            return []
        keypoints = result.keypoints.data.cpu().numpy()
        rows: list[dict[str, Any]] = []
        for index, (track_id, conf, xyxy) in enumerate(
            zip(boxes.id.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist())
        ):
            if conf < det_thresh or index >= len(keypoints):
                continue
            row: dict[str, Any] = {
                "frame_idx": frame_idx,
                "track_id": int(track_id),
                "conf": round(float(conf), 4),
                "x1": round(float(xyxy[0]), 2),
                "y1": round(float(xyxy[1]), 2),
                "x2": round(float(xyxy[2]), 2),
                "y2": round(float(xyxy[3]), 2),
            }
            for name, (px, py, pconf) in zip(KEYPOINT_NAMES, keypoints[index]):
                row[f"{name}_x"] = round(float(px), 2)
                row[f"{name}_y"] = round(float(py), 2)
                row[f"{name}_conf"] = round(float(pconf), 4)
            rows.append(row)
        return rows

    def _yolo_device(self, device: str) -> str | torch.device:
        if device.startswith("cuda:"):
            return torch.device(device)
        return device

    def meta(self, context: MediaContext, backend: str, det_thresh: float, tracker_config: dict[str, Any]) -> dict[str, Any]:
        """Everything poses.csv depends on; stored in summary.json["yolo"].

        Settings applied after YOLO (person selection, mode, keypoint threshold, orientation) are deliberately
        left out, so changing them reuses poses.csv instead of rerunning YOLO.
        """

        media_path = Path(context.media_path).resolve()
        return {
            "cache_version": POSE_CACHE_VERSION,
            "media_path": str(media_path),
            "media_mtime_ns": media_path.stat().st_mtime_ns,
            "backend": backend,
            "det_thresh": float(det_thresh),
            "stride": int(context.person_stride),
            "tracker_config": tracker_config,
        }
