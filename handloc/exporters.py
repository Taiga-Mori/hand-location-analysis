import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .annotate import FrameAnnotator
from .constants import FRAME_COLUMNS, HANDS
from .progress import update_progress
from .types import AppPaths, MediaContext


class AnnotationExporter:
    """Export the hand location CSV files and the annotated video."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self._available_ffmpeg_encoders: set[str] | None = None

    def write_frames_csv(self, frames: pd.DataFrame, context: MediaContext) -> Path:
        """Normalized wrist coordinates, one row per valid frame, track, and hand."""

        tables = []
        if not frames.empty:
            for hand in HANDS:
                valid = frames[frames[f"{hand}_valid"]]
                tables.append(
                    pd.DataFrame(
                        {
                            "frame_idx": valid["frame_idx"],
                            "track_id": valid["track_id"],
                            "hand": hand,
                            "x": valid[f"{hand}_x_norm"],
                            "y": valid[f"{hand}_y_norm"],
                        }
                    )
                )
        table = pd.concat(tables) if tables else pd.DataFrame(columns=FRAME_COLUMNS)
        table = table.sort_values(["frame_idx", "track_id", "hand"], ascending=[True, True, False])
        table.to_csv(context.frames_path, index=False, float_format="%.4f")
        return context.frames_path

    def write_locations_csv(self, segments: pd.DataFrame, context: MediaContext) -> Path:
        segments.to_csv(context.locations_path, index=False)
        return context.locations_path

    def make_heatmaps(self, frames: pd.DataFrame, context: MediaContext) -> dict[str, Path]:
        """Wrist-position heatmaps over the whole video (right, left, both) on the middle frame.

        Counts every valid frame of the target person, blurs the counts, and normalizes each image to its own
        maximum. The person's grid is drawn faintly at its median position over the video, for reference.
        """

        capture = cv2.VideoCapture(str(context.media_path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, context.total_frames // 2)
            ok, background = capture.read()
        finally:
            capture.release()
        if not ok:
            background = np.zeros((context.height, context.width, 3), np.uint8)
        height, width = background.shape[:2]
        sigma = 0.015 * max(width, height)

        counts = {}
        for hand in HANDS:
            grid = np.zeros((height, width), np.float32)
            if not frames.empty:
                valid = frames[frames[f"{hand}_valid"]]
                xs = np.clip(np.round(valid[f"{hand}_wrist_x"].to_numpy(dtype=float)), 0, width - 1).astype(int)
                ys = np.clip(np.round(valid[f"{hand}_wrist_y"].to_numpy(dtype=float)), 0, height - 1).astype(int)
                np.add.at(grid, (ys, xs), 1.0)
            counts[hand] = grid
        counts["both"] = counts["right"] + counts["left"]

        reference = self._median_grids(background, frames)
        paths = {}
        for name, grid in [("right", counts["right"]), ("left", counts["left"]), ("both", counts["both"])]:
            image = reference.copy()
            total = int(grid.sum())
            if total:
                heat = cv2.GaussianBlur(grid, (0, 0), sigma)
                heat /= heat.max()
                colored = cv2.applyColorMap(np.uint8(heat * 255), cv2.COLORMAP_JET).astype(np.float32)
                alpha = np.clip((heat - 0.02) / 0.98, 0, 1)[..., None] * 0.75  # faint areas stay transparent
                image = (image * (1 - alpha) + colored * alpha).astype(np.uint8)
            path = context.output_dir / f"heatmap_{name}.png"
            cv2.imwrite(str(path), image)
            paths[name] = path
        return paths

    def _median_grids(self, background: np.ndarray, frames: pd.DataFrame) -> np.ndarray:
        if frames.empty:
            return background.copy()
        annotator = FrameAnnotator(background.shape[1], background.shape[0])
        layer = background.copy()
        columns = ["grid_x_first", "grid_x_second", "grid_y_top", "grid_y_bottom", "x1", "y1", "x2", "y2"]
        for track_id, group in frames.groupby("track_id", sort=True):
            usable = group[group["right_valid"] | group["left_valid"]]
            if usable.empty:
                continue
            row = usable[columns].median().to_dict()
            mirrored = annotator.mirrored(row)
            edges = annotator._grid_edges(row)
            color = annotator.id_to_color(track_id)
            annotator._draw_grid(layer, edges, mirrored, color)
            annotator._draw_track_id(layer, {"track_id": int(track_id)}, color, edges)
        return cv2.addWeighted(layer, 0.5, background, 0.5, 0.0)

    def make_video(self, frames: pd.DataFrame, context: MediaContext, progress_bar=None) -> Path:
        rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in frames.to_dict("records"):
            rows_by_frame[int(row["frame_idx"])].append(row)

        encoder = self._preferred_h264_encoder()
        if encoder is not None:
            try:
                return self._render(rows_by_frame, context, encoder, progress_bar)
            except RuntimeError as exc:
                print(f"Video encoder {encoder} failed; falling back to libx264.\n{exc}", flush=True)
        return self._render(rows_by_frame, context, "libx264", progress_bar)

    def _render(
        self,
        rows_by_frame: dict[int, list[dict[str, Any]]],
        context: MediaContext,
        encoder: str,
        progress_bar=None,
    ) -> Path:
        capture = cv2.VideoCapture(str(context.media_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {context.media_path}")
        width, height = context.width, context.height
        annotator = FrameAnnotator(width, height)
        command = [
            str(self.paths.ffmpeg_path),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-framerate",
            f"{context.fps:.6f}",
            "-i",
            "-",
            "-i",
            str(context.media_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            *self._h264_encoder_args(encoder, width, height, context.fps),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(context.video_path),
        ]
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            capture.release()
            raise RuntimeError(f"ffmpeg was not found. Expected ffmpeg path: {self.paths.ffmpeg_path}") from exc

        label = "Rendering annotated video..."
        update_interval = max(1, context.total_frames // 200)
        frame_idx = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                annotator.draw(frame, rows_by_frame.get(frame_idx, []))
                try:
                    process.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    break
                frame_idx += 1
                if frame_idx % update_interval == 0:
                    update_progress(progress_bar, frame_idx, context.total_frames, label)
        finally:
            capture.release()
            if process.stdin:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with {encoder} (exit code {return_code}).\nCommand: {' '.join(command)}\n{stderr.strip()}")
        update_progress(progress_bar, context.total_frames, context.total_frames, label)
        return context.video_path

    def _preferred_h264_encoder(self) -> str | None:
        if sys.platform == "darwin" and self._ffmpeg_encoder_available("h264_videotoolbox"):
            return "h264_videotoolbox"
        if self._ffmpeg_encoder_available("h264_nvenc"):
            return "h264_nvenc"
        return None

    def _h264_encoder_args(self, encoder: str, width: int, height: int, fps: float) -> list[str]:
        bitrate = f"{max(2, int(width * height * fps * 0.1 / 1_000_000))}M"
        if encoder == "h264_videotoolbox":
            return ["-c:v", "h264_videotoolbox", "-allow_sw", "1", "-b:v", bitrate, "-pix_fmt", "yuv420p"]
        if encoder == "h264_nvenc":
            return ["-c:v", "h264_nvenc", "-b:v", bitrate, "-pix_fmt", "yuv420p"]
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]

    def _ffmpeg_encoder_available(self, encoder: str) -> bool:
        if self._available_ffmpeg_encoders is None:
            self._available_ffmpeg_encoders = self._detect_ffmpeg_encoders()
        return encoder in self._available_ffmpeg_encoders

    def _detect_ffmpeg_encoders(self) -> set[str]:
        ffmpeg = str(self.paths.ffmpeg_path) if self.paths.ffmpeg_path.exists() else shutil.which("ffmpeg")
        if ffmpeg is None:
            return set()
        try:
            result = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], check=True, capture_output=True, text=True)
        except Exception:
            return set()
        encoders: set[str] = set()
        for line in (result.stdout or "").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith("V"):
                encoders.add(parts[1])
        return encoders
