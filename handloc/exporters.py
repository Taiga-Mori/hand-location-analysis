import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from .annotate import FrameAnnotator
from .progress import update_progress
from .types import AppPaths, MediaContext


class AnnotationExporter:
    """Export the hand location CSV files and the annotated video."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self._available_ffmpeg_encoders: set[str] | None = None

    def write_frames_csv(self, frames: pd.DataFrame, context: MediaContext) -> Path:
        frames.to_csv(context.frames_path, index=False, float_format="%.2f")
        return context.frames_path

    def write_segments_csv(self, segments: pd.DataFrame, context: MediaContext) -> Path:
        segments.to_csv(context.segments_path, index=False)
        return context.segments_path

    def make_video(self, frames: pd.DataFrame, context: MediaContext, orientation: str, progress_bar=None) -> Path:
        rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in frames.to_dict("records"):
            rows_by_frame[int(row["frame_idx"])].append(row)

        encoder = self._preferred_h264_encoder()
        if encoder is not None:
            try:
                return self._render(rows_by_frame, context, orientation, encoder, progress_bar)
            except RuntimeError as exc:
                print(f"Video encoder {encoder} failed; falling back to libx264.\n{exc}", flush=True)
        return self._render(rows_by_frame, context, orientation, "libx264", progress_bar)

    def _render(
        self,
        rows_by_frame: dict[int, list[dict[str, Any]]],
        context: MediaContext,
        orientation: str,
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
                for row in rows_by_frame.get(frame_idx, []):
                    mirrored = orientation == "auto" and float(row["grid_x_right"]) > float(row["grid_x_left"])
                    annotator.draw_person(frame, row, mirrored)
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
