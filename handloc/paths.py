import shutil
import sys
from pathlib import Path

from .types import AppPaths


class PathManager:
    """Resolve working, cache, config, and binary paths."""

    def __init__(self) -> None:
        app_dir = Path.home() / ".handloc"
        app_dir.mkdir(exist_ok=True)

        if hasattr(sys, "_MEIPASS"):
            working_dir = Path(sys._MEIPASS)
        else:
            working_dir = Path(__file__).resolve().parent.parent

        if sys.platform == "darwin":
            bundled_ffmpeg = working_dir / "ffmpeg" / "mac" / "ffmpeg"
        elif sys.platform.startswith("win"):
            bundled_ffmpeg = working_dir / "ffmpeg" / "win" / "ffmpeg.exe"
        else:
            bundled_ffmpeg = working_dir / "ffmpeg" / "AMD" / "ffmpeg"
        ffmpeg_path = bundled_ffmpeg if bundled_ffmpeg.exists() else Path(shutil.which("ffmpeg") or "ffmpeg")

        self.paths = AppPaths(
            working_dir=working_dir,
            app_dir=app_dir,
            botsort_template_path=working_dir / "config" / "botsort.yaml",
            botsort_runtime_path=app_dir / "botsort_runtime.yaml",
            ffmpeg_path=ffmpeg_path,
        )

    def get(self) -> AppPaths:
        return self.paths
