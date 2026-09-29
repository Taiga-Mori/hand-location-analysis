import logging
import os
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")

from ultralytics import YOLO

try:
    from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER
except Exception:
    ULTRALYTICS_LOGGER = None

from .types import AppPaths


class ModelManager:
    """Download, cache, and serve the YOLO pose model."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.yolo_pose: YOLO | None = None
        self.loaded_backend: str | None = None
        self._configure_download_environment()

    def _configure_download_environment(self) -> None:
        if ULTRALYTICS_LOGGER is not None:
            ULTRALYTICS_LOGGER.setLevel(logging.ERROR)
        try:
            import certifi

            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        except Exception:
            pass

    def weights_path(self, backend: str) -> Path:
        return self.paths.pose_model_path(backend)

    def load(self, backend: str) -> YOLO:
        """Load (and download on first use) the requested pose model.

        A fresh model instance is created for each run so tracker state never leaks between videos.
        """

        weights = self.weights_path(backend)
        if not weights.exists():
            print(f"Downloading {weights.name} to {weights.parent} ...", flush=True)
        # Ultralytics downloads official assets to the given path when it does not exist yet.
        self.yolo_pose = YOLO(str(weights), task="pose")
        self.loaded_backend = backend
        return self.yolo_pose
