import os

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")

from .pipeline import HandLocationAnalyzer

try:
    import cv2

    cv2.setLogLevel(0)
except Exception:
    pass

__all__ = ["HandLocationAnalyzer"]
