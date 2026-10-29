"""Reads a video file, a network stream (rtsp/http), a camera, or a folder of images, and yields sampled frames."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Frame:
    index: int          # frame number in the source (image number for a folder)
    time_s: float       # seconds from the start (image number for a folder)
    image: "object"     # BGR numpy array
    name: str = ""      # file name, for a folder of images


class VideoSource:
    def __init__(self, source: str, sample_every_seconds: float = 1.0, every_n_frames: int | None = None,
                 max_seconds: float | None = None):
        self.source = source
        self.sample_every_seconds = sample_every_seconds
        self.max_seconds = max_seconds
        self.is_folder = Path(source).is_dir()
        self.is_camera = source.isdigit()
        self.is_live = False
        if self.is_folder:
            # every image is analysed once, at its own size (the same way the end-to-end test read them)
            self.paths = sorted(p for p in Path(source).iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not self.paths:
                raise IOError(f"No images found in folder: {source}")
            first = cv2.imread(str(self.paths[0]))
            self.height, self.width = first.shape[:2] if first is not None else (0, 0)
            self.fps, self.step, self.total_frames = 1.0, 1, len(self.paths)
            self.cap = None
            return
        self.cap = cv2.VideoCapture(int(source) if self.is_camera else source)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video source: {source}")
        self.is_live = self.is_camera or "://" in source
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 0 < fps < 1000 else 25.0
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.total_frames = n if n > 0 and not self.is_live else None
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.step = max(1, int(every_n_frames) if every_n_frames else int(round(self.fps * sample_every_seconds)))

    @property
    def expected_samples(self):
        if self.total_frames is None:
            return None
        if self.is_folder:
            return self.total_frames
        n = self.total_frames
        if self.max_seconds:
            n = min(n, int(self.max_seconds * self.fps))
        return (n + self.step - 1) // self.step

    def frames(self):
        if self.is_folder:
            yield from self._folder_frames()
        else:
            yield from self._video_frames()

    def _folder_frames(self):
        for i, p in enumerate(self.paths):
            img = cv2.imread(str(p))
            if img is not None:
                yield Frame(i, float(i), img, p.name)

    def _video_frames(self):
        """Files: every `step`-th frame. Live streams: one frame per interval of wall time."""
        idx = -1
        start = time.monotonic()
        next_live = 0.0
        try:
            while True:
                if not self.cap.grab():
                    break
                idx += 1
                if self.is_live:
                    t = time.monotonic() - start
                    if self.max_seconds and t > self.max_seconds:
                        break
                    if t < next_live:
                        continue
                    next_live = t + self.sample_every_seconds
                else:
                    t = idx / self.fps
                    if self.max_seconds and t > self.max_seconds:
                        break
                    if idx % self.step:
                        continue
                ok, img = self.cap.retrieve()
                if ok and img is not None:
                    yield Frame(idx, round(t, 3), img)
        finally:
            self.cap.release()
