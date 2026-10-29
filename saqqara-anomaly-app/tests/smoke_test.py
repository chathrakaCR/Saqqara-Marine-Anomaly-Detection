"""Quick check that the models load and a short synthetic video runs end to end.

    python tests/smoke_test.py            (from the app folder, with the model files in models/)
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import Pipeline  # noqa: E402
from app.runner import process_video  # noqa: E402


def main():
    tmp = Path(tempfile.mkdtemp())
    video = tmp / "synthetic.mp4"
    rng = np.random.default_rng(0)
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))
    for i in range(50):
        frame = np.clip(np.full((720, 1280, 3), (120, 90, 30), np.int16) + rng.normal(0, 4, (720, 1280, 1)), 0, 255).astype(np.uint8)
        if i >= 25:                                   # a bright foam-like patch in the second half
            cv2.ellipse(frame, (640, 360), (160, 90), 0, 0, 360, (235, 235, 235), -1)
        w.write(frame)
    w.release()
    res = process_video(str(video), tmp / "out", Pipeline(), sample_every_seconds=1.0, save_video=True)
    s = res["summary"]
    assert s["frames_analysed"] == 5, s
    assert (tmp / "out" / "frames.csv").exists() and (tmp / "out" / "annotated.mp4").exists()
    print("OK:", s["frames_analysed"], "frames,", s["counts"], "| results in", tmp / "out")


if __name__ == "__main__":
    main()
