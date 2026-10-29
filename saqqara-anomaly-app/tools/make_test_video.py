"""Make a test video from labelled photos, plus a file saying what is in it when.

Each photo is shown for --seconds-per-image (default 1 s), in random order. Photos are scaled to fill the frame
(centre-cropped, no black bars). A truth file is written next to the video, so tools/score_run.py can compare
the app's answers with the right ones.

Inside the container (folders must be under ./data):
  docker compose run --rm marine-anomaly python tools/make_test_video.py \
      --folder prawn=/data/Test_data/PRAWNS --folder tuna=/data/Test_data/TUNA \
      --folder none=/data/Test_data/SEA --folder cetacean=/data/cetacean_test \
      --per-class 10 --out /data/test_video.mp4
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
ALIASES = {"sea": "none", "clean": "none", "water": "none", "prawns": "prawn", "cetaceans": "cetacean"}


def fill(img, W, H):
    h, w = img.shape[:2]
    s = max(W / w, H / h)
    img = cv2.resize(img, (max(W, round(w * s)), max(H, round(h * s))), interpolation=cv2.INTER_AREA)
    y0, x0 = (img.shape[0] - H) // 2, (img.shape[1] - W) // 2
    return img[y0:y0 + H, x0:x0 + W]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", action="append", required=True, help="LABEL=PATH, e.g. tuna=/data/Test_data/TUNA (repeat)")
    p.add_argument("--per-class", type=int, default=10, help="photos per label (0 = all)")
    p.add_argument("--seconds-per-image", type=float, default=1.0)
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--size", default="1920x1080")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="/data/test_video.mp4")
    a = p.parse_args()

    W, H = map(int, a.size.lower().split("x"))
    rng = random.Random(a.seed)
    items = []
    for spec in a.folder:
        label, path = spec.split("=", 1)
        label = ALIASES.get(label.strip().lower(), label.strip().lower())
        files = sorted(f for f in Path(path).iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise SystemExit(f"No images in {path}")
        if a.per_class and len(files) > a.per_class:
            files = rng.sample(files, a.per_class)
        items += [(label, f) for f in files]
        print(f"{label:<9} {len(files)} photos from {path}")
    rng.shuffle(items)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    hold = max(1, round(a.seconds_per_image * a.fps))
    truth = []
    t = 0.0
    for label, f in items:
        img = cv2.imread(str(f))
        if img is None:
            continue
        frame = fill(img, W, H)
        for _ in range(hold):
            writer.write(frame)
        dur = hold / a.fps
        truth.append({"start_s": round(t, 3), "end_s": round(t + dur, 3), "label": label, "file": f.name})
        t += dur
    writer.release()
    truth_path = out.with_name(out.stem + "_truth.csv")
    with open(truth_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["start_s", "end_s", "label", "file"])
        w.writeheader()
        w.writerows(truth)
    print(f"Wrote {out} ({len(truth)} photos, {t:.0f} s) and {truth_path}")
    print(f"Analyse it with one frame every {a.seconds_per_image:g} s so each photo is looked at once.")


if __name__ == "__main__":
    main()
