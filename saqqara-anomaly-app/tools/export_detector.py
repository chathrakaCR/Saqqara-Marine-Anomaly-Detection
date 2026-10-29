"""One-off: convert the trained cetacean detector (.pt) to ONNX for the app, and check the two give the same answers.

Run this on the machine where the detector was trained (it needs the training framework installed there),
not inside the container:

    python tools/export_detector.py --weights path/to/cetacean_detector_v2.pt --out models/ --check-images some/folder

It writes models/cetacean_detector_v2.onnx (copy cetacean_detector_v2.json next to it as well).
With --check-images it runs both versions on those images and reports the largest confidence difference.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default=str(ROOT / "models"))
    p.add_argument("--imgsz", type=int, default=352)
    p.add_argument("--check-images", default=None, help="folder of .jpg/.png images to compare the two versions on")
    a = p.parse_args()

    from ultralytics import YOLO  # training framework, only needed for this conversion

    weights = Path(a.weights)
    model = YOLO(str(weights))
    onnx_path = Path(model.export(format="onnx", imgsz=a.imgsz, dynamic=True, simplify=True, opset=17))
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / (weights.stem + ".onnx")
    if onnx_path.resolve() != target.resolve():
        shutil.move(str(onnx_path), target)
    info = weights.with_suffix(".json")
    if info.exists() and info.resolve() != (out_dir / info.name).resolve():
        shutil.copy(info, out_dir / info.name)
    print(f"Wrote {target}" + (f" and {out_dir / info.name}" if info.exists() else " (copy the detector's .json there too)"))

    if a.check_images:
        import cv2
        from app.track_b import CetaceanDetector
        det = CetaceanDetector({"img_size": a.imgsz, "confidence_cutoff": 0.10, "min_box_confidence": 0.05,
                                "nms_iou": 0.7, "max_boxes": 300}, target)
        paths = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in Path(a.check_images).glob(ext))[:200]
        worst, flips = 0.0, 0
        for path in paths:
            r = model.predict(str(path), imgsz=a.imgsz, conf=0.05, verbose=False)[0]
            c_pt = float(r.boxes.conf.max()) if len(r.boxes) else 0.0
            dets = det.detect(cv2.imread(str(path)))
            c_onnx = max((c for c, _ in dets), default=0.0)
            worst = max(worst, abs(c_pt - c_onnx))
            flips += (c_pt >= 0.10) != (c_onnx >= 0.10)
        print(f"Checked {len(paths)} images: largest confidence difference {worst:.4f}; "
              f"{flips} images would get a different cetacean answer at cut-off 0.10")


if __name__ == "__main__":
    main()
