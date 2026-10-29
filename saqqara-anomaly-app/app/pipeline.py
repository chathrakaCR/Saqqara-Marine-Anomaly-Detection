"""The full system for one video frame: Stage 1 -> Track B -> Stage 2 -> Track A -> final answer.

Decision order (as in the end-to-end test, detector v2):
  1. Stage 1 not flagged                      -> none
  2. detector's best box >= cut-off (0.10)    -> cetacean   (checked first: Track A has no cetacean class)
  3. Stage 2 finds no usable mask             -> none
  4. Track A says prawn / tuna                -> prawn / tuna
  5. Track A says other, or cannot measure    -> none
The detector is only run on frames Stage 1 flags; the end-to-end test ran it on every frame but only
counted it after the Stage 1 gate, so the answers are the same.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
import torch

from .config import load_config
from .stage1 import Stage1Detector
from .stage2 import Stage2Segmenter
from .track_a import TrackAClassifier
from .track_b import CetaceanDetector

log = logging.getLogger("pipeline")

LABELS = ("prawn", "tuna", "cetacean", "none")


@dataclass
class FrameResult:
    label: str = "none"
    decided_at: str = "stage1"
    reason: str = ""
    stage1_flagged: bool = False
    stage1_rule: str = ""
    stage1_box: tuple | None = None          # in frame pixels
    cetacean_confidence: float | None = None
    cetacean_boxes: list = field(default_factory=list)     # [(conf, (x0, y0, x1, y1))] >= cut-off, frame pixels
    stage2_mask_px: int = 0
    stage2_max_prob: float | None = None
    track_a_label: str | None = None
    track_a_probs: dict | None = None
    processing_ms: float = 0.0
    mask: np.ndarray | None = field(default=None, repr=False)   # Stage 2 mask resized to the frame (uint8 0/1)

    def to_dict(self):
        d = asdict(self)
        d.pop("mask", None)
        return d


class Pipeline:
    def __init__(self, cfg: dict | None = None, device: str | None = None):
        self.cfg = cfg or load_config()
        m = self.cfg["models"]
        missing = [f"{k}: {p}" for k, p in m.items() if k != "detector_info" and not p.exists()]
        if missing:
            raise FileNotFoundError("Model files not found (see models/README.md):\n  " + "\n  ".join(missing))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        s1 = self.cfg["stage1"]
        self.img_size = tuple(s1["img_size"])
        self.stage1 = Stage1Detector(s1, m["stage1_bucket_ref"])
        self.stage2 = Stage2Segmenter(self.cfg["stage2"], self.img_size, m["segformer_checkpoint"], self.device)
        self.track_a = TrackAClassifier(self.cfg["track_a"], self.img_size, m["track_a_bundle"])
        self.track_b = CetaceanDetector(self.cfg["track_b"], m["detector_onnx"], m["detector_info"])
        for w in self.track_a.load_warnings:
            log.warning("Track A model file: %s", w)
        diff = self.track_a.settings_mismatch(s1, self.cfg["stage2"], m["segformer_checkpoint"].name)
        if diff:
            msg = f"Settings differ from what Track A was trained with: {diff}"
            if not self.cfg.get("allow_settings_mismatch"):
                raise RuntimeError(msg + "\nFix config/pipeline_config.json, or set ALLOW_SETTINGS_MISMATCH=1 to run anyway.")
            log.warning(msg)
        log.info("Pipeline ready on %s. Detector cut-off %.2f at %d px.", self.device, self.track_b.cutoff,
                 self.track_b.img_size)

    def describe(self) -> dict:
        return {"device": str(self.device), "stage1": self.cfg["stage1"], "stage2": self.cfg["stage2"],
                "track_a": {**self.cfg["track_a"], "classes": self.track_a.classes, "features": self.track_a.columns},
                "track_b": {"img_size": self.track_b.img_size, "confidence_cutoff": self.track_b.cutoff}}

    def analyse(self, frame_bgr: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        r = FrameResult()
        H, W = frame_bgr.shape[:2]
        sx, sy = W / self.img_size[0], H / self.img_size[1]

        def to_frame(b):
            return (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)

        s1 = self.stage1.run(frame_bgr)
        r.stage1_flagged, r.stage1_rule = s1["flagged"], s1["rule"]
        if not s1["flagged"]:
            r.reason = "Stage 1 did not flag the frame"
            return self._done(r, t0)
        r.stage1_box = to_frame(s1["box"])

        # Track B first
        dets = self.track_b.detect(frame_bgr)
        r.cetacean_confidence = max((c for c, _ in dets), default=0.0)
        r.cetacean_boxes = [(c, b) for c, b in dets if c >= self.track_b.cutoff]
        if r.cetacean_confidence >= self.track_b.cutoff:
            r.label, r.decided_at = "cetacean", "track_b"
            r.reason = f"Detector: best box {r.cetacean_confidence:.2f} >= {self.track_b.cutoff}"
            return self._done(r, t0)

        # Stage 2 on the whole frame
        s2 = self.stage2.run(s1["rgb"], s1["box"])
        r.stage2_mask_px, r.stage2_max_prob = s2["mask_px"], s2["max_prob"]
        if s2["frame_mask"] is not None:
            r.mask = cv2.resize(s2["frame_mask"], (W, H), interpolation=cv2.INTER_NEAREST)
        if s2["mask_px"] == 0:
            r.decided_at = "stage2"
            r.reason = f"Stage 2 found no usable mask (max prob {s2['max_prob']:.2f})"
            return self._done(r, t0)

        # Track A
        label, probs, note = self.track_a.classify(s2["crop"], s2["crop_mask"], s2["crop_box"], (W, H))
        r.track_a_label, r.track_a_probs, r.decided_at = label, probs, "track_a"
        if label in ("prawn", "tuna"):
            r.label = label
            r.reason = f"Track A said {label} ({max(probs.values()):.2f})"
        elif label == "other":
            r.reason = "Track A called it 'other' (false alarm)"
        else:
            r.reason = note
        return self._done(r, t0)

    @staticmethod
    def _done(r, t0):
        r.processing_ms = round((time.perf_counter() - t0) * 1000, 1)
        return r
