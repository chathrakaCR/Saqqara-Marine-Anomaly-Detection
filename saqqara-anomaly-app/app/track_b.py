"""Stage 3 Track B - trained cetacean detector (detector v2), run with ONNX Runtime.

The detector looks at the whole frame at 352 px and draws a box, with a confidence, around each cetacean.
A frame is "cetacean" when its best box has confidence >= the cut-off chosen on validation (0.10).

Pre- and post-processing reproduce the training framework's own prediction step exactly:
letterbox to 352 px (sides rounded up to a multiple of 32, grey 114 padding), RGB 0-1 input,
confidence filter, non-maximum suppression (IoU 0.7), boxes mapped back to frame pixels.
Use tools/export_detector.py to make the .onnx file from the trained .pt file.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(img, new_size, stride=32, auto=True):
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_size - new_unpad[0], new_size - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw, dh = dw / 2, dh / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def nms(boxes, scores, iou_thr):
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=int)


class CetaceanDetector:
    def __init__(self, cfg: dict, onnx_path, info_path=None):
        self.img_size = int(cfg["img_size"])
        self.cutoff = float(cfg["confidence_cutoff"])
        self.min_conf = float(cfg["min_box_confidence"])
        self.iou = float(cfg["nms_iou"])
        self.max_boxes = int(cfg["max_boxes"])
        self.info = {}
        if info_path and Path(info_path).exists():
            with open(info_path) as f:
                self.info = json.load(f)
            # the training notebook stores the validated cut-off and image size; they take priority
            self.cutoff = float(self.info.get("confidence_cutoff", self.cutoff))
            self.img_size = int(self.info.get("img_size", self.img_size))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=ort.get_available_providers())
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        # dynamic input -> same minimal padding as training-time prediction; fixed input -> square letterbox
        self.dynamic = not (isinstance(h, int) and isinstance(w, int))
        if not self.dynamic:
            self.img_size = int(h)

    def detect(self, frame_bgr):
        """Returns a list of (confidence, (x0, y0, x1, y1)) in frame pixels, best first."""
        h0, w0 = frame_bgr.shape[:2]
        img = letterbox(frame_bgr, self.img_size, auto=self.dynamic)
        h1, w1 = img.shape[:2]
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
        pred = self.session.run(None, {self.input_name: x})[0][0]          # (4 + n_classes, n_anchors)
        scores = pred[4:].max(axis=0)
        keep = scores > self.min_conf
        if not keep.any():
            return []
        xywh, scores = pred[:4, keep].T, scores[keep]
        boxes = np.empty_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2; boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2; boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        idx = nms(boxes, scores, self.iou)[: self.max_boxes]
        boxes, scores = boxes[idx], scores[idx]
        gain = min(h1 / h0, w1 / w0)
        pad_x, pad_y = round((w1 - w0 * gain) / 2 - 0.1), round((h1 - h0 * gain) / 2 - 0.1)
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= gain
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)
        return [(float(s), tuple(float(v) for v in b)) for s, b in zip(scores, boxes)]
