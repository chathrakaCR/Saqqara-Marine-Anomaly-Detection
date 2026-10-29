"""Draws the pipeline's answer on a frame."""
from __future__ import annotations

import cv2
import numpy as np

COLOURS = {  # BGR
    "prawn": (214, 120, 42),
    "tuna": (52, 104, 235),
    "cetacean": (60, 170, 60),
    "none": (140, 140, 140),
}
STAGE1_COLOUR = (0, 215, 255)


def draw(frame_bgr: np.ndarray, r, time_s: float | None = None) -> np.ndarray:
    img = frame_bgr.copy()
    H, W = img.shape[:2]
    th = max(2, int(round(min(H, W) / 360)))
    scale = max(0.6, min(H, W) / 720)
    colour = COLOURS.get(r.label, COLOURS["none"])

    if r.mask is not None and r.mask.any():
        overlay = img.copy()
        overlay[r.mask > 0] = colour
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        contours, _ = cv2.findContours(r.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, colour, th)
    if r.stage1_box is not None:
        x0, y0, x1, y1 = map(int, r.stage1_box)
        cv2.rectangle(img, (x0, y0), (x1, y1), STAGE1_COLOUR, max(1, th - 1), cv2.LINE_AA)
    for conf, (x0, y0, x1, y1) in r.cetacean_boxes:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), COLOURS["cetacean"], th, cv2.LINE_AA)
        cv2.putText(img, f"{conf:.2f}", (int(x0), max(0, int(y0) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                    COLOURS["cetacean"], max(1, th - 1), cv2.LINE_AA)

    text = r.label.upper() if r.label != "none" else ("NONE (flagged, rejected)" if r.stage1_flagged else "NONE")
    if time_s is not None:
        text = f"{time_s:7.1f}s  {text}"
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    cv2.rectangle(img, (0, 0), (tw + 20, tht + base + 16), colour, -1)
    cv2.putText(img, text, (10, tht + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th, cv2.LINE_AA)
    return img
