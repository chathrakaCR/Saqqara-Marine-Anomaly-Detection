"""Stage 2 - SegFormer-B0 mask on the whole frame, then the crop that Track A measures.

Ported from Stage_1234_Integrated_Pipeline_Test (cells 2 and 4, route "whole_frame").
The model is built from the standard SegFormer-B0 architecture and the full fine-tuned checkpoint is loaded,
so no download is needed at run time.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from transformers import SegformerConfig, SegformerForSemanticSegmentation

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pad_box(box, pad, W, H):
    x0, y0, x1, y1 = box
    return max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)


def boxes_overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def largest_blob(mask):
    labeled, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return (labeled == (int(np.argmax(sizes)) + 1)).astype(mask.dtype)


class Stage2Segmenter:
    def __init__(self, cfg: dict, img_size, checkpoint_path, device):
        self.prob = cfg["prob_threshold"]
        self.min_px = cfg["min_mask_pixels"]
        self.keep_largest = cfg["keep_largest_blob"]
        self.crop_pad = cfg["crop_padding_px"]
        if cfg.get("route", "whole_frame") != "whole_frame":
            raise ValueError("Only the final 'whole_frame' Stage 2 route is supported")
        self.img_size = tuple(img_size)
        self.device = device
        config = SegformerConfig(num_labels=2, id2label={0: "background", 1: "anomaly"},
                                 label2id={"background": 0, "anomaly": 1})      # default config = SegFormer-B0
        self.model = SegformerForSemanticSegmentation(config)
        state = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()

    def probs(self, rgb_u8):
        arr = np.array(Image.fromarray(rgb_u8).resize(self.img_size, Image.BILINEAR), dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(pixel_values=x).logits
            logits = F.interpolate(logits, size=(self.img_size[1], self.img_size[0]), mode="bilinear", align_corners=False)
            return F.softmax(logits, dim=1)[0, 1].cpu().numpy()

    def run(self, rgb, s1_box):
        """rgb: 512x512 uint8 frame from Stage 1. s1_box: Stage 1 box (unpadded, frame coords).
        Returns dict with frame_mask (512x512 0/1), crop_box, crop (float 0-1), crop_mask, mask_px, max_prob."""
        H, W = rgb.shape[:2]
        s1_padded = pad_box(s1_box, self.crop_pad, W, H)
        probs = self.probs(rgb)
        pred = (probs > self.prob).astype(np.uint8)
        out = {"s1_padded_box": s1_padded, "max_prob": float(probs.max()), "frame_mask": None,
               "crop_box": None, "crop": None, "crop_mask": None, "mask_px": 0}
        if self.keep_largest and pred.any():
            pred = largest_blob(pred)
        if pred.sum() < self.min_px:
            return out
        ys, xs = np.where(pred > 0)
        mask_box = pad_box((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), self.crop_pad, W, H)
        if boxes_overlap(mask_box, s1_padded):
            crop_box = (min(mask_box[0], s1_padded[0]), min(mask_box[1], s1_padded[1]),
                        max(mask_box[2], s1_padded[2]), max(mask_box[3], s1_padded[3]))
        else:
            crop_box = mask_box
        x0, y0, x1, y1 = crop_box
        crop = np.array(Image.fromarray(rgb[y0:y1, x0:x1]).resize(self.img_size, Image.BILINEAR), dtype=np.float32) / 255.0
        crop_mask = cv2.resize(pred[y0:y1, x0:x1], self.img_size, interpolation=cv2.INTER_NEAREST)
        out.update(frame_mask=pred, crop_box=crop_box, crop=crop, crop_mask=crop_mask, mask_px=int(crop_mask.sum()))
        return out
