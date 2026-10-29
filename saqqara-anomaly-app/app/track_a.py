"""Stage 3 Track A - prawn / tuna / other from 12 measurements of the Stage 2 mask.

Feature code ported from Stage_1234_Integrated_Pipeline_Test (cell 5). The trained logistic regression,
its feature columns and the settings it was trained with come from track_a_model.pkl.
"""
from __future__ import annotations

import pickle
import warnings

import cv2
import numpy as np


def _sobel_mag(gray):
    return np.sqrt(cv2.Sobel(gray, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(gray, cv2.CV_32F, 0, 1) ** 2)


class TrackAClassifier:
    def __init__(self, cfg: dict, img_size, bundle_path):
        self.ring_px = cfg["ring_px"]
        self.min_native_px = cfg["min_native_px"]
        self.min_native_mask_px = cfg["min_native_mask_px"]
        self.img_size = tuple(img_size)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with open(bundle_path, "rb") as f:
                self.bundle = pickle.load(f)
        self.load_warnings = [str(w.message) for w in caught]
        self.model = self.bundle["model"]
        self.columns = list(self.bundle["columns"])
        self.classes = list(self.bundle["classes"])

    def settings_mismatch(self, stage1_cfg: dict, stage2_cfg: dict, segformer_name: str) -> dict:
        """Compare this app's settings with the ones stored when Track A was trained (same check as the notebook)."""
        mine = {"prob": stage2_cfg["prob_threshold"], "min_px": stage2_cfg["min_mask_pixels"],
                "dilation": stage1_cfg["dilation_iters"], "crop_pad": stage2_cfg["crop_padding_px"],
                "ring_px": self.ring_px, "min_native_px": self.min_native_px,
                "min_native_mask_px": self.min_native_mask_px}
        stored = self.bundle.get("settings", {})
        diff = {k: (stored.get(k), v) for k, v in mine.items() if k in stored and stored.get(k) != v}
        s1 = self.bundle.get("stage1_config")
        if s1:
            for k in ("img_size", "n_segments", "compactness", "glcm_levels", "glint_thresh", "bucket_edges",
                      "z_thresh", "min_features_flagged", "min_component_area_px", "local_z_thresh", "min_area_local"):
                if k in s1 and not np.allclose(np.array(s1[k], dtype=float), np.array(stage1_cfg[k], dtype=float)):
                    diff[f"stage1.{k}"] = (s1[k], stage1_cfg[k])
        ckpt = str(self.bundle.get("segformer_checkpoint", "")).replace("\\", "/").split("/")[-1]
        if ckpt and ckpt != segformer_name:
            diff["segformer_checkpoint"] = (ckpt, segformer_name)
        return diff

    def _native_texture(self, crop_u8, mask, bbox, src_size):
        x0, y0, x1, y1 = bbox
        src_w, src_h = src_size
        fw, fh = self.img_size
        nw = int(np.clip(round((x1 - x0) * src_w / fw), self.min_native_px, fw))
        nh = int(np.clip(round((y1 - y0) * src_h / fh), self.min_native_px, fh))
        small = cv2.resize(crop_u8, (nw, nh), interpolation=cv2.INTER_AREA)
        m_small = cv2.resize(mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA) > 0.5
        if m_small.sum() < self.min_native_mask_px:
            return None
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        L = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
        return float(L[m_small].std()), float(_sobel_mag(gray)[m_small].mean()), nw

    def features(self, crop, mask, bbox, src_size):
        """Returns (features, None) or (None, reason). src_size = (width, height) of the original video frame."""
        mask_area = int(mask.sum())
        if mask_area == 0:
            return None, "empty mask"
        inside = mask.astype(bool)
        ys, xs = np.where(inside)
        mask_h, mask_w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        crop_u8 = (np.clip(crop, 0, 1) * 255).astype(np.uint8)
        lab = cv2.cvtColor(crop_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
        gray = cv2.cvtColor(crop_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad = _sobel_mag(gray)
        L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.ring_px + 1, 2 * self.ring_px + 1))
        ring = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) & ~inside
        if ring.sum() < 50:
            ring = ~inside
        if ring.sum() < 50:
            return None, "mask covers almost the whole crop"
        native = self._native_texture(crop_u8, mask, bbox, src_size)
        if native is None:
            return None, "mask too small at native scale"
        in_L_std_native, in_edge_native, native_w = native
        box_w = max(1, bbox[2] - bbox[0])
        return {
            "fill_ratio": mask_area / mask.size,
            "elongation": max(mask_w, mask_h) / max(1, min(mask_w, mask_h)),
            "extent": mask_area / (mask_w * mask_h),
            "in_L_std": float(L[inside].std()), "in_edge": float(grad[inside].mean()),
            "in_L_std_native": in_L_std_native, "in_edge_native": in_edge_native,
            "contrast_ring": float(L[inside].mean() - L[ring].mean()),
            "in_L": float(L[inside].mean()), "ring_L": float(L[ring].mean()),
            "in_a": float(a[inside].mean()), "ring_a": float(a[ring].mean()),
            "in_b": float(b[inside].mean()), "ring_b": float(b[ring].mean()),
            "crop_scale": self.img_size[0] / box_w,
            "total_upscale": self.img_size[0] * self.img_size[0] / (box_w * src_size[0]),
            "src_w": float(src_size[0]), "native_w": float(native_w),
        }, None

    def classify(self, crop, mask, bbox, src_size):
        """Returns (label, {class: prob}, note)."""
        feats, why = self.features(crop, mask, bbox, src_size)
        if feats is None:
            return None, None, f"Track A could not measure the mask: {why}"
        x = np.array([[feats[c] for c in self.columns]], dtype=float)
        probs = self.model.predict_proba(x)[0]
        label = self.classes[int(np.argmax(probs))]
        return label, {c: float(p) for c, p in zip(self.classes, probs)}, ""
