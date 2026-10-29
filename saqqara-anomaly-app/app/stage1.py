"""Stage 1 - unsupervised anomaly detector (texture rule OR local-brightness rule).

Ported unchanged from Stage_01_Pipeline_FINAL_v2 / Stage_1234_Integrated_Pipeline_Test (cell 3).
The only change: it takes a decoded video frame (BGR array) instead of a file path.
"""
from __future__ import annotations

import pickle

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.segmentation import slic

FEATURE_KEYS = ["contrast_L", "homogeneity_L", "energy_L", "contrast_a", "lbp_var"]


class Stage1Detector:
    def __init__(self, cfg: dict, bucket_ref_path):
        self.img_size = tuple(cfg["img_size"])
        self.n_segments = cfg["n_segments"]
        self.compactness = cfg["compactness"]
        self.glcm_levels = cfg["glcm_levels"]
        self.glint_thresh = cfg["glint_thresh"]
        self.bucket_edges = np.array(cfg["bucket_edges"])
        self.z_thresh = cfg["z_thresh"]
        self.min_features_flagged = cfg["min_features_flagged"]
        self.min_component_area_px = cfg["min_component_area_px"]
        self.local_z_thresh = cfg["local_z_thresh"]
        self.min_area_local = cfg["min_area_local"]
        self.dilation_iters = cfg["dilation_iters"]
        with open(bucket_ref_path, "rb") as f:
            self.bucket_ref = pickle.load(f)

    # ---- preprocessing ----
    def suppress_glint(self, img_bgr, dilate_iter=2):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        glint_mask = (lab[:, :, 0] > self.glint_thresh).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        glint_mask = cv2.dilate(glint_mask, kernel, iterations=dilate_iter)
        return cv2.morphologyEx(glint_mask, cv2.MORPH_CLOSE, kernel)

    def preprocess(self, frame_bgr):
        img_bgr = cv2.resize(frame_bgr, self.img_size)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        glint_mask = self.suppress_glint(img_bgr)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return lab, glint_mask, rgb

    # ---- superpixel texture features ----
    def _quantize(self, patch):
        return (patch.astype(np.float32) / 256 * self.glcm_levels).astype(np.uint8)

    def _glcm_props(self, patch_q):
        glcm = graycomatrix(patch_q, distances=[1], angles=[0], levels=self.glcm_levels, symmetric=True, normed=True)
        return (float(graycoprops(glcm, "contrast")[0, 0]), float(graycoprops(glcm, "homogeneity")[0, 0]),
                float(graycoprops(glcm, "energy")[0, 0]))

    def superpixel_features(self, img_lab, glint_mask):
        L_channel, a_channel = img_lab[:, :, 0], img_lab[:, :, 1]
        segments = slic(img_lab, n_segments=self.n_segments, compactness=self.compactness, start_label=1)
        features = []
        for seg_id, bbox in enumerate(ndimage.find_objects(segments), start=1):
            if bbox is None:
                continue
            y_slice, x_slice = bbox
            local_mask = segments[y_slice, x_slice] == seg_id
            if glint_mask[y_slice, x_slice][local_mask].mean() > 0.5:
                continue
            patch_L, patch_a = L_channel[y_slice, x_slice], a_channel[y_slice, x_slice]
            if patch_L.shape[0] < 2 or patch_L.shape[1] < 2:
                continue
            contrast_L, homogeneity_L, energy_L = self._glcm_props(self._quantize(patch_L))
            contrast_a, _, _ = self._glcm_props(self._quantize(patch_a))
            lbp_var = float(np.var(local_binary_pattern(patch_L, P=8, R=1, method="uniform")))
            features.append({"seg_id": int(seg_id), "contrast_L": contrast_L, "homogeneity_L": homogeneity_L,
                             "energy_L": energy_L, "contrast_a": contrast_a, "lbp_var": lbp_var,
                             "mean_L": float(patch_L.mean())})
        return segments, features

    # ---- sea-state bucket ----
    def bucket(self, img_lab, glint_mask):
        L = img_lab[:, :, 0]
        valid = glint_mask == 0
        score = float(np.mean(L[valid] if valid.any() else L))
        return int(np.searchsorted(self.bucket_edges, score))

    # ---- the two rules ----
    def _region_with_bbox(self, flagged_mask, min_area):
        if self.dilation_iters > 0:
            flagged_mask = ndimage.binary_dilation(flagged_mask, iterations=self.dilation_iters)
        labeled, n_components = ndimage.label(flagged_mask)
        if n_components == 0:
            return False, 0, None
        sizes = ndimage.sum(flagged_mask, labeled, range(1, n_components + 1))
        biggest = int(np.argmax(sizes)) + 1
        max_component = float(sizes[biggest - 1])
        if max_component < min_area:
            return False, max_component, None
        y_slice, x_slice = ndimage.find_objects(labeled)[biggest - 1]
        return True, max_component, (x_slice.start, y_slice.start, x_slice.stop, y_slice.stop)

    def texture_rule(self, segments, feats, ref):
        def deviates(f):
            z = [abs(f[k] - ref[k][0]) / (ref[k][1] + 1e-6) for k in FEATURE_KEYS]
            return sum(1 for v in z if v > self.z_thresh) >= self.min_features_flagged
        ids = {f["seg_id"] for f in feats if deviates(f)}
        if not ids:
            return False, 0, None
        return self._region_with_bbox(np.isin(segments, list(ids)), self.min_component_area_px)

    def local_brightness_rule(self, segments, feats):
        if not feats:
            return False, 0, None
        mean_Ls = np.array([f["mean_L"] for f in feats])
        m, s = mean_Ls.mean(), mean_Ls.std()
        ids = {f["seg_id"] for f in feats if abs(f["mean_L"] - m) / (s + 1e-6) > self.local_z_thresh}
        if not ids:
            return False, 0, None
        return self._region_with_bbox(np.isin(segments, list(ids)), self.min_area_local)

    def run(self, frame_bgr):
        """Returns dict: flagged, box (512x512 frame coords or None), rule, rgb (512x512 uint8)."""
        lab, glint_mask, rgb = self.preprocess(frame_bgr)
        segments, feats = self.superpixel_features(lab, glint_mask)
        ref = self.bucket_ref[self.bucket(lab, glint_mask)]
        tex = self.texture_rule(segments, feats, ref)
        loc = self.local_brightness_rule(segments, feats)
        if tex[0] and loc[0]:
            flagged, area, box = tex if tex[1] >= loc[1] else loc
            rule = "texture+brightness"
        elif tex[0]:
            flagged, area, box = tex
            rule = "texture"
        elif loc[0]:
            flagged, area, box = loc
            rule = "brightness"
        else:
            flagged, area, box, rule = False, 0, None, ""
        return {"flagged": bool(flagged and box is not None), "box": box, "area": float(area), "rule": rule, "rgb": rgb}
