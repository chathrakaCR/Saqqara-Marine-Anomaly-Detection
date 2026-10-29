"""Runs the pipeline over a video and writes the results.

Output folder:
  frames.csv          one row per analysed frame (answer, reason, confidences, boxes, timing)
  events.json         consecutive frames with the same answer, merged into sightings with start/end times
  summary.json        settings used, counts per answer, timing
  detections/         annotated JPEG of every frame whose answer is prawn, tuna or cetacean
  frames/             (optional, --save-frames) every analysed frame as extracted from the video
  annotated.mp4       (optional, --save-video) every analysed frame with the answer drawn on it
"""
from __future__ import annotations

import csv
import json
import logging
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2

from .annotate import draw
from .pipeline import LABELS, Pipeline
from .video import VideoSource

log = logging.getLogger("runner")

CSV_FIELDS = ["frame_index", "time_s", "source_file", "label", "decided_at", "reason", "stage1_flagged", "stage1_rule",
              "stage1_box", "cetacean_confidence", "cetacean_boxes", "stage2_mask_px", "stage2_max_prob",
              "track_a_label", "track_a_probs", "processing_ms", "image"]


def _fmt_box(b):
    return "" if b is None else ",".join(f"{v:.0f}" for v in b)


def build_events(rows, gap_s):
    events, cur = [], None
    for row in rows:
        lab = row["label"]
        if lab == "none":
            continue
        if cur and cur["label"] == lab and row["time_s"] - cur["end_s"] <= gap_s:
            cur["end_s"] = row["time_s"]; cur["frames"] += 1
            if (row["cetacean_confidence"] or 0) > (cur["best_confidence"] or 0):
                cur["best_confidence"] = row["cetacean_confidence"]
            cur["best_image"] = cur["best_image"] or row["image"]
        else:
            cur = {"label": lab, "start_s": row["time_s"], "end_s": row["time_s"], "frames": 1,
                   "best_confidence": row["cetacean_confidence"] if lab == "cetacean" else None,
                   "best_image": row["image"]}
            events.append(cur)
    return events


def process_video(source: str, out_dir, pipeline: Pipeline, sample_every_seconds: float | None = None,
                  every_n_frames: int | None = None, max_seconds: float | None = None, save_frames: bool = False,
                  save_video: bool = False, progress=None, stop_event: threading.Event | None = None) -> dict:
    out = Path(out_dir)
    (out / "detections").mkdir(parents=True, exist_ok=True)
    if save_frames:
        (out / "frames").mkdir(exist_ok=True)
    vcfg = pipeline.cfg["video"]
    every = sample_every_seconds or vcfg["sample_every_seconds"]
    src = VideoSource(source, every, every_n_frames, max_seconds)
    log.info("Source %s: %dx%d, %.2f fps, analysing every %d frames%s", source, src.width, src.height, src.fps,
             src.step, f" (~{src.expected_samples} frames)" if src.expected_samples else " (live)")

    writer, writer_size = None, None
    rows, counts = [], Counter()
    t_start = time.time()
    with open(out / "frames.csv", "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=CSV_FIELDS)
        w.writeheader()
        for fr in src.frames():
            if stop_event is not None and stop_event.is_set():
                log.info("Stopped by request")
                break
            r = pipeline.analyse(fr.image)
            counts[r.label] += 1
            stem = f"img_{fr.index:05d}_{Path(fr.name).stem}" if fr.name else f"frame_{fr.index:07d}_t{fr.time_s:09.2f}"
            if save_frames:
                cv2.imwrite(str(out / "frames" / f"{stem}.jpg"), fr.image)
            image_name = ""
            annotated = None
            if r.label != "none" or save_video:
                annotated = draw(fr.image, r, None if fr.name else fr.time_s)
            if r.label != "none":
                image_name = f"detections/{stem}_{r.label}.jpg"
                cv2.imwrite(str(out / image_name), annotated)
            if save_video:
                if writer is None:
                    h, wd = annotated.shape[:2]
                    fps_out = max(1.0, min(30.0, src.fps / src.step))
                    writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (wd, h))
                    writer_size = (wd, h)
                frame_out = annotated
                if (annotated.shape[1], annotated.shape[0]) != writer_size:      # folder images can differ in size
                    frame_out = cv2.resize(annotated, writer_size, interpolation=cv2.INTER_AREA)
                writer.write(frame_out)
            row = {"frame_index": fr.index, "time_s": fr.time_s, "source_file": fr.name, "label": r.label, "decided_at": r.decided_at,
                   "reason": r.reason, "stage1_flagged": r.stage1_flagged, "stage1_rule": r.stage1_rule,
                   "stage1_box": _fmt_box(r.stage1_box),
                   "cetacean_confidence": None if r.cetacean_confidence is None else round(r.cetacean_confidence, 4),
                   "cetacean_boxes": ";".join(f"{c:.3f}@{_fmt_box(b)}" for c, b in r.cetacean_boxes),
                   "stage2_mask_px": r.stage2_mask_px,
                   "stage2_max_prob": None if r.stage2_max_prob is None else round(r.stage2_max_prob, 4),
                   "track_a_label": r.track_a_label or "",
                   "track_a_probs": json.dumps({k: round(v, 4) for k, v in r.track_a_probs.items()}) if r.track_a_probs else "",
                   "processing_ms": r.processing_ms, "image": image_name}
            w.writerow(row)
            fcsv.flush()
            rows.append(row)
            if progress:
                progress(len(rows), src.expected_samples, row)
    if writer is not None:
        writer.release()

    # frames further apart than this start a new sighting (never less than 1.5 sampling intervals)
    gap = max(vcfg["event_gap_seconds"], 1.5 * (every if src.is_live else src.step / src.fps))
    events = build_events(rows, gap)
    with open(out / "events.json", "w") as f:
        json.dump(events, f, indent=2)
    elapsed = time.time() - t_start
    summary = {
        "source": source, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video": {"width": src.width, "height": src.height, "fps": round(src.fps, 3), "live": src.is_live, "image_folder": src.is_folder},
        "analysed_every_n_frames": src.step, "frames_analysed": len(rows),
        "counts": {k: counts.get(k, 0) for k in LABELS},
        "frames_flagged_by_stage1": sum(1 for r in rows if r["stage1_flagged"]),
        "events": len(events), "seconds_taken": round(elapsed, 1),
        "avg_ms_per_frame": round(sum(r["processing_ms"] for r in rows) / max(1, len(rows)), 1),
        "settings": pipeline.describe(),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return {"summary": summary, "events": events}
