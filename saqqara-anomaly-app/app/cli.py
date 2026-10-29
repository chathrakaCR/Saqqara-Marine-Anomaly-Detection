"""Command line: analyse one video file, stream or camera.

  python -m app.cli --input /data/flight.mp4 --output /data/results/flight
  python -m app.cli --input rtsp://192.168.1.20:8554/drone --every 2 --max-seconds 600
"""
from __future__ import annotations

import argparse
import logging
import sys

from .pipeline import Pipeline
from .runner import process_video


def main(argv=None):
    p = argparse.ArgumentParser(description="Marine anomaly detection on a video (P000285DS pipeline)")
    p.add_argument("--input", "-i", required=True, help="video file, rtsp/http stream URL, or camera number (e.g. 0)")
    p.add_argument("--output", "-o", default="/data/results/run", help="folder for the results")
    p.add_argument("--every", type=float, default=None, help="analyse one frame every N seconds (default from config: 1.0)")
    p.add_argument("--every-n-frames", type=int, default=None, help="analyse every N-th frame instead (files only)")
    p.add_argument("--max-seconds", type=float, default=None, help="stop after this many seconds of video")
    p.add_argument("--save-frames", action="store_true", help="also save every analysed frame as a JPEG")
    p.add_argument("--save-video", action="store_true", help="also write annotated.mp4")
    p.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pipe = Pipeline(device=a.device)

    def progress(done, total, row):
        tag = f"{done}/{total}" if total else str(done)
        print(f"[{tag}] t={row['time_s']:8.2f}s  {row['label']:<9} {row['reason']}  ({row['processing_ms']:.0f} ms)",
              flush=True)

    res = process_video(a.input, a.output, pipe, a.every, a.every_n_frames, a.max_seconds, a.save_frames,
                        a.save_video, progress=progress)
    s = res["summary"]
    print(f"\nDone: {s['frames_analysed']} frames, counts {s['counts']}, {s['events']} sightings. Results in {a.output}")
    for e in res["events"]:
        print(f"  {e['label']:<9} {e['start_s']:8.1f}s - {e['end_s']:8.1f}s  ({e['frames']} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
