"""Copy the cetacean detector's own TEST photos (never used in training) into data/cetacean_test.

Run on your own machine (not in the container), from the app folder:
  python tools/copy_cetacean_test.py --info models/cetacean_detector_v2.json --out data/cetacean_test

One photo per photo group, the same 61 photos the end-to-end test scored.
Use --all for every test file (442 photos, including near-copies).
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--info", default="models/cetacean_detector_v2.json")
    p.add_argument("--out", default="data/cetacean_test")
    p.add_argument("--all", action="store_true")
    a = p.parse_args()

    with open(a.info) as f:
        info = json.load(f)
    entries = [e for e in info["files"]["test"] if e["kind"] == "cetacean"]
    if not a.all:
        first = {}
        for e in sorted(entries, key=lambda e: e["source"]):
            first.setdefault(e["group"], e)
        entries = list(first.values())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    copied, missing = 0, []
    for e in entries:
        src = Path(e["source"])
        if src.exists():
            shutil.copy2(src, out / src.name)
            copied += 1
        else:
            missing.append(str(src))
    print(f"Copied {copied} photos to {out}")
    if missing:
        print(f"{len(missing)} not found, e.g. {missing[0]}")


if __name__ == "__main__":
    main()
