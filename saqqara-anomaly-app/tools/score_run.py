"""Compare a run's answers with the right ones: confusion matrix, accuracy, precision / recall / F1.

Test video made by make_test_video.py:
  python tools/score_run.py --results /data/results/test_video --truth /data/test_video_truth.csv
Folder of photos that are all one class:
  python tools/score_run.py --results /data/results/tuna_folder --label tuna
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CLASSES = ["prawn", "tuna", "cetacean", "none"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="the run's output folder (holds frames.csv)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--truth", help="truth CSV written by make_test_video.py")
    g.add_argument("--label", help="every frame in this run should be this class (prawn, tuna, cetacean, none)")
    a = p.parse_args()

    with open(Path(a.results) / "frames.csv") as f:
        rows = list(csv.DictReader(f))
    if a.truth:
        with open(a.truth) as f:
            segs = [(float(r["start_s"]), float(r["end_s"]), r["label"], r["file"]) for r in csv.DictReader(f)]
    pairs = []
    for r in rows:
        if a.label:
            pairs.append((a.label, r["label"], r.get("source_file", "")))
            continue
        t = float(r["time_s"])
        seg = next((s for s in segs if s[0] <= t + 1e-6 < s[1]), None)
        if seg:
            pairs.append((seg[2], r["label"], seg[3]))

    if not pairs:
        raise SystemExit("No frames to score")
    true_classes = [c for c in CLASSES if any(t == c for t, _, _ in pairs)]
    cm = {t: {p_: 0 for p_ in CLASSES} for t in true_classes}
    for t, p_, _ in pairs:
        cm[t][p_] += 1
    correct = sum(cm[c][c] for c in true_classes)

    print(f"\n{len(pairs)} frames scored. Rows = right answer, columns = app's answer\n")
    print(f"{'':<12}" + "".join(f"{c:>10}" for c in CLASSES) + f"{'total':>9}")
    for t in true_classes:
        print(f"{t:<12}" + "".join(f"{cm[t][c]:>10}" for c in CLASSES) + f"{sum(cm[t].values()):>9}")
    print(f"\nAccuracy: {correct}/{len(pairs)} = {correct / len(pairs):.1%}\n")

    metrics = {}
    print(f"{'class':<12}{'precision':>10}{'recall':>10}{'F1':>8}")
    for c in true_classes:
        tp = cm[c][c]
        pred = sum(cm[t][c] for t in true_classes)
        prec = tp / pred if pred else 0.0
        rec = tp / sum(cm[c].values())
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        metrics[c] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}
        print(f"{c:<12}{prec:>10.3f}{rec:>10.3f}{f1:>8.3f}")
    if len(true_classes) > 1:
        macro = sum(m["f1"] for m in metrics.values()) / len(metrics)
        print(f"{'macro F1':<12}{'':>20}{macro:>8.3f}")
    wrong = [(t, p_, f) for t, p_, f in pairs if t != p_]
    if wrong:
        print(f"\nWrong answers ({len(wrong)}):")
        for t, p_, f in wrong[:40]:
            print(f"  {f or '?':<50} right: {t:<9} app: {p_}")
    with open(Path(a.results) / "score.json", "w") as f:
        json.dump({"confusion": cm, "accuracy": correct / len(pairs), "per_class": metrics,
                   "frames": len(pairs)}, f, indent=2)
    print(f"\nSaved {Path(a.results) / 'score.json'}")


if __name__ == "__main__":
    main()
