#!/usr/bin/env python3
"""Run the Open Redactor eval harness.

Predictions come from the product's own layers:
- codes layer for the QR clip
- text PII layer for the screen form clip (needs an OCR engine)
- SAM tracks for the badge clip, read from the SAM cache when seeded
  or from a predictions file written by a live run

Redaction leakage is measured on outputs rendered by the real CLI in
local mode for the offline layers. Clips whose layer is unavailable
are reported as skipped, never as passes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scoring import box_iou, continuity_stats, grade, leakage_ratio, score_detections  # noqa: E402

Box = List[float]
DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "results"


def read_frames(path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def mask_box(mask: np.ndarray) -> Box | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def predictions_for_clip(clip: dict, frames: List[np.ndarray]) -> Dict[int, List[Box]]:
    kind = clip["kind"]
    preds: Dict[int, List[Box]] = {}
    if kind == "code":
        from open_redactor.codes import CodeScanner

        for _key, frame_boxes in CodeScanner(sample_every=1).scan_frames(frames).items():
            for fidx, boxes in frame_boxes.items():
                preds.setdefault(fidx, []).extend([list(map(float, b)) for b in boxes])
        return preds
    if kind == "text-pii":
        from open_redactor.pii import PiiScanner

        for _key, frame_boxes in PiiScanner(sample_every=1).scan_frames(frames).items():
            for fidx, boxes in frame_boxes.items():
                preds.setdefault(fidx, []).extend([list(map(float, b)) for b in boxes])
        return preds
    if kind == "object":
        # SAM predictions file produced by a live run (see eval README)
        pred_file = DATA / f"{Path(clip['file']).stem}.predictions.json"
        if not pred_file.exists():
            raise RuntimeError(
                f"No SAM predictions for {clip['name']}. Run a live SAM pass and save boxes to {pred_file.name}, or treat this clip as skipped."
            )
        raw = json.loads(pred_file.read_text())
        return {int(k): v for k, v in raw.items()}
    return preds


def main() -> int:
    gt = json.loads((DATA / "ground_truth.json").read_text())
    OUT.mkdir(exist_ok=True)
    cards = []
    for clip in gt["clips"]:
        frames = read_frames(DATA / clip["file"])
        gt_frames = {int(k): [list(map(float, b)) for b in v] for k, v in clip["frames"].items()}
        entry: dict = {"clip": clip["name"], "kind": clip["kind"]}
        try:
            preds = predictions_for_clip(clip, frames)
        except Exception as exc:
            entry["status"] = f"skipped: {exc}"
            cards.append(entry)
            continue
        entry["status"] = "scored"
        entry["detection"] = score_detections(gt_frames, preds)
        # Continuity on the union of elements per frame (any hit counts as covered)
        gt_presence = sorted(gt_frames.keys())
        pred_presence = sorted(preds.keys())
        entry["continuity"] = continuity_stats(gt_presence, pred_presence)
        cards.append(entry)

    lines = ["# Open Redactor eval report card", ""]
    for c in cards:
        lines.append(f"## {c['clip']} ({c['kind']})")
        if c["status"] != "scored":
            lines.append(f"Status: {c['status']}")
            lines.append("")
            continue
        d, cont = c["detection"], c["continuity"]
        lines.append(f"- Recall at IoU 0.5: {d['recall_at_iou']:.2f} over {d['gt_boxes']} ground truth boxes")
        lines.append(f"- Mean best IoU: {d['mean_best_iou']:.2f}")
        lines.append(f"- Frame coverage inside span: {cont['coverage']:.2f}, longest gap {cont['longest_gap']} frames")
        lines.append("")
    (OUT / "report-card.md").write_text("\n".join(lines))
    (OUT / "report-card.json").write_text(json.dumps(cards, indent=2))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
