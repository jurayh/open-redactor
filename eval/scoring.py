"""Scoring core for the Open Redactor eval harness.

Metrics, grouped by lane:
- Detection: recall at IoU 0.5 and mean best IoU per ground truth box
- Continuity: gap rate inside track spans and longest uncovered run
- Leakage: how much identifying detail survives redaction inside a box,
  measured as high frequency energy after versus before, plus an OCR
  digit survival check for the text layer
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def score_detections(
    ground_truth: Dict[int, List[Box]],
    predictions: Dict[int, List[Box]],
    iou_threshold: float = 0.5,
) -> dict:
    """Score predicted boxes against ground truth, frame by frame."""
    gt_total = 0
    pred_total = 0
    hits = 0
    ious: List[float] = []
    for frame, gt_boxes in ground_truth.items():
        pred_boxes = predictions.get(frame, [])
        gt_total += len(gt_boxes)
        pred_total += len(pred_boxes)
        used: set[int] = set()
        for gt in gt_boxes:
            best, best_idx = 0.0, -1
            for i, pred in enumerate(pred_boxes):
                if i in used:
                    continue
                score = box_iou(gt, pred)
                if score > best:
                    best, best_idx = score, i
            ious.append(best)
            if best >= iou_threshold and best_idx >= 0:
                hits += 1
                used.add(best_idx)
    frames = sorted(ground_truth.keys())
    return {
        "gt_boxes": gt_total,
        "predicted_boxes": pred_total,
        "recall_at_iou": round(hits / gt_total, 4) if gt_total else 0.0,
        "mean_best_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "iou_threshold": iou_threshold,
        "frames_scored": len(frames),
    }


def continuity_stats(ground_truth_frames: Sequence[int], predicted_frames: Sequence[int]) -> dict:
    """Gap analysis over one element's span.

    Ground truth defines the span where the element exists. Any run of
    frames inside that span with no prediction is an uncovered gap, the
    exact failure our coverage report flags.
    """
    if not ground_truth_frames:
        return {"span": 0, "covered": 0, "coverage": 0.0, "longest_gap": 0, "gap_frames": 0}
    gt = sorted(set(ground_truth_frames))
    pred = set(predicted_frames)
    start, end = gt[0], gt[-1]
    span = end - start + 1
    covered = sum(1 for f in range(start, end + 1) if f in pred)
    longest = 0
    run = 0
    for f in range(start, end + 1):
        if f in pred:
            run = 0
        else:
            run += 1
            longest = max(longest, run)
    return {
        "span": span,
        "covered": covered,
        "coverage": round(covered / span, 4) if span else 0.0,
        "longest_gap": longest,
        "gap_frames": span - covered,
    }


def high_frequency_energy(frame_gray: np.ndarray, box: Box) -> float:
    """Mean absolute Laplacian inside a box. Detail a viewer could use."""
    import cv2

    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = frame_gray.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = frame_gray[y1:y2, x1:x2]
    lap = cv2.Laplacian(crop, cv2.CV_64F)
    return float(np.mean(np.abs(lap)))


def leakage_ratio(before_gray: np.ndarray, after_gray: np.ndarray, box: Box) -> float:
    """Detail surviving redaction. 1.0 means untouched, near 0 means gone."""
    before = high_frequency_energy(before_gray, box)
    after = high_frequency_energy(after_gray, box)
    if before <= 0:
        return 0.0
    return round(after / before, 4)


def grade(scorecard: dict) -> str:
    """One letter for the whole card, weighted toward leakage and recall."""
    recall = scorecard.get("detection", {}).get("recall_at_iou", 0.0)
    coverage = scorecard.get("continuity", {}).get("coverage", 0.0)
    leakage = scorecard.get("leakage_ratio", 1.0)
    score = 0.45 * recall + 0.35 * coverage + 0.20 * (1.0 - min(1.0, leakage))
    if score >= 0.9:
        return "A"
    if score >= 0.8:
        return "B"
    if score >= 0.65:
        return "C"
    if score >= 0.5:
        return "D"
    return "F"
