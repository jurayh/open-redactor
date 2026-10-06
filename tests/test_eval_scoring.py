import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from scoring import box_iou, continuity_stats, grade, leakage_ratio, score_detections


def test_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_score_detections():
    gt = {0: [(0.0, 0.0, 10.0, 10.0)], 1: [(0.0, 0.0, 10.0, 10.0)]}
    pred = {0: [(0.0, 0.0, 10.0, 10.0)]}
    s = score_detections(gt, pred)
    assert s["recall_at_iou"] == 0.5


def test_continuity_gap():
    c = continuity_stats([0, 1, 2, 3, 4], [0, 1, 4])
    assert c["longest_gap"] == 2
    assert c["coverage"] == 0.6


def test_leakage_drops_after_blur():
    import cv2

    img = np.zeros((40, 40), dtype=np.uint8)
    img[::2, :] = 255
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    assert leakage_ratio(img, blurred, (0, 0, 40, 40)) < 0.3
    assert leakage_ratio(img, img, (0, 0, 40, 40)) == 1.0


def test_grade():
    card = {"detection": {"recall_at_iou": 1.0}, "continuity": {"coverage": 1.0}, "leakage_ratio": 0.05}
    assert grade(card) == "A"
