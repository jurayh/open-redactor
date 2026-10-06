"""QR code and barcode scanning for sampled frames.

Uses OpenCV detectors that ship with opencv-python. QR codes carry URLs,
Wi-Fi passwords, payment links, and contact cards, so any readable code
in frame is treated as sensitive and covered. Boxes are held between
samples the same way text PII boxes are.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

Box = Tuple[int, int, int, int]


def boxes_from_points(points) -> List[Box]:
    """Turn detector corner points into axis aligned boxes."""
    boxes: List[Box] = []
    if points is None:
        return boxes
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return boxes
    # Shapes seen in the wild: (n, 4, 2), (n, 1, 4, 2), or (4, 2) for one code
    arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1]) if arr.ndim >= 3 else arr.reshape(1, -1, 2)
    for poly in arr:
        xs, ys = poly[:, 0], poly[:, 1]
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    return boxes


def detect_codes_in_frame(frame) -> List[Box]:
    """Detect QR codes, and barcodes when the OpenCV build provides them."""
    import cv2

    boxes: List[Box] = []
    try:
        detector = cv2.QRCodeDetector()
        ok, decoded, points, _ = detector.detectAndDecodeMulti(frame)
        if points is not None:
            boxes.extend(boxes_from_points(points))
        elif ok:
            single_ok, single_points = detector.detect(frame)
            if single_ok:
                boxes.extend(boxes_from_points(single_points))
    except Exception:
        pass
    # Barcode detector lives in opencv-contrib builds only
    try:
        barcode_detector = cv2.barcode.BarcodeDetector()  # type: ignore[attr-defined]
        ok, decoded_info, decoded_type, points = barcode_detector.detectAndDecodeWithType(frame)
        if points is not None:
            boxes.extend(boxes_from_points(points))
    except Exception:
        pass
    return boxes


class CodeScanner:
    """Sample frames, detect codes, and hold each box until the next sample."""

    def __init__(self, sample_every: int = 10) -> None:
        self.sample_every = sample_every

    def scan_frames(self, frames) -> Dict[str, Dict[int, List[Box]]]:
        result: Dict[str, Dict[int, List[Box]]] = {}
        total = len(frames)
        count = 0
        for idx in range(0, total, max(1, self.sample_every)):
            boxes = detect_codes_in_frame(frames[idx])
            for box in boxes:
                count += 1
                key = f"code:{count}"
                hold_until = min(total, idx + max(1, self.sample_every))
                for f in range(idx, hold_until):
                    result.setdefault(key, {}).setdefault(f, []).append(box)
                print(f"Code: QR or barcode covered near frame {idx}")
        return result
