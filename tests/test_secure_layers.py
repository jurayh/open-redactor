import cv2
import numpy as np

from open_redactor.codes import CodeScanner
from open_redactor.pipeline import apply_secure_fill
from tests.test_codes import make_qr_frame


def test_secure_fill_makes_qr_undecodable():
    frame = make_qr_frame()
    tracks = CodeScanner(sample_every=1).scan_frames([frame])
    assert tracks
    mask = np.zeros(frame.shape[:2], dtype=bool)
    for frame_boxes in tracks.values():
        for x1, y1, x2, y2 in frame_boxes[0]:
            mask[y1:y2, x1:x2] = True

    filled = apply_secure_fill(frame, mask)
    data, points, _straight = cv2.QRCodeDetector().detectAndDecode(filled)

    assert data == ""
    assert np.all(filled[mask] == 0)


def test_secure_fill_leaves_other_pixels_alone():
    frame = np.full((8, 8, 3), 200, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:4, 2:4] = True
    filled = apply_secure_fill(frame, mask)
    assert np.all(filled[~mask] == 200)
    assert np.all(filled[mask] == 0)
