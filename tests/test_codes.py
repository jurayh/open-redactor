import numpy as np

from open_redactor.codes import CodeScanner, boxes_from_points, detect_codes_in_frame
from open_redactor.presets import PRESETS


def make_qr_frame():
    import io

    import segno
    from PIL import Image

    qr = segno.make("https://example.com/secret-wifi-password")
    buf = io.BytesIO()
    qr.save(buf, kind="png", scale=8, border=2)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((240, 240))
    canvas = np.full((240, 320, 3), 255, dtype=np.uint8)
    canvas[0:240, 40:280] = np.asarray(img)
    return canvas[:, :, ::-1].copy()  # to BGR like pipeline frames


def test_boxes_from_points():
    pts = np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]])
    assert boxes_from_points(pts) == [(0, 0, 10, 10)]
    assert boxes_from_points(None) == []


def test_detect_real_qr():
    frame = make_qr_frame()
    boxes = detect_codes_in_frame(frame)
    assert len(boxes) >= 1
    x1, y1, x2, y2 = boxes[0]
    assert x2 > x1 and y2 > y1


def test_scanner_holds_box():
    frame = make_qr_frame()
    frames = [frame.copy() for _ in range(15)]
    result = CodeScanner(sample_every=10).scan_frames(frames)
    assert result
    first = next(iter(result.values()))
    assert 0 in first and 9 in first


def test_location_and_badges():
    assert "house number" in PRESETS["location"]["targets"]
    assert "street sign" in PRESETS["location"]["targets"]
    assert "name badge" in PRESETS["documents"]["targets"]
