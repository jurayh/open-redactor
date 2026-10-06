from pathlib import Path

import cv2
import numpy as np

from open_redactor.image_pipeline import SUPPORTED_IMAGE_SUFFIXES, run_image_pipeline


def make_photo(path: Path):
    img = np.full((120, 160, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_photo_local_clean_copy(tmp_path: Path):
    src = tmp_path / "photo.jpg"
    make_photo(src)
    out = tmp_path / "photo.redacted.jpg"
    summary = run_image_pipeline(
        input_path=src, output_path=out, targets=["face"], local=True, provider="sam"
    )
    assert out.exists()
    assert summary["photo"] is True
    read_back = cv2.imread(str(out))
    assert read_back is not None and read_back.shape == (120, 160, 3)


def test_photo_qr_codes_layer(tmp_path: Path):
    import io

    import segno
    from PIL import Image

    qr = segno.make("https://example.com/secret")
    buf = io.BytesIO()
    qr.save(buf, kind="png", scale=8, border=2)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((200, 200))
    canvas = np.full((240, 320, 3), 255, dtype=np.uint8)
    canvas[20:220, 60:260] = np.asarray(img)
    src = tmp_path / "qr.png"
    cv2.imwrite(str(src), canvas[:, :, ::-1].copy())
    out = tmp_path / "qr.redacted.png"
    summary = run_image_pipeline(
        input_path=src, output_path=out, targets=["face"], local=True, provider="sam", codes=True
    )
    assert summary["objects"] >= 1
    redacted = cv2.imread(str(out))
    original = cv2.imread(str(src))
    # The covered QR region must differ from the sharp original
    changed = np.abs(redacted.astype(int) - original.astype(int)).sum()
    assert changed > 1000


def test_supported_images():
    assert ".jpg" in SUPPORTED_IMAGE_SUFFIXES and ".png" in SUPPORTED_IMAGE_SUFFIXES
