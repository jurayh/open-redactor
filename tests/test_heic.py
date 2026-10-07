"""HEIC and HEIF photo input: decode, redact, and write back in format."""

import io
from pathlib import Path

import cv2
import numpy as np
import pytest

pillow_heif = pytest.importorskip("pillow_heif")
from PIL import Image  # noqa: E402

from open_redactor.api import IMAGE_SUFFIXES  # noqa: E402
from open_redactor.image_pipeline import (  # noqa: E402
    SUPPORTED_IMAGE_SUFFIXES,
    read_photo,
    run_image_pipeline,
)

QR_PAYLOAD = "https://example.com/heic-secret"


def make_heic_photo(path: Path, with_qr: bool = True) -> Path:
    pillow_heif.register_heif_opener()
    canvas = np.full((240, 320, 3), 255, dtype=np.uint8)
    if with_qr:
        import segno

        qr = segno.make(QR_PAYLOAD)
        buf = io.BytesIO()
        qr.save(buf, kind="png", scale=8, border=2)
        buf.seek(0)
        qr_img = Image.open(buf).convert("RGB").resize((160, 160))
        canvas[40:200, 80:240] = np.asarray(qr_img)
    Image.fromarray(canvas, "RGB").save(str(path), quality=95)
    return path


def decode_qr_from(path: Path) -> str:
    frame = read_photo(path)
    data, _points, _straight = cv2.QRCodeDetector().detectAndDecode(frame)
    return data


def test_heic_suffixes_registered():
    assert ".heic" in SUPPORTED_IMAGE_SUFFIXES
    assert ".heif" in SUPPORTED_IMAGE_SUFFIXES
    assert ".heic" in IMAGE_SUFFIXES


def test_heic_read_photo_decodes_frame(tmp_path: Path):
    src = make_heic_photo(tmp_path / "plain.heic", with_qr=False)
    frame = read_photo(src)
    assert frame.shape == (240, 320, 3)


def test_heic_qr_covered_and_written_as_heic(tmp_path: Path):
    src = make_heic_photo(tmp_path / "qr.heic")
    assert decode_qr_from(src) == QR_PAYLOAD
    out = tmp_path / "qr.redacted.heic"
    result = run_image_pipeline(
        input_path=src, output_path=out, targets=["face"],
        local=True, provider="sam", codes=True,
    )
    assert out.exists()
    assert result["photo"] is True
    pillow_heif.register_heif_opener()
    with Image.open(out) as img:
        assert img.format == "HEIF"
    assert decode_qr_from(out) == ""


def test_heic_exclude_keeps_qr_visible(tmp_path: Path):
    src = make_heic_photo(tmp_path / "qr.heic")
    out = tmp_path / "qr.redacted.heic"
    result = run_image_pipeline(
        input_path=src, output_path=out, targets=["face"],
        local=True, provider="sam", codes=True, exclude_tracks=["code:*"],
    )
    assert decode_qr_from(out) == QR_PAYLOAD
    assert any(key.startswith("code:") for key in result["kept_visible"])


def test_heif_extension_also_works(tmp_path: Path):
    src = make_heic_photo(tmp_path / "qr.heif")
    out = tmp_path / "qr.redacted.heif"
    run_image_pipeline(
        input_path=src, output_path=out, targets=["face"],
        local=True, provider="sam", codes=True,
    )
    assert out.exists()
    assert decode_qr_from(out) == ""
