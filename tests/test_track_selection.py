"""Keep and exclude track selection: unit tests plus end to end runs."""

import io
import json
from pathlib import Path

import cv2
import numpy as np

from open_redactor.image_pipeline import run_image_pipeline
from open_redactor.masks import select_tracks
from open_redactor.pipeline import run_pipeline

QR_PAYLOAD = "https://example.com/secret"


def _tracks(*keys: str):
    return {key: {0: np.ones((4, 4), dtype=bool)} for key in keys}


def test_select_no_patterns_redacts_everything():
    redacted, kept, excluded, unmatched = select_tracks(_tracks("person:0", "face:0"))
    assert sorted(redacted) == ["face:0", "person:0"]
    assert kept == [] and excluded == [] and unmatched == []


def test_select_keep_exact_and_glob():
    redacted, kept, excluded, unmatched = select_tracks(
        _tracks("person:0", "person:1", "face:0"), keep=["person:0"]
    )
    assert kept == ["person:0"]
    assert sorted(redacted) == ["face:0", "person:1"]
    redacted, kept, _, _ = select_tracks(
        _tracks("person:0", "person:1", "face:0"), keep=["person:*"]
    )
    assert kept == ["person:0", "person:1"]
    assert sorted(redacted) == ["face:0"]


def test_select_exclude_and_unmatched():
    redacted, kept, excluded, unmatched = select_tracks(
        _tracks("person:0", "code:qr-0"), exclude=["code:qr-0"], keep=["missing:*"]
    )
    assert excluded == ["code:qr-0"]
    assert kept == []
    assert sorted(redacted) == ["person:0"]
    assert unmatched == ["missing:*"]


def _qr_frame(width=320, height=240) -> np.ndarray:
    import segno
    from PIL import Image

    qr = segno.make(QR_PAYLOAD)
    buf = io.BytesIO()
    qr.save(buf, kind="png", scale=8, border=2)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((160, 160))
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[40:200, 80:240] = np.asarray(img)
    return canvas[:, :, ::-1].copy()


def _write_qr_clip(path: Path, frames: int = 10) -> Path:
    frame = _qr_frame()
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(frames):
        writer.write(frame)
    writer.release()
    return path


def _decode_first_frame(path: Path) -> str:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    assert ok
    data, _points, _straight = cv2.QRCodeDetector().detectAndDecode(frame)
    return data


def _run_qr_video(tmp_path: Path, **kwargs):
    src = _write_qr_clip(tmp_path / "qr.mp4")
    out = tmp_path / "qr.redacted.mp4"
    details = run_pipeline(
        input_path=src,
        output_path=out,
        targets=["face"],
        local=True,
        provider="sam",
        codes=True,
        **kwargs,
    )
    return out, details


def test_video_default_run_covers_qr(tmp_path: Path):
    out, _details = _run_qr_video(tmp_path)
    assert _decode_first_frame(out) == ""


def test_video_exclude_leaves_qr_visible(tmp_path: Path):
    out, details = _run_qr_video(tmp_path, exclude_tracks=["code:*"])
    assert _decode_first_frame(out) == QR_PAYLOAD
    summary = json.loads(Path(details["summary"]).read_text()) if details.get("summary") else None
    assert summary is None or summary["elements_found"] >= 0


def test_video_shadow_audit_reports_kept_visible(tmp_path: Path):
    _out, details = _run_qr_video(tmp_path, shadow=True, keep_tracks=["code:*"])
    summary = json.loads(Path(details["summary"]).read_text())
    assert summary["elements_found"] == 0
    assert any(key.startswith("code:") for key in summary["kept_visible"])


def test_video_shadow_audit_without_selection_finds_code(tmp_path: Path):
    _out, details = _run_qr_video(tmp_path, shadow=True)
    summary = json.loads(Path(details["summary"]).read_text())
    assert summary["elements_found"] >= 1
    assert summary["kept_visible"] == []


def test_photo_exclude_leaves_qr_visible(tmp_path: Path):
    src = tmp_path / "qr.png"
    cv2.imwrite(str(src), _qr_frame(320, 240))
    out = tmp_path / "qr.redacted.png"
    result = run_image_pipeline(
        input_path=src,
        output_path=out,
        targets=["face"],
        local=True,
        provider="sam",
        codes=True,
        exclude_tracks=["code:*"],
    )
    img = cv2.imread(str(out))
    data, _points, _straight = cv2.QRCodeDetector().detectAndDecode(img)
    assert data == QR_PAYLOAD
    assert any(key.startswith("code:") for key in result["kept_visible"])


def test_photo_default_run_covers_qr(tmp_path: Path):
    src = tmp_path / "qr.png"
    cv2.imwrite(str(src), _qr_frame(320, 240))
    out = tmp_path / "qr.redacted.png"
    run_image_pipeline(
        input_path=src,
        output_path=out,
        targets=["face"],
        local=True,
        provider="sam",
        codes=True,
    )
    img = cv2.imread(str(out))
    data, _points, _straight = cv2.QRCodeDetector().detectAndDecode(img)
    assert data == ""
