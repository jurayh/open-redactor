"""0.2.9 trust bundle: adaptive blur, mask source, doctor, batch roll-up."""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from open_redactor.analytics import build_summary, summary_text
from open_redactor.cli import main as cli_main
from open_redactor.cli import write_batch_rollup
from open_redactor.doctor import main as doctor_main
from open_redactor.pipeline import (
    apply_redaction_to_frame,
    blur_kernel_for_region,
    build_blur_layers,
    track_typical_min_dim,
)


def test_kernel_scales_with_region_size():
    assert blur_kernel_for_region(21, 20) == 21
    assert blur_kernel_for_region(21, 34) == 41
    assert blur_kernel_for_region(21, 82) == 81
    assert blur_kernel_for_region(21, 170) == 161
    assert blur_kernel_for_region(21, 900) == 301
    # A caller strength above the buckets stays in charge
    assert blur_kernel_for_region(101, 30) == 101


def test_track_typical_min_dim_uses_median():
    small = np.zeros((100, 100), dtype=bool)
    small[10:40, 10:30] = True  # 20 wide, 30 tall, min dim 20
    big = np.zeros((100, 100), dtype=bool)
    big[0:90, 0:90] = True  # min dim 90
    assert track_typical_min_dim({0: small, 1: small, 2: big}) == 20
    assert track_typical_min_dim({}) == 0


def test_adaptive_layers_hide_a_large_pattern():
    # A checkerboard stands in for plate characters: high contrast, regular.
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    board = (np.indices((80, 320)).sum(axis=0) // 8) % 2
    frame[60:140, 40:360] = (board * 255).astype(np.uint8)[..., None]
    mask = np.zeros((200, 400), dtype=bool)
    mask[60:140, 40:360] = True
    tracks = {"license plate:0": {0: mask}}

    base = apply_redaction_to_frame(frame, mask, mode="blur", strength=21)
    layers = build_blur_layers(tracks, total_frames=1, shape=(200, 400), margin=0, carry_frames=0, base_strength=21)
    assert len(layers) == 1 and layers[0][1] == 81
    adapted = frame
    for layer_masks, kernel in layers:
        adapted = apply_redaction_to_frame(adapted, layer_masks[0], mode="blur", strength=kernel)

    def high_freq_energy(img: np.ndarray) -> float:
        gray = img[60:140, 40:360].astype(float).mean(axis=2)
        return float(np.abs(np.diff(gray, axis=1)).mean())

    assert high_freq_energy(adapted) < high_freq_energy(base) * 0.5


def test_summary_records_mask_source():
    summary = build_summary({}, 10, sam_mask_source="pixel")
    assert summary["sam_mask_source"] == "pixel"
    assert "SAM mask source: pixel" in summary_text(summary)
    default = build_summary({}, 10)
    assert default["sam_mask_source"] == "none"
    assert "SAM mask source" not in summary_text(default)


def test_doctor_reports_and_matches_core_availability(capsys):
    code = doctor_main([])
    out = capsys.readouterr().out
    assert "Open Redactor doctor" in out
    assert "Pixel mask decoder" in out
    assert "API key env var" in out
    expected = 0 if (shutil.which("ffmpeg") and shutil.which("ffprobe")) else 1
    assert code == expected


def test_batch_rollup_sums_summaries(tmp_path: Path):
    def write_summary(name: str, elements: int, counts: dict, kept: int) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps({
            "elements_found": elements,
            "severity_counts": counts,
            "frames_with_sensitive_elements": 5,
            "kept_visible": ["person:0"] * kept,
            "exposure_score": 12.0,
        }))
        return path

    a = write_summary("a.summary.json", 2, {"critical": 1, "high": 1, "medium": 0, "low": 0}, 1)
    b = write_summary("b.summary.json", 3, {"critical": 0, "high": 2, "medium": 1, "low": 0}, 0)
    dest = write_batch_rollup([a, b], tmp_path)
    rollup = json.loads(dest.read_text())
    assert rollup["files_processed"] == 2
    assert rollup["elements_found"] == 5
    assert rollup["severity_counts"] == {"critical": 1, "high": 3, "medium": 1, "low": 0}
    assert rollup["kept_visible_count"] == 1


def _tiny_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(5):
        writer.write(np.full((48, 64, 3), 120, dtype=np.uint8))
    writer.release()


def test_cli_batch_writes_rollup(tmp_path: Path):
    _tiny_video(tmp_path / "one.mp4")
    _tiny_video(tmp_path / "two.mp4")
    code = cli_main([str(tmp_path), "--batch", "--local"])
    assert code == 0
    rollup_path = tmp_path / "batch-summary.json"
    assert rollup_path.exists()
    rollup = json.loads(rollup_path.read_text())
    assert rollup["files_processed"] == 2
