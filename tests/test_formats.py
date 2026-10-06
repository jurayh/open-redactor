from pathlib import Path

import pytest

from open_redactor.pipeline import SUPPORTED_INPUT_SUFFIXES, ensure_mp4_for_api, probe_video


def test_supported_set():
    assert {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"} <= SUPPORTED_INPUT_SUFFIXES


def test_probe_rejects_unknown(tmp_path: Path):
    f = tmp_path / "clip.xyz"
    f.write_bytes(b"nope")
    with pytest.raises(ValueError, match="Unsupported input format"):
        probe_video(f)


def test_ensure_mp4_passthrough(tmp_path: Path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x")
    assert ensure_mp4_for_api(f, tmp_path) == f
