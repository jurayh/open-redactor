import subprocess
from pathlib import Path

from open_redactor.pipeline import audio_filter_for


def make_av_clip(path: Path):
    subprocess.check_call([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "color=c=blue:size=320x240:rate=10:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ])


def audio_streams(path: Path) -> int:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
    ], text=True).strip()
    return len(out.splitlines()) if out else 0


def test_filter_mapping():
    assert audio_filter_for("keep") is None
    assert audio_filter_for("mute") == ""
    assert "rubberband=pitch=0.8" in audio_filter_for("pitch")


def test_mute_and_pitch_end_to_end(tmp_path: Path):
    from open_redactor.cli import main

    src = tmp_path / "clip.mp4"
    make_av_clip(src)
    muted = tmp_path / "muted.mp4"
    assert main([str(src), "--local", "--audio", "mute", "--output", str(muted)]) == 0
    assert audio_streams(muted) == 0

    pitched = tmp_path / "pitched.mp4"
    assert main([str(src), "--local", "--audio", "pitch", "--output", str(pitched)]) == 0
    assert audio_streams(pitched) == 1

    kept = tmp_path / "kept.mp4"
    assert main([str(src), "--local", "--output", str(kept)]) == 0
    assert audio_streams(kept) == 1
