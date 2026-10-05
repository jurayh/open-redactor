import numpy as np

from open_redactor.masks import build_coverage_report


def test_coverage_report_zero_tracks():
    text = build_coverage_report(tracks={}, total_frames=10, carry_frames=4, targets=["face"])
    assert "nothing matched" in text
    assert "Total frames: 10" in text


def test_coverage_report_counts_track():
    m = np.ones((4, 4), dtype=bool)
    tracks = {"face:0": {0: m, 1: m, 2: m}}
    text = build_coverage_report(tracks=tracks, total_frames=5, carry_frames=4, targets=["face"])
    assert "Track face:0" in text
    assert "3 detected frames" in text
    assert "Uncovered gaps" in text


def test_coverage_report_flags_long_gap():
    m = np.ones((4, 4), dtype=bool)
    tracks = {"face:0": {0: m, 10: m}}
    text = build_coverage_report(tracks=tracks, total_frames=12, carry_frames=2, targets=["face"])
    assert "NEEDS REVIEW" in text
