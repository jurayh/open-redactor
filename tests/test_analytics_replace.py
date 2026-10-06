import numpy as np

from open_redactor.analytics import build_summary, severity_for, summary_text
from open_redactor.pii import luhn_ok
from open_redactor.replace import (
    apply_replacement,
    fake_card_number,
    fake_house_number,
    fake_plate,
)


def test_severity_levels():
    assert severity_for("passport") == "critical"
    assert severity_for("credit card number") == "critical"
    assert severity_for("face") == "high"
    assert severity_for("license plate") == "medium"


def test_summary_counts():
    tracks = {
        "face:0": {0: np.ones((4, 4), dtype=bool), 1: np.ones((4, 4), dtype=bool)},
        "pii:credit card number:1": {1: np.ones((4, 4), dtype=bool)},
    }
    s = build_summary(tracks, total_frames=4, audio_mode="mute")
    assert s["elements_found"] == 2
    assert s["severity_counts"]["critical"] == 1
    assert s["frames_with_sensitive_elements"] == 2
    assert "Most sensitive first" in summary_text(s)
    assert s["elements"][0]["severity"] == "critical"


def test_fake_card_is_luhn_valid_and_deterministic():
    a = fake_card_number("pii:credit card number:1")
    assert a == fake_card_number("pii:credit card number:1")
    assert a.startswith("4242")
    assert luhn_ok(a)
    assert fake_plate("x") != "" and fake_house_number("x").isdigit()


def test_replace_face_draws_avatar():
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    out = apply_replacement(frame, mask, "face:0")
    assert out.shape == frame.shape
    # Region changed and is no longer flat white
    assert np.abs(out.astype(int) - frame.astype(int)).sum() > 0
    assert out[50, 50].tolist() != [255, 255, 255]


def test_replace_card_patch():
    frame = np.full((60, 200, 3), 255, dtype=np.uint8)
    mask = np.zeros((60, 200), dtype=bool)
    mask[20:40, 10:190] = True
    out = apply_replacement(frame, mask, "pii:credit card number:1")
    assert out.shape == frame.shape
