"""Spoken PII layer: transcript words to muted spans."""

import json
import subprocess
from pathlib import Path

import numpy as np

from open_redactor.cli import main as cli_main
from open_redactor.speech import (
    SpeechHit,
    WordSpan,
    find_speech_pii,
    load_transcript,
    mute_span_filter,
)


def words(text: str, start: float = 0.0, step: float = 0.3) -> list[WordSpan]:
    return [
        WordSpan(w, start + i * step, start + i * step + 0.25)
        for i, w in enumerate(text.split())
    ]


def test_spoken_phone_number_words():
    hits = find_speech_pii(words("call me at four one five five five five two six seven one please"))
    assert len(hits) == 1
    assert hits[0].kind == "phone number"


def test_spoken_card_requires_luhn():
    good = "four two four two four two four two four two four two four two four two"
    hits = find_speech_pii(words(f"my card is {good}"))
    assert len(hits) == 1 and hits[0].kind == "credit card number"
    bad = "four two four two four two four two four two four two four two four three"
    assert find_speech_pii(words(f"my card is {bad}")) == []


def test_spoken_ssn_uses_issuing_rules():
    hits = find_speech_pii(words("the number is zero seven eight zero five one one two three"))
    assert len(hits) == 1 and hits[0].kind == "ssn"
    # Area 666 is never issued
    assert find_speech_pii(words("six six six zero five one one two three")) == []


def test_digit_tokens_from_transcriber():
    hits = find_speech_pii(words("call 415 555 2671 today"))
    assert len(hits) == 1 and hits[0].kind == "phone number"


def test_spoken_email_from_at_dot_tokens():
    hits = find_speech_pii(words("write to alex dot example at mail dot com soon"))
    assert len(hits) == 1
    assert hits[0].kind == "email"
    assert hits[0].text == "alex.example@mail.com"


def test_clean_speech_has_no_hits():
    assert find_speech_pii(words("we went to the park and played all afternoon")) == []


def test_margin_padding_applies():
    hits = find_speech_pii(words("four one five five five five two six seven one", start=5.0), margin=0.5)
    assert len(hits) == 1
    assert hits[0].start == 4.5


def test_load_transcript_segments_shape(tmp_path: Path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"segments": [{"words": [
        {"word": " hello", "start": 0.0, "end": 0.4},
        {"word": " world", "start": 0.4, "end": 0.9},
    ]}]}))
    loaded = load_transcript(path)
    assert [w.word for w in loaded] == ["hello", "world"]
    assert loaded[1].end == 0.9


def test_mute_span_filter_format():
    out = mute_span_filter([SpeechHit("phone number", 1.0, 2.5, "x")])
    assert "between(t,1.00,2.50)" in out
    assert "volume=0" in out


def _video_with_silent_audio(path: Path) -> None:
    subprocess.check_call([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "color=c=gray:s=64x48:r=10:d=3",
        "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
        "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        str(path),
    ])


def test_cli_transcript_run_lands_in_summary(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    _video_with_silent_audio(video)
    transcript = tmp_path / "clip.json"
    transcript.write_text(json.dumps({"words": [
        {"word": w.word, "start": w.start, "end": w.end}
        for w in words("call four one five five five five two six seven one now", start=0.2, step=0.2)
    ]}))
    code = cli_main([str(video), "--local", "--transcript", str(transcript)])
    assert code == 0
    summary = json.loads((tmp_path / "clip.redacted.summary.json").read_text())
    speech = [e for e in summary["elements"] if e["track"].startswith("speech:")]
    assert len(speech) == 1
    assert speech[0]["kind"] == "phone number"
    assert speech[0]["severity"] == "high"


def test_cli_speech_pii_without_transcriber_errors(tmp_path: Path, monkeypatch):
    import open_redactor.speech as speech_module

    monkeypatch.setattr(speech_module, "transcriber_available", lambda: False)
    video = tmp_path / "clip.mp4"
    _video_with_silent_audio(video)
    code = cli_main([str(video), "--local", "--speech-pii"])
    assert code == 2


def test_glued_digit_tokens_from_real_transcribers():
    # Whisper often emits a spoken number as digit tokens with the
    # separators still attached, or the whole number as one token.
    spaced = [
        WordSpan("415", 3.1, 3.5),
        WordSpan("-555", 3.9, 4.4),
        WordSpan("-2671", 5.5, 6.2),
    ]
    hits = find_speech_pii(spaced)
    assert len(hits) == 1 and hits[0].kind == "phone number"
    glued = [WordSpan("415-555-2671", 3.1, 6.2)]
    hits = find_speech_pii(glued)
    assert len(hits) == 1 and hits[0].kind == "phone number"
