"""Spoken PII detection for the audio track.

Visual layers cannot hear. A phone number read aloud, a card number
dictated on a support call, an email spelled out in a voice note all
survive visual redaction. This module closes that channel: it takes
word level transcript timings, finds the same verified PII the text
layer looks for, and returns time spans for the renderer to mute.

Detection runs three paths over the words. Plain transcript text goes
through the text layer patterns for values the transcriber wrote as
digits or with a real at sign. Runs of number words are rebuilt into
digit strings and verified with the same rules, Luhn for cards, SSA
issuing rules for Social Security numbers, so a spoken number must be
a plausible real one before anything is muted. Spoken emails are
rebuilt from at and dot tokens and validated as addresses.

Transcription itself is pluggable. faster-whisper does it locally when
the speech extra is installed, and any Whisper style JSON transcript
with word timings can be supplied instead, which keeps the layer
testable and usable with transcripts from other tools.
"""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .pii import EMAIL_RE, find_pii_in_text, luhn_ok

NUMBER_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


@dataclass
class WordSpan:
    word: str
    start: float
    end: float


@dataclass
class SpeechHit:
    kind: str
    start: float
    end: float
    text: str


def transcriber_available() -> bool:
    try:
        return importlib.util.find_spec("faster_whisper") is not None
    except Exception:
        return False


def load_transcript(path: Path) -> List[WordSpan]:
    """Load word timings from a Whisper style JSON transcript.

    Accepts segments with a words list, or a top level words list. Each
    word entry needs word, start, and end fields.
    """
    data = json.loads(Path(path).read_text())
    raw_words: list = []
    if isinstance(data, dict) and isinstance(data.get("words"), list):
        raw_words = data["words"]
    elif isinstance(data, dict) and isinstance(data.get("segments"), list):
        for segment in data["segments"]:
            if isinstance(segment, dict) and isinstance(segment.get("words"), list):
                raw_words.extend(segment["words"])
    words: List[WordSpan] = []
    for entry in raw_words:
        try:
            text = str(entry["word"]).strip().strip(".,!?;:\"'-").strip()
            if text:
                words.append(WordSpan(text, float(entry["start"]), float(entry["end"])))
        except Exception:
            continue
    return words


def transcribe(path: Path, model: str = "base") -> List[WordSpan]:
    """Transcribe a media file locally with faster-whisper."""
    from faster_whisper import WhisperModel

    whisper = WhisperModel(model, device="cpu", compute_type="int8")
    segments, _info = whisper.transcribe(str(path), word_timestamps=True, vad_filter=False)
    words: List[WordSpan] = []
    for segment in segments:
        for entry in segment.words or []:
            text = str(entry.word).strip().strip(".,!?;:\"'-").strip()
            if text:
                words.append(WordSpan(text, float(entry.start), float(entry.end)))
    return words


def _word_digits(word: str) -> Optional[str]:
    low = word.lower()
    if low in NUMBER_WORDS:
        return NUMBER_WORDS[low]
    if low.isdigit():
        return low
    # Transcribers often glue a number together with its separators,
    # like 415-555-2671 or (415) as one token. A token made only of
    # digits and separators contributes its digits to the run.
    if any(c.isdigit() for c in low) and re.fullmatch(r"[\d\s.,\-()+]+", low):
        return re.sub(r"\D", "", low)
    return None


def _digit_runs(words: Sequence[WordSpan]) -> List[Tuple[str, int, int]]:
    """Maximal runs of digit bearing words as (digits, first, last)."""
    runs: List[Tuple[str, int, int]] = []
    digits: List[str] = []
    first = 0
    for idx, word in enumerate(words):
        piece = _word_digits(word.word)
        if piece is not None:
            if not digits:
                first = idx
            digits.append(piece)
        else:
            if digits:
                runs.append(("".join(digits), first, idx - 1))
                digits = []
    if digits:
        runs.append(("".join(digits), first, len(words) - 1))
    return runs


def _verify_digits(digits: str) -> Optional[str]:
    """Classify a digit string with the text layer verification rules."""
    if 13 <= len(digits) <= 19 and luhn_ok(digits):
        return "credit card number"
    if len(digits) == 9:
        formatted = f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
        if any(h.kind == "ssn" for h in find_pii_in_text(formatted)):
            return "ssn"
    if len(digits) in (10, 11):
        local = digits[-10:]
        formatted = f"({local[:3]}) {local[3:6]}-{local[6:]}"
        if any(h.kind == "phone number" for h in find_pii_in_text(formatted)):
            return "phone number"
    return None


def _spoken_email_hits(words: Sequence[WordSpan]) -> List[SpeechHit]:
    """Rebuild addresses from at and dot tokens and validate them."""
    hits: List[SpeechHit] = []
    lowers = [w.word.lower() for w in words]
    for idx, low in enumerate(lowers):
        if low != "at" or idx == 0 or idx == len(words) - 1:
            continue
        # Left side: an alnum token, then (dot, alnum) pairs walking back
        if not lowers[idx - 1].isalnum():
            continue
        left_parts = [lowers[idx - 1]]
        left_idx = idx - 1
        while left_idx >= 2 and lowers[left_idx - 1] == "dot" and lowers[left_idx - 2].isalnum():
            left_parts = [lowers[left_idx - 2], "."] + left_parts
            left_idx -= 2
        # Right side: an alnum token, then (dot, alnum) pairs walking forward
        if not lowers[idx + 1].isalnum():
            continue
        right_parts = [lowers[idx + 1]]
        right_idx = idx + 1
        while right_idx + 2 < len(words) and lowers[right_idx + 1] == "dot" and lowers[right_idx + 2].isalnum():
            right_parts = right_parts + [".", lowers[right_idx + 2]]
            right_idx += 2
        candidate = f"{''.join(left_parts)}@{''.join(right_parts)}"
        if EMAIL_RE.fullmatch(candidate):
            hits.append(SpeechHit("email", words[left_idx].start, words[right_idx].end, candidate))
    return hits


def find_speech_pii(words: Sequence[WordSpan], margin: float = 0.35) -> List[SpeechHit]:
    """Find verified spoken PII and return padded, merged time spans."""
    if not words:
        return []
    raw: List[SpeechHit] = []

    # Path A: patterns over the plain transcript text
    joined = " ".join(w.word for w in words)
    offsets: List[Tuple[int, int]] = []
    pos = 0
    for w in words:
        offsets.append((pos, pos + len(w.word)))
        pos += len(w.word) + 1
    for hit in find_pii_in_text(joined):
        first = next((i for i, (s, e) in enumerate(offsets) if e > hit.start), None)
        last = next((i for i in range(len(offsets) - 1, -1, -1) if offsets[i][0] < hit.end), None)
        if first is not None and last is not None and last >= first:
            raw.append(SpeechHit(hit.kind, words[first].start, words[last].end, hit.text))

    # Path B: number words rebuilt into verified digit strings
    claimed = [(h.start, h.end) for h in raw]
    for digits, first, last in _digit_runs(words):
        kind = _verify_digits(digits)
        if kind is None:
            continue
        start, end = words[first].start, words[last].end
        if any(start < e and end > s for s, e in claimed):
            continue
        raw.append(SpeechHit(kind, start, end, digits))
        claimed.append((start, end))

    # Path C: spoken emails
    for hit in _spoken_email_hits(words):
        if any(hit.start < e and hit.end > s for s, e in claimed):
            continue
        raw.append(hit)
        claimed.append((hit.start, hit.end))

    # Pad by the margin and merge overlapping spans
    padded = [
        SpeechHit(h.kind, max(0.0, h.start - margin), h.end + margin, h.text)
        for h in sorted(raw, key=lambda h: h.start)
    ]
    merged: List[SpeechHit] = []
    for hit in padded:
        if merged and hit.start <= merged[-1].end:
            prev = merged[-1]
            if hit.end > prev.end:
                prev.end = hit.end
            if hit.kind != prev.kind:
                prev.text = f"{prev.text} / {hit.text}"
                prev.kind = prev.kind if len(prev.text) <= len(hit.text) else hit.kind
        else:
            merged.append(SpeechHit(hit.kind, hit.start, hit.end, hit.text))
    return merged


def mute_span_filter(spans: Sequence[SpeechHit]) -> str:
    """ffmpeg audio filter chain that silences the given spans."""
    return ",".join(
        f"volume=enable='between(t,{h.start:.2f},{h.end:.2f})':volume=0" for h in spans
    )
