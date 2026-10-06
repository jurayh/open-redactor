"""Text PII detection for frames.

Two layers. Pattern matching on OCR text finds card numbers, US Social
Security numbers, phone numbers, and emails. Card numbers are verified
with the Luhn check so random long numbers do not get flagged. OCR is
optional and lazy through pytesseract plus the tesseract binary. When
OCR is not installed the scanner returns nothing with a clear message
and never invents detections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

Box = Tuple[int, int, int, int]


@dataclass
class PiiHit:
    kind: str
    text: str
    start: int
    end: int


CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
SSN_RE = re.compile(r"(?<!\d)(\d{3})-(\d{2})-(\d{4})(?!\d)")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def luhn_ok(digits: str) -> bool:
    """Standard Luhn check used by payment card numbers."""
    nums = [int(c) for c in digits if c.isdigit()]
    if len(nums) < 13 or len(nums) > 19:
        return False
    total = 0
    alt = False
    for n in reversed(nums):
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alt = not alt
    return total % 10 == 0


def find_pii_in_text(text: str) -> List[PiiHit]:
    """Find verified PII in one block of OCR text."""
    hits: List[PiiHit] = []
    taken: List[Tuple[int, int]] = []

    def free(start: int, end: int) -> bool:
        return all(end <= s or start >= e for s, e in taken)

    for m in EMAIL_RE.finditer(text):
        if free(m.start(), m.end()):
            hits.append(PiiHit("email", m.group(0), m.start(), m.end()))
            taken.append((m.start(), m.end()))
    for m in SSN_RE.finditer(text):
        area, group, serial = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # SSA rules: area 0, 666, and 900 plus are never issued, group and serial are never 0
        if area == 0 or area == 666 or area >= 900 or group == 0 or serial == 0:
            continue
        if free(m.start(), m.end()):
            hits.append(PiiHit("ssn", m.group(0), m.start(), m.end()))
            taken.append((m.start(), m.end()))
    for m in CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if luhn_ok(digits) and free(m.start(), m.end()):
            hits.append(PiiHit("credit card number", m.group(0), m.start(), m.end()))
            taken.append((m.start(), m.end()))
    for m in PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        # Skip anything already claimed as a card and skip short local numbers
        if len(digits) >= 10 and free(m.start(), m.end()):
            # A phone span inside a Luhn valid card was already taken above
            hits.append(PiiHit("phone number", m.group(0), m.start(), m.end()))
            taken.append((m.start(), m.end()))
    return sorted(hits, key=lambda h: h.start)


def ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
        import shutil

        return shutil.which("tesseract") is not None
    except Exception:
        return False


@dataclass
class PiiScanner:
    """Scan sampled frames with OCR and return box tracks per kind."""

    sample_every: int = 10
    tracks: Dict[str, Dict[int, List[Box]]] = field(default_factory=dict)

    def scan_frames(self, frames) -> Dict[str, Dict[int, List[Box]]]:
        """Frames are BGR numpy arrays. Returns {kind-index: {frame: [boxes]}}.

        OCR runs on sampled frames only and each hit box is held until the
        next sample, which matches how documents sit still on screen. The
        pipeline carry logic handles the final smoothing.
        """
        if not ocr_available():
            print(
                "Text PII scan needs pytesseract and the tesseract binary. "
                "Skipping text scan, object targets still run."
            )
            return {}
        import pytesseract
        from PIL import Image

        result: Dict[str, Dict[int, List[Box]]] = {}
        counters: Dict[str, int] = {}
        total = len(frames)
        for idx in range(0, total, max(1, self.sample_every)):
            frame = frames[idx]
            rgb = frame[:, :, ::-1]
            data = pytesseract.image_to_data(Image.fromarray(rgb), output_type=pytesseract.Output.DATAFRAME)
            if data is None or data.empty:
                continue
            # Group words into lines by block, paragraph, and line numbers
            lines: Dict[Tuple[int, int, int], List[Tuple[str, Box]]] = {}
            for _, row in data.iterrows():
                word = str(row.get("text", "") or "").strip()
                if not word or float(row.get("conf", -1) or -1) < 0:
                    continue
                key = (int(row.get("block_num", 0)), int(row.get("par_num", 0)), int(row.get("line_num", 0)))
                box = (
                    int(row["left"]),
                    int(row["top"]),
                    int(row["left"] + row["width"]),
                    int(row["top"] + row["height"]),
                )
                lines.setdefault(key, []).append((word, box))
            for words in lines.values():
                line_text = " ".join(w for w, _ in words)
                hits = find_pii_in_text(line_text)
                if not hits:
                    continue
                # Map character spans back to word boxes, simple span cover
                char_boxes: List[Tuple[int, int, Box]] = []
                pos = 0
                for word, box in words:
                    char_boxes.append((pos, pos + len(word), box))
                    pos += len(word) + 1
                for hit in hits:
                    covered = [b for s, e, b in char_boxes if e > hit.start and s < hit.end]
                    if not covered:
                        continue
                    x1 = min(b[0] for b in covered)
                    y1 = min(b[1] for b in covered)
                    x2 = max(b[2] for b in covered)
                    y2 = max(b[3] for b in covered)
                    counters[hit.kind] = counters.get(hit.kind, 0) + 1
                    track_key = f"{hit.kind}:{counters[hit.kind]}"
                    hold_until = min(total, idx + max(1, self.sample_every))
                    for f in range(idx, hold_until):
                        result.setdefault(track_key, {}).setdefault(f, []).append((x1, y1, x2, y2))
                    print(f"Text PII: {hit.kind} found near frame {idx}")
        return result
