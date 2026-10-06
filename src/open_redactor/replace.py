"""Replacement instead of masking.

Swap sensitive regions for generated stand-ins so the result keeps a
natural feel without leaking the real thing. Faces and people get a
neutral synthetic avatar drawn per track. Plates, house numbers, and
signs get a plaque with a fake value. Text PII gets a fake value patch.
Anything else gets an inpainted fill from its surroundings.

Fakes are deterministic per track key, so the same track keeps the
same fake face tone and the same fake number across the whole clip.
Fake card numbers are Luhn valid in format only and use test ranges.
"""

from __future__ import annotations

import hashlib
import random
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Box = Tuple[int, int, int, int]


def _rng_for(track_key: str) -> random.Random:
    digest = hashlib.sha256(track_key.encode()).hexdigest()
    return random.Random(int(digest[:8], 16))


def _luhn_complete(prefix_digits: str) -> str:
    nums = [int(c) for c in prefix_digits]
    total = 0
    alt = True  # the check digit position flips parity
    for n in reversed(nums):
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alt = not alt
    check = (10 - (total % 10)) % 10
    return prefix_digits + str(check)


def fake_card_number(track_key: str) -> str:
    """A format valid fake. Uses the 4242 test range, never a real account."""
    rng = _rng_for(track_key)
    tail = "".join(str(rng.randint(0, 9)) for _ in range(11))
    digits = _luhn_complete("424242424242"[:4] + tail)[:16]
    digits = "4242" + digits[4:]
    # Recompute the check digit after forcing the test prefix
    digits = _luhn_complete(digits[:15])
    return " ".join(digits[i : i + 4] for i in range(0, 16, 4))


def fake_plate(track_key: str) -> str:
    rng = _rng_for(track_key)
    letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(3))
    nums = "".join(str(rng.randint(0, 9)) for _ in range(3))
    return f"{letters} {nums}"


def fake_house_number(track_key: str) -> str:
    return str(_rng_for(track_key).randint(12, 9899))


def fake_phone(track_key: str) -> str:
    rng = _rng_for(track_key)
    return f"555-01{rng.randint(10, 99)}"  # 555-01xx is the reserved fictional range


def fake_email(track_key: str) -> str:
    return "person@example.com"


def mask_box(mask: np.ndarray) -> Box | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _draw_text_patch(frame: np.ndarray, box: Box, text: str, bg=(245, 245, 245), fg=(30, 30, 30)) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], fill=bg)
    size = max(8, int((y2 - y1) * 0.62))
    try:
        font = ImageFont.load_default(size=size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((x1 + 3, y1 + max(1, (y2 - y1 - size) // 2)), text, fill=fg, font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _draw_avatar(frame: np.ndarray, box: Box, track_key: str) -> np.ndarray:
    """Neutral synthetic person: head and shoulders silhouette, no real features."""
    rng = _rng_for(track_key)
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    skin = (rng.randint(195, 225), rng.randint(160, 185), rng.randint(125, 155))
    shirt = (rng.randint(70, 120), rng.randint(90, 130), rng.randint(120, 165))
    cx = (x1 + x2) // 2
    head_rx = max(4, int(bw * 0.34))
    head_ry = max(4, int(bh * 0.40))
    head_cy = y1 + int(bh * 0.42)
    draw.ellipse([cx - head_rx, head_cy - head_ry, cx + head_rx, head_cy + head_ry], fill=skin)
    shoulder_w = int(bw * 0.92)
    shoulder_top = y1 + int(bh * 0.80)
    draw.ellipse([cx - shoulder_w // 2, shoulder_top, cx + shoulder_w // 2, y2 + int(bh * 0.45)], fill=shirt)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def apply_replacement(frame: np.ndarray, mask: np.ndarray, track_key: str) -> np.ndarray:
    """Replace one track region in one frame based on its kind."""
    if mask is None or not np.any(mask):
        return frame
    box = mask_box(mask)
    if box is None:
        return frame
    kind = track_key.lower()
    if "face" in kind or "person" in kind:
        return _draw_avatar(frame, box, track_key)
    if "license plate" in kind:
        return _draw_text_patch(frame, box, fake_plate(track_key), bg=(250, 250, 250))
    if "house number" in kind or "street sign" in kind:
        return _draw_text_patch(frame, box, fake_house_number(track_key), bg=(90, 90, 90), fg=(255, 255, 255))
    if "credit card number" in kind:
        return _draw_text_patch(frame, box, fake_card_number(track_key))
    if "phone number" in kind:
        return _draw_text_patch(frame, box, fake_phone(track_key))
    if "email" in kind:
        return _draw_text_patch(frame, box, fake_email(track_key))
    if "ssn" in kind:
        return _draw_text_patch(frame, box, "000-00-0000")
    # Generic natural fill: inpaint from surroundings where possible
    try:
        mask_u8 = (mask.astype(np.uint8)) * 255
        return cv2.inpaint(frame, mask_u8, 3, cv2.INPAINT_TELEA)
    except Exception:
        out = frame.copy()
        color = frame[~mask].mean(axis=0) if np.any(~mask) else (128, 128, 128)
        out[mask] = color
        return out
