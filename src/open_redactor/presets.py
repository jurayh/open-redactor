"""Presets for common redaction jobs."""

from __future__ import annotations

PRESETS: dict[str, dict] = {
    "family": {
        "description": "Family sharing with kids, plates, and screens covered generously",
        "targets": ["person", "face", "license plate", "screen"],
        "mode": "blur",
        "strength": 31,
        "mask_margin": 14,
        "carry_frames": 6,
    },
    "street": {
        "description": "Street interviews and vlogs with bystanders and plates pixelated",
        "targets": ["person", "face", "license plate"],
        "mode": "pixelate",
        "strength": 18,
        "mask_margin": 16,
        "carry_frames": 6,
    },
    "documents": {
        "description": "Passports, cards, IDs, and paperwork on camera or on screen. Pair with --pii-text for numbers",
        "targets": ["passport", "credit card", "driver license", "id card", "document"],
        "mode": "blur",
        "strength": 41,
        "mask_margin": 12,
        "carry_frames": 6,
    },
    "screen-share": {
        "description": "Screen recordings and demos with monitors and faces blurred",
        "targets": ["screen", "face"],
        "mode": "blur",
        "strength": 35,
        "mask_margin": 8,
        "carry_frames": 3,
    },
}


def apply_preset(name: str) -> dict:
    """Return a copy of a preset or raise a clear error."""
    if name not in PRESETS:
        choices = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown preset '{name}'. Choose one of: {choices}")
    return dict(PRESETS[name])
