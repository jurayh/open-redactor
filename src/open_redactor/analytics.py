"""Analytics and severity for redaction runs and shadow audits.

Severity reflects how identifying an element is on its own.
Critical items can identify a person or account alone. High items
identify with a little context. Medium items help locate or profile.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

SEVERITY_WEIGHTS = {"critical": 10, "high": 6, "medium": 3, "low": 1}

SEVERITY_MAP = {
    "passport": "critical",
    "credit card": "critical",
    "credit card number": "critical",
    "ssn": "critical",
    "driver license": "critical",
    "id card": "critical",
    "face": "high",
    "person": "high",
    "email": "high",
    "phone number": "high",
    "house number": "high",
    "code": "high",
    "license plate": "medium",
    "screen": "medium",
    "name badge": "medium",
    "street sign": "medium",
    "document": "medium",
    "mailbox": "medium",
    "lanyard": "medium",
}


def severity_for(name: str) -> str:
    low = name.lower()
    for key, level in SEVERITY_MAP.items():
        if key in low:
            return level
    return "medium"


def kind_from_track_key(track_key: str) -> str:
    """Track keys look like phrase:id, pii:kind:n, or code:code:n."""
    parts = track_key.split(":")
    if parts[0] == "pii" and len(parts) >= 2:
        return parts[1]
    if parts[0] == "code":
        return "code"
    return parts[0]


def build_summary(
    tracks: Dict[str, Dict[int, object]],
    total_frames: int,
    audio_mode: str = "keep",
    shadow: bool = False,
    kept_visible: Optional[List[str]] = None,
    sam_mask_source: str = "none",
    extra_elements: Optional[List[dict]] = None,
) -> dict:
    elements: List[dict] = []
    frames_with_hits: set[int] = set()
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for key in sorted(tracks.keys()):
        frame_map = tracks[key]
        if not frame_map:
            continue
        kind = kind_from_track_key(key)
        level = severity_for(kind)
        counts[level] += 1
        detected = sorted(frame_map.keys())
        for f in detected:
            frames_with_hits.add(f)
        elements.append(
            {
                "track": key,
                "kind": kind,
                "severity": level,
                "severity_weight": SEVERITY_WEIGHTS[level],
                "frames_detected": len(detected),
                "first_frame": detected[0],
                "last_frame": detected[-1],
                "share_of_clip": round(len(detected) / total_frames, 3) if total_frames else 0.0,
            }
        )
    for element in extra_elements or []:
        elements.append(element)
        counts[element["severity"]] += 1
        for f in range(element["first_frame"], element["last_frame"] + 1):
            frames_with_hits.add(f)
    elements.sort(key=lambda e: (-e["severity_weight"], -e["frames_detected"], e["track"]))
    risk = sum(e["severity_weight"] * max(0.2, e["share_of_clip"]) for e in elements)
    if audio_mode == "keep":
        audio_note = "Audio kept. Original voices remain a separate identifying channel."
    elif audio_mode == "mute":
        audio_note = "Audio muted. No voice channel remains."
    else:
        audio_note = "Audio pitch shifted. Voices are disguised."
    return {
        "mode": "shadow audit, nothing was redacted or rendered" if shadow else "redaction",
        "total_frames": total_frames,
        "frames_with_sensitive_elements": len(frames_with_hits),
        "share_of_frames_affected": round(len(frames_with_hits) / total_frames, 3) if total_frames else 0.0,
        "elements_found": len(elements),
        "severity_counts": counts,
        "exposure_score": round(risk, 1),
        "audio_mode": audio_mode,
        "audio_note": audio_note,
        "kept_visible": list(kept_visible or []),
        "sam_mask_source": sam_mask_source,
        "elements": elements,
    }


def summary_text(summary: dict) -> str:
    lines = []
    lines.append("Open Redactor analytics")
    lines.append(f"Mode: {summary['mode']}")
    lines.append(
        f"Frames affected: {summary['frames_with_sensitive_elements']} of {summary['total_frames']} "
        f"({summary['share_of_frames_affected']:.0%})"
    )
    c = summary["severity_counts"]
    lines.append(
        f"Elements: {summary['elements_found']} total, {c['critical']} critical, {c['high']} high, {c['medium']} medium"
    )
    lines.append(f"Exposure score before action: {summary['exposure_score']}")
    source = summary.get("sam_mask_source", "none")
    if source != "none":
        source_notes = {
            "pixel": "decoded pixel masks",
            "mixed": "a mix of pixel and box masks",
            "box": "box masks, pixel decode unavailable",
        }
        lines.append(f"SAM mask source: {source} ({source_notes.get(source, source)})")
    lines.append(f"Audio: {summary['audio_note']}")
    if summary.get("kept_visible"):
        lines.append(f"Left visible by request, not redacted: {', '.join(summary['kept_visible'])}")
    if summary["elements"]:
        lines.append("")
        lines.append("Most sensitive first:")
        for e in summary["elements"]:
            lines.append(
                f"- [{e['severity'].upper()}] {e['track']}: {e['frames_detected']} frames, "
                f"frames {e['first_frame']} to {e['last_frame']}"
            )
    else:
        lines.append("No sensitive elements were detected.")
    return "\n".join(lines) + "\n"


def write_summary_files(summary: dict, base_output: Path, shadow: bool = False) -> Dict[str, Path]:
    json_path = base_output.with_suffix(".summary.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2))
    paths = {"json": json_path}
    if shadow:
        audit_path = base_output.with_suffix(".audit.txt")
        audit_path.write_text(summary_text(summary))
        paths["audit"] = audit_path
    return paths
