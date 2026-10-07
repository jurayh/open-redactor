"""Mask helpers for padding, carry-forward, smoothing, and gap fill.

These helpers are pure numpy so they are easy to test and reuse.
They implement the failure-mode mitigations from the spec:
pad masks by a margin, carry tracks forward, smooth edges over time,
and fill single-frame gaps inside a track.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class TrackMask:
    """One object mask in one frame."""

    frame_index: int
    track_id: str
    mask: np.ndarray
    box: Optional[tuple[int, int, int, int]] = None


def ensure_binary(mask: np.ndarray) -> np.ndarray:
    """Return a boolean mask from any numeric mask."""
    if mask.dtype == bool:
        return mask.copy()
    return mask > 0


def pad_mask(mask: np.ndarray, margin: int) -> np.ndarray:
    """Dilate a boolean mask by margin pixels using a simple box dilation.

    Uses cumulative shifts so it works without scipy.
    """
    m = ensure_binary(mask)
    if margin <= 0:
        return m
    # Horizontal dilation
    padded = m.copy()
    for dx in range(1, margin + 1):
        padded[:, dx:] |= m[:, :-dx]
        padded[:, :-dx] |= m[:, dx:]
    # Vertical dilation on the horizontally padded result
    out = padded.copy()
    for dy in range(1, margin + 1):
        out[dy:, :] |= padded[:-dy, :]
        out[:-dy, :] |= padded[dy:, :]
    return out


def merge_masks(masks: Sequence[np.ndarray], shape: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Merge many masks into one boolean mask."""
    if not masks:
        if shape is None:
            raise ValueError("shape is required when masks is empty")
        return np.zeros(shape, dtype=bool)
    merged = ensure_binary(masks[0])
    for m in masks[1:]:
        merged |= ensure_binary(m)
    return merged


def smooth_mask_temporal(masks: Sequence[np.ndarray], radius: int = 1) -> List[np.ndarray]:
    """Smooth mask edges over time with a simple temporal majority vote.

    For each frame, pixels that appear in a majority of the window are kept.
    This reduces one-frame flicker without dropping real coverage.
    """
    if not masks:
        return []
    if radius <= 0:
        return [ensure_binary(m) for m in masks]
    binary = [ensure_binary(m) for m in masks]
    out: List[np.ndarray] = []
    n = len(binary)
    for i in range(n):
        window = binary[max(0, i - radius): min(n, i + radius + 1)]
        stacked = np.stack(window, axis=0).astype(np.float32)
        avg = stacked.mean(axis=0)
        # Keep pixels that are present in at least half the window
        # and always keep the current frame coverage
        smoothed = (avg >= 0.5) | binary[i]
        out.append(smoothed)
    return out


def fill_single_frame_gaps(
    frames: Dict[int, np.ndarray],
    max_gap: int = 1,
) -> Dict[int, np.ndarray]:
    """Fill short gaps inside a track by carrying the last mask forward.

    frames maps frame_index to mask for one track.
    Gaps of length <= max_gap are filled with the previous mask.
    Longer gaps are left empty so the caller can log them.
    """
    if not frames:
        return {}
    result = dict(frames)
    indices = sorted(frames.keys())
    # Walk sorted indices and fill gaps between consecutive present frames
    for a, b in zip(indices, indices[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            last = frames[a]
            for f in range(a + 1, b):
                result[f] = last.copy()
    return result


def carry_forward(
    track_frames: Dict[int, np.ndarray],
    total_frames: int,
    carry_frames: int,
) -> Dict[int, np.ndarray]:
    """Carry each track forward for carry_frames after its last detection.

    track_frames maps frame_index to mask for one track.
    Returns a new dict with held masks added after the last known frame
    and after any isolated dropouts.
    """
    if not track_frames or carry_frames <= 0:
        return dict(track_frames)
    result = dict(track_frames)
    sorted_idx = sorted(track_frames.keys())
    last_idx = sorted_idx[-1]
    # Extend after last detection
    last_mask = track_frames[last_idx]
    for offset in range(1, carry_frames + 1):
        f = last_idx + offset
        if f >= total_frames:
            break
        if f not in result:
            result[f] = last_mask.copy()
    # Hold through short internal dropouts as well
    for a, b in zip(sorted_idx, sorted_idx[1:]):
        gap = b - a - 1
        if 0 < gap <= carry_frames:
            mask_a = track_frames[a]
            for f in range(a + 1, b):
                if f not in result:
                    result[f] = mask_a.copy()
    return result


def apply_padding_to_sequence(
    frames: Dict[int, np.ndarray],
    margin: int,
) -> Dict[int, np.ndarray]:
    """Pad every mask in a per-frame dict."""
    if margin <= 0:
        return dict(frames)
    return {idx: pad_mask(mask, margin) for idx, mask in frames.items()}


def build_per_frame_masks(
    tracks: Dict[str, Dict[int, np.ndarray]],
    total_frames: int,
    shape: tuple[int, int],
    margin: int = 10,
    carry_frames: int = 4,
    smooth_radius: int = 1,
) -> List[np.ndarray]:
    """Build merged per-frame redaction masks with all mitigations applied.

    tracks maps track_id to {frame_index: mask}.
    Steps per track: fill single-frame gaps, carry forward, pad.
    Then merge per frame and smooth temporally.
    """
    # Per-track processing
    processed_tracks: Dict[str, Dict[int, np.ndarray]] = {}
    for track_id, frames in tracks.items():
        filled = fill_single_frame_gaps(frames, max_gap=1)
        carried = carry_forward(filled, total_frames=total_frames, carry_frames=carry_frames)
        padded = apply_padding_to_sequence(carried, margin=margin)
        processed_tracks[track_id] = padded

    # Merge per frame
    merged: List[np.ndarray] = []
    for f in range(total_frames):
        frame_masks = [pt[f] for pt in processed_tracks.values() if f in pt]
        if frame_masks:
            merged.append(merge_masks(frame_masks, shape=shape))
        else:
            merged.append(np.zeros(shape, dtype=bool))

    # Temporal smoothing
    smoothed = smooth_mask_temporal(merged, radius=smooth_radius)
    return smoothed


def coverage_gaps(
    tracks: Dict[str, Dict[int, np.ndarray]],
    carry_frames: int,
) -> List[tuple[str, int, int, int]]:
    """Return gaps longer than the carry window.

    Each item is (track_id, gap_start, gap_end, gap_length).
    Use this to fail loud on coverage gaps.
    """
    gaps: List[tuple[str, int, int, int]] = []
    for track_id, frames in tracks.items():
        if not frames:
            continue
        idx = sorted(frames.keys())
        for a, b in zip(idx, idx[1:]):
            gap = b - a - 1
            if gap > carry_frames:
                gaps.append((track_id, a + 1, b - 1, gap))
    return gaps


def select_tracks(
    tracks: Dict[str, Dict[int, np.ndarray]],
    keep: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> tuple[Dict[str, Dict[int, np.ndarray]], List[str], List[str], List[str]]:
    """Split tracks into the redaction set and the tracks left visible.

    Patterns match full track keys (for example ``person:0`` or
    ``code:qr-0``) exactly or as fnmatch globs (``person:*``). A keep
    pattern names a track to leave visible while everything else is
    still redacted, the "blur everyone except this person" case. An
    exclude pattern drops a named track from redaction, the false
    positive case. Both remove tracks from the returned redaction set.

    Returns (redacted, kept_visible, excluded, unmatched_patterns).
    """
    keep_patterns = list(keep or [])
    exclude_patterns = list(exclude or [])
    if not keep_patterns and not exclude_patterns:
        return dict(tracks), [], [], []

    def matches(key: str, patterns: Sequence[str]) -> bool:
        return any(fnmatch.fnmatchcase(key, pattern) for pattern in patterns)

    kept_visible = sorted(key for key in tracks if matches(key, keep_patterns))
    excluded = sorted(
        key for key in tracks if key not in kept_visible and matches(key, exclude_patterns)
    )
    skip = set(kept_visible) | set(excluded)
    redacted = {key: value for key, value in tracks.items() if key not in skip}
    matched = {pattern for pattern in keep_patterns + exclude_patterns
               if any(fnmatch.fnmatchcase(key, pattern) for key in tracks)}
    unmatched = [pattern for pattern in keep_patterns + exclude_patterns if pattern not in matched]
    return redacted, kept_visible, excluded, unmatched


def build_coverage_report(
    tracks: Dict[str, Dict[int, np.ndarray]],
    total_frames: int,
    carry_frames: int,
    targets: List[str],
    audio_mode: str = "keep",
    visible: Optional[List[str]] = None,
) -> str:
    """Build a plain text coverage report.

    Counts detected frames, carried frames, longest gap, and uncovered gaps.
    Carried frames are estimated from the processed track span.
    """
    lines: List[str] = []
    lines.append("Open Redactor coverage report")
    lines.append(f"Targets: {', '.join(targets) if targets else 'none'}")
    lines.append(f"Total frames: {total_frames}")
    lines.append(f"Carry window: {carry_frames} frames")
    lines.append("")

    if not tracks:
        lines.append("Tracks: 0")
        lines.append("Detections: 0 frames with detections")
        if visible:
            lines.append(f"Left visible by request, not redacted: {', '.join(visible)}")
            lines.append("Result: every detected track was left visible by request. The copy matches the original.")
        else:
            lines.append("Result: nothing matched. A clean copy is safe to review in the contact sheet.")
        return "\n".join(lines) + "\n"

    total_detected = 0
    longest_gap = 0
    uncovered: List[tuple[str, int, int, int]] = []

    for track_id in sorted(tracks.keys()):
        frames = tracks[track_id]
        detected = len(frames)
        total_detected += detected
        idx = sorted(frames.keys())
        first = idx[0] if idx else 0
        last = idx[-1] if idx else 0
        span = (last - first + 1) if idx else 0
        carried_est = max(0, span - detected)
        lines.append(f"Track {track_id}: {detected} detected frames, span {first} to {last}, about {carried_est} filled or carried inside span")
        for a, b in zip(idx, idx[1:]):
            gap = b - a - 1
            if gap > longest_gap:
                longest_gap = gap
            if gap > carry_frames:
                uncovered.append((track_id, a + 1, b - 1, gap))

    lines.append("")
    lines.append(f"Frames with detections across all tracks: {total_detected}")
    lines.append(f"Longest internal gap: {longest_gap} frames")
    if uncovered:
        lines.append(f"Uncovered gaps longer than carry window: {len(uncovered)}")
        for track_id, start, end, length in uncovered:
            lines.append(f"  {track_id} frames {start} to {end} length {length} NEEDS REVIEW")
        lines.append("Result: review the contact sheet around those frames before sharing.")
    else:
        lines.append("Uncovered gaps longer than carry window: 0")
        lines.append("Result: continuous coverage inside track spans with current carry settings.")
    if visible:
        lines.append(f"Left visible by request, not redacted: {', '.join(visible)}")
    if audio_mode == "mute":
        lines.append("Audio: muted, the output has no audio track.")
    elif audio_mode == "pitch":
        lines.append("Audio: pitch shifted, voices are disguised but speech remains.")
    else:
        lines.append("Note: audio mode is keep, so original voices can still identify people. Use --audio mute or --audio pitch to redact them.")
    return "\n".join(lines) + "\n"
