"""Photo pipeline. One image in, redacted image out, same layers as video."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .masks import build_per_frame_masks
from .pipeline import apply_redaction_to_frame, apply_secure_fill
from .sam_client import LocalSamStub, SamApiClient, SegmentationResult

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def run_image_pipeline(
    input_path: Path,
    output_path: Path,
    targets: List[str],
    mode: str = "blur",
    strength: int = 21,
    mask_margin: int = 10,
    local: bool = False,
    api_key_env: str = "MODEL_API_KEY",
    backend: str | None = None,
    endpoint: str | None = None,
    provider: str | None = None,
    pii_text: bool = False,
    codes: bool = False,
) -> Dict[str, object]:
    frame = cv2.imread(str(input_path))
    if frame is None:
        raise ValueError(f"Could not read image: {input_path}")
    height, width = frame.shape[:2]
    shape = (height, width)
    print(f"Input photo: {input_path} {width}x{height}")
    print(f"Targets: {', '.join(targets)}")

    resolved_backend = backend or ("local" if local else "api")
    resolved_provider = provider or ("grounding-sam" if resolved_backend == "local" else "sam")
    client: object
    if resolved_backend == "local":
        if resolved_provider == "grounding-sam":
            from .local_gsam import LocalGroundingSamClient

            client = LocalGroundingSamClient()
        else:
            client = LocalSamStub()
    else:
        client = SamApiClient.from_env(api_key_env, endpoint=endpoint)
        if resolved_backend == "hosted":
            print(f"Hosted backend endpoint: {client.endpoint}")  # type: ignore[attr-defined]

    all_tracks: Dict[str, Dict[int, np.ndarray]] = {}
    total_objects = 0
    for phrase in targets:
        try:
            result: SegmentationResult = client.segment_image(input_path, phrase, shape=shape)  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"Warning: SAM failed for phrase '{phrase}': {exc}")
            result = SegmentationResult(phrase=phrase)
        total_objects += len(result.objects)
        for track_id, frame_map in result.tracks.items():
            merged: Dict[int, np.ndarray] = {}
            for _fidx, m in frame_map.items():
                merged[0] = m
            if merged:
                all_tracks[f"{phrase}:{track_id}"] = merged

    frames = [frame]
    if pii_text:
        from .pii import PiiScanner

        for key, frame_boxes in PiiScanner(sample_every=1).scan_frames(frames).items():
            for fidx, boxes in frame_boxes.items():
                m = np.zeros(shape, dtype=bool)
                for x1, y1, x2, y2 in boxes:
                    m[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = True
                if np.any(m):
                    all_tracks[f"pii:{key}"] = {0: m}
                    total_objects += 1
    if codes:
        from .codes import CodeScanner

        for key, frame_boxes in CodeScanner(sample_every=1).scan_frames(frames).items():
            for fidx, boxes in frame_boxes.items():
                m = np.zeros(shape, dtype=bool)
                for x1, y1, x2, y2 in boxes:
                    m[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = True
                if np.any(m):
                    all_tracks[f"code:{key}"] = {0: m}
                    total_objects += 1

    if total_objects == 0:
        print("Nothing matched. Writing a clean copy.")
    masks = build_per_frame_masks(
        tracks=all_tracks, total_frames=1, shape=shape, margin=mask_margin, carry_frames=0, smooth_radius=1
    )
    if mode == "replace":
        from .replace import apply_replacement
        redacted = frame
        for track_key, track_map in all_tracks.items():
            single = build_per_frame_masks(
                tracks={track_key: track_map}, total_frames=1, shape=shape,
                margin=mask_margin, carry_frames=0, smooth_radius=1,
            )
            if np.any(single[0]):
                redacted = apply_replacement(redacted, single[0], track_key)
    else:
        base_frame = frame
        code_tracks = {key: value for key, value in all_tracks.items() if key.startswith("code:")}
        if code_tracks:
            code_masks = build_per_frame_masks(
                tracks=code_tracks, total_frames=1, shape=shape,
                margin=mask_margin, carry_frames=0, smooth_radius=1,
            )
            base_frame = apply_secure_fill(frame, code_masks[0])
            print("Codes: opaque fill applied so the payload cannot be decoded from a blurred pattern")
        redacted = apply_redaction_to_frame(base_frame, masks[0], mode=mode if mode in ("blur", "pixelate") else "blur", strength=strength)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), redacted):
        raise RuntimeError(f"Could not write output image: {output_path}")
    print(f"Wrote: {output_path}")
    return {"input": str(input_path), "output": str(output_path), "objects": total_objects, "photo": True}
