"""Pipeline for Open Redactor.

Stages:
1. Ingest
2. Segment and track
3. Pad and smooth
4. Render
5. Contact sheet
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .masks import build_coverage_report, build_per_frame_masks, coverage_gaps, select_tracks
from .sam_client import LocalSamStub, SamApiClient, SegmentationResult


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    has_audio: bool


SUPPORTED_INPUT_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def ensure_mp4_for_api(path: Path, tmp_dir: Path) -> Path:
    """Return an MP4 version of the input for APIs that take MP4 data URIs.

    The pipeline itself reads any supported format with OpenCV. Only the
    SAM API upload needs MP4, so non-MP4 inputs are transcoded once into
    a temp file that lives for the duration of the run.
    """
    if path.suffix.lower() == ".mp4":
        return path
    out = tmp_dir / (path.stem + ".api.mp4")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(out)]
    subprocess.check_call(cmd)
    print(f"Transcoded {path.suffix} input to MP4 for the SAM API: {out.name}")
    return out


def probe_video(path: Path) -> VideoInfo:
    """Probe a video with ffprobe and return basic info."""
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
        raise ValueError(f"Unsupported input format '{path.suffix}'. Supported inputs: {supported}. Output is always MP4.")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
    except Exception as exc:
        raise RuntimeError(f"ffprobe failed for {path}: {exc}") from exc

    import json

    data = json.loads(out)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    s = streams[0]
    width = int(s.get("width", 0))
    height = int(s.get("height", 0))
    r_frame_rate = s.get("r_frame_rate", "30/1")
    try:
        num, den = r_frame_rate.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    except Exception:
        fps = 30.0
    nb_frames = s.get("nb_frames")
    duration = float(s.get("duration", 0.0)) if s.get("duration") else 0.0

    # Fall back to OpenCV for frame count when ffprobe is vague
    cap = cv2.VideoCapture(str(path))
    cv_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cv_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
    cap.release()
    frame_count = int(nb_frames) if nb_frames and str(nb_frames).isdigit() else cv_frame_count
    if frame_count <= 0:
        frame_count = cv_frame_count
    if fps <= 0:
        fps = cv_fps or 30.0
    if duration <= 0 and fps > 0 and frame_count > 0:
        duration = frame_count / fps

    # Audio presence check
    audio_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        audio_out = subprocess.check_output(audio_cmd, text=True).strip()
        has_audio = bool(audio_out)
    except Exception:
        has_audio = False

    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        has_audio=has_audio,
    )


def extract_frames(path: Path) -> List[np.ndarray]:
    """Extract all frames as BGR numpy arrays. Fine for v1 clip sizes."""
    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def apply_redaction_to_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    mode: str = "blur",
    strength: int = 21,
) -> np.ndarray:
    """Apply blur or pixelate inside mask and return a new frame."""
    if mask is None or not np.any(mask):
        return frame.copy()
    out = frame.copy()
    h, w = mask.shape
    if mode == "pixelate":
        block = max(2, int(strength))
        # Downscale the masked region in blocks
        small_h = max(1, h // block)
        small_w = max(1, w // block)
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        out[mask] = pixelated[mask]
        return out
    # Blur mode
    ksize = int(strength)
    if ksize % 2 == 0:
        ksize += 1
    if ksize < 1:
        ksize = 1
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    out[mask] = blurred[mask]
    return out


KERNEL_BUCKETS = (41, 81, 161, 301)


def _odd_at_least(value: int, floor: int) -> int:
    k = max(int(value), int(floor))
    return k if k % 2 == 1 else k + 1


def blur_kernel_for_region(base_strength: int, min_dim: int) -> int:
    """Blur kernel for one region, scaled to the region size.

    A fixed kernel leaves large high contrast regions readable: a plate
    80 pixels tall blurred at the default strength still shows its
    characters. The kernel therefore grows with the smaller side of the
    region, at three quarters of that side, rounded up to a small set of
    buckets so a frame needs only a few blurred layers. Small regions
    such as faces keep the caller's strength untouched.
    """
    base = _odd_at_least(base_strength, 1)
    needed = _odd_at_least(int(np.ceil(0.75 * max(0, min_dim))), 1)
    if needed <= base:
        return base
    for bucket in KERNEL_BUCKETS:
        if needed <= bucket:
            return bucket
    return KERNEL_BUCKETS[-1]


def mask_min_dim(mask: np.ndarray) -> int:
    """Smaller side of the mask bounding box, or 0 for an empty mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0
    return int(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


def track_typical_min_dim(frame_map: Dict[int, np.ndarray]) -> int:
    """Median smaller-side size across a track's detected frames."""
    dims = [mask_min_dim(m) for m in frame_map.values()]
    dims = [d for d in dims if d > 0]
    if not dims:
        return 0
    return int(np.median(dims))


def build_blur_layers(
    tracks: Dict[str, Dict[int, np.ndarray]],
    total_frames: int,
    shape: tuple[int, int],
    margin: int,
    carry_frames: int,
    base_strength: int,
) -> List[tuple[List[np.ndarray], int]]:
    """Group tracks by adaptive blur kernel and build masks per group.

    Returns (per-frame masks, kernel) layers ordered by kernel. Tracks
    whose regions are small stay in the base layer at the caller's
    strength, so the common case renders exactly as before.
    """
    groups: Dict[int, Dict[str, Dict[int, np.ndarray]]] = {}
    for key, frame_map in tracks.items():
        kernel = blur_kernel_for_region(base_strength, track_typical_min_dim(frame_map))
        groups.setdefault(kernel, {})[key] = frame_map
    layers: List[tuple[List[np.ndarray], int]] = []
    for kernel in sorted(groups.keys()):
        masks = build_per_frame_masks(
            tracks=groups[kernel],
            total_frames=total_frames,
            shape=shape,
            margin=margin,
            carry_frames=carry_frames,
            smooth_radius=1,
        )
        layers.append((masks, kernel))
    return layers


def apply_secure_fill(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cover a region with a solid fill that cannot be decoded or sharpened.

    Gaussian blur is not a safe treatment for QR codes and barcodes. Their
    payload is a high contrast binary pattern, and a decoder can threshold
    the blurred pattern back into modules. Code regions therefore get an
    opaque fill before the caller's chosen blur or pixelate treatment is
    applied to the rest of the frame.
    """
    if mask is None or not np.any(mask):
        return frame.copy()
    out = frame.copy()
    out[mask] = (0, 0, 0)
    return out


AUDIO_MODES = ("keep", "mute", "pitch")


def audio_filter_for(mode: str, pitch_factor: float = 0.8) -> str | None:
    """Return the ffmpeg audio filter for a mode, or None to drop audio.

    keep copies the source track. mute returns an empty string sentinel
    handled by the caller, which simply omits the audio stream. pitch
    shifts voices down with rubberband while keeping tempo and duration,
    so speech stays intelligible but no longer sounds like the speaker.
    """
    if mode not in AUDIO_MODES:
        raise ValueError(f"Unknown audio mode '{mode}'. Choose keep, mute, or pitch.")
    if mode == "keep":
        return None
    if mode == "mute":
        return ""
    return f"rubberband=pitch={pitch_factor}"


def render_video(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    info: VideoInfo,
    output_path: Path,
    mode: str = "blur",
    strength: int = 21,
    audio_mode: str = "keep",
    pitch_factor: float = 0.8,
    shadow: bool = False,
    mask_layers: Optional[List[tuple[List[np.ndarray], int]]] = None,
) -> Path:
    """Render redacted frames to MP4 and mux audio when present."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp video-only file then mux audio
    tmp_path = output_path.with_suffix(".video-only.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, info.fps, (info.width, info.height))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter for output")
    total = len(frames)
    start_t = time.time()
    last_pct = -10
    for idx, frame in enumerate(frames):
        if mask_layers:
            redacted = frame
            for layer_masks, kernel in mask_layers:
                layer_mask = layer_masks[idx] if idx < len(layer_masks) else np.zeros((info.height, info.width), dtype=bool)
                redacted = apply_redaction_to_frame(redacted, layer_mask, mode="blur", strength=kernel)
        else:
            mask = masks[idx] if idx < len(masks) else np.zeros((info.height, info.width), dtype=bool)
            redacted = apply_redaction_to_frame(frame, mask, mode=mode, strength=strength)
        writer.write(redacted)
        if total > 0:
            pct = int((idx+1)/total*100)
            if pct >= last_pct+10 or idx+1==total:
                last_pct = pct
                elapsed = time.time()-start_t
                eta = (elapsed/max(1,idx+1))*(total-idx-1)
                print(f"Rendering {pct}% frame {idx+1}/{total} ETA {eta:.0f}s", flush=True)
    writer.release()

    if info.has_audio:
        # Mux audio from source, applying the audio redaction mode
        afilter = audio_filter_for(audio_mode, pitch_factor=pitch_factor)
        if afilter == "":
            print("Audio: muted, output has no audio track")
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(tmp_path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                str(output_path),
            ]
        elif afilter:
            print(f"Audio: pitch shifted by factor {pitch_factor}, tempo preserved")
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(tmp_path),
                "-i", str(info.path),
                "-c:v", "libx264",
                "-af", afilter,
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                str(output_path),
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(tmp_path),
                "-i", str(info.path),
                "-c:v", "libx264",
                "-c:a", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                str(output_path),
            ]
        try:
            subprocess.check_call(cmd)
            tmp_path.unlink(missing_ok=True)
            return output_path
        except Exception as exc:
            if audio_mode != "keep":
                # Never silently ship original audio when redaction failed
                print(f"Audio redaction failed ({exc}). Writing video with no audio instead of the original track.")
                fallback = [
                    "ffmpeg", "-y", "-v", "error",
                    "-i", str(tmp_path),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                    str(output_path),
                ]
                try:
                    subprocess.check_call(fallback)
                    tmp_path.unlink(missing_ok=True)
                    return output_path
                except Exception:
                    pass
            tmp_path.replace(output_path)
            return output_path
    else:
        # Re-encode to H.264 for broad compatibility
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(tmp_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        try:
            subprocess.check_call(cmd)
            tmp_path.unlink(missing_ok=True)
        except Exception:
            tmp_path.replace(output_path)
        return output_path


def make_contact_sheet(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    output_path: Path,
    samples: int = 9,
    thumb_width: int = 320,
) -> Path:
    """Create a PNG grid that shows what was redacted."""
    if not frames:
        raise ValueError("No frames for contact sheet")
    total = len(frames)
    if total <= samples:
        indices = list(range(total))
    else:
        step = total / samples
        indices = [int(i * step) for i in range(samples)]

    thumbs: List[Image.Image] = []
    for idx in indices:
        frame = frames[idx]
        mask = masks[idx] if idx < len(masks) else np.zeros(frame.shape[:2], dtype=bool)
        # Draw mask outline in red on a copy
        vis = frame.copy()
        if np.any(mask):
            mask_u8 = (mask.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)
            # Light red overlay
            overlay = vis.copy()
            overlay[mask] = (0.7 * overlay[mask] + 0.3 * np.array([0, 0, 255])).astype(np.uint8)
            vis = overlay
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        # Resize keeping aspect
        w, h = img.size
        scale = thumb_width / max(1, w)
        new_h = max(1, int(h * scale))
        img = img.resize((thumb_width, new_h))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 110, 18], fill=(0, 0, 0))
        draw.text((4, 3), f"frame {idx}", fill=(255, 255, 255))
        thumbs.append(img)

    if not thumbs:
        raise ValueError("No thumbnails created")

    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    thumb_h = thumbs[0].height
    sheet_w = cols * thumb_width
    sheet_h = rows * thumb_h
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(20, 20, 20))
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * thumb_width
        y = (i // cols) * thumb_h
        sheet.paste(thumb, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(output_path))
    return output_path


def run_pipeline(
    input_path: Path,
    output_path: Path,
    targets: List[str],
    mode: str = "blur",
    strength: int = 21,
    mask_margin: int = 10,
    carry_frames: int = 4,
    contact_sheet: bool = False,
    local: bool = False,
    api_key_env: str = "MODEL_API_KEY",
    preview: bool = False,
    report: bool = True,
    use_cache: bool = True,
    backend: str | None = None,
    endpoint: str | None = None,
    provider: str | None = None,
    pii_text: bool = False,
    codes: bool = False,
    audio_mode: str = "keep",
    pitch_factor: float = 0.8,
    shadow: bool = False,
    keep_tracks: Optional[List[str]] = None,
    exclude_tracks: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Run the full redaction pipeline and return a summary dict.

    Preview mode renders only the first 3 seconds plus a contact sheet
    and a coverage report so the user can confirm coverage before a full run.
    Report mode writes a coverage text file next to the output. It defaults
    to on for normal runs and is always on in preview.
    """
    info = probe_video(input_path)
    frames = extract_frames(input_path)
    total_frames = len(frames)
    shape = (info.height, info.width)

    print(f"Input: {input_path} {info.width}x{info.height} {info.fps:.2f} fps {total_frames} frames")
    print(f"Targets: {', '.join(targets)}")
    print(f"Mode: {mode} strength={strength} margin={mask_margin} carry={carry_frames} local={local}")

    # Segment and track. Backend is api, hosted, or local.
    # --local stays as an alias for the local backend.
    resolved_backend = backend or ("local" if local else "api")
    resolved_provider = provider or ("grounding-sam" if resolved_backend == "local" else "sam")
    client: object
    if resolved_backend == "local":
        if resolved_provider == "grounding-sam":
            from .local_gsam import LocalGroundingSamClient
            client = LocalGroundingSamClient()
        else:
            client = LocalSamStub()
    elif resolved_backend in ("api", "hosted"):
        client = SamApiClient.from_env(api_key_env, use_cache=use_cache, endpoint=endpoint)
        if resolved_backend == "hosted":
            print(f"Hosted backend endpoint: {client.endpoint}")
    else:
        raise ValueError(f"Unknown backend '{resolved_backend}'. Choose api, hosted, or local.")

    # API and hosted backends need an MP4 upload. Local providers read the original.
    api_input = input_path
    _api_tmp = None
    if resolved_backend in ("api", "hosted") and input_path.suffix.lower() != ".mp4":
        import tempfile
        _api_tmp = tempfile.TemporaryDirectory(prefix="open-redactor-api-")
        api_input = ensure_mp4_for_api(input_path, Path(_api_tmp.name))

    all_tracks: Dict[str, Dict[int, np.ndarray]] = {}
    total_objects = 0
    sam_sources: List[str] = []
    for phrase in targets:
        try:
            result: SegmentationResult = client.segment_video(api_input, phrase, shape=shape)  # type: ignore[arg-type]
        except Exception as exc:
            # API failures fall back to empty result with a clear log so the run still writes a copy
            print(f"Warning: SAM failed for phrase '{phrase}': {exc}")
            result = SegmentationResult(phrase=phrase)
        total_objects += len(result.objects)
        if result.objects:
            sam_sources.append(result.mask_source)
        for track_id, frame_map in result.tracks.items():
            # Namespace track ids by phrase to avoid collisions
            key = f"{phrase}:{track_id}"
            if key not in all_tracks:
                all_tracks[key] = {}
            for fidx, m in frame_map.items():
                all_tracks[key][fidx] = m

    # Optional text PII layer: OCR numbers and IDs that object phrases miss
    if pii_text:
        from .pii import PiiScanner
        scanner = PiiScanner()
        pii_tracks = scanner.scan_frames(frames)
        for track_key, frame_boxes in pii_tracks.items():
            mask_map: Dict[int, np.ndarray] = {}
            for fidx, boxes in frame_boxes.items():
                m = np.zeros(shape, dtype=bool)
                for (x1, y1, x2, y2) in boxes:
                    x1c, x2c = max(0, x1), min(shape[1], x2)
                    y1c, y2c = max(0, y1), min(shape[0], y2)
                    if x2c > x1c and y2c > y1c:
                        m[y1c:y2c, x1c:x2c] = True
                if np.any(m):
                    mask_map[fidx] = m
            if mask_map:
                all_tracks[f"pii:{track_key}"] = mask_map
                total_objects += 1

    # Optional codes layer: QR codes and barcodes covered as boxes
    if codes:
        from .codes import CodeScanner
        code_tracks = CodeScanner().scan_frames(frames)
        for track_key, frame_boxes in code_tracks.items():
            mask_map: Dict[int, np.ndarray] = {}
            for fidx, boxes in frame_boxes.items():
                m = np.zeros(shape, dtype=bool)
                for (x1, y1, x2, y2) in boxes:
                    x1c, x2c = max(0, x1), min(shape[1], x2)
                    y1c, y2c = max(0, y1), min(shape[0], y2)
                    if x2c > x1c and y2c > y1c:
                        m[y1c:y2c, x1c:x2c] = True
                if np.any(m):
                    mask_map[fidx] = m
            if mask_map:
                all_tracks[f"code:{track_key}"] = mask_map
                total_objects += 1

    # Keep or exclude tracks by key. Kept tracks stay visible while the
    # rest are redacted, excluded tracks are dropped as false positives.
    left_visible: List[str] = []
    if keep_tracks or exclude_tracks:
        all_tracks, kept_visible, excluded, unmatched = select_tracks(
            all_tracks, keep=keep_tracks, exclude=exclude_tracks
        )
        left_visible = kept_visible + excluded
        for key in kept_visible:
            print(f"Kept visible, not redacted: {key}")
        for key in excluded:
            print(f"Excluded, not redacted: {key}")
        for pattern in unmatched:
            available = ", ".join(sorted(all_tracks.keys())) or "none"
            print(
                f"Warning: no track matches '{pattern}'. "
                f"Track keys in this run: {available}"
            )

    if total_objects == 0:
        print("Nothing matched. Writing a clean copy.")

    # Log coverage gaps longer than carry window
    gaps = coverage_gaps(all_tracks, carry_frames=carry_frames)
    for track_id, start, end, length in gaps:
        print(f"Coverage gap in {track_id}: frames {start} to {end} length {length}")

    # Pad and smooth
    merged_masks = build_per_frame_masks(
        tracks=all_tracks,
        total_frames=total_frames,
        shape=shape,
        margin=mask_margin,
        carry_frames=carry_frames,
        smooth_radius=1,
    )

    # Codes get an opaque fill in blur and pixelate modes. A blurred QR
    # pattern can still be thresholded and decoded, so the code layer must
    # not depend on the decorative treatment used for faces and objects.
    secure_tracks = {key: value for key, value in all_tracks.items() if key.startswith("code:")}
    secure_masks = build_per_frame_masks(
        tracks=secure_tracks,
        total_frames=total_frames,
        shape=shape,
        margin=mask_margin,
        carry_frames=carry_frames,
        smooth_radius=1,
    ) if secure_tracks else []

    # Analytics summary, shared by shadow audits and real runs
    from .analytics import build_summary, summary_text, write_summary_files
    if not sam_sources:
        sam_mask_source = "none"
    elif all(s == "pixel" for s in sam_sources):
        sam_mask_source = "pixel"
    elif all(s == "box" for s in sam_sources):
        sam_mask_source = "box"
    else:
        sam_mask_source = "mixed"
    summary = build_summary(all_tracks, total_frames, audio_mode=audio_mode, shadow=shadow, kept_visible=left_visible, sam_mask_source=sam_mask_source)

    if shadow:
        paths = write_summary_files(summary, output_path, shadow=True)
        contact_path = output_path.with_suffix(".contact.png")
        make_contact_sheet(frames, merged_masks, contact_path)
        print(summary_text(summary))
        print(f"Audit report: {paths['audit']}")
        print(f"Summary JSON: {paths['json']}")
        print(f"Contact sheet: {contact_path}")
        print("Shadow mode: no redacted video was rendered and the original is untouched.")
        return {
            "input": str(input_path),
            "output": None,
            "audit": str(paths["audit"]),
            "summary": str(paths["json"]),
            "contact_sheet": str(contact_path),
            "frames": total_frames,
            "objects": total_objects,
            "shadow": True,
            "preview": False,
        }

    # Replace mode works per track so each kind gets its own stand-in
    render_frames = frames
    render_masks = merged_masks
    if secure_masks and mode in ("blur", "pixelate"):
        render_frames = [
            apply_secure_fill(frame, secure_masks[idx]) if idx < len(secure_masks) else frame.copy()
            for idx, frame in enumerate(frames)
        ]
        print("Codes: opaque fill applied so the payload cannot be decoded from a blurred pattern")
    if mode == "replace":
        from .replace import apply_replacement
        per_track_masks = {}
        for track_key, track_map in all_tracks.items():
            per_track_masks[track_key] = build_per_frame_masks(
                tracks={track_key: track_map},
                total_frames=total_frames,
                shape=shape,
                margin=mask_margin,
                carry_frames=carry_frames,
                smooth_radius=1,
            )
        replaced_frames = []
        for idx, frame in enumerate(frames):
            out_frame = frame
            for track_key, tmask in per_track_masks.items():
                if idx < len(tmask) and np.any(tmask[idx]):
                    out_frame = apply_replacement(out_frame, tmask[idx], track_key)
            replaced_frames.append(out_frame)
        render_frames = replaced_frames
        render_masks = [np.zeros(shape, dtype=bool) for _ in range(total_frames)]
        print("Replace mode: sensitive regions swapped for generated stand-ins")

    # Adaptive blur layers: tracks grouped by a kernel scaled to region
    # size, so a large plate or sign gets a far stronger cover than the
    # base strength while faces keep the caller's strength.
    blur_layers = None
    if mode == "blur" and all_tracks:
        blur_layers = build_blur_layers(
            tracks=all_tracks,
            total_frames=total_frames,
            shape=shape,
            margin=mask_margin,
            carry_frames=carry_frames,
            base_strength=strength,
        )
        top_kernel = max(k for _m, k in blur_layers)
        if top_kernel > strength:
            print(f"Adaptive blur: large regions use up to kernel {top_kernel} so they stay unreadable")

    # Coverage report
    report_path: Optional[Path] = None
    if report or preview:
        report_path = output_path.with_suffix(".coverage.txt")
        report_text = build_coverage_report(
            tracks=all_tracks,
            total_frames=total_frames,
            carry_frames=carry_frames,
            targets=targets,
            audio_mode=audio_mode,
            visible=left_visible,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text)
        print(f"Coverage report: {report_path}")
        # Print the short result line for the terminal
        for line in report_text.splitlines():
            if line.startswith("Result:"):
                print(line)

    if preview:
        # Render only the first 3 seconds as a sample
        sample_frames = int(info.fps * 3) if info.fps > 0 else 90
        sample_frames = max(1, min(sample_frames, total_frames))
        preview_output = output_path.with_name(output_path.stem + ".preview" + output_path.suffix)
        render_video(
            frames=render_frames[:sample_frames],
            masks=render_masks[:sample_frames],
            info=info,
            output_path=preview_output,
            mode=mode,
            strength=strength,
            audio_mode=audio_mode,
            pitch_factor=pitch_factor,
            mask_layers=blur_layers,
        )
        print(f"Preview sample: {preview_output} ({sample_frames} frames)")
        contact_path = output_path.with_suffix(".contact.png")
        make_contact_sheet(frames, merged_masks, contact_path)
        print(f"Contact sheet: {contact_path}")
        print("Preview only. Run without --preview for the full clip.")
        return {
            "input": str(input_path),
            "output": str(preview_output),
            "targets": targets,
            "frames": total_frames,
            "objects": total_objects,
            "contact_sheet": str(contact_path),
            "report": str(report_path) if report_path else None,
            "preview": True,
        }

    # Render full clip
    render_video(
        frames=render_frames,
        masks=render_masks,
        info=info,
        output_path=output_path,
        mode=mode,
        strength=strength,
        audio_mode=audio_mode,
        pitch_factor=pitch_factor,
        mask_layers=blur_layers,
    )
    if info.has_audio and audio_mode != "keep":
        print(f"Audio redaction applied: {audio_mode}")
    print(f"Wrote: {output_path}")

    contact_path = None
    if contact_sheet:
        contact_path = output_path.with_suffix(".contact.png")
        make_contact_sheet(frames, merged_masks, contact_path)
        print(f"Contact sheet: {contact_path}")

    paths = write_summary_files(summary, output_path, shadow=False)
    print(summary_text(summary))
    print(f"Summary JSON: {paths['json']}")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "targets": targets,
        "frames": total_frames,
        "objects": total_objects,
        "contact_sheet": str(contact_path) if contact_path else None,
        "report": str(report_path) if report_path else None,
        "summary": str(paths["json"]),
        "preview": False,
    }
