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

from .masks import build_coverage_report, build_per_frame_masks, coverage_gaps
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


def probe_video(path: Path) -> VideoInfo:
    """Probe a video with ffprobe and return basic info."""
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() != ".mp4":
        raise ValueError("v1 accepts MP4 input only")

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


def render_video(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    info: VideoInfo,
    output_path: Path,
    mode: str = "blur",
    strength: int = 21,
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
        # Mux audio from source
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(tmp_path),
            "-i",
            str(info.path),
            "-c:v",
            "libx264",
            "-c:a",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            str(output_path),
        ]
        try:
            subprocess.check_call(cmd)
            tmp_path.unlink(missing_ok=True)
            return output_path
        except Exception:
            # Fall back to video-only file
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

    # Segment and track
    client: SamApiClient | LocalSamStub
    if local:
        client = LocalSamStub()
    else:
        from .sam_client import default_cache_dir
        client = SamApiClient.from_env(api_key_env, use_cache=use_cache)

    all_tracks: Dict[str, Dict[int, np.ndarray]] = {}
    total_objects = 0
    for phrase in targets:
        try:
            result: SegmentationResult = client.segment_video(input_path, phrase, shape=shape)  # type: ignore[arg-type]
        except Exception as exc:
            # API failures fall back to empty result with a clear log so the run still writes a copy
            print(f"Warning: SAM failed for phrase '{phrase}': {exc}")
            result = SegmentationResult(phrase=phrase)
        total_objects += len(result.objects)
        for track_id, frame_map in result.tracks.items():
            # Namespace track ids by phrase to avoid collisions
            key = f"{phrase}:{track_id}"
            if key not in all_tracks:
                all_tracks[key] = {}
            for fidx, m in frame_map.items():
                all_tracks[key][fidx] = m

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

    # Coverage report
    report_path: Optional[Path] = None
    if report or preview:
        report_path = output_path.with_suffix(".coverage.txt")
        report_text = build_coverage_report(
            tracks=all_tracks,
            total_frames=total_frames,
            carry_frames=carry_frames,
            targets=targets,
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
            frames=frames[:sample_frames],
            masks=merged_masks[:sample_frames],
            info=info,
            output_path=preview_output,
            mode=mode,
            strength=strength,
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
        frames=frames,
        masks=merged_masks,
        info=info,
        output_path=output_path,
        mode=mode,
        strength=strength,
    )
    print(f"Wrote: {output_path}")

    contact_path = None
    if contact_sheet:
        contact_path = output_path.with_suffix(".contact.png")
        make_contact_sheet(frames, merged_masks, contact_path)
        print(f"Contact sheet: {contact_path}")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "targets": targets,
        "frames": total_frames,
        "objects": total_objects,
        "contact_sheet": str(contact_path) if contact_path else None,
        "report": str(report_path) if report_path else None,
        "preview": False,
    }
