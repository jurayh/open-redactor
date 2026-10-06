"""SAM 3.1 client for API mode and local stub.

API facts verified from https://dev.meta.ai/docs/sam/overview
- POST https://api.meta.ai/v1/responses
- model sam-3.1
- input is a message with input_text and input_image or input_video
- stream true is recommended for video
- metadata mask_encoding one_bit
- output_text holds special-token lines with boxes and masks per frame
- @meta-sam/parser decodes rasters in the JS ecosystem
- Zero matches is a valid outcome

This Python client keeps the same contract and provides a clear
extension point for a real parser and for local weights.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore


DEFAULT_ENDPOINT = "https://api.meta.ai/v1/responses"
DEFAULT_MODEL = "sam-3.1"
DEFAULT_TARGETS: List[str] = ["person", "face", "license plate", "screen"]


@dataclass
class DetectedObject:
    track_id: str
    frame_index: int
    box: tuple[int, int, int, int]
    mask: Optional[np.ndarray] = None
    phrase: str = ""
    mask_payload: Optional[str] = None
    mask_width: Optional[int] = None
    mask_height: Optional[int] = None


@dataclass
class SegmentationResult:
    phrase: str
    objects: List[DetectedObject] = field(default_factory=list)
    # per-frame masks keyed by track then frame
    tracks: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)
    raw_output_text: str = ""


def resolve_targets(
    targets: Optional[List[str]],
    add_targets: Optional[List[str]],
    use_defaults: bool,
) -> List[str]:
    """Resolve the final target list from CLI flags.

    Rules from the spec:
    - No phrase runs the default set
    - --target replaces defaults unless --targets-default is set
    - --add-target always adds on top of defaults
    """
    if add_targets:
        base = list(DEFAULT_TARGETS)
        merged = base + [t for t in add_targets if t not in base]
        if targets:
            merged += [t for t in targets if t not in merged]
        return merged
    if targets:
        if use_defaults:
            merged = list(DEFAULT_TARGETS)
            merged += [t for t in targets if t not in merged]
            return merged
        return list(targets)
    return list(DEFAULT_TARGETS)


def build_request_body(phrase: str, video_path: Path, stream: bool = True) -> Dict[str, Any]:
    """Build the SAM 3.1 request body for one phrase.

    Live verified shape on 2026-10-01:
    input_video needs video_url with a data URI for local files.
    A hosted https URL works the same way.
    """
    video_b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
    return {
        "model": DEFAULT_MODEL,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": phrase},
                    {
                        "type": "input_video",
                        "video_url": f"data:video/mp4;base64,{video_b64}",
                    },
                ],
            }
        ],
        "stream": stream,
        "metadata": {"mask_encoding": "one_bit"},
    }


def build_image_request_body(phrase: str, image_b64: str, stream: bool = True) -> Dict[str, Any]:
    """Build a SAM 3.1 request body for a single image."""
    return {
        "model": DEFAULT_MODEL,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": phrase},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_b64}",
                    },
                ],
            }
        ],
        "stream": stream,
        "metadata": {"mask_encoding": "one_bit"},
    }


def parse_output_text(output_text: str, phrase: str, shape: tuple[int, int]) -> SegmentationResult:
    """Parse SAM output_text into tracks.

    Live wire format verified on 2026-10-01:
    Tokens look like <0f>0<|box;x1=160;y1=65;x2=479;y2=452;w=640;h=546|><|mask;...|>
    The number before f is the frame index. The number after is the object ordinal.
    The one_bit mask raster is decoded by @meta-sam/parser in the JS ecosystem.
    This parser extracts boxes and builds box masks so the Python pipeline
    can pad, carry, and render today. Swap in a raster decoder when available.

    Zero matches returns an empty result and is valid.
    """
    result = SegmentationResult(phrase=phrase, raw_output_text=output_text or "")
    if not output_text or not output_text.strip():
        return result

    h, w = shape
    # Live SAM token format with optional mask token right after the box
    import re

    token_pat = re.compile(
        r"<(\d+)f>(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>"
        r"(?:<\|mask;x=0;y=0;data=(\d+),(\d+),([^|]+)\|>)?"
    )
    found = False
    pending_decode: list[DetectedObject] = []
    for m in token_pat.finditer(output_text):
        found = True
        frame_index = int(m.group(1))
        ordinal = m.group(2)
        x0, y0, x1, y1 = int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6))
        track_id = str(ordinal)
        mask = np.zeros((h, w), dtype=bool)
        mask[max(0, y0): max(0, y1), max(0, x0): max(0, x1)] = True
        det = DetectedObject(
            track_id=track_id,
            frame_index=frame_index,
            box=(x0, y0, x1, y1),
            mask=mask,
            phrase=phrase,
        )
        # Mask token groups are 9 height, 10 width, 11 payload in the token header order
        # Live tokens use data=height,width,payload and the raster is width by height
        if m.group(9) and m.group(11):
            try:
                mh = int(m.group(9))
                mw = int(m.group(10))
                det.mask_payload = m.group(11)
                det.mask_width = mw
                det.mask_height = mh
                pending_decode.append(det)
            except Exception:
                pass
        result.objects.append(det)
        result.tracks.setdefault(track_id, {})[frame_index] = mask
    if found:
        # Try pixel-perfect raster decode when Node and @meta-sam/parser are present
        if pending_decode:
            try:
                from .mask_decode import decode_masks_batch

                items = [
                    {"payload": d.mask_payload, "width": d.mask_width, "height": d.mask_height}
                    for d in pending_decode
                ]
                decoded = decode_masks_batch(items)  # type: ignore[arg-type]
                if decoded is not None:
                    placed = 0
                    for det, raster in zip(pending_decode, decoded):
                        if raster is None:
                            continue
                        x0, y0, x1, y1 = det.box
                        full = np.zeros((h, w), dtype=bool)
                        # Raster covers the box area. Resize to box size when needed.
                        rh, rw = raster.shape
                        box_h = max(0, y1 - y0)
                        box_w = max(0, x1 - x0)
                        if box_h > 0 and box_w > 0:
                            if (rh, rw) != (box_h, box_w):
                                # Nearest neighbor resize via numpy indexing
                                yi = (np.linspace(0, rh - 1, box_h)).astype(int)
                                xi = (np.linspace(0, rw - 1, box_w)).astype(int)
                                raster = raster[np.ix_(yi, xi)]
                            full[max(0, y0): max(0, y1), max(0, x0): max(0, x1)] = raster
                            det.mask = full
                            result.tracks[det.track_id][det.frame_index] = full
                            placed += 1
                    if placed:
                        print(f"Decoded {placed} pixel masks via @meta-sam/parser")
            except Exception as exc:
                print(f"Mask decode unavailable, using box masks: {exc}")
        return result

    # JSON lines fallback for local testing and recorded fixtures
    lines = [ln.strip() for ln in output_text.strip().splitlines() if ln.strip()]
    parsed_any = False
    for line in lines:
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            parsed_any = True
            frame_index = int(obj.get("frame", obj.get("frame_index", 0)))
            track_id = str(obj.get("track_id", obj.get("id", "0")))
            box = obj.get("box", [0, 0, w, h])
            if len(box) != 4:
                box = [0, 0, w, h]
            x0, y0, x1, y1 = [int(v) for v in box]
            mask = np.zeros((h, w), dtype=bool)
            mask[max(0, y0): max(0, y1), max(0, x0): max(0, x1)] = True
            det = DetectedObject(
                track_id=track_id,
                frame_index=frame_index,
                box=(x0, y0, x1, y1),
                mask=mask,
                phrase=phrase,
            )
            result.objects.append(det)
            result.tracks.setdefault(track_id, {})[frame_index] = mask
    if parsed_any:
        return result

    return result


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "open-redactor" / "sam"


def cache_key_for(video_path: Path, phrase: str) -> str:
    h = hashlib.sha256()
    try:
        h.update(video_path.read_bytes())
    except Exception:
        h.update(str(video_path).encode())
    h.update(phrase.encode())
    h.update(DEFAULT_MODEL.encode())
    return h.hexdigest()


class SamApiClient:
    """Client for the Meta Model API SAM 3.1 endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.cache_dir = cache_dir if cache_dir is not None else default_cache_dir()
        self.use_cache = use_cache

    @classmethod
    def from_env(
        cls,
        env_name: str = "MODEL_API_KEY",
        use_cache: bool = True,
        endpoint: str | None = None,
    ) -> "SamApiClient":
        key = os.environ.get(env_name)
        kwargs: dict = {"api_key": key, "use_cache": use_cache}
        if endpoint:
            kwargs["endpoint"] = endpoint
        elif os.environ.get("OPEN_REDACTOR_ENDPOINT"):
            kwargs["endpoint"] = os.environ["OPEN_REDACTOR_ENDPOINT"]
        return cls(**kwargs)

    def segment_video(self, video_path: Path, phrase: str, shape: tuple[int, int]) -> SegmentationResult:
        """Run one phrase against one video via the API.

        Returns an empty result when the model finds nothing.
        Raises a clear error when httpx is missing or no key is set.
        """
        if httpx is None:
            raise RuntimeError("httpx is required for API mode. Install with pip install httpx")

        # Cache check first. A hit avoids paying for detection again on
        # re-renders and works with no key at all, fully offline.
        cache_path: Optional[Path] = None
        if self.use_cache:
            try:
                key = cache_key_for(video_path, phrase)
                cache_path = self.cache_dir / f"{key}.txt"
                if cache_path.exists():
                    cached = cache_path.read_text()
                    print(f"SAM cache hit for '{phrase}'. Skipping API call.")
                    return parse_output_text(cached, phrase=phrase, shape=shape)
            except Exception:
                cache_path = None
        if not self.api_key:
            raise RuntimeError("No API key found. Set the env var named by --api-key-env")

        body = build_request_body(phrase, video_path, stream=True)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        output_parts: List[str] = []
        # Streaming response handling. Live events use response.output_text.delta
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", self.endpoint, json=body, headers=headers) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            event = json.loads(payload)
                        except json.JSONDecodeError:
                            output_parts.append(payload)
                            continue
                        if event.get("type") == "response.output_text.delta":
                            delta = event.get("delta", "")
                            if isinstance(delta, str) and delta:
                                output_parts.append(delta)
                            continue
                        if event.get("type") == "response.completed":
                            resp_obj = event.get("response", {})
                            for item in resp_obj.get("output", []) or []:
                                for content in item.get("content", []) or []:
                                    txt = content.get("text") or ""
                                    if txt and not output_parts:
                                        output_parts.append(txt)
                            continue
                        delta = event.get("delta") or event.get("text") or ""
                        if isinstance(delta, str) and delta:
                            output_parts.append(delta)
                    else:
                        try:
                            event = json.loads(line)
                            for item in event.get("output", []) or []:
                                for content in item.get("content", []) or []:
                                    txt = content.get("text") or ""
                                    if txt:
                                        output_parts.append(txt)
                        except json.JSONDecodeError:
                            output_parts.append(line)
        output_text = "".join(output_parts)
        if self.use_cache and cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(output_text)
            except Exception:
                pass
        return parse_output_text(output_text, phrase=phrase, shape=shape)

    def segment_image(self, image_path: Path, phrase: str, shape: tuple[int, int]) -> SegmentationResult:
        """Run one phrase against one photo via the image API shape."""
        if httpx is None:
            raise RuntimeError("httpx is required for API mode. Install with pip install httpx")
        cache_path: Optional[Path] = None
        if self.use_cache:
            try:
                key = cache_key_for(image_path, phrase + ":image")
                cache_path = self.cache_dir / f"{key}.txt"
                if cache_path.exists():
                    print(f"SAM cache hit for '{phrase}'. Skipping API call.")
                    return parse_output_text(cache_path.read_text(), phrase=phrase, shape=shape)
            except Exception:
                cache_path = None
        if not self.api_key:
            raise RuntimeError("No API key found. Set the env var named by --api-key-env")
        image_b64 = base64.b64encode(image_path.read_bytes()).decode()
        body = build_image_request_body(phrase, image_b64, stream=True)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        output_parts: List[str] = []
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", self.endpoint, json=body, headers=headers) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        output_parts.append(payload)
                        continue
                    if event.get("type") == "response.output_text.delta":
                        delta = event.get("delta", "")
                        if isinstance(delta, str) and delta:
                            output_parts.append(delta)
                    elif event.get("type") == "response.completed":
                        for item in (event.get("response", {}).get("output", []) or []):
                            for content in item.get("content", []) or []:
                                txt = content.get("text") or ""
                                if txt and not output_parts:
                                    output_parts.append(txt)
        output_text = "".join(output_parts)
        if self.use_cache and cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(output_text)
            except Exception:
                pass
        return parse_output_text(output_text, phrase=phrase, shape=shape)


class LocalSamStub:
    """Local mode stub.

    Real local inference loads open SAM weights.
    The stub keeps the CLI runnable and testable on machines
    without weights and returns zero matches with a clear log path.
    Swap run_segmentation with a real model loader when weights are present.
    """

    def __init__(self, weights_path: Optional[Path] = None) -> None:
        self.weights_path = weights_path

    def segment_image(self, image_path: Path, phrase: str, shape: tuple[int, int]) -> SegmentationResult:
        return SegmentationResult(phrase=phrase, raw_output_text="")

    def segment_video(self, video_path: Path, phrase: str, shape: tuple[int, int]) -> SegmentationResult:
        # Extension point for real local weights.
        # For v1 the stub returns an empty result so the pipeline
        # still exercises padding, rendering, and contact sheet code.
        print("Local backend: open SAM weights are not loaded in this build, returning zero matches. Use the api or hosted backend for live detection, or see docs/using-open-redactor.md for the local roadmap.")
        return SegmentationResult(phrase=phrase, raw_output_text="")
