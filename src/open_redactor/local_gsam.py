"""Local Grounding SAM provider.

Phrase grounding with Grounding DINO, mask cutting with SAM 2, and a
small IoU tracker that keeps identities stable across frames. All model
imports are lazy so the base CLI install stays light. Install the heavy
stack with: pip install "open-redactor[local]"

Video approach for v1: run grounding plus segmentation on each frame
and link detections into tracks by box overlap. This is slower than a
native video propagator but it is real, inspectable, and works with the
image models published on Hugging Face today.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .sam_client import DetectedObject, SegmentationResult

GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM_MODEL = "facebook/sam2-hiera-tiny"

Box = Tuple[float, float, float, float]


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IouTracker:
    """Greedy IoU linking. A detection joins the best overlapping track."""

    def __init__(self, min_iou: float = 0.3) -> None:
        self.min_iou = min_iou
        self.last_box: Dict[int, Box] = {}
        self.next_id = 0

    def assign(self, boxes: List[Box]) -> List[int]:
        ids: List[int] = []
        used: set[int] = set()
        for box in boxes:
            best_id, best_iou = None, self.min_iou
            for tid, prev in self.last_box.items():
                if tid in used:
                    continue
                score = box_iou(box, prev)
                if score >= best_iou:
                    best_id, best_iou = tid, score
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            used.add(best_id)
            self.last_box[best_id] = box
            ids.append(best_id)
        return ids


def local_stack_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


class LocalGroundingSamClient:
    """Real local provider. Returns zero matches with a clear message
    when the optional local stack is not installed, so the CLI never
    fails silently or pretends it detected something."""

    def __init__(
        self,
        grounding_model: str = GROUNDING_MODEL,
        sam_model: str = SAM_MODEL,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: Optional[str] = None,
    ) -> None:
        self.grounding_model = grounding_model
        self.sam_model = sam_model
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self._gd = None
        self._gd_processor = None
        self._sam = None
        self._sam_processor = None

    def _load(self) -> bool:
        if self._gd is not None:
            return True
        try:
            import torch
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
                Sam2Model,
                Sam2Processor,
            )
        except Exception as exc:
            print(
                "Local provider needs the optional stack. "
                'Install with: pip install "open-redactor[local]" '
                f"Import failed: {exc}"
            )
            return False
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        print(f"Local Grounding SAM on {device}: {self.grounding_model} + {self.sam_model}")
        self._gd_processor = AutoProcessor.from_pretrained(self.grounding_model)
        self._gd = AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_model).to(device)
        self._sam_processor = Sam2Processor.from_pretrained(self.sam_model)
        self._sam = Sam2Model.from_pretrained(self.sam_model).to(device)
        return True

    def segment_video(self, video_path: Path, phrase: str, shape: tuple[int, int]) -> SegmentationResult:
        if not self._load():
            return SegmentationResult(phrase=phrase, raw_output_text="")
        import cv2
        import torch
        from PIL import Image

        height, width = shape
        result = SegmentationResult(phrase=phrase, mask_source="pixel")
        tracker = IouTracker()
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        with torch.no_grad():
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                # Grounding DINO: phrase in, boxes out
                inputs = self._gd_processor(images=image, text=[[phrase]], return_tensors="pt").to(self.device)
                outputs = self._gd(**inputs)
                target_sizes = [(height, width)]
                # transformers 5 renamed box_threshold to threshold
                import inspect

                gd_params = inspect.signature(
                    self._gd_processor.post_process_grounded_object_detection
                ).parameters
                box_kwarg = "threshold" if "threshold" in gd_params else "box_threshold"
                detections = self._gd_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    text_threshold=self.text_threshold,
                    target_sizes=target_sizes,
                    **{box_kwarg: self.box_threshold},
                )[0]
                boxes: List[Box] = [tuple(map(float, b)) for b in detections["boxes"].tolist()]
                if boxes:
                    track_ids = tracker.assign(boxes)
                    # SAM 2 image segmentation from box prompts
                    sam_inputs = self._sam_processor(
                        images=image,
                        input_boxes=[[list(b) for b in boxes]],
                        return_tensors="pt",
                    ).to(self.device)
                    sam_outputs = self._sam(**sam_inputs)
                    masks = self._sam_processor.post_process_masks(
                        sam_outputs.pred_masks.cpu(),
                        sam_inputs["original_sizes"].cpu(),
                    )[0]
                    for i, tid in enumerate(track_ids):
                        mask_arr = masks[i]
                        if mask_arr.ndim == 3:
                            mask_arr = mask_arr[0]
                        mask_bool = np.asarray(mask_arr).astype(bool)
                        if mask_bool.shape != (height, width):
                            mask_bool = np.asarray(
                                Image.fromarray(mask_bool.astype(np.uint8) * 255).resize(
                                    (width, height)
                                )
                            ).astype(bool)
                        key = str(tid)
                        result.tracks.setdefault(key, {})[frame_idx] = mask_bool
                        result.objects.append(
                            DetectedObject(
                                track_id=key,
                                frame_index=frame_idx,
                                box=tuple(int(round(v)) for v in boxes[i]),
                                mask=mask_bool,
                                phrase=phrase,
                            )
                        )
                frame_idx += 1
        cap.release()
        print(f"Local Grounding SAM: '{phrase}' produced {len(result.tracks)} tracks over {frame_idx} frames")
        return result


    def segment_image(self, image_path: Path, phrase: str, shape: tuple[int, int]) -> SegmentationResult:
        """Run the local stack on one photo by way of a one frame clip."""
        import tempfile

        import cv2

        frame = cv2.imread(str(image_path))
        if frame is None:
            return SegmentationResult(phrase=phrase, raw_output_text="")
        with tempfile.TemporaryDirectory(prefix="open-redactor-photo-") as td:
            tmp = Path(td) / "photo.mp4"
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 10, (shape[1], shape[0]))
            writer.write(frame)
            writer.release()
            return self.segment_video(tmp, phrase, shape=shape)
