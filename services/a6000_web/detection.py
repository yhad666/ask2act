from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

from .phrase_extractor import InstructionPhraseExtractor
from .schemas import Candidate


DEFAULT_BASE_TERMS = ("cup", "bottle", "spoon", "fork", "knife", "plate")


def image_bytes_to_data_url(image_bytes: bytes, mime_type: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def to_numpy_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms_per_class(boxes, scores, labels, iou_thr=0.5):
    b_arr = np.array(boxes, float)
    s_arr = np.array(scores, float)
    l_arr = np.array(labels, dtype=object)
    keep_b, keep_s, keep_l = [], [], []
    for cls in np.unique(l_arr):
        idx = np.where(l_arr == cls)[0]
        cls_boxes, cls_scores = b_arr[idx], s_arr[idx]
        order = np.argsort(-cls_scores)
        while len(order) > 0:
            top = order[0]
            keep_b.append(cls_boxes[top])
            keep_s.append(cls_scores[top])
            keep_l.append(cls)
            rest = order[1:]
            order = np.array([j for j in rest if iou_xyxy(cls_boxes[top], cls_boxes[j]) <= iou_thr])
    return np.array(keep_b), np.array(keep_s), np.array(keep_l)


def _load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass
class DetectionResult:
    prepared_image_bytes: bytes
    image_data_url: str
    overlay_data_url: str
    candidates: List[Candidate]
    phrases: List[str]
    detection_prompt: str


class GroundingDinoDetector:
    def __init__(
        self,
        phrase_extractor: InstructionPhraseExtractor,
        model_id: str = "IDEA-Research/grounding-dino-base",
        base_terms: Sequence[str] = DEFAULT_BASE_TERMS,
        box_threshold: float = 0.20,
        text_threshold: float = 0.30,
        nms_iou: float = 0.50,
        max_per_image: int = 100,
        rotate_clockwise_90: bool = True,
        device: str | None = None,
    ) -> None:
        self.phrase_extractor = phrase_extractor
        self.model_id = model_id
        self.base_terms = tuple(base_terms)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou = nms_iou
        self.max_per_image = max_per_image
        self.rotate_clockwise_90 = rotate_clockwise_90
        self.device = self._resolve_device(device)
        self.processor = None
        self.model = None

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        if device:
            return device

        env_device = os.getenv("ASK2ACT_DETECTOR_DEVICE", "").strip()
        if env_device:
            return env_device

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                return "cuda:1"
            return "cuda:0"

        return "cpu"

    def _ensure_model(self) -> None:
        if self.processor is not None and self.model is not None:
            return
        self.processor = GroundingDinoProcessor.from_pretrained(self.model_id)
        self.model = GroundingDinoForObjectDetection.from_pretrained(self.model_id).to(self.device).eval()

    def _prepare_image(self, image_bytes: bytes) -> Image.Image:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = ImageOps.exif_transpose(image)
        if self.rotate_clockwise_90:
            image = image.transpose(Image.ROTATE_270)
        return image

    def _post_process(self, outputs, input_ids, height: int, width: int):
        for kwargs in (
            dict(threshold=self.box_threshold, text_threshold=self.text_threshold),
            dict(box_threshold=self.box_threshold, text_threshold=self.text_threshold),
        ):
            try:
                return self.processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    input_ids=input_ids,
                    target_sizes=[(height, width)],
                    **kwargs,
                )[0]
            except TypeError:
                continue
        raise RuntimeError("GroundingDINO post_process signature mismatch")

    def build_detection_prompt(self, instruction: str) -> Tuple[List[str], str]:
        phrases, terms = self.phrase_extractor.extract_terms(instruction, self.base_terms)
        prompt = " . ".join(terms).strip()
        if prompt:
            prompt += " ."
        else:
            prompt = " . ".join(self.base_terms) + " ."
        return phrases, prompt

    def render_overlay(
        self,
        prepared_image_bytes: bytes,
        candidates: Sequence[Candidate],
        selected_candidate_id: str | None = None,
        success: bool | None = None,
        banner_text: str | None = None,
    ) -> str:
        image = Image.open(io.BytesIO(prepared_image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = _load_font(max(18, int(min(image.size) * 0.04)))
        line_w = max(3, int(min(image.size) * 0.006))

        for cand in candidates:
            x1, y1, x2, y2 = cand.bbox_xyxy
            is_selected = cand.candidate_id == selected_candidate_id
            if is_selected and success is True:
                color = "#2e8b57"
            elif is_selected and success is False:
                color = "#d74e4e"
            elif is_selected:
                color = "#f4b942"
            else:
                color = "#36c1ff"

            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
            tag = f"{cand.candidate_id} · {cand.label} · {cand.score:.2f}"
            bbox = draw.textbbox((0, 0), tag, font=font)
            tw = bbox[2] - bbox[0] + 14
            th = bbox[3] - bbox[1] + 10
            box_y0 = max(0, y1 - th)
            draw.rectangle([x1, box_y0, x1 + tw, box_y0 + th], fill=color)
            draw.text((x1 + 7, box_y0 + 5), tag, fill="black", font=font)

        if banner_text:
            banner_font = _load_font(max(24, int(min(image.size) * 0.05)))
            bbox = draw.textbbox((0, 0), banner_text, font=banner_font)
            tw = bbox[2] - bbox[0] + 24
            th = bbox[3] - bbox[1] + 16
            color = "#2e8b57" if success else "#d74e4e"
            draw.rectangle([16, 16, 16 + tw, 16 + th], fill=color)
            draw.text((28, 24), banner_text, fill="white", font=banner_font)

        return image_bytes_to_data_url(pil_to_png_bytes(image))

    def detect(self, image_bytes: bytes, instruction: str) -> DetectionResult:
        self._ensure_model()
        prepared = self._prepare_image(image_bytes)
        phrases, prompt = self.build_detection_prompt(instruction)

        width, height = prepared.size
        inputs = self.processor(images=prepared, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self._post_process(outputs, inputs["input_ids"], height, width)

        labels_raw = results.get("text_labels", results.get("labels", []))
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])

        candidates: List[Candidate] = []
        if len(boxes) > 0:
            labels = [str(label) for label in labels_raw]
            boxes_np = to_numpy_safe(boxes)
            scores_np = to_numpy_safe(scores)
            keep_boxes, keep_scores, keep_labels = nms_per_class(boxes_np, scores_np, labels, self.nms_iou)
            if len(keep_boxes) > self.max_per_image:
                order = np.argsort(-np.asarray(keep_scores))[: self.max_per_image]
                keep_boxes = keep_boxes[order]
                keep_scores = np.asarray(keep_scores)[order]
                keep_labels = np.asarray(keep_labels, dtype=object)[order]

            for idx, (bbox, score, label) in enumerate(zip(keep_boxes, keep_scores, keep_labels), start=1):
                x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]
                candidates.append(
                    Candidate(
                        candidate_id=f"cand_{idx:03d}",
                        label=str(label),
                        score=float(score),
                        bbox_xyxy=[x1, y1, x2, y2],
                        mask_rle=None,
                    )
                )

        prepared_bytes = pil_to_png_bytes(prepared)
        return DetectionResult(
            prepared_image_bytes=prepared_bytes,
            image_data_url=image_bytes_to_data_url(prepared_bytes),
            overlay_data_url=self.render_overlay(prepared_bytes, candidates),
            candidates=candidates,
            phrases=phrases,
            detection_prompt=prompt,
        )
