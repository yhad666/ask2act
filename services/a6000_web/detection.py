from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

from .phrase_extractor import InstructionPhraseExtractor
from .schemas import Candidate


DEFAULT_BASE_TERMS = ("cup", "bottle", "spoon", "fork", "knife", "plate")
SAME_CLASS_RECALL_TERMS = {
    "fork": ("fork", "forks", "plastic fork", "table fork", "dining fork"),
}
RELATION_PROMPT_WORDS = {
    "above",
    "at",
    "behind",
    "below",
    "beside",
    "between",
    "in",
    "inside",
    "near",
    "next",
    "on",
    "over",
    "under",
    "with",
}
SPATIAL_MODIFIER_WORDS = {
    "back",
    "backmost",
    "bottom",
    "bottommost",
    "center",
    "centered",
    "central",
    "front",
    "frontmost",
    "left",
    "leftmost",
    "middle",
    "right",
    "rightmost",
    "top",
    "topmost",
}
RELATIVE_SIZE_WORDS = {"biggest", "largest", "smallest"}
COLOR_ATTRIBUTE_WORDS = {
    "black",
    "blue",
    "brown",
    "clear",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "transparent",
    "white",
    "yellow",
}
MATERIAL_ATTRIBUTE_WORDS = {
    "ceramic",
    "glass",
    "metal",
    "metallic",
    "paper",
    "plastic",
    "rubber",
    "steel",
    "wood",
    "wooden",
}
SIZE_ATTRIBUTE_WORDS = {"big", "large", "little", "medium", "small", "smaller", "larger"}


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


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        value = (item or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _prompt_from_terms(terms: Sequence[str]) -> str:
    prompt = " . ".join(_dedupe_keep_order(terms)).strip()
    if prompt:
        return _normalize_grounding_prompt(f"{prompt} .")
    return ""


def _normalize_grounding_prompt(prompt: str) -> str:
    cleaned = re.sub(r"\s+", " ", (prompt or "").strip())
    if not cleaned:
        return ""
    cleaned = re.sub(r"\s*\.\s*$", "", cleaned).strip()
    return f"{cleaned} ."


def _phrase_words(phrase: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_-]*", phrase.lower())


def _contains_visual_modifier(phrases: Sequence[str]) -> bool:
    for phrase in phrases:
        if set(_phrase_words(phrase)) & _visual_modifier_words():
            return True
    return False


def _contains_attribute_modifier(phrases: Sequence[str]) -> bool:
    attribute_words = COLOR_ATTRIBUTE_WORDS | MATERIAL_ATTRIBUTE_WORDS | SIZE_ATTRIBUTE_WORDS
    for phrase in phrases:
        if set(_phrase_words(phrase)) & attribute_words:
            return True
    return False


def _visual_modifier_words() -> set[str]:
    return (
        RELATION_PROMPT_WORDS
        | SPATIAL_MODIFIER_WORDS
        | RELATIVE_SIZE_WORDS
        | COLOR_ATTRIBUTE_WORDS
        | MATERIAL_ATTRIBUTE_WORDS
        | SIZE_ATTRIBUTE_WORDS
    )


def _base_terms_in_phrases(phrases: Sequence[str], base_terms: Sequence[str]) -> List[str]:
    out: List[str] = []
    for phrase in phrases:
        lowered = phrase.lower()
        for term in base_terms:
            pattern = rf"\b{re.escape(term.lower())}s?\b"
            if re.search(pattern, lowered) and term.lower() not in out:
                out.append(term.lower())
    return out


def _object_like_terms_in_phrases(phrases: Sequence[str]) -> List[str]:
    out: List[str] = []
    skip_words = _visual_modifier_words() | {
        "half",
        "one",
        "side",
    }
    for phrase in phrases:
        for word in _phrase_words(phrase):
            if word in skip_words:
                continue
            if word in out:
                continue
            out.append(word)
    return out


def _attribute_relaxed_terms(phrases: Sequence[str], base_terms: Sequence[str]) -> List[str]:
    out: List[str] = []
    base_term_set = {term.lower() for term in base_terms}
    keep_attributes = COLOR_ATTRIBUTE_WORDS | MATERIAL_ATTRIBUTE_WORDS
    for phrase in phrases:
        words = _phrase_words(phrase)
        bases = [word for word in words if word in base_term_set]
        if not bases:
            continue
        kept = [word for word in words if word in keep_attributes]
        if not kept:
            continue
        for base in bases:
            relaxed = " ".join([*kept, base])
            if relaxed != phrase.lower() and relaxed not in out:
                out.append(relaxed)
    return out


def _target_terms_from_phrases(phrases: Sequence[str]) -> List[str]:
    return _dedupe_keep_order(phrase.lower() for phrase in phrases if (phrase or "").strip())


def _singularize(word: str) -> str:
    word = (word or "").lower()
    if len(word) > 3 and word.endswith("ies"):
        return f"{word[:-3]}y"
    if len(word) > 3 and word.endswith("es"):
        return word[:-2]
    if len(word) > 2 and word.endswith("s"):
        return word[:-1]
    return word


def _label_matches_target_term(label: str, target_term: str) -> bool:
    label_words = [_singularize(word) for word in _phrase_words(label)]
    target_words = [_singularize(word) for word in _phrase_words(target_term)]
    if not label_words or not target_words or len(target_words) > len(label_words):
        return False
    for start in range(0, len(label_words) - len(target_words) + 1):
        if label_words[start : start + len(target_words)] == target_words:
            return True
    return False


def _filter_candidates_by_target_terms(candidates: Sequence[Candidate], target_terms: Sequence[str]) -> List[Candidate]:
    terms = _target_terms_from_phrases(target_terms)
    if not terms:
        return []
    return [
        candidate
        for candidate in candidates
        if any(_label_matches_target_term(candidate.label, term) for term in terms)
    ]


def _same_class_recall_terms(phrases: Sequence[str]) -> List[str]:
    out: List[str] = []
    for phrase in phrases:
        words = _phrase_words(phrase)
        if len(words) != 1:
            continue
        out.extend(SAME_CLASS_RECALL_TERMS.get(_singularize(words[0]), ()))
    return _dedupe_keep_order(out)


def _load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _candidate_display_label(candidate: Candidate) -> str:
    if candidate.display_id is not None:
        display_id = str(candidate.display_id)
    else:
        suffix = candidate.candidate_id.rsplit("_", 1)[-1]
        try:
            display_id = str(int(suffix))
        except ValueError:
            display_id = candidate.candidate_id
    label = (candidate.label or "").strip()
    if label:
        return f"{display_id} {label}"
    return display_id


def _rect_overlap_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    left = max(min(ax1, ax2), min(bx1, bx2))
    top = max(min(ay1, ay2), min(by1, by2))
    right = min(max(ax1, ax2), max(bx1, bx2))
    bottom = min(max(ay1, ay2), max(by1, by2))
    return max(0.0, right - left) * max(0.0, bottom - top)


def _label_position_for_bbox(
    image_size: Tuple[int, int],
    bbox_xyxy: Sequence[float],
    label_size: Tuple[int, int],
    gap: int,
    avoid_bboxes: Sequence[Sequence[float]] | None = None,
) -> Tuple[int, int]:
    image_w, image_h = image_size
    label_w, label_h = label_size
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    max_x = max(0.0, float(image_w - label_w))
    max_y = max(0.0, float(image_h - label_h))

    preferred = [
        (center_x - label_w / 2.0, top - label_h - gap),
        (center_x - label_w / 2.0, bottom + gap),
        (left - label_w - gap, center_y - label_h / 2.0),
        (right + gap, center_y - label_h / 2.0),
    ]
    candidates: List[Tuple[int, float, float]] = []
    for idx, (origin_x, origin_y) in enumerate(preferred):
        if 0 <= origin_x <= max_x and 0 <= origin_y <= max_y:
            candidates.append((idx, origin_x, origin_y))

    if not candidates:
        for idx, (origin_x, origin_y) in enumerate(preferred):
            candidates.append(
                (
                    idx,
                    max(0.0, min(origin_x, max_x)),
                    max(0.0, min(origin_y, max_y)),
                )
            )

    own_rect = (left, top, right, bottom)
    avoid_rects = list(avoid_bboxes or [])

    def score(option: Tuple[int, float, float]) -> Tuple[int, float, float, int]:
        idx, origin_x, origin_y = option
        label_rect = (origin_x, origin_y, origin_x + label_w, origin_y + label_h)
        own_overlap = _rect_overlap_area(label_rect, own_rect)
        avoid_overlap = sum(_rect_overlap_area(label_rect, rect) for rect in avoid_rects)
        return (1 if own_overlap > 0 else 0, avoid_overlap, own_overlap, idx)

    _, best_x, best_y = min(candidates, key=score)
    return int(round(best_x)), int(round(best_y))


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
        box_threshold: float | None = None,
        text_threshold: float | None = None,
        nms_iou: float = 0.50,
        max_per_image: int = 100,
        rotate_clockwise_90: bool | None = None,
        device: str | None = None,
    ) -> None:
        self.phrase_extractor = phrase_extractor
        self.model_id = model_id
        self.base_terms = tuple(base_terms)
        self.box_threshold = float(os.getenv("ASK2ACT_DINO_BOX_THRESHOLD", "0.38")) if box_threshold is None else box_threshold
        self.text_threshold = (
            float(os.getenv("ASK2ACT_DINO_TEXT_THRESHOLD", "0.30")) if text_threshold is None else text_threshold
        )
        self.nms_iou = nms_iou
        self.max_per_image = max_per_image
        self.rotate_clockwise_90 = (
            os.getenv("ASK2ACT_DINO_ROTATE_CLOCKWISE_90", "1").strip().lower() in {"1", "true", "yes", "on"}
            if rotate_clockwise_90 is None
            else rotate_clockwise_90
        )
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

    def _post_process(
        self,
        outputs,
        input_ids,
        height: int,
        width: int,
        *,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ):
        box_threshold = self.box_threshold if box_threshold is None else box_threshold
        text_threshold = self.text_threshold if text_threshold is None else text_threshold
        for kwargs in (
            dict(threshold=box_threshold, text_threshold=text_threshold),
            dict(box_threshold=box_threshold, text_threshold=text_threshold),
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
        prompt = _normalize_grounding_prompt(_prompt_from_terms(terms))
        if not prompt:
            prompt = _prompt_from_terms(self.base_terms)
        return phrases, prompt

    def _build_broad_fallback_detection_prompt(self, phrases: Sequence[str], primary_prompt: str) -> str | None:
        terms = _object_like_terms_in_phrases(phrases)
        terms = [*terms, *[term.lower() for term in self.base_terms]]
        fallback_prompt = _prompt_from_terms(terms)
        if not fallback_prompt or fallback_prompt == primary_prompt:
            return None
        return fallback_prompt

    def _build_attribute_relaxed_detection_prompt(self, phrases: Sequence[str], primary_prompt: str) -> str | None:
        terms = _attribute_relaxed_terms(phrases, self.base_terms)
        if not terms:
            return None
        fallback_prompt = _prompt_from_terms(terms)
        if not fallback_prompt or fallback_prompt == primary_prompt:
            return None
        return fallback_prompt

    def _build_attribute_object_recall_prompt(self, phrases: Sequence[str], primary_prompt: str) -> str | None:
        if not _contains_visual_modifier(phrases):
            return None
        terms = _object_like_terms_in_phrases(phrases)
        fallback_prompt = _prompt_from_terms(terms)
        if not fallback_prompt or fallback_prompt == primary_prompt:
            return None
        return fallback_prompt

    def _build_context_detection_prompt(self, phrases: Sequence[str], primary_prompt: str) -> str | None:
        target_terms = _target_terms_from_phrases(phrases)
        if not target_terms:
            return None
        prompt = _prompt_from_terms([*target_terms, *[term.lower() for term in self.base_terms]])
        if not prompt or prompt == primary_prompt:
            return None
        return prompt

    def _build_same_class_recall_prompt(self, phrases: Sequence[str], primary_prompt: str) -> str | None:
        terms = _same_class_recall_terms(phrases)
        if not terms:
            return None
        prompt = _prompt_from_terms(terms)
        if not prompt or prompt == primary_prompt:
            return None
        return prompt

    def _run_grounding_dino(
        self,
        prepared: Image.Image,
        prompt: str,
        *,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> List[Candidate]:
        width, height = prepared.size
        inputs = self.processor(images=prepared, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self._post_process(
            outputs,
            inputs["input_ids"],
            height,
            width,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        labels_raw = results.get("text_labels", results.get("labels", []))
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])

        candidates: List[Candidate] = []
        if len(boxes) <= 0:
            return candidates

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
                    display_id=idx,
                    label=str(label),
                    score=float(score),
                    bbox_xyxy=[x1, y1, x2, y2],
                    mask_rle=None,
                )
            )
        return candidates

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
        font_size = max(18, int(min(image.size) * 0.035))
        font = _load_font(font_size)
        line_w = max(2, int(min(image.size) * 0.003))
        pad_x = max(4, int(font_size * 0.25))
        pad_y = max(3, int(font_size * 0.18))
        label_gap = max(3, line_w + 1)
        candidate_bboxes = [cand.bbox_xyxy for cand in candidates]

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

            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w + (1 if is_selected else 0))
            tag = _candidate_display_label(cand)
            bbox = draw.textbbox((0, 0), tag, font=font)
            tw = bbox[2] - bbox[0] + (2 * pad_x)
            th = bbox[3] - bbox[1] + (2 * pad_y)
            box_x0, box_y0 = _label_position_for_bbox(
                image.size,
                cand.bbox_xyxy,
                (tw, th),
                label_gap,
                candidate_bboxes,
            )
            draw.rectangle([box_x0, box_y0, box_x0 + tw, box_y0 + th], fill=color)
            draw.text((box_x0 + pad_x, box_y0 + pad_y), tag, fill="black", font=font)

        if banner_text:
            banner_font = _load_font(max(24, int(min(image.size) * 0.05)))
            bbox = draw.textbbox((0, 0), banner_text, font=banner_font)
            tw = bbox[2] - bbox[0] + 24
            th = bbox[3] - bbox[1] + 16
            color = "#2e8b57" if success else "#d74e4e"
            draw.rectangle([16, 16, 16 + tw, 16 + th], fill=color)
            draw.text((28, 24), banner_text, fill="white", font=banner_font)

        return image_bytes_to_data_url(pil_to_png_bytes(image))

    def detect(
        self,
        image_bytes: bytes,
        instruction: str,
        *,
        enable_attribute_object_recall: bool = False,
        attribute_object_recall_candidate_threshold: int = 0,
    ) -> DetectionResult:
        self._ensure_model()
        prepared = self._prepare_image(image_bytes)
        phrases, prompt = self.build_detection_prompt(instruction)
        using_attribute_object_recall = False

        candidates = self._run_grounding_dino(prepared, prompt)
        selected_prompt = prompt
        same_class_recall_prompt = self._build_same_class_recall_prompt(phrases, prompt)
        if same_class_recall_prompt and len(candidates) <= 1:
            same_class_recall_candidates = self._run_grounding_dino(prepared, same_class_recall_prompt)
            if len(same_class_recall_candidates) > len(candidates):
                candidates = same_class_recall_candidates
                selected_prompt = same_class_recall_prompt

        if enable_attribute_object_recall and len(candidates) <= attribute_object_recall_candidate_threshold:
            object_recall_prompt = self._build_attribute_object_recall_prompt(phrases, prompt)
            if object_recall_prompt:
                object_recall_candidates = self._run_grounding_dino(prepared, object_recall_prompt)
                if len(object_recall_candidates) > len(candidates):
                    candidates = object_recall_candidates
                    selected_prompt = object_recall_prompt
                    using_attribute_object_recall = True

        attribute_relaxed_prompt = self._build_attribute_relaxed_detection_prompt(phrases, prompt)
        if attribute_relaxed_prompt and len(candidates) == 0 and not using_attribute_object_recall:
            attribute_relaxed_candidates = self._run_grounding_dino(prepared, attribute_relaxed_prompt)
            if len(attribute_relaxed_candidates) > len(candidates):
                candidates = attribute_relaxed_candidates
                selected_prompt = attribute_relaxed_prompt

        context_prompt = self._build_context_detection_prompt(phrases, selected_prompt)
        if context_prompt and len(candidates) == 0 and not using_attribute_object_recall:
            context_candidates = self._run_grounding_dino(prepared, context_prompt)
            target_candidates = _filter_candidates_by_target_terms(context_candidates, phrases)
            if target_candidates:
                candidates = target_candidates
                selected_prompt = context_prompt

        broad_fallback_prompt = self._build_broad_fallback_detection_prompt(phrases, selected_prompt)
        should_try_broad_fallback = (
            broad_fallback_prompt is not None
            and (_contains_visual_modifier(phrases) or not _base_terms_in_phrases(phrases, self.base_terms))
            and len(candidates) == 0
        )
        if broad_fallback_prompt and should_try_broad_fallback:
            broad_fallback_candidates = self._run_grounding_dino(prepared, broad_fallback_prompt)
            if len(broad_fallback_candidates) > 0:
                candidates = broad_fallback_candidates
                selected_prompt = broad_fallback_prompt

        prepared_bytes = pil_to_png_bytes(prepared)
        return DetectionResult(
            prepared_image_bytes=prepared_bytes,
            image_data_url=image_bytes_to_data_url(prepared_bytes),
            overlay_data_url=self.render_overlay(prepared_bytes, candidates),
            candidates=candidates,
            phrases=phrases,
            detection_prompt=selected_prompt,
        )
