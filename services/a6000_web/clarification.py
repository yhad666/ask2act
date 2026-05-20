from __future__ import annotations

import base64
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from PIL import Image

from .schemas import Candidate, QuestionTurn, ResolvedTarget, ScoredQuestion, SessionState


def compress_to_data_url(image_bytes: bytes, max_side: int = 768, jpeg_quality: int = 80) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    scale = min(max_side / max(width, height), 1.0)
    if scale < 1.0:
        image = image.resize((int(width * scale), int(height * scale)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def data_url_to_bytes(data_url: str) -> bytes:
    if not data_url:
        raise ValueError("Empty data URL")
    _, sep, payload = data_url.partition(",")
    if not sep or not payload:
        raise ValueError("Invalid data URL")
    return base64.b64decode(payload)


def _find_all_complete_json_object_spans(text: str) -> List[Tuple[int, int]]:
    if not text:
        return []
    in_str = False
    escape = False
    stack = []
    spans = []
    start_idx = None
    for idx, char in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_str = False
            continue
        if char == '"':
            in_str = True
            continue
        if char == "{":
            if not stack:
                start_idx = idx
            stack.append("{")
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    spans.append((start_idx, idx))
                    start_idx = None
    return spans


def extract_protocol_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output")

    # Prefer the final answer region after the model's optional reasoning block.
    # If the model includes JSON-like scratch content in <think>, this keeps us
    # focused on the protocol JSON that should actually drive the robot.
    search_texts = []
    think_end = text.lower().rfind("</think>")
    if think_end >= 0:
        search_texts.append(text[think_end + len("</think>") :].strip())
    search_texts.append(text)

    parsed: List[Dict[str, Any]] = []
    for search_text in search_texts:
        spans = _find_all_complete_json_object_spans(search_text)
        for start, end in spans:
            chunk = search_text[start : end + 1]
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                parsed.append(obj)
        for obj in reversed(parsed):
            head = str(obj.get("Head") or "").strip().lower()
            if head in {"probose", "propose", "proposal"} and ("Task_ID" in obj):
                normalized = dict(obj)
                normalized["Head"] = "probose"
                return normalized
            if head in {"decision", "decide"} and ("Task_ID" in obj):
                normalized = dict(obj)
                normalized["Head"] = "decision"
                return normalized
    if not parsed:
        raise ValueError("No balanced JSON object found in model output")
    raise ValueError("No valid Ask2Act protocol JSON found in model output")


def efe_neg_entropy(py: float, pn: float, eps: float = 1e-12) -> float:
    py = max(min(py, 1.0), 0.0)
    pn = max(min(pn, 1.0), 0.0)
    py = max(py, eps)
    pn = max(pn, eps)
    return py * math.log(py) + pn * math.log(pn)


def calc_py_pn_from_count(count_obj: Dict[str, Any]) -> Tuple[float, float]:
    total = int(count_obj.get("total") or 0)
    y_count = int(count_obj.get("y") or 0)
    n_count = int(count_obj.get("n") or 0)
    if total <= 0:
        return 0.5, 0.5
    if y_count + n_count != total:
        total = max(y_count + n_count, 1)
    return y_count / total, n_count / total


QUESTION_COLOR_WORDS = {
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
QUESTION_SPATIAL_WORDS = {
    "back",
    "backmost",
    "behind",
    "below",
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
    "under",
    "upper",
    "lower",
}
QUESTION_RELATION_WORDS = {
    "above",
    "beside",
    "between",
    "closer",
    "farther",
    "near",
    "next",
    "over",
}
QUESTION_SIZE_WORDS = {"big", "bigger", "biggest", "large", "larger", "largest", "little", "small", "smaller", "smallest"}
QUESTION_SIZE_TRAITS = {
    "big": "large",
    "bigger": "large",
    "biggest": "large",
    "large": "large",
    "larger": "large",
    "largest": "large",
    "little": "small",
    "small": "small",
    "smaller": "small",
    "smallest": "small",
}
QUESTION_SPATIAL_TRAITS = {
    "back": "back",
    "backmost": "back",
    "bottom": "bottom",
    "bottommost": "bottom",
    "front": "front",
    "frontmost": "front",
    "left": "left",
    "leftmost": "left",
    "lower": "bottom",
    "right": "right",
    "rightmost": "right",
    "top": "top",
    "topmost": "top",
    "upper": "top",
}
QUESTION_SPATIAL_OPPOSITES = {
    "back": "front",
    "bottom": "top",
    "front": "back",
    "left": "right",
    "right": "left",
    "top": "bottom",
}
QUESTION_SHAPE_MATERIAL_WORDS = {
    "ceramic",
    "glass",
    "handle",
    "lid",
    "metal",
    "metallic",
    "open",
    "paper",
    "plastic",
    "rubber",
    "steel",
    "transparent",
    "wood",
    "wooden",
}
QUESTION_START_WORDS = {"is", "are", "does", "do", "has", "have", "can", "was", "were"}
QUESTION_FORBIDDEN_TERMS = ("marked", "tag", "display", "candidate", "cand_", "#", "bbox", "bounding box")


class ClarificationEngine:
    def __init__(
        self,
        base_url: str,
        model: str,
        system_prompt_path: str,
        max_side: int = 768,
        jpeg_quality: int = 80,
        gen_max_tokens: int = 2000,
        temperature: float = 0.0,
        repetition_penalty: float = 1.10,
        think_hint: bool = False,
        request_timeout_s: float = 120.0,
    ) -> None:
        self.client = OpenAI(base_url=base_url.rstrip("/"), api_key="EMPTY", timeout=request_timeout_s, max_retries=0)
        self.model = model
        self.max_side = max_side
        self.jpeg_quality = jpeg_quality
        self.gen_max_tokens = gen_max_tokens
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.system_prompt_path = Path(system_prompt_path)
        self.system_prompt = self.system_prompt_path.read_text(encoding="utf-8").strip()
        if think_hint:
            self.system_prompt += (
                "\n\nRESPONSE HINT:\n"
                "- You may use <think> to fully update the plausible candidate set before answering.\n"
                "- Keep reasoning concise, but do not skip candidate elimination or count checks.\n"
                "- The final robot-driving output must be the last JSON object in the response.\n"
            )

    def _chat(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens or self.gen_max_tokens,
            temperature=self.temperature,
            extra_body={"repetition_penalty": self.repetition_penalty},
        )

    @staticmethod
    def _has_forbidden_question_reference(protocol: Dict[str, Any]) -> bool:
        for question in protocol.get("Question") or []:
            text = str(question.get("text") or "").lower()
            if any(term in text for term in QUESTION_FORBIDDEN_TERMS):
                return True
            if any(char.isdigit() for char in text):
                return True
        return False

    @staticmethod
    def _has_candidate_state(protocol: Dict[str, Any]) -> bool:
        state = protocol.get("Candidate_State")
        if not isinstance(state, dict):
            return False
        return isinstance(state.get("plausible"), list) and isinstance(state.get("eliminated"), list)

    @staticmethod
    def _protocol_requests_object_recall(protocol: Dict[str, Any]) -> bool:
        for key in ("Need_Object_Recall", "need_object_recall", "Object_Recall", "Requires_Object_Recall"):
            value = protocol.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "y"}:
                return True
        return False

    @staticmethod
    def _response_content(response: Any) -> str:
        try:
            return response.choices[0].message.content or ""
        except Exception:
            return ""

    @staticmethod
    def _response_finish_reason(response: Any) -> str:
        try:
            return str(response.choices[0].finish_reason or "")
        except Exception:
            return ""

    @staticmethod
    def _is_reasoning_only_output(raw: str) -> bool:
        text = (raw or "").strip().lower()
        if not text:
            return True
        return "<think" in text and "</think>" not in text

    def _repair_token_budget(self, preferred: int) -> int:
        return max(768, min(max(int(self.gen_max_tokens or 0), 768), preferred))

    @classmethod
    def _extract_response_protocol(cls, response: Any) -> Tuple[str, Dict[str, Any]]:
        raw = cls._response_content(response)
        finish_reason = cls._response_finish_reason(response).lower()
        if cls._is_reasoning_only_output(raw):
            raise ValueError(f"Model output ended before final protocol JSON; finish_reason={finish_reason or 'unknown'}")
        try:
            return raw, extract_protocol_json(raw)
        except Exception as exc:
            if finish_reason in {"length", "max_tokens"}:
                raise ValueError("Model output hit the token budget before valid protocol JSON") from exc
            raise

    def _generate_protocol(self, messages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        response = self._chat(messages=messages)
        try:
            raw, protocol = self._extract_response_protocol(response)
        except Exception as first_error:
            follow = (
                "Your previous response was unusable for the robot because it did not end in valid Ask2Act JSON. "
                f"Reason: {type(first_error).__name__}.\n"
                "If it was truncated or only contained <think>, discard that reasoning and produce the final object now.\n"
                "Output ONLY one JSON object now. Do NOT output <think>, markdown, explanations, or code fences.\n"
                'The object Head must be "probose" or "decision", and it must include Task_ID.\n'
                "Use the exact schema from the system prompt. Start with { and end with }.\n"
            )
            second = self._chat(
                messages=messages + [{"role": "user", "content": follow}],
                max_tokens=self._repair_token_budget(1800),
            )
            try:
                raw_second, protocol = self._extract_response_protocol(second)
                raw = raw_second
            except Exception as second_error:
                final_repair = (
                    "Still invalid. Return ONLY the final Ask2Act protocol JSON. "
                    f"Reason: {type(second_error).__name__}.\n"
                    "No <think>. No prose. No markdown. No code fence.\n"
                    "Start with { and end with }."
                )
                third = self._chat(
                    messages=messages + [{"role": "user", "content": final_repair}],
                    max_tokens=self._repair_token_budget(1600),
                )
                raw_third, protocol = self._extract_response_protocol(third)
                raw = raw_third

        if self._has_forbidden_question_reference(protocol):
            repair = (
                "REWRITE the final protocol JSON.\n"
                "Your previous questions mentioned candidate numbers/tags/marks/ids. That is forbidden.\n"
                "Ask only about visible object properties such as color, left/right position, relative position, "
                "size, or shape. Do not mention numbers, marks, tags, display IDs, candidate IDs, or bbox values.\n"
                "Prefer balanced visible-attribute splits when possible.\n"
                "Keep the same protocol schema, Candidate_State, count fields, yes_candidates, and no_candidates. "
                "Output only one JSON object; no <think>."
            )
            repaired = self._chat(
                messages=messages + [{"role": "assistant", "content": json.dumps(protocol, ensure_ascii=False)}]
                + [{"role": "user", "content": repair}],
                max_tokens=self._repair_token_budget(1200),
            )
            raw_repaired, repaired_protocol = self._extract_response_protocol(repaired)
            if self._has_forbidden_question_reference(repaired_protocol):
                raise ValueError("VLM proposed a question about candidate numbers/tags after repair")
            raw = raw_repaired
            protocol = repaired_protocol

        if not self._has_candidate_state(protocol):
            repair = (
                "REWRITE the final protocol JSON to include bookkeeping Candidate_State.\n"
                "Do not change the user-facing question text unless necessary.\n"
                'Add "Candidate_State": {"plausible": [candidate_id, ...], "eliminated": [candidate_id, ...]}.\n'
                "Candidate_State must use candidate_id strings from Candidates, not display numbers. "
                "Output only one JSON object; no <think>, markdown, prose, or code fence."
            )
            repaired = self._chat(
                messages=messages + [{"role": "assistant", "content": json.dumps(protocol, ensure_ascii=False)}]
                + [{"role": "user", "content": repair}],
                max_tokens=self._repair_token_budget(1400),
            )
            try:
                raw_repaired, repaired_protocol = self._extract_response_protocol(repaired)
                if self._has_candidate_state(repaired_protocol):
                    return raw_repaired, repaired_protocol
            except Exception:
                pass

        return raw, protocol

    @staticmethod
    def _candidate_display_id(candidate: Candidate) -> int | str:
        if candidate.display_id is not None:
            return candidate.display_id
        suffix = candidate.candidate_id.rsplit("_", 1)[-1]
        try:
            return int(suffix)
        except ValueError:
            return candidate.candidate_id

    @classmethod
    def _candidate_payload(cls, candidates: List[Candidate]) -> Dict[str, Any]:
        def rounded(values: List[float] | Tuple[float, ...], ndigits: int = 1) -> List[float]:
            return [round(float(value), ndigits) for value in values]

        centers: Dict[str, Tuple[float, float]] = {}
        for candidate in candidates:
            x1, y1, x2, y2 = [float(value) for value in candidate.bbox_xyxy]
            centers[candidate.candidate_id] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        left_to_right = {
            candidate.candidate_id: rank
            for rank, candidate in enumerate(
                sorted(candidates, key=lambda item: (centers[item.candidate_id][0], centers[item.candidate_id][1])),
                start=1,
            )
        }
        top_to_bottom = {
            candidate.candidate_id: rank
            for rank, candidate in enumerate(
                sorted(candidates, key=lambda item: (centers[item.candidate_id][1], centers[item.candidate_id][0])),
                start=1,
            )
        }

        payload: Dict[str, Any] = {}
        for candidate in candidates:
            x1, y1, x2, y2 = [float(value) for value in candidate.bbox_xyxy]
            cx, cy = centers[candidate.candidate_id]
            display_id = cls._candidate_display_id(candidate)
            payload[candidate.candidate_id] = {
                "display_id": display_id,
                "visual_tag": str(display_id),
                "label": candidate.label,
                "bbox": rounded([x1, y1, x2, y2]),
                "bbox_center": rounded([cx, cy]),
                "left_to_right_rank": left_to_right[candidate.candidate_id],
                "top_to_bottom_rank": top_to_bottom[candidate.candidate_id],
                "score": round(float(candidate.score), 3),
            }
        return payload

    @classmethod
    def ensure_candidate_state(cls, session: SessionState) -> None:
        all_ids = [candidate.candidate_id for candidate in session.candidates]
        all_id_set = set(all_ids)
        eliminated = {candidate_id for candidate_id in session.eliminated_candidate_ids if candidate_id in all_id_set}
        if not session.plausible_candidate_ids and not eliminated:
            plausible = set(all_ids)
        else:
            plausible = {
                candidate_id
                for candidate_id in session.plausible_candidate_ids
                if candidate_id in all_id_set and candidate_id not in eliminated
            }
        session.plausible_candidate_ids = [candidate_id for candidate_id in all_ids if candidate_id in plausible]
        session.eliminated_candidate_ids = [candidate_id for candidate_id in all_ids if candidate_id in eliminated]
        session.last_removed_candidate_ids = [
            candidate_id for candidate_id in session.last_removed_candidate_ids if candidate_id in all_id_set
        ]

    @classmethod
    def _candidate_state_payload(cls, session: SessionState) -> Dict[str, Any]:
        cls.ensure_candidate_state(session)
        return {
            "plausible": list(session.plausible_candidate_ids),
            "eliminated": list(session.eliminated_candidate_ids),
            "last_removed": list(session.last_removed_candidate_ids),
        }

    @staticmethod
    def _question_mode_instruction(question_mode: str | None) -> str:
        mode = (question_mode or "proposed_efe").strip()
        if mode == "vlm_best_question":
            return (
                "Put your own best next yes/no question as Question id=1. "
                "Choose it by considering answer consistency, visibility, balanced split, and diversity. "
                "Questions id=2..4 are alternates."
            )
        if mode == "first_question":
            return (
                "Put a simple natural first clarification question as Question id=1. "
                "It should be valid and visible, but it is a first-question baseline, not optimized by backend EFE."
            )
        if mode == "random_question":
            return (
                "Return four distinct useful questions from different visual categories when possible; "
                "the backend will choose one randomly for this baseline."
            )
        if mode == "proposed_efe":
            return (
                "Return four diverse valid questions. The backend will score them by balanced yes/no split "
                "with a small diversity preference."
            )
        return "Follow the Ask2Act protocol and return valid diverse clarification questions."

    @classmethod
    def _base_payload(
        cls,
        task_id: int,
        round_idx: int,
        image_id: str,
        target: str,
        candidates: List[Candidate],
        *,
        question_mode: str | None = None,
        prompt_type: str | None = None,
        candidate_state: Dict[str, Any] | None = None,
        single_candidate_audit: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            "Task_ID": task_id,
            "Round": round_idx,
            "Image": image_id,
            "Target_Instruction": target,
            "Candidates": cls._candidate_payload(candidates),
        }
        if question_mode:
            payload["Question_Mode"] = question_mode
            payload["Question_Mode_Instruction"] = cls._question_mode_instruction(question_mode)
        if prompt_type:
            payload["Prompt_Type"] = prompt_type
        if candidate_state is not None:
            payload["Candidate_State"] = candidate_state
        if single_candidate_audit:
            payload["Single_Candidate_Audit"] = True
            payload["Single_Candidate_Audit_Instruction"] = (
                "This is a partial-description trial with only one detector candidate. "
                "Inspect whether the annotated image suggests other plausible unboxed objects. "
                "Do not accept the single candidate only because it is alone; accept only if it visibly matches the instruction."
            )
        return payload

    @classmethod
    def _start_payload(
        cls,
        task_id: int,
        image_id: str,
        target: str,
        candidates: List[Candidate],
        **kwargs,
    ) -> Dict[str, Any]:
        return {"Head": "start", **cls._base_payload(task_id, 1, image_id, target, candidates, **kwargs)}

    @classmethod
    def _answer_payload(
        cls,
        task_id: int,
        round_idx: int,
        image_id: str,
        target: str,
        candidates: List[Candidate],
        asked_history: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        payload = {"Head": "answer", **cls._base_payload(task_id, round_idx, image_id, target, candidates, **kwargs)}
        payload["Asked"] = asked_history
        return payload

    @staticmethod
    def _score_questions(protocol: Dict[str, Any], *, sort_by_efe: bool = True) -> List[ScoredQuestion]:
        questions = protocol.get("Question", [])
        scored: List[ScoredQuestion] = []
        for question in questions:
            try:
                count = question.get("count") or {}
                normalized_count = {
                    "total": int(count.get("total", 0)),
                    "y": int(count.get("y", 0)),
                    "n": int(count.get("n", 0)),
                }
                py, pn = calc_py_pn_from_count(normalized_count)
                scored.append(
                    ScoredQuestion(
                        id=int(question["id"]),
                        text=str(question["text"]),
                        count=normalized_count,
                        score_py=py,
                        score_pn=pn,
                        efe_score=efe_neg_entropy(py, pn),
                        yes_candidate_ids=[str(item) for item in question.get("yes_candidates") or []],
                        no_candidate_ids=[str(item) for item in question.get("no_candidates") or []],
                    )
                )
            except Exception:
                continue
        if sort_by_efe:
            return sorted(scored, key=lambda item: item.efe_score)
        return scored

    @staticmethod
    def _normalized_question_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower().rstrip("?"))

    @staticmethod
    def _question_words(text: str) -> set[str]:
        return set(re.findall(r"[a-z][a-z0-9_-]*", (text or "").lower()))

    @classmethod
    def question_category(cls, text: str) -> str:
        words = cls._question_words(text)
        if words & QUESTION_SHAPE_MATERIAL_WORDS:
            return "shape_material"
        if words & QUESTION_SPATIAL_WORDS:
            return "spatial"
        if words & QUESTION_RELATION_WORDS:
            return "relation"
        if words & QUESTION_SIZE_WORDS:
            return "size"
        if words & QUESTION_COLOR_WORDS:
            return "color"
        return "other"

    @classmethod
    def _is_legal_question_text(cls, text: str) -> bool:
        normalized = cls._normalized_question_text(text)
        if not normalized:
            return False
        if any(term in normalized for term in QUESTION_FORBIDDEN_TERMS):
            return False
        if any(char.isdigit() for char in normalized):
            return False
        words = cls._question_words(normalized)
        if not words:
            return False
        first = normalized.split(" ", 1)[0].rstrip("?")
        return first in QUESTION_START_WORDS

    @classmethod
    def _candidate_ids(cls, session: SessionState) -> List[str]:
        cls.ensure_candidate_state(session)
        return [candidate.candidate_id for candidate in session.candidates]

    @classmethod
    def _plausible_ids(cls, session: SessionState) -> List[str]:
        cls.ensure_candidate_state(session)
        if session.plausible_candidate_ids:
            return list(session.plausible_candidate_ids)
        return cls._candidate_ids(session)

    @staticmethod
    def _ordered_subset(items: List[str], keep: set[str]) -> List[str]:
        return [item for item in items if item in keep]

    @classmethod
    def _coerce_split_ids(cls, raw_value: Any, allowed_ids: set[str]) -> set[str]:
        if raw_value is None:
            return set()
        if isinstance(raw_value, dict):
            values = raw_value.keys()
        elif isinstance(raw_value, list):
            values = raw_value
        else:
            values = [raw_value]
        return {str(value) for value in values if str(value) in allowed_ids}

    @staticmethod
    def _raw_split_ids(raw_value: Any) -> List[str] | None:
        if raw_value is None:
            return []
        if isinstance(raw_value, dict):
            values = list(raw_value.keys())
        elif isinstance(raw_value, list):
            values = list(raw_value)
        else:
            return None
        return [str(value) for value in values]

    @classmethod
    def _protocol_question_count_matches(
        cls,
        question: Dict[str, Any],
        *,
        plausible_ids: List[str],
        yes_ids: set[str],
        no_ids: set[str],
    ) -> bool:
        count = question.get("count")
        if not isinstance(count, dict):
            return False
        try:
            total = int(count.get("total"))
            y_count = int(count.get("y"))
            n_count = int(count.get("n"))
        except Exception:
            return False
        return total == len(plausible_ids) and y_count == len(yes_ids) and n_count == len(no_ids) and y_count + n_count == total

    @classmethod
    def _exclusive_question_traits(cls, text: str) -> Dict[str, set[str]]:
        words = cls._question_words(text)
        traits: Dict[str, set[str]] = {}
        colors = words & QUESTION_COLOR_WORDS
        if colors:
            traits["color"] = set(colors)
        size_traits = {QUESTION_SIZE_TRAITS[word] for word in words if word in QUESTION_SIZE_TRAITS}
        if size_traits:
            traits["size"] = size_traits
        spatial_traits = {QUESTION_SPATIAL_TRAITS[word] for word in words if word in QUESTION_SPATIAL_TRAITS}
        if spatial_traits:
            traits["spatial"] = spatial_traits
        return traits

    @classmethod
    def _question_contradicts_history(cls, session: SessionState, text: str) -> bool:
        traits = cls._exclusive_question_traits(text)
        if not traits:
            return False
        for turn in session.question_history:
            previous_traits = cls._exclusive_question_traits(turn.text)
            if not previous_traits:
                continue
            for category, current_values in traits.items():
                previous_values = previous_traits.get(category) or set()
                if not previous_values:
                    continue
                if turn.answer == "n" and current_values & previous_values:
                    return True
                if turn.answer != "y":
                    continue
                if category in {"color", "size"} and current_values.isdisjoint(previous_values):
                    return True
                if category == "spatial":
                    for value in current_values:
                        if QUESTION_SPATIAL_OPPOSITES.get(value) in previous_values:
                            return True
        return False

    @classmethod
    def _question_from_parts(
        cls,
        *,
        question_id: int,
        text: str,
        plausible_ids: List[str],
        yes_ids: set[str],
        no_ids: set[str],
    ) -> ScoredQuestion | None:
        plausible_set = set(plausible_ids)
        yes_ids &= plausible_set
        no_ids &= plausible_set
        if not cls._is_legal_question_text(text):
            return None
        if yes_ids & no_ids:
            return None
        if yes_ids | no_ids != plausible_set:
            return None
        total = len(plausible_ids)
        if total <= 0:
            return None
        if total > 1 and (not yes_ids or not no_ids):
            return None
        yes_ordered = cls._ordered_subset(plausible_ids, yes_ids)
        no_ordered = cls._ordered_subset(plausible_ids, no_ids)
        count = {"total": total, "y": len(yes_ordered), "n": len(no_ordered)}
        py, pn = calc_py_pn_from_count(count)
        question_text = text.strip()
        if not question_text.endswith("?"):
            question_text += "?"
        return ScoredQuestion(
            id=question_id,
            text=question_text,
            count=count,
            score_py=py,
            score_pn=pn,
            efe_score=efe_neg_entropy(py, pn),
            yes_candidate_ids=yes_ordered,
            no_candidate_ids=no_ordered,
        )

    @classmethod
    def _question_from_protocol_question(
        cls,
        question: Dict[str, Any],
        session: SessionState,
        *,
        question_id: int,
    ) -> ScoredQuestion | None:
        plausible_ids = cls._plausible_ids(session)
        allowed_ids = set(plausible_ids)
        text = str(question.get("text") or "").strip()
        yes_raw = cls._raw_split_ids(
            question.get("yes_candidates") or question.get("yes_candidate_ids") or question.get("y_candidates")
        )
        no_raw = cls._raw_split_ids(
            question.get("no_candidates") or question.get("no_candidate_ids") or question.get("n_candidates")
        )
        if yes_raw is None or no_raw is None:
            return None
        if len(set(yes_raw)) != len(yes_raw) or len(set(no_raw)) != len(no_raw):
            return None
        if (set(yes_raw) | set(no_raw)) - allowed_ids:
            return None
        yes_ids = set(yes_raw)
        no_ids = set(no_raw)
        if not yes_ids and not no_ids:
            return None
        if not cls._protocol_question_count_matches(
            question,
            plausible_ids=plausible_ids,
            yes_ids=yes_ids,
            no_ids=no_ids,
        ):
            return None
        candidate_question = cls._question_from_parts(
            question_id=question_id,
            text=text,
            plausible_ids=plausible_ids,
            yes_ids=yes_ids,
            no_ids=no_ids,
        )
        if candidate_question is None:
            return None
        if cls._question_contradicts_history(session, candidate_question.text):
            return None
        return candidate_question

    @classmethod
    def _candidate_center(cls, candidate: Candidate) -> Tuple[float, float]:
        x1, y1, x2, y2 = [float(value) for value in candidate.bbox_xyxy]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @classmethod
    def _fallback_object_noun(cls, candidates: Sequence[Candidate]) -> str:
        skip_words = (
            QUESTION_COLOR_WORDS
            | QUESTION_SHAPE_MATERIAL_WORDS
            | QUESTION_SIZE_WORDS
            | QUESTION_SPATIAL_WORDS
            | {"object", "item", "target"}
        )
        counts: Dict[str, int] = {}
        order: List[str] = []
        for candidate in candidates:
            for word in cls._question_words(candidate.label):
                if word in skip_words:
                    continue
                if word not in counts:
                    counts[word] = 0
                    order.append(word)
                counts[word] += 1
        if not order:
            return "object"
        return sorted(order, key=lambda word: (-counts[word], order.index(word)))[0]

    @classmethod
    def _fallback_question_candidates(cls, session: SessionState) -> List[Tuple[str, set[str]]]:
        plausible_ids = cls._plausible_ids(session)
        plausible_set = set(plausible_ids)
        candidates = [candidate for candidate in session.candidates if candidate.candidate_id in plausible_set]
        out: List[Tuple[str, set[str]]] = []
        noun = cls._fallback_object_noun(candidates)

        def add(text: str, yes_ids: set[str]) -> None:
            yes_ids = {candidate_id for candidate_id in yes_ids if candidate_id in plausible_set}
            if not yes_ids or yes_ids == plausible_set:
                return
            normalized = cls._normalized_question_text(text)
            if normalized in {cls._normalized_question_text(existing_text) for existing_text, _ in out}:
                return
            out.append((text, yes_ids))

        for word_set, template in (
            (QUESTION_COLOR_WORDS, "Is it the {word} " + noun + "?"),
            (QUESTION_SHAPE_MATERIAL_WORDS, "Is it the {word} " + noun + "?"),
            (QUESTION_SIZE_WORDS, "Is it the {word} " + noun + "?"),
        ):
            seen_words: List[str] = []
            for candidate in candidates:
                words = cls._question_words(candidate.label)
                for word in sorted(words & word_set):
                    if word not in seen_words:
                        seen_words.append(word)
            for word in seen_words:
                add(template.format(word=word), {candidate.candidate_id for candidate in candidates if word in cls._question_words(candidate.label)})

        if len(candidates) > 1:
            by_x = sorted(candidates, key=lambda item: (cls._candidate_center(item)[0], cls._candidate_center(item)[1]))
            by_y = sorted(candidates, key=lambda item: (cls._candidate_center(item)[1], cls._candidate_center(item)[0]))
            half = max(1, len(candidates) // 2)
            left_ids = {candidate.candidate_id for candidate in by_x[:half]}
            right_ids = {candidate.candidate_id for candidate in by_x[-half:]}
            upper_ids = {candidate.candidate_id for candidate in by_y[:half]}
            lower_ids = {candidate.candidate_id for candidate in by_y[-half:]}
            add("Is it on the left side?", left_ids)
            add("Is it on the right side?", right_ids)
            add("Is it toward the top?", upper_ids)
            add("Is it toward the bottom?", lower_ids)
            add(f"Is it the leftmost {noun}?", {by_x[0].candidate_id})
            add(f"Is it the rightmost {noun}?", {by_x[-1].candidate_id})
            add(f"Is it the topmost {noun}?", {by_y[0].candidate_id})
            add(f"Is it the bottommost {noun}?", {by_y[-1].candidate_id})

        return out

    @classmethod
    def _fallback_questions(
        cls,
        session: SessionState,
        *,
        start_id: int = 1,
        exclude_texts: set[str] | None = None,
    ) -> List[ScoredQuestion]:
        plausible_ids = cls._plausible_ids(session)
        plausible_set = set(plausible_ids)
        excluded = set(exclude_texts or set())
        for turn in session.question_history:
            excluded.add(cls._normalized_question_text(turn.text))

        questions: List[ScoredQuestion] = []
        for text, yes_ids in cls._fallback_question_candidates(session):
            if cls._normalized_question_text(text) in excluded:
                continue
            if cls._question_contradicts_history(session, text):
                continue
            question = cls._question_from_parts(
                question_id=start_id + len(questions),
                text=text,
                plausible_ids=plausible_ids,
                yes_ids=set(yes_ids),
                no_ids=plausible_set - set(yes_ids),
            )
            if question is None:
                continue
            questions.append(question)
            excluded.add(cls._normalized_question_text(question.text))
            if len(questions) >= 4:
                break

        if len(questions) < 4:
            current_texts = {cls._normalized_question_text(question.text) for question in questions}
            for text, yes_ids in cls._fallback_question_candidates(session):
                if cls._normalized_question_text(text) in current_texts:
                    continue
                if cls._question_contradicts_history(session, text):
                    continue
                question = cls._question_from_parts(
                    question_id=start_id + len(questions),
                    text=text,
                    plausible_ids=plausible_ids,
                    yes_ids=set(yes_ids),
                    no_ids=plausible_set - set(yes_ids),
                )
                if question is None:
                    continue
                questions.append(question)
                current_texts.add(cls._normalized_question_text(question.text))
                if len(questions) >= 4:
                    break

        if len(questions) < 4:
            current_texts = {cls._normalized_question_text(question.text) for question in questions}
            for text, yes_ids in cls._fallback_question_candidates(session):
                if cls._normalized_question_text(text) in current_texts:
                    continue
                question = cls._question_from_parts(
                    question_id=start_id + len(questions),
                    text=text,
                    plausible_ids=plausible_ids,
                    yes_ids=set(yes_ids),
                    no_ids=plausible_set - set(yes_ids),
                )
                if question is None:
                    continue
                questions.append(question)
                current_texts.add(cls._normalized_question_text(question.text))
                if len(questions) >= 4:
                    break

        return questions

    @classmethod
    def _fallback_protocol(cls, session: SessionState, *, task_id: int = 1, reason: str = "backend_fallback") -> Dict[str, Any]:
        cls.ensure_candidate_state(session)
        plausible_ids = cls._plausible_ids(session)
        if len(plausible_ids) == 1:
            return {
                "Head": "decision",
                "Task_ID": task_id,
                "Round": session.current_round,
                "Grasp": "yes",
                "Target": {"name": plausible_ids[0]},
                "Candidate_State": cls._candidate_state_payload(session),
                "Reason": f"{reason}: one plausible candidate remains.",
            }
        round_idx = session.current_round + (1 if session.question_history else 0)
        questions = cls._fallback_questions(session, start_id=1)
        head = "decision" if session.question_history else "probose"
        protocol: Dict[str, Any] = {
            "Head": head,
            "Task_ID": task_id,
            "Round": round_idx,
            "Candidate_State": cls._candidate_state_payload(session),
            "Question": [
                {
                    "id": question.id,
                    "text": question.text,
                    "count": dict(question.count),
                    "yes_candidates": list(question.yes_candidate_ids),
                    "no_candidates": list(question.no_candidate_ids),
                }
                for question in questions
            ],
            "Reason": f"{reason}: generated backend-safe clarification questions.",
        }
        if head == "decision":
            protocol["Grasp"] = "no"
        return protocol

    @classmethod
    def _rank_question_key(cls, question: ScoredQuestion, session: SessionState | None) -> Tuple[float, float, int]:
        total = int(question.count.get("total") or 0)
        y_count = int(question.count.get("y") or 0)
        n_count = int(question.count.get("n") or 0)
        if total <= 0:
            balance = 1.0
        else:
            balance = abs(y_count - n_count) / max(total, 1)

        known_questions = set()
        previous_categories: List[str] = []
        if session is not None:
            for turn in session.question_history:
                known_questions.add(cls._normalized_question_text(turn.text))
                previous_categories.append(cls.question_category(turn.text))

        category = cls.question_category(question.text)
        category_prior = {
            "spatial": 0.00,
            "relation": 0.02,
            "shape_material": 0.03,
            "size": 0.04,
            "color": 0.08,
            "other": 0.10,
        }.get(category, 0.10)
        repeat_category_penalty = 0.12 * previous_categories.count(category)
        duplicate_penalty = 1.0 if cls._normalized_question_text(question.text) in known_questions else 0.0
        no_split_penalty = 1.0 if total > 1 and (y_count <= 0 or n_count <= 0) else 0.0
        isolate_penalty = 0.08 if total > 2 and min(y_count, n_count) == 1 else 0.0
        score = balance + category_prior + repeat_category_penalty + duplicate_penalty + no_split_penalty + isolate_penalty
        return score, question.efe_score, question.id

    @classmethod
    def rank_questions_for_mode(
        cls,
        protocol: Dict[str, Any],
        session: SessionState | None,
        *,
        mode: str | None,
    ) -> List[ScoredQuestion]:
        if session is None:
            questions = cls._score_questions(protocol, sort_by_efe=False)
        else:
            used_texts = {cls._normalized_question_text(turn.text) for turn in session.question_history}
            questions = []
            for raw_question in protocol.get("Question") or []:
                if not isinstance(raw_question, dict):
                    continue
                question = cls._question_from_protocol_question(
                    raw_question,
                    session,
                    question_id=len(questions) + 1,
                )
                if question is None:
                    continue
                normalized = cls._normalized_question_text(question.text)
                if normalized in used_texts:
                    continue
                questions.append(question)
                used_texts.add(normalized)
            if len(questions) < 4:
                questions.extend(
                    cls._fallback_questions(
                        session,
                        start_id=len(questions) + 1,
                        exclude_texts={cls._normalized_question_text(question.text) for question in questions},
                    )
                )
            questions = questions[:4]
        if mode == "proposed_efe":
            return sorted(questions, key=lambda question: cls._rank_question_key(question, session))
        return questions

    @classmethod
    def _protocol_state_ids(cls, protocol_state: Dict[str, Any], *keys: str, allowed_ids: set[str]) -> set[str]:
        values: List[Any] = []
        for key in keys:
            raw_value = protocol_state.get(key)
            if raw_value is None:
                continue
            if isinstance(raw_value, dict):
                values.extend(raw_value.keys())
            elif isinstance(raw_value, list):
                values.extend(raw_value)
            else:
                values.append(raw_value)
        return {str(value) for value in values if str(value) in allowed_ids}

    @classmethod
    def sync_candidate_state_from_protocol(
        cls,
        session: SessionState,
        protocol: Dict[str, Any],
        *,
        source: str,
    ) -> None:
        cls.ensure_candidate_state(session)
        all_ids = [candidate.candidate_id for candidate in session.candidates]
        allowed_ids = set(all_ids)
        previous_plausible = set(session.plausible_candidate_ids)
        previous_eliminated = set(session.eliminated_candidate_ids)
        next_plausible = set(previous_plausible)
        next_eliminated = set(previous_eliminated)
        protocol_state = protocol.get("Candidate_State")
        has_protocol_state = isinstance(protocol_state, dict)

        if has_protocol_state:
            protocol_plausible = cls._protocol_state_ids(
                protocol_state,
                "plausible",
                "remaining",
                "still_plausible",
                "still_possible",
                allowed_ids=allowed_ids,
            )
            protocol_eliminated = cls._protocol_state_ids(
                protocol_state,
                "eliminated",
                "removed",
                "last_removed",
                "ruled_out",
                allowed_ids=allowed_ids,
            )
            if protocol_plausible:
                next_plausible = protocol_plausible & previous_plausible
                next_eliminated |= previous_plausible - protocol_plausible
            next_eliminated |= protocol_eliminated
            next_plausible -= next_eliminated

        head = str(protocol.get("Head") or "").lower()
        if head == "decision" and protocol.get("Grasp") == "yes":
            target = (protocol.get("Target") or {}).get("name")
            if target in allowed_ids and target in previous_plausible and target not in next_eliminated:
                next_plausible = {str(target)}
                next_eliminated = (allowed_ids - next_plausible) | previous_eliminated

        rejected_empty_state = False
        if not next_plausible and previous_plausible:
            rejected_empty_state = True
            next_plausible = set(previous_plausible)
            next_eliminated = set(previous_eliminated)

        last_removed = previous_plausible - next_plausible
        session.plausible_candidate_ids = [candidate_id for candidate_id in all_ids if candidate_id in next_plausible]
        session.eliminated_candidate_ids = [candidate_id for candidate_id in all_ids if candidate_id in next_eliminated]
        session.last_removed_candidate_ids = [candidate_id for candidate_id in all_ids if candidate_id in last_removed]
        session.candidate_state_history.append(
            {
                "source": source,
                "round": protocol.get("Round"),
                "head": protocol.get("Head"),
                "has_protocol_state": has_protocol_state,
                "rejected_empty_state": rejected_empty_state,
                "plausible": list(session.plausible_candidate_ids),
                "eliminated": list(session.eliminated_candidate_ids),
                "last_removed": list(session.last_removed_candidate_ids),
            }
        )

    @staticmethod
    def _resolve_target(protocol: Dict[str, Any], session: SessionState) -> ResolvedTarget:
        target = protocol.get("Target") or {}
        target_name = target.get("name")
        ClarificationEngine.ensure_candidate_state(session)
        if session.plausible_candidate_ids and target_name not in session.plausible_candidate_ids:
            raise ValueError(
                f"Resolved target '{target_name}' is not plausible after the recorded clarification answers"
            )
        for candidate in session.candidates:
            if candidate.candidate_id == target_name:
                return ResolvedTarget(
                    candidate_id=candidate.candidate_id,
                    display_id=candidate.display_id,
                    label=candidate.label,
                    score=candidate.score,
                    bbox_xyxy=candidate.bbox_xyxy,
                    mask_rle=candidate.mask_rle,
                )
        raise ValueError(f"Resolved target '{target_name}' not found in candidate set")

    @classmethod
    def apply_answer_split(cls, session: SessionState, question: ScoredQuestion, answer: str) -> bool:
        cls.ensure_candidate_state(session)
        plausible_ids = cls._plausible_ids(session)
        plausible_set = set(plausible_ids)
        yes_ids = set(question.yes_candidate_ids) & plausible_set
        no_ids = set(question.no_candidate_ids) & plausible_set
        if not yes_ids and not no_ids:
            return False
        if yes_ids | no_ids != plausible_set or yes_ids & no_ids:
            return False
        keep = yes_ids if answer == "y" else no_ids
        if not keep:
            return False
        previous_plausible = set(session.plausible_candidate_ids or plausible_ids)
        next_plausible = previous_plausible & keep
        next_eliminated = (set(session.eliminated_candidate_ids) | (previous_plausible - next_plausible)) & set(
            cls._candidate_ids(session)
        )
        all_ids = cls._candidate_ids(session)
        session.plausible_candidate_ids = [candidate_id for candidate_id in all_ids if candidate_id in next_plausible]
        session.eliminated_candidate_ids = [candidate_id for candidate_id in all_ids if candidate_id in next_eliminated]
        session.last_removed_candidate_ids = [
            candidate_id for candidate_id in all_ids if candidate_id in (previous_plausible - next_plausible)
        ]
        session.candidate_state_history.append(
            {
                "source": "backend_question_split",
                "round": session.current_round,
                "question_id": question.id,
                "answer": answer,
                "has_protocol_state": True,
                "plausible": list(session.plausible_candidate_ids),
                "eliminated": list(session.eliminated_candidate_ids),
                "last_removed": list(session.last_removed_candidate_ids),
            }
        )
        return True

    def initialize_session(self, session: SessionState, task_id: int = 1) -> None:
        session.vlm_messages = []
        self.ensure_candidate_state(session)
        try:
            annotated_image_bytes = data_url_to_bytes(session.candidate_overlay_data_url)
        except Exception:
            annotated_image_bytes = session.observation_image_bytes
        image_data_url = compress_to_data_url(
            annotated_image_bytes,
            max_side=self.max_side,
            jpeg_quality=self.jpeg_quality,
        )
        payload = self._start_payload(
            task_id,
            session.observation_id or session.session_id,
            session.instruction,
            session.candidates,
            question_mode=session.question_mode,
            prompt_type=session.prompt_type,
            candidate_state=self._candidate_state_payload(session),
            single_candidate_audit=session.single_candidate_audit,
        )
        session.vlm_messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                ],
            },
        ]
        self.advance_without_answer(session)

    def direct_select(self, session: SessionState, task_id: int = 1) -> None:
        session.vlm_messages = []
        self.ensure_candidate_state(session)
        try:
            annotated_image_bytes = data_url_to_bytes(session.candidate_overlay_data_url)
        except Exception:
            annotated_image_bytes = session.observation_image_bytes
        image_data_url = compress_to_data_url(
            annotated_image_bytes,
            max_side=self.max_side,
            jpeg_quality=self.jpeg_quality,
        )
        payload = {
            "Head": "start",
            **self._base_payload(
                task_id,
                1,
                session.observation_id or session.session_id,
                session.instruction,
                session.candidates,
                question_mode=session.question_mode or "vlm_direct",
                prompt_type=session.prompt_type,
                candidate_state=self._candidate_state_payload(session),
                single_candidate_audit=session.single_candidate_audit,
            ),
            "Offline_Baseline": "vlm_direct_target_selection",
            "Instruction": (
                "Select the intended target candidate directly from the annotated image and candidate metadata. "
                "Do not ask clarification questions. Output a decision JSON with Grasp=yes and Target.name."
            ),
        }
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                ],
            },
        ]
        try:
            raw = ""
            try:
                raw, protocol = self._extract_response_protocol(self._chat(messages=messages))
            except Exception:
                protocol = {}
            try:
                valid_direct = str(protocol.get("Head") or "").lower() == "decision" and protocol.get("Grasp") == "yes"
            except Exception:
                valid_direct = False
            if not valid_direct:
                repair = (
                    "For this offline non-interactive baseline, questions are forbidden. "
                    "Return ONLY one Ask2Act decision JSON object now. "
                    'It must have Head=\"decision\", Grasp=\"yes\", and Target.name equal to one candidate_id. '
                    "No <think>, markdown, prose, or code fence."
                )
                raw, protocol = self._extract_response_protocol(
                    self._chat(
                        messages=messages + [{"role": "user", "content": repair}],
                        max_tokens=self._repair_token_budget(1000),
                    )
                )
            if str(protocol.get("Head") or "").lower() != "decision" or protocol.get("Grasp") != "yes":
                raise ValueError("VLM direct baseline did not return a decision")
            self.sync_candidate_state_from_protocol(session, protocol, source="vlm_direct")
            resolved = self._resolve_target(protocol, session)
        except Exception as exc:
            plausible_ids = self._plausible_ids(session)
            candidates = [candidate for candidate in session.candidates if candidate.candidate_id in set(plausible_ids)]
            if not candidates:
                raise
            candidate = max(candidates, key=lambda item: item.score)
            session.plausible_candidate_ids = [candidate.candidate_id]
            session.eliminated_candidate_ids = [
                item.candidate_id for item in session.candidates if item.candidate_id != candidate.candidate_id
            ]
            session.last_removed_candidate_ids = list(session.eliminated_candidate_ids)
            protocol = {
                "Head": "decision",
                "Task_ID": task_id,
                "Round": 1,
                "Grasp": "yes",
                "Target": {"name": candidate.candidate_id},
                "Candidate_State": self._candidate_state_payload(session),
                "Reason": f"Backend legal-output fallback after VLM direct failure: {type(exc).__name__}.",
            }
            raw = json.dumps(protocol, ensure_ascii=False)
            resolved = self._resolve_target(protocol, session)

        session.vlm_messages = messages + [{"role": "assistant", "content": json.dumps(protocol, ensure_ascii=False)}]
        session.last_protocol_json = protocol
        session.resolved_target = resolved
        session.current_question = None
        session.current_questions = []
        session.status = "resolved"

    def advance_without_answer(self, session: SessionState) -> None:
        try:
            raw, protocol = self._generate_protocol(session.vlm_messages)
        except Exception as exc:
            protocol = self._fallback_protocol(session, reason=f"vlm_invalid_output:{type(exc).__name__}")
            raw = json.dumps(protocol, ensure_ascii=False)
        session.last_protocol_json = protocol
        session.vlm_messages.append({"role": "assistant", "content": json.dumps(protocol, ensure_ascii=False)})
        self.sync_candidate_state_from_protocol(session, protocol, source="vlm")
        if self._protocol_requests_object_recall(protocol):
            session.current_question = None
            session.current_questions = []
            session.status = "needs_object_recall"
            return
        head = protocol.get("Head")
        if head == "decision" and protocol.get("Grasp") == "yes":
            try:
                session.resolved_target = self._resolve_target(protocol, session)
                session.current_question = None
                session.current_questions = []
                session.status = "resolved"
                return
            except Exception:
                protocol = self._fallback_protocol(session, reason="invalid_vlm_decision")
                session.last_protocol_json = protocol
                session.vlm_messages[-1] = {"role": "assistant", "content": json.dumps(protocol, ensure_ascii=False)}
                self.sync_candidate_state_from_protocol(session, protocol, source="backend_fallback")
                if protocol.get("Head") == "decision" and protocol.get("Grasp") == "yes":
                    session.resolved_target = self._resolve_target(protocol, session)
                    session.current_question = None
                    session.current_questions = []
                    session.status = "resolved"
                    return

        session.current_questions = self.rank_questions_for_mode(protocol, session, mode=session.question_mode or "proposed_efe")
        if len(session.current_questions) != 4:
            fallback_protocol = self._fallback_protocol(session, reason="not_enough_legal_questions")
            session.last_protocol_json = fallback_protocol
            session.vlm_messages[-1] = {"role": "assistant", "content": json.dumps(fallback_protocol, ensure_ascii=False)}
            protocol = fallback_protocol
            self.sync_candidate_state_from_protocol(session, protocol, source="backend_fallback")
            if protocol.get("Head") == "decision" and protocol.get("Grasp") == "yes":
                session.resolved_target = self._resolve_target(protocol, session)
                session.current_question = None
                session.current_questions = []
                session.status = "resolved"
                return
            session.current_questions = self.rank_questions_for_mode(
                protocol,
                session,
                mode=session.question_mode or "proposed_efe",
            )
        if len(session.current_questions) != 4:
            raise ValueError(f"Backend could not generate 4 legal clarification questions, got {len(session.current_questions)}")
        session.current_question = session.current_questions[0]
        session.current_round = int(protocol.get("Round") or session.current_round)
        session.status = "awaiting_answer"

    def answer_current_question(self, session: SessionState, answer: str, task_id: int = 1) -> None:
        if session.current_question is None:
            raise ValueError("No active clarification question to answer")
        self.ensure_candidate_state(session)
        answered_question = session.current_question
        round_key = f"Round {session.current_round}"
        session.asked_history.setdefault(round_key, [])
        session.asked_history[round_key].append(
            {
                "id": answered_question.id,
                "text": answered_question.text,
                "answer": answer,
            }
        )
        session.question_history.append(
            QuestionTurn(
                round_index=session.current_round,
                question_id=answered_question.id,
                text=answered_question.text,
                answer=answer,
                score_py=answered_question.score_py,
                score_pn=answered_question.score_pn,
                efe_score=answered_question.efe_score,
            )
        )
        self.apply_answer_split(session, answered_question, answer)
        payload = self._answer_payload(
            task_id,
            session.current_round,
            session.observation_id or session.session_id,
            session.instruction,
            session.candidates,
            session.asked_history,
            question_mode=session.question_mode,
            prompt_type=session.prompt_type,
            candidate_state=self._candidate_state_payload(session),
            single_candidate_audit=session.single_candidate_audit,
        )
        session.vlm_messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})
        if len(session.plausible_candidate_ids) == 1:
            target_id = session.plausible_candidate_ids[0]
            protocol = {
                "Head": "decision",
                "Task_ID": task_id,
                "Round": session.current_round,
                "Grasp": "yes",
                "Target": {"name": target_id},
                "Candidate_State": self._candidate_state_payload(session),
                "Reason": "Backend question split left exactly one plausible candidate.",
            }
            session.last_protocol_json = protocol
            session.vlm_messages.append({"role": "assistant", "content": json.dumps(protocol, ensure_ascii=False)})
            session.resolved_target = self._resolve_target(protocol, session)
            session.current_question = None
            session.current_questions = []
            session.status = "resolved"
            return
        self.advance_without_answer(session)
