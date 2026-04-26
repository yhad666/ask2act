from __future__ import annotations

import base64
import io
import json
import math
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
    spans = _find_all_complete_json_object_spans(text)
    if not spans:
        raise ValueError("No balanced JSON object found in model output")
    parsed: List[Dict[str, Any]] = []
    for start, end in spans:
        chunk = text[start : end + 1]
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            parsed.append(obj)
    for obj in reversed(parsed):
        if obj.get("Head") in {"probose", "decision"} and ("Task_ID" in obj):
            return obj
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
    ) -> None:
        self.client = OpenAI(base_url=base_url.rstrip("/"), api_key="EMPTY")
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
                "\n\nSOFT THINKING HINT:\n"
                "- You may use <think>, but keep it concise while staying correct.\n"
                "- Focus on the minimum reasoning needed to produce the final JSON.\n"
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
        forbidden_terms = ("marked", "tag", "display", "candidate", "cand_", "#")
        for question in protocol.get("Question") or []:
            text = str(question.get("text") or "").lower()
            if any(term in text for term in forbidden_terms):
                return True
            if any(char.isdigit() for char in text):
                return True
        return False

    def _generate_protocol(self, messages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        response = self._chat(messages=messages)
        raw = response.choices[0].message.content or ""
        try:
            protocol = extract_protocol_json(raw)
        except Exception:
            follow = (
                "CONTINUE.\n"
                "You MAY use <think>, but you MUST end with EXACTLY ONE FINAL protocol JSON.\n"
                "Make <think> as short as possible while staying correct.\n"
                "No extra text after the JSON.\n"
            )
            second = self._chat(
                messages=messages + [{"role": "user", "content": follow}],
                max_tokens=min(self.gen_max_tokens, 1500),
            )
            raw_second = second.choices[0].message.content or ""
            protocol = extract_protocol_json(raw_second)
            raw = raw_second

        if self._has_forbidden_question_reference(protocol):
            repair = (
                "REWRITE the final protocol JSON.\n"
                "Your previous questions mentioned candidate numbers/tags/marks/ids. That is forbidden.\n"
                "Ask only about visible object properties such as color, left/right position, relative position, "
                "size, or shape. Do not mention numbers, marks, tags, display IDs, candidate IDs, or bbox values.\n"
                "For each visible-trait question, prefer the split whose count.y/count.n is closest to half/half "
                "over questions that isolate a single object.\n"
                "Keep the same protocol schema and count fields. Output exactly one final JSON object."
            )
            repaired = self._chat(
                messages=messages + [{"role": "assistant", "content": json.dumps(protocol, ensure_ascii=False)}]
                + [{"role": "user", "content": repair}],
                max_tokens=min(self.gen_max_tokens, 1200),
            )
            raw_repaired = repaired.choices[0].message.content or ""
            repaired_protocol = extract_protocol_json(raw_repaired)
            if self._has_forbidden_question_reference(repaired_protocol):
                raise ValueError("VLM proposed a question about candidate numbers/tags after repair")
            return raw_repaired, repaired_protocol

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
    def _base_payload(cls, task_id: int, round_idx: int, image_id: str, target: str, candidates: List[Candidate]) -> Dict[str, Any]:
        return {
            "Task_ID": task_id,
            "Round": round_idx,
            "Image": image_id,
            "Target_Instruction": target,
            "Candidates": cls._candidate_payload(candidates),
        }

    @classmethod
    def _start_payload(cls, task_id: int, image_id: str, target: str, candidates: List[Candidate]) -> Dict[str, Any]:
        return {"Head": "start", **cls._base_payload(task_id, 1, image_id, target, candidates)}

    @classmethod
    def _answer_payload(
        cls,
        task_id: int,
        round_idx: int,
        image_id: str,
        target: str,
        candidates: List[Candidate],
        asked_history: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {"Head": "answer", **cls._base_payload(task_id, round_idx, image_id, target, candidates)}
        payload["Asked"] = asked_history
        return payload

    @staticmethod
    def _score_questions(protocol: Dict[str, Any]) -> List[ScoredQuestion]:
        questions = protocol.get("Question", [])
        scored: List[ScoredQuestion] = []
        for question in questions:
            count = question.get("count") or {}
            py, pn = calc_py_pn_from_count(count)
            scored.append(
                ScoredQuestion(
                    id=int(question["id"]),
                    text=str(question["text"]),
                    count={
                        "total": int(count.get("total", 0)),
                        "y": int(count.get("y", 0)),
                        "n": int(count.get("n", 0)),
                    },
                    score_py=py,
                    score_pn=pn,
                    efe_score=efe_neg_entropy(py, pn),
                )
            )
        return sorted(scored, key=lambda item: item.efe_score)

    @staticmethod
    def _resolve_target(protocol: Dict[str, Any], session: SessionState) -> ResolvedTarget:
        target = protocol.get("Target") or {}
        target_name = target.get("name")
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

    def initialize_session(self, session: SessionState, task_id: int = 1) -> None:
        try:
            annotated_image_bytes = data_url_to_bytes(session.candidate_overlay_data_url)
        except Exception:
            annotated_image_bytes = session.observation_image_bytes
        image_data_url = compress_to_data_url(
            annotated_image_bytes,
            max_side=self.max_side,
            jpeg_quality=self.jpeg_quality,
        )
        payload = self._start_payload(task_id, session.observation_id or session.session_id, session.instruction, session.candidates)
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

    def advance_without_answer(self, session: SessionState) -> None:
        raw, protocol = self._generate_protocol(session.vlm_messages)
        session.last_protocol_json = protocol
        session.vlm_messages.append({"role": "assistant", "content": json.dumps(protocol, ensure_ascii=False)})
        head = protocol.get("Head")
        if head == "decision" and protocol.get("Grasp") == "yes":
            session.resolved_target = self._resolve_target(protocol, session)
            session.current_question = None
            session.current_questions = []
            session.status = "resolved"
            return

        session.current_questions = self._score_questions(protocol)
        if len(session.current_questions) != 4:
            raise ValueError(f"Expected 4 clarification questions, got {len(session.current_questions)}")
        session.current_question = session.current_questions[0]
        session.current_round = int(protocol.get("Round") or session.current_round)
        session.status = "awaiting_answer"

    def answer_current_question(self, session: SessionState, answer: str, task_id: int = 1) -> None:
        if session.current_question is None:
            raise ValueError("No active clarification question to answer")
        round_key = f"Round {session.current_round}"
        session.asked_history.setdefault(round_key, [])
        session.asked_history[round_key].append(
            {
                "id": session.current_question.id,
                "text": session.current_question.text,
                "answer": answer,
            }
        )
        session.question_history.append(
            QuestionTurn(
                round_index=session.current_round,
                question_id=session.current_question.id,
                text=session.current_question.text,
                answer=answer,
                score_py=session.current_question.score_py,
                score_pn=session.current_question.score_pn,
                efe_score=session.current_question.efe_score,
            )
        )
        payload = self._answer_payload(
            task_id,
            session.current_round,
            session.observation_id or session.session_id,
            session.instruction,
            session.candidates,
            session.asked_history,
        )
        session.vlm_messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})
        self.advance_without_answer(session)
