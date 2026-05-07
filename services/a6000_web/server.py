from __future__ import annotations

import base64
import io
import json
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .clarification import ClarificationEngine
from .detection import GroundingDinoDetector
from .grasp_runtime import LocalGraspRuntime
from .offline_experiments import (
    OfflineExperimentStore,
    compact_session_view,
    data_url_to_bytes as offline_data_url_to_bytes,
)
from .phrase_extractor import InstructionPhraseExtractor
from .schemas import (
    ConfirmSessionRequest,
    ExecuteSessionRequest,
    GraspPlanResult,
    OfflineExperimentRequest,
    OfflineSceneRequest,
    OfflineTrialFinishRequest,
    OfflineTrialStartRequest,
    OfflineTrialStepRequest,
    ResolvedTarget,
    SessionState,
    StartSessionRequest,
    StepSessionRequest,
)
from .stretch_transport import StretchTransportClient


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

VLLM_BASE_URL = os.getenv("ASK2ACT_VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLLM_MODEL = os.getenv("ASK2ACT_VLLM_MODEL", "mimo-vl")
SYSTEM_PROMPT_PATH = os.getenv(
    "ASK2ACT_SYSTEM_PROMPT_PATH",
    str(ROOT / "prompts" / "system_prompt.txt"),
)
STRETCH_TRANSPORT_MODE = os.getenv("ASK2ACT_STRETCH_TRANSPORT", "mock")
STRETCH_ZMQ_ENDPOINT = os.getenv("ASK2ACT_STRETCH_ZMQ_ENDPOINT", "tcp://127.0.0.1:5557")
STRETCH_TIMEOUT_MS = int(os.getenv("ASK2ACT_STRETCH_TIMEOUT_MS", "30000"))
STRETCH_OBSERVE_TIMEOUT_MS = int(os.getenv("ASK2ACT_STRETCH_OBSERVE_TIMEOUT_MS", str(STRETCH_TIMEOUT_MS)))
STRETCH_EXECUTE_TIMEOUT_MS = int(os.getenv("ASK2ACT_STRETCH_EXECUTE_TIMEOUT_MS", str(STRETCH_TIMEOUT_MS)))
STRETCH_MOCK_IMAGE_PATH = os.getenv("ASK2ACT_STRETCH_MOCK_IMAGE_PATH", "")
PIPELINE_MODE = os.getenv("ASK2ACT_PIPELINE_MODE", "mock")
PIPELINE_SCENE_CONFIG = os.getenv("ASK2ACT_SCENE_CONFIG_PATH", "")
PIPELINE_GRASP_CONFIG = os.getenv("ASK2ACT_GRASP_CONFIG_PATH", "")
PIPELINE_RUN_ROOT = os.getenv("ASK2ACT_PIPELINE_RUN_ROOT", "")
PIPELINE_HEADLESS = os.getenv("ASK2ACT_PIPELINE_HEADLESS", "1") != "0"
PIPELINE_SHOW_VIEWER = os.getenv("ASK2ACT_PIPELINE_SHOW_VIEWER", "0") == "1"
AUTO_EXECUTE_ON_RESOLVE = os.getenv("ASK2ACT_AUTO_EXECUTE_ON_RESOLVE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
REAL_REPLAN_AFTER_BASE_REACH = os.getenv("ASK2ACT_REAL_REPLAN_AFTER_BASE_REACH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
REAL_BASE_REACH_REPLAN_MAX_ATTEMPTS = int(os.getenv("ASK2ACT_REAL_BASE_REACH_REPLAN_MAX_ATTEMPTS", "2"))
REAL_REOBSERVE_TARGET_LOCK_AMBIGUITY_MARGIN = float(
    os.getenv("ASK2ACT_REOBSERVE_TARGET_LOCK_AMBIGUITY_MARGIN", "0.02")
)
SESSION_RECORD_ROOT = Path(os.getenv("ASK2ACT_SESSION_RECORD_ROOT", str(ROOT / "artifacts" / "session_records"))).expanduser()
OFFLINE_EXPERIMENT_ROOT = Path(
    os.getenv("ASK2ACT_OFFLINE_EXPERIMENT_ROOT", str(ROOT / "artifacts" / "offline_experiments"))
).expanduser()
OFFLINE_MAX_ROUNDS = int(os.getenv("ASK2ACT_OFFLINE_MAX_ROUNDS", "6"))
GEN_MAX_TOKENS_REQUESTED = int(os.getenv("ASK2ACT_GEN_MAX_TOKENS", "4096"))
GEN_MAX_TOKENS_CAP = int(os.getenv("ASK2ACT_GEN_MAX_TOKENS_CAP", "4096"))
GEN_MAX_TOKENS = max(256, min(GEN_MAX_TOKENS_REQUESTED, GEN_MAX_TOKENS_CAP))

SESSIONS: Dict[str, SessionState] = {}
OFFLINE_TRIAL_SESSIONS: Dict[str, str] = {}

phrase_extractor = InstructionPhraseExtractor()
detector = GroundingDinoDetector(phrase_extractor=phrase_extractor)
clarifier = ClarificationEngine(
    base_url=VLLM_BASE_URL,
    model=VLLM_MODEL,
    system_prompt_path=SYSTEM_PROMPT_PATH,
    gen_max_tokens=GEN_MAX_TOKENS,
    think_hint=os.getenv("ASK2ACT_THINK_HINT", "0") == "1",
)
stretch_transport = StretchTransportClient(
    mode=STRETCH_TRANSPORT_MODE,
    zmq_endpoint=STRETCH_ZMQ_ENDPOINT,
    timeout_ms=STRETCH_TIMEOUT_MS,
    observe_timeout_ms=STRETCH_OBSERVE_TIMEOUT_MS,
    execute_timeout_ms=STRETCH_EXECUTE_TIMEOUT_MS,
    mock_image_path=STRETCH_MOCK_IMAGE_PATH,
)
grasp_runtime = LocalGraspRuntime(
    mode=PIPELINE_MODE,
    scene_config_path=PIPELINE_SCENE_CONFIG,
    grasp_config_path=PIPELINE_GRASP_CONFIG,
    run_root=PIPELINE_RUN_ROOT,
    headless=PIPELINE_HEADLESS,
    show_viewer_ui=PIPELINE_SHOW_VIEWER,
)
offline_store = OfflineExperimentStore(OFFLINE_EXPERIMENT_ROOT)

app = FastAPI(title="Ask2Act A6000 Service")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def decode_data_url(data_url: str) -> bytes:
    if not data_url or "," not in data_url:
        raise ValueError("Invalid image data URL")
    _, payload = data_url.split(",", 1)
    return base64.b64decode(payload)


def build_session_view(session: SessionState) -> Dict[str, Any]:
    return {
        "session_id": session.session_id,
        "status": session.status,
        "instruction": session.instruction,
        "instruction_phrases": session.instruction_phrases,
        "detection_prompt": session.detection_prompt,
        "observation_id": session.observation_id,
        "observation_source": session.observation_source,
        "observation_metadata": _public_observation_metadata(session.observation_raw_response),
        "observation_image_data_url": session.observation_image_data_url,
        "candidate_overlay_data_url": session.candidate_overlay_data_url,
        "final_image_data_url": session.final_image_data_url,
        "candidates": [candidate.model_dump() for candidate in session.candidates],
        "current_round": session.current_round,
        "current_question": session.current_question.model_dump() if session.current_question else None,
        "current_questions": [question.model_dump() for question in session.current_questions],
        "question_history": [turn.model_dump() for turn in session.question_history],
        "resolved_target": session.resolved_target.model_dump() if session.resolved_target else None,
        "grasp_plan_result": session.grasp_plan_result.model_dump() if session.grasp_plan_result else None,
        "execution_result": session.execution_result,
        "confirmation_result": session.confirmation_result,
        "error_message": session.error_message,
        "stretch_transport_mode": stretch_transport.mode,
        "pipeline_mode": grasp_runtime.mode,
    }


def _public_observation_metadata(raw_response: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not raw_response:
        return None
    hidden_keys = {"image_base64", "image_data_url", "depth_npy_base64"}
    out: Dict[str, Any] = {}
    for key, value in raw_response.items():
        if key in hidden_keys:
            continue
        if key == "detail" and isinstance(value, dict):
            out[key] = {inner_key: inner_value for inner_key, inner_value in value.items() if inner_key not in hidden_keys}
        else:
            out[key] = value
    return out


def finalize_resolved_target(session: SessionState, success: bool | None = None, banner_text: str | None = None) -> None:
    if session.resolved_target is None:
        return
    session.final_image_data_url = detector.render_overlay(
        prepared_image_bytes=session.observation_image_bytes,
        candidates=session.candidates,
        selected_candidate_id=session.resolved_target.candidate_id,
        success=success,
        banner_text=banner_text,
    )


def _write_session_record(session: SessionState) -> Path:
    SESSION_RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    record_path = SESSION_RECORD_ROOT / f"session_{stamp}_{session.session_id}.json"
    if session.confirmation_result is not None:
        session.confirmation_result["record_path"] = str(record_path)
    record = build_session_view(session)
    record["recorded_at_epoch_s"] = time.time()
    record_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    index_path = SESSION_RECORD_ROOT / "session_records.jsonl"
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "recorded_at_epoch_s": record["recorded_at_epoch_s"],
                    "session_id": session.session_id,
                    "status": session.status,
                    "success": session.confirmation_result.get("success") if session.confirmation_result else None,
                    "record_path": str(record_path),
                },
                default=str,
            )
            + "\n"
        )
    return record_path


def resolve_single_candidate(session: SessionState) -> None:
    if len(session.candidates) != 1:
        return
    candidate = session.candidates[0]
    session.resolved_target = ResolvedTarget(
        candidate_id=candidate.candidate_id,
        display_id=candidate.display_id,
        label=candidate.label,
        score=candidate.score,
        bbox_xyxy=candidate.bbox_xyxy,
        mask_rle=candidate.mask_rle,
    )
    session.current_question = None
    session.current_questions = []
    session.status = "resolved"
    session.last_protocol_json = {
        "Head": "decision",
        "Grasp": "yes",
        "Target": {"name": candidate.candidate_id},
        "Reason": "Only one GroundingDINO candidate was returned; clarification was skipped.",
    }


def maybe_auto_execute(session: SessionState) -> None:
    if not AUTO_EXECUTE_ON_RESOLVE:
        return
    if session.resolved_target is None:
        return
    if session.execution_result is not None:
        return
    execute_resolved_session(session, dry_run=False, raise_on_error=False)


def _find_base_reach_waypoint(plan_result: Dict[str, Any]) -> Dict[str, Any] | None:
    dispatch_payload = plan_result.get("dispatch_payload") or {}
    trajectory = dispatch_payload.get("trajectory") or []
    if not isinstance(trajectory, list):
        return None
    for waypoint in trajectory:
        if not isinstance(waypoint, dict):
            continue
        if str(waypoint.get("name") or "") == "base_translate_for_reach":
            return waypoint
    return None


def _make_base_reach_preposition_payload(plan_result: Dict[str, Any], waypoint: Dict[str, Any]) -> Dict[str, Any]:
    dispatch_payload = dict(plan_result.get("dispatch_payload") or {})
    dispatch_payload["planner_backend"] = "real_base_reach_preposition"
    dispatch_payload["trajectory"] = [waypoint]
    dispatch_payload["preposition_only"] = True
    dispatch_payload["note"] = "Execute base reach correction only; A6000 will reobserve and replan before grasping."
    return dispatch_payload


def _bbox_center_xy(bbox: Any) -> tuple[float, float] | None:
    try:
        values = [float(value) for value in bbox]
    except Exception:
        return None
    if len(values) != 4:
        return None
    return (0.5 * (values[0] + values[2]), 0.5 * (values[1] + values[3]))


def _bbox_wh(bbox: Any) -> tuple[float, float] | None:
    try:
        values = [float(value) for value in bbox]
    except Exception:
        return None
    if len(values) != 4:
        return None
    return (abs(values[2] - values[0]), abs(values[3] - values[1]))


def _label_matches(candidate_label: str | None, target_label: str | None) -> bool:
    target = (target_label or "").strip().lower()
    label = (candidate_label or "").strip().lower()
    return bool(target and (label == target or target in label or label in target))


def _candidate_visual_signature(image_bytes: bytes | None, bbox_xyxy: Any) -> Dict[str, Any] | None:
    if not image_bytes:
        return None
    try:
        import numpy as np
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        values = [float(value) for value in bbox_xyxy]
        if len(values) != 4:
            return None
        x0, y0, x1, y1 = values
        x0, x1 = sorted((max(0.0, min(float(width), x0)), max(0.0, min(float(width), x1))))
        y0, y1 = sorted((max(0.0, min(float(height), y0)), max(0.0, min(float(height), y1))))
        if x1 - x0 < 4.0 or y1 - y0 < 4.0:
            return None
        pad_x = 0.12 * (x1 - x0)
        pad_y = 0.12 * (y1 - y0)
        crop_box = (
            int(round(x0 + pad_x)),
            int(round(y0 + pad_y)),
            int(round(x1 - pad_x)),
            int(round(y1 - pad_y)),
        )
        crop = image.crop(crop_box).resize((24, 24))
        arr = np.asarray(crop, dtype=np.float32) / 255.0
        mean_rgb = arr.reshape(-1, 3).mean(axis=0)
        hist_parts = []
        for channel in range(3):
            hist, _ = np.histogram(arr[:, :, channel], bins=8, range=(0.0, 1.0), density=False)
            hist = hist.astype(np.float32)
            hist_parts.append(hist / max(float(hist.sum()), 1.0))
        return {
            "mean_rgb": mean_rgb.tolist(),
            "hist_rgb": np.concatenate(hist_parts).tolist(),
            "image_size": [width, height],
        }
    except Exception:
        return None


def _visual_signature_distance(a: Dict[str, Any] | None, b: Dict[str, Any] | None) -> float | None:
    if not a or not b:
        return None
    try:
        import numpy as np

        mean_a = np.asarray(a["mean_rgb"], dtype=np.float32)
        mean_b = np.asarray(b["mean_rgb"], dtype=np.float32)
        hist_a = np.asarray(a["hist_rgb"], dtype=np.float32)
        hist_b = np.asarray(b["hist_rgb"], dtype=np.float32)
        mean_dist = float(np.linalg.norm(mean_a - mean_b) / max(np.sqrt(3.0), 1e-6))
        hist_dist = float(0.5 * np.sum(np.abs(hist_a - hist_b)) / 3.0)
        return 0.65 * mean_dist + 0.35 * hist_dist
    except Exception:
        return None


def _candidate_layout_signature(target: Any, candidates: Any, image_size: Any = None) -> Dict[str, Any] | None:
    target_center = _bbox_center_xy(getattr(target, "bbox_xyxy", None))
    if target_center is None:
        return None
    tx, ty = target_center
    target_label = getattr(target, "label", None)
    peers = [
        candidate
        for candidate in candidates or []
        if _label_matches(getattr(candidate, "label", None), target_label)
        and _bbox_center_xy(getattr(candidate, "bbox_xyxy", None)) is not None
    ]
    if not peers:
        return None

    centers = [(candidate, _bbox_center_xy(candidate.bbox_xyxy)) for candidate in peers]
    centers = [(candidate, center) for candidate, center in centers if center is not None]
    if not centers:
        return None

    try:
        width = float(image_size[0]) if image_size and len(image_size) == 2 else 0.0
        height = float(image_size[1]) if image_size and len(image_size) == 2 else 0.0
    except Exception:
        width = 0.0
        height = 0.0
    max_x = max([abs(center[0]) for _, center in centers] + [abs(tx), width, 1.0])
    max_y = max([abs(center[1]) for _, center in centers] + [abs(ty), height, 1.0])
    norm_x = max(width, max_x, 1.0)
    norm_y = max(height, max_y, 1.0)
    norm = max(norm_x, norm_y, 1.0)

    n = len(centers)
    denom = max(n - 1, 1)
    sorted_x = sorted(centers, key=lambda item: item[1][0])
    sorted_y = sorted(centers, key=lambda item: item[1][1])
    target_id = getattr(target, "candidate_id", None)

    def rank_in(sorted_items, axis: int) -> float:
        best_index = 0
        best_distance = float("inf")
        for index, (candidate, center) in enumerate(sorted_items):
            if target_id is not None and getattr(candidate, "candidate_id", None) == target_id:
                return index / denom
            distance = abs(center[axis] - (tx if axis == 0 else ty))
            if distance < best_distance:
                best_distance = distance
                best_index = index
        return best_index / denom

    left: list[float] = []
    right: list[float] = []
    above: list[float] = []
    below: list[float] = []
    vectors: list[tuple[float, float, float]] = []
    for candidate, center in centers:
        if target_id is not None and getattr(candidate, "candidate_id", None) == target_id:
            continue
        cx, cy = center
        dx = (cx - tx) / norm_x
        dy = (cy - ty) / norm_y
        if cx < tx - 1.0:
            left.append((tx - cx) / norm)
        if cx > tx + 1.0:
            right.append((cx - tx) / norm)
        if cy < ty - 1.0:
            above.append((ty - cy) / norm)
        if cy > ty + 1.0:
            below.append((cy - ty) / norm)
        vectors.append((dx, dy, float((dx * dx + dy * dy) ** 0.5)))
    vectors.sort(key=lambda item: item[2])
    nearest_vectors = [[dx, dy] for dx, dy, _ in vectors[:4]]
    while len(nearest_vectors) < 4:
        nearest_vectors.append([1.0, 1.0])

    wh = _bbox_wh(getattr(target, "bbox_xyxy", None)) or (0.0, 0.0)
    return {
        "count": n,
        "x_rank": rank_in(sorted_x, 0),
        "y_rank": rank_in(sorted_y, 1),
        "left_count": len(left) / denom,
        "right_count": len(right) / denom,
        "above_count": len(above) / denom,
        "below_count": len(below) / denom,
        "nearest_left": min(left) if left else 1.0,
        "nearest_right": min(right) if right else 1.0,
        "nearest_above": min(above) if above else 1.0,
        "nearest_below": min(below) if below else 1.0,
        "bbox_wh": [wh[0] / norm_x, wh[1] / norm_y],
        "nearest_vectors": nearest_vectors,
    }


def _layout_signature_distance(a: Dict[str, Any] | None, b: Dict[str, Any] | None) -> float | None:
    if not a or not b:
        return None
    try:
        scalar_keys = [
            "x_rank",
            "y_rank",
            "left_count",
            "right_count",
            "above_count",
            "below_count",
            "nearest_left",
            "nearest_right",
            "nearest_above",
            "nearest_below",
        ]
        scalar_distance = sum(abs(float(a[key]) - float(b[key])) for key in scalar_keys) / len(scalar_keys)
        size_a = a.get("bbox_wh") or [0.0, 0.0]
        size_b = b.get("bbox_wh") or [0.0, 0.0]
        size_distance = 0.5 * (abs(float(size_a[0]) - float(size_b[0])) + abs(float(size_a[1]) - float(size_b[1])))
        vectors_a = a.get("nearest_vectors") or []
        vectors_b = b.get("nearest_vectors") or []
        vector_distance = 0.0
        vector_count = max(min(len(vectors_a), len(vectors_b)), 1)
        for vec_a, vec_b in zip(vectors_a[:vector_count], vectors_b[:vector_count]):
            dx = float(vec_a[0]) - float(vec_b[0])
            dy = float(vec_a[1]) - float(vec_b[1])
            vector_distance += float((dx * dx + dy * dy) ** 0.5)
        vector_distance /= vector_count
        count_distance = min(abs(float(a.get("count", 0)) - float(b.get("count", 0))) / max(float(a.get("count", 1)), 1.0), 1.0)
        return 0.50 * scalar_distance + 0.35 * vector_distance + 0.10 * size_distance + 0.05 * count_distance
    except Exception:
        return None


def _select_reobserved_target(
    original: ResolvedTarget,
    candidates,
    *,
    previous_candidates=None,
    previous_image_bytes: bytes | None = None,
    reobserved_image_bytes: bytes | None = None,
) -> ResolvedTarget:
    if not candidates:
        raise RuntimeError("Reobserve after base reach returned no GroundingDINO candidates")

    matching = [candidate for candidate in candidates if _label_matches(candidate.label, original.label)]
    pool = matching or list(candidates)
    original_signature = _candidate_visual_signature(previous_image_bytes, original.bbox_xyxy)
    original_image_size = original_signature.get("image_size") if original_signature else None
    original_layout = _candidate_layout_signature(original, previous_candidates or [], original_image_size)

    original_center = _bbox_center_xy(original.bbox_xyxy)
    if original_center is not None:
        ox, oy = original_center

        def target_lock_rank(candidate) -> Dict[str, Any]:
            center = _bbox_center_xy(candidate.bbox_xyxy)
            candidate_signature = _candidate_visual_signature(reobserved_image_bytes, candidate.bbox_xyxy)
            appearance_distance = _visual_signature_distance(original_signature, candidate_signature)
            if center is None:
                continuity_distance = float("inf")
            else:
                cx, cy = center
                # After arm-axis base preposition, the target can move vertically
                # in the camera view, while left/right ordering is usually stable.
                # Prefer the same visual slot over the highest-scoring same-label cup.
                continuity_distance = abs(cx - ox) + 0.35 * abs(cy - oy)
            image_size = candidate_signature.get("image_size") if candidate_signature else None
            if isinstance(image_size, list) and len(image_size) == 2:
                norm = max(float(image_size[0]), float(image_size[1]), 1.0)
            elif center is not None:
                cx, cy = center
                norm = max(abs(cx), abs(cy), abs(ox), abs(oy), 1.0)
            else:
                norm = 1.0
            continuity_score = min(float(continuity_distance / norm), 1.0)
            candidate_layout = _candidate_layout_signature(candidate, candidates, image_size)
            layout_distance = _layout_signature_distance(original_layout, candidate_layout)

            components: list[tuple[float, float]] = []
            if appearance_distance is not None:
                components.append((0.35, float(appearance_distance)))
            if layout_distance is not None:
                components.append((0.45, float(layout_distance)))
            components.append((0.20 if len(components) >= 2 else 0.35, continuity_score))
            total_weight = sum(weight for weight, _ in components)
            combined_distance = sum(weight * value for weight, value in components) / max(total_weight, 1e-6)
            return {
                "candidate": candidate,
                "combined_distance": combined_distance,
                "appearance_distance": appearance_distance,
                "layout_distance": layout_distance,
                "continuity_score": continuity_score,
                "score": -float(candidate.score),
            }

        ranked = sorted(
            [target_lock_rank(candidate) for candidate in pool],
            key=lambda item: (item["combined_distance"], item["score"]),
        )
        if len(ranked) > 1:
            best = float(ranked[0]["combined_distance"])
            second = float(ranked[1]["combined_distance"])
            if second - best < REAL_REOBSERVE_TARGET_LOCK_AMBIGUITY_MARGIN:
                best_candidate = ranked[0]["candidate"]
                second_candidate = ranked[1]["candidate"]
                raise RuntimeError(
                    "Target lock after base reach is ambiguous between "
                    f"{getattr(best_candidate, 'candidate_id', '?')} and {getattr(second_candidate, 'candidate_id', '?')} "
                    f"(distance gap {second - best:.4f}); refusing to grasp the wrong same-label object."
                )
        selected = ranked[0]["candidate"]
    else:
        selected = max(pool, key=lambda candidate: float(candidate.score))
    return ResolvedTarget(
        candidate_id=selected.candidate_id,
        display_id=selected.display_id,
        label=selected.label,
        score=selected.score,
        bbox_xyxy=selected.bbox_xyxy,
        mask_rle=selected.mask_rle,
    )


def _refresh_session_observation_after_base_reach(session: SessionState, *, attempt_index: int) -> Dict[str, Any]:
    observation = stretch_transport.fetch_observation(
        session_id=session.session_id,
        instruction=f"{session.instruction} (reobserve after base reach correction {attempt_index})",
    )
    detection = detector.detect(image_bytes=observation.image_bytes, instruction=session.instruction)
    if session.resolved_target is None:
        raise RuntimeError("Cannot reselect target after base reach because session has no resolved target")
    previous_image_bytes = session.observation_image_bytes
    reselected_target = _select_reobserved_target(
        session.resolved_target,
        detection.candidates,
        previous_candidates=session.candidates,
        previous_image_bytes=previous_image_bytes,
        reobserved_image_bytes=detection.prepared_image_bytes,
    )

    session.observation_id = observation.observation_id or session.session_id
    session.observation_source = f"stretch_{stretch_transport.mode}_reobserve_after_base_reach"
    session.observation_image_bytes = detection.prepared_image_bytes
    session.observation_image_data_url = detection.image_data_url
    session.candidate_overlay_data_url = detection.overlay_data_url
    session.candidates = detection.candidates
    session.observation_raw_response = observation.raw_response
    session.resolved_target = reselected_target
    finalize_resolved_target(session)
    return {
        "attempt_index": attempt_index,
        "observation_id": session.observation_id,
        "candidate_count": len(detection.candidates),
        "selected_candidate": reselected_target.model_dump(),
    }


def execute_resolved_session(session: SessionState, *, dry_run: bool, raise_on_error: bool = True) -> None:
    if session.resolved_target is None:
        raise HTTPException(status_code=409, detail="session has no resolved target")

    try:
        preposition_attempts: list[Dict[str, Any]] = []
        max_attempts = max(0, REAL_BASE_REACH_REPLAN_MAX_ATTEMPTS)
        for attempt_index in range(max_attempts + 1):
            plan_result = grasp_runtime.plan_for_target(
                session.resolved_target,
                observation_metadata=session.observation_raw_response,
                dry_run=dry_run,
                rotate_clockwise_90=detector.rotate_clockwise_90,
            )
            base_reach_waypoint = _find_base_reach_waypoint(plan_result)
            should_preposition = (
                REAL_REPLAN_AFTER_BASE_REACH
                and not dry_run
                and grasp_runtime.mode in {"real_pointcloud", "real_depth"}
                and base_reach_waypoint is not None
            )
            if not should_preposition:
                break
            if attempt_index >= max_attempts:
                raise RuntimeError(
                    "Base reach correction is still required after reobserve/replan attempts; "
                    "refusing to execute stale grasp coordinates."
                )

            preposition_payload = _make_base_reach_preposition_payload(plan_result, base_reach_waypoint)
            preposition_transport = stretch_transport.dispatch_grasp(
                session_id=session.session_id,
                instruction=session.instruction,
                observation_id=session.observation_id,
                resolved_target={
                    "candidate_id": session.resolved_target.candidate_id,
                    "bbox_xyxy": session.resolved_target.bbox_xyxy,
                    "mask_rle": session.resolved_target.mask_rle,
                },
                grasp_plan=preposition_payload,
                dry_run=False,
            )
            if not bool(preposition_transport.get("ok", True)):
                raise RuntimeError(f"Base reach preposition failed: {preposition_transport}")
            reobserve = _refresh_session_observation_after_base_reach(session, attempt_index=attempt_index + 1)
            preposition_attempts.append(
                {
                    "attempt_index": attempt_index + 1,
                    "base_reach_waypoint": base_reach_waypoint,
                    "transport_result": preposition_transport,
                    "reobserve": reobserve,
                }
            )
        else:
            raise RuntimeError("Internal error while planning after base reach preposition")

        # Normalize through the Pydantic model for a stable schema.
        session.grasp_plan_result = GraspPlanResult.model_validate(plan_result["plan_summary"])

        transport_result = stretch_transport.dispatch_grasp(
            session_id=session.session_id,
            instruction=session.instruction,
            observation_id=session.observation_id,
            resolved_target={
                "candidate_id": session.resolved_target.candidate_id,
                "bbox_xyxy": session.resolved_target.bbox_xyxy,
                "mask_rle": session.resolved_target.mask_rle,
            },
            grasp_plan=plan_result["dispatch_payload"],
            dry_run=dry_run,
        )

        execution_debug_path = None
        run_dir_raw = plan_result.get("run_dir") or (plan_result.get("plan_summary") or {}).get("pipeline_run_dir")
        if run_dir_raw:
            try:
                run_dir = Path(str(run_dir_raw)).expanduser()
                run_dir.mkdir(parents=True, exist_ok=True)
                execution_debug_path = run_dir / "stretch_execute_result.json"
                execution_debug_path.write_text(
                    json.dumps(
                        {
                            "session_id": session.session_id,
                            "dry_run": dry_run,
                            "preposition_attempts": preposition_attempts,
                            "transport_result": transport_result,
                        },
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
            except Exception as debug_exc:
                execution_debug_path = f"failed_to_write: {debug_exc}"

        success = bool(plan_result.get("ok", True)) and bool(transport_result.get("ok", True))
        session.execution_result = {
            "ok": success,
            "dry_run": dry_run,
            "auto_execute": AUTO_EXECUTE_ON_RESOLVE and not dry_run,
            "base_reach_preposition_attempts": preposition_attempts,
            "execution_debug_path": None if execution_debug_path is None else str(execution_debug_path),
            "plan_result": plan_result,
            "transport_result": transport_result,
        }
        session.status = "executed" if success else "execution_failed"
        if dry_run:
            banner = "DRY RUN"
        elif success:
            banner = "EXECUTION SENT"
        else:
            banner = "EXECUTION FAILED"
        finalize_resolved_target(session, success=success, banner_text=banner)
    except Exception as exc:
        session.status = "execution_failed"
        session.error_message = str(exc)
        session.execution_result = {"ok": False, "dry_run": dry_run, "auto_execute": AUTO_EXECUTE_ON_RESOLVE and not dry_run, "error": str(exc)}
        finalize_resolved_target(session, success=False, banner_text="EXECUTION FAILED")
        if raise_on_error:
            raise HTTPException(status_code=502, detail=str(exc)) from exc


def create_session(
    request: StartSessionRequest,
    *,
    initialize_clarification: bool = True,
    auto_execute: bool = True,
) -> SessionState:
    instruction = (request.instruction or "").strip()
    if not instruction:
        raise HTTPException(status_code=400, detail="instruction is required")

    session_id = str(uuid.uuid4())
    try:
        if request.observation_image_data_url:
            image_bytes = decode_data_url(request.observation_image_data_url)
            observation_id = request.observation_id or session_id
            observation_source = "browser_upload"
            observation_raw_response = None
        elif request.fetch_observation:
            observation = stretch_transport.fetch_observation(session_id=session_id, instruction=instruction)
            image_bytes = observation.image_bytes
            observation_id = observation.observation_id or session_id
            observation_source = f"stretch_{stretch_transport.mode}"
            observation_raw_response = observation.raw_response
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide observation_image_data_url or enable fetch_observation",
            )

        detection = detector.detect(image_bytes=image_bytes, instruction=instruction)
        session = SessionState(
            session_id=session_id,
            instruction=instruction,
            instruction_phrases=detection.phrases,
            detection_prompt=detection.detection_prompt,
            observation_id=observation_id,
            observation_source=observation_source,
            observation_image_bytes=detection.prepared_image_bytes,
            observation_image_data_url=detection.image_data_url,
            candidate_overlay_data_url=detection.overlay_data_url,
            candidates=detection.candidates,
            vlm_messages=[],
            observation_raw_response=observation_raw_response,
        )
        if not session.candidates:
            session.status = "failed_no_candidates"
            session.error_message = "GroundingDINO returned no candidates for the current observation."
            session.final_image_data_url = session.observation_image_data_url
            return session

        if len(session.candidates) == 1:
            resolve_single_candidate(session)
        elif initialize_clarification:
            clarifier.initialize_session(session)
        else:
            session.status = "detected"
        if session.resolved_target is not None:
            finalize_resolved_target(session)
            if auto_execute:
                maybe_auto_execute(session)
        return session
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _candidate_to_resolved(candidate) -> ResolvedTarget:
    return ResolvedTarget(
        candidate_id=candidate.candidate_id,
        display_id=candidate.display_id,
        label=candidate.label,
        score=candidate.score,
        bbox_xyxy=candidate.bbox_xyxy,
        mask_rle=candidate.mask_rle,
    )


def _expected_candidate_id_from_display(session: SessionState, display_id: int | None) -> str | None:
    if display_id is None:
        return None
    for candidate in session.candidates:
        if candidate.display_id == display_id:
            return candidate.candidate_id
    return None


def _apply_offline_question_method(session: SessionState, method: str, *, trial_id: str) -> None:
    if session.resolved_target is not None or session.status != "awaiting_answer":
        return
    protocol = session.last_protocol_json or {}
    if method == "proposed_efe":
        session.current_questions = clarifier._score_questions(protocol, sort_by_efe=True)
        if session.current_questions:
            session.current_question = session.current_questions[0]
        return

    questions = clarifier._score_questions(protocol, sort_by_efe=False)
    if not questions:
        return
    session.current_questions = questions
    if method in {"first_question", "vlm_best_question"}:
        session.current_question = questions[0]
    elif method == "random_question":
        seed = f"{trial_id}:{session.current_round}:{len(session.question_history)}"
        session.current_question = random.Random(seed).choice(questions)


def _apply_offline_trial_method(session: SessionState, method: str, *, trial_id: str) -> None:
    if session.status == "failed_no_candidates":
        return
    if session.resolved_target is not None:
        return
    if method == "top_score":
        if not session.candidates:
            return
        session.resolved_target = _candidate_to_resolved(max(session.candidates, key=lambda item: item.score))
        session.status = "resolved"
        session.current_question = None
        session.current_questions = []
        session.last_protocol_json = {
            "Head": "decision",
            "Task_ID": 1,
            "Grasp": "yes",
            "Target": {"name": session.resolved_target.candidate_id},
            "Reason": "Offline top-score baseline selected the highest GroundingDINO score.",
        }
    elif method == "random_candidate":
        if not session.candidates:
            return
        candidate = random.Random(trial_id).choice(session.candidates)
        session.resolved_target = _candidate_to_resolved(candidate)
        session.status = "resolved"
        session.current_question = None
        session.current_questions = []
        session.last_protocol_json = {
            "Head": "decision",
            "Task_ID": 1,
            "Grasp": "yes",
            "Target": {"name": candidate.candidate_id},
            "Reason": "Offline random-candidate baseline selected uniformly from the CP-gated candidates.",
        }
    elif method == "vlm_direct":
        clarifier.direct_select(session)
    else:
        _apply_offline_question_method(session, method, trial_id=trial_id)

    if session.resolved_target is not None:
        finalize_resolved_target(session)


def _offline_session_status(session: SessionState) -> str:
    if session.status == "awaiting_answer":
        return "awaiting_answer"
    if session.resolved_target is not None:
        return "resolved"
    if session.status == "failed_no_candidates":
        return "failed_no_candidates"
    if session.status == "offline_max_rounds":
        return "max_rounds"
    return session.status


def _update_offline_trial_from_session(
    experiment_id: str,
    trial: Dict[str, Any],
    session: SessionState,
    *,
    status: str | None = None,
) -> Dict[str, Any]:
    view = build_session_view(session)
    expected_candidate_id = trial.get("expected_candidate_id") or _expected_candidate_id_from_display(
        session,
        trial.get("expected_display_id"),
    )
    trial.update(
        {
            "session_id": session.session_id,
            "status": status or _offline_session_status(session),
            "candidate_count": len(session.candidates),
            "question_count": len(session.question_history),
            "resolved_candidate_id": session.resolved_target.candidate_id if session.resolved_target else None,
            "expected_candidate_id": expected_candidate_id,
            "session_snapshot": compact_session_view(view),
            "latency_s": time.time() - float(trial.get("started_at_epoch_s") or time.time()),
        }
    )
    if expected_candidate_id and session.resolved_target is not None:
        trial["auto_outcome"] = "correct" if session.resolved_target.candidate_id == expected_candidate_id else "wrong"
    offline_store.write_trial(experiment_id, trial)
    return trial


def _offline_trial_view(experiment_id: str, trial_id: str) -> Dict[str, Any]:
    trial = offline_store.read_trial(experiment_id, trial_id)
    session = None
    session_id = trial.get("session_id") or OFFLINE_TRIAL_SESSIONS.get(trial_id)
    if session_id:
        active = SESSIONS.get(str(session_id))
        if active is not None:
            session = build_session_view(active)
    return {
        "experiment": offline_store.read_experiment(experiment_id),
        "trial": trial,
        "session": session,
        "metrics": offline_store.metrics(experiment_id),
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/offline")
def offline_index():
    return FileResponse(STATIC_DIR / "offline.html")


@app.get("/health")
def health():
    return {
        "ok": True,
        "sessions": len(SESSIONS),
        "vllm_base_url": VLLM_BASE_URL,
        "vllm_model": VLLM_MODEL,
        "gen_max_tokens": GEN_MAX_TOKENS,
        "stretch_transport_mode": stretch_transport.mode,
        "stretch_zmq_endpoint": STRETCH_ZMQ_ENDPOINT if stretch_transport.mode == "zmq" else None,
        "pipeline_mode": grasp_runtime.mode,
        "auto_execute_on_resolve": AUTO_EXECUTE_ON_RESOLVE,
        "stretch_observe_timeout_ms": STRETCH_OBSERVE_TIMEOUT_MS,
        "stretch_execute_timeout_ms": STRETCH_EXECUTE_TIMEOUT_MS,
    }


@app.get("/api/offline/experiments")
def list_offline_experiments():
    return {"experiments": offline_store.list_experiments(), "root": str(OFFLINE_EXPERIMENT_ROOT)}


@app.post("/api/offline/experiments")
def create_offline_experiment(request: OfflineExperimentRequest):
    try:
        experiment = offline_store.create_experiment(
            experiment_id=request.experiment_id,
            name=request.name,
            experiment_type=request.experiment_type,
            notes=request.notes or "",
        )
        return {
            "experiment": experiment,
            "scenes": offline_store.list_scenes(experiment["experiment_id"]),
            "trials": offline_store.list_trials(experiment["experiment_id"]),
            "metrics": offline_store.metrics(experiment["experiment_id"]),
            "root": str(OFFLINE_EXPERIMENT_ROOT),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/offline/experiments/{experiment_id}")
def get_offline_experiment(experiment_id: str):
    try:
        return {
            "experiment": offline_store.read_experiment(experiment_id),
            "scenes": offline_store.list_scenes(experiment_id),
            "trials": offline_store.list_trials(experiment_id),
            "metrics": offline_store.metrics(experiment_id),
            "root": str(OFFLINE_EXPERIMENT_ROOT),
        }
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/offline/experiments/{experiment_id}/scenes")
def save_offline_scene(experiment_id: str, request: OfflineSceneRequest):
    try:
        instruction = f"offline scene capture {request.scene_id}".strip()
        if request.observation_image_data_url:
            image_bytes, mime_type = offline_data_url_to_bytes(request.observation_image_data_url)
            observation_id = request.observation_id or f"{experiment_id}:{request.scene_id}"
            observation_source = "browser_upload"
            observation_metadata = None
        elif request.fetch_observation:
            observation = stretch_transport.fetch_observation(session_id=str(uuid.uuid4()), instruction=instruction)
            image_bytes = observation.image_bytes
            mime_type = observation.mime_type or "image/png"
            observation_id = observation.observation_id or f"{experiment_id}:{request.scene_id}"
            observation_source = f"stretch_{stretch_transport.mode}"
            observation_metadata = _public_observation_metadata(observation.raw_response)
        else:
            raise HTTPException(status_code=400, detail="Provide observation_image_data_url or enable fetch_observation")
        scene = offline_store.save_scene(
            experiment_id=experiment_id,
            scene_id=request.scene_id,
            scene_type=request.scene_type,
            object_categories=request.object_categories,
            notes=request.notes or "",
            image_bytes=image_bytes,
            mime_type=mime_type,
            observation_id=observation_id,
            observation_source=observation_source,
            observation_metadata=observation_metadata,
        )
        return {
            "scene": scene,
            "scenes": offline_store.list_scenes(experiment_id),
            "metrics": offline_store.metrics(experiment_id),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/offline/experiments/{experiment_id}/scenes/{scene_id}")
def get_offline_scene(experiment_id: str, scene_id: str):
    try:
        return {"scene": offline_store.scene_view(experiment_id, scene_id)}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/offline/experiments/{experiment_id}/trials/start")
def start_offline_trial(experiment_id: str, request: OfflineTrialStartRequest):
    try:
        scene = offline_store.scene_view(experiment_id, request.scene_id)
        method = request.method
        trial_id = f"trial_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        initialize_clarification = method in {"proposed_efe", "first_question", "random_question", "vlm_best_question"}
        session = create_session(
            StartSessionRequest(
                instruction=request.prompt,
                fetch_observation=False,
                observation_image_data_url=scene["observation_image_data_url"],
                observation_id=scene.get("observation_id") or f"{experiment_id}:{request.scene_id}",
            ),
            initialize_clarification=initialize_clarification,
            auto_execute=False,
        )
        _apply_offline_trial_method(session, method, trial_id=trial_id)
        if session.status == "awaiting_answer" and len(session.question_history) >= OFFLINE_MAX_ROUNDS:
            session.status = "offline_max_rounds"
            session.current_question = None
        SESSIONS[session.session_id] = session
        OFFLINE_TRIAL_SESSIONS[trial_id] = session.session_id
        trial = {
            "trial_id": trial_id,
            "experiment_id": experiment_id,
            "experiment_type": offline_store.read_experiment(experiment_id).get("experiment_type"),
            "scene_id": request.scene_id,
            "scene_type": scene.get("scene_type"),
            "object_categories": scene.get("object_categories") or [],
            "prompt": request.prompt,
            "prompt_type": request.prompt_type,
            "method": method,
            "expected_candidate_id": request.expected_candidate_id,
            "expected_display_id": request.expected_display_id,
            "notes": request.notes or "",
            "started_at_epoch_s": time.time(),
            "scene_observation_path": scene.get("observation_path"),
        }
        _update_offline_trial_from_session(experiment_id, trial, session)
        offline_store.append_event(experiment_id, {"event": "trial_started", "trial_id": trial_id, "method": method})
        return _offline_trial_view(experiment_id, trial_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/offline/experiments/{experiment_id}/trials/{trial_id}")
def get_offline_trial(experiment_id: str, trial_id: str):
    try:
        return _offline_trial_view(experiment_id, trial_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/offline/experiments/{experiment_id}/trials/{trial_id}/step")
def step_offline_trial(experiment_id: str, trial_id: str, request: OfflineTrialStepRequest):
    try:
        trial = offline_store.read_trial(experiment_id, trial_id)
        session_id = trial.get("session_id") or OFFLINE_TRIAL_SESSIONS.get(trial_id)
        session = SESSIONS.get(str(session_id))
        if session is None:
            raise HTTPException(status_code=409, detail="active session is not in memory; start a new trial or mark this one aborted")
        if session.status != "awaiting_answer" or session.current_question is None:
            raise HTTPException(status_code=409, detail="trial is not waiting for a clarification answer")
        clarifier.answer_current_question(session, request.answer)
        if session.resolved_target is not None:
            finalize_resolved_target(session)
        elif len(session.question_history) >= OFFLINE_MAX_ROUNDS:
            session.status = "offline_max_rounds"
            session.current_question = None
            session.current_questions = []
        else:
            _apply_offline_question_method(session, str(trial.get("method") or "proposed_efe"), trial_id=trial_id)
        _update_offline_trial_from_session(experiment_id, trial, session)
        offline_store.append_event(experiment_id, {"event": "trial_answered", "trial_id": trial_id, "answer": request.answer})
        return _offline_trial_view(experiment_id, trial_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/offline/experiments/{experiment_id}/trials/{trial_id}/finish")
def finish_offline_trial(experiment_id: str, trial_id: str, request: OfflineTrialFinishRequest):
    try:
        trial = offline_store.read_trial(experiment_id, trial_id)
        session_id = trial.get("session_id") or OFFLINE_TRIAL_SESSIONS.get(trial_id)
        session = SESSIONS.get(str(session_id)) if session_id else None
        if request.expected_candidate_id:
            trial["expected_candidate_id"] = request.expected_candidate_id
        if request.expected_display_id is not None:
            trial["expected_display_id"] = request.expected_display_id
        trial["outcome"] = request.outcome
        trial["operator_note"] = request.note or ""
        trial["finished_at_epoch_s"] = time.time()
        trial["status"] = "finished"
        if session is not None:
            _update_offline_trial_from_session(experiment_id, trial, session, status="finished")
        else:
            trial["latency_s"] = trial["finished_at_epoch_s"] - float(trial.get("started_at_epoch_s") or trial["finished_at_epoch_s"])
            offline_store.write_trial(experiment_id, trial)
        offline_store.append_event(
            experiment_id,
            {"event": "trial_finished", "trial_id": trial_id, "outcome": request.outcome},
        )
        return _offline_trial_view(experiment_id, trial_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/offline/experiments/{experiment_id}/metrics")
def get_offline_metrics(experiment_id: str):
    try:
        return offline_store.metrics(experiment_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/sessions/start")
def start_session(request: StartSessionRequest):
    session = create_session(request)
    SESSIONS[session.session_id] = session
    return build_session_view(session)


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    return build_session_view(session)


@app.post("/api/sessions/{session_id}/step")
def step_session(session_id: str, request: StepSessionRequest):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    if session.status != "awaiting_answer" or session.current_question is None:
        raise HTTPException(status_code=409, detail="session is not waiting for a clarification answer")
    try:
        clarifier.answer_current_question(session, request.answer)
        if session.resolved_target is not None:
            finalize_resolved_target(session)
            maybe_auto_execute(session)
        return build_session_view(session)
    except Exception as exc:
        session.status = "error"
        session.error_message = str(exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/sessions/{session_id}/execute")
def execute_session(session_id: str, request: ExecuteSessionRequest):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    if session.resolved_target is None:
        raise HTTPException(status_code=409, detail="session has no resolved target")

    execute_resolved_session(session, dry_run=request.dry_run, raise_on_error=True)
    return build_session_view(session)


@app.post("/api/sessions/{session_id}/confirm")
def confirm_session(session_id: str, request: ConfirmSessionRequest):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    if session.execution_result is None:
        raise HTTPException(status_code=409, detail="session has not executed yet")

    session.confirmation_result = {
        "success": bool(request.success),
        "note": (request.note or "").strip(),
        "confirmed_at_epoch_s": time.time(),
        "reset_ready": bool(request.reset_ready),
    }
    session.status = "confirmed_success" if request.success else "confirmed_failure"
    if request.success:
        session.error_message = None
    finalize_resolved_target(
        session,
        success=bool(request.success),
        banner_text="CONFIRMED SUCCESS" if request.success else "CONFIRMED FAILURE",
    )
    record_path = _write_session_record(session)
    return build_session_view(session)
