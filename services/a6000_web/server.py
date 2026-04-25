from __future__ import annotations

import base64
import json
import os
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
from .phrase_extractor import InstructionPhraseExtractor
from .schemas import (
    ConfirmSessionRequest,
    ExecuteSessionRequest,
    GraspPlanResult,
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
SESSION_RECORD_ROOT = Path(os.getenv("ASK2ACT_SESSION_RECORD_ROOT", str(ROOT / "artifacts" / "session_records"))).expanduser()

SESSIONS: Dict[str, SessionState] = {}

phrase_extractor = InstructionPhraseExtractor()
detector = GroundingDinoDetector(phrase_extractor=phrase_extractor)
clarifier = ClarificationEngine(
    base_url=VLLM_BASE_URL,
    model=VLLM_MODEL,
    system_prompt_path=SYSTEM_PROMPT_PATH,
    gen_max_tokens=int(os.getenv("ASK2ACT_GEN_MAX_TOKENS", "2000")),
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


def execute_resolved_session(session: SessionState, *, dry_run: bool, raise_on_error: bool = True) -> None:
    if session.resolved_target is None:
        raise HTTPException(status_code=409, detail="session has no resolved target")

    try:
        plan_result = grasp_runtime.plan_for_target(
            session.resolved_target,
            observation_metadata=session.observation_raw_response,
            dry_run=dry_run,
            rotate_clockwise_90=detector.rotate_clockwise_90,
        )
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

        success = bool(plan_result.get("ok", True)) and bool(transport_result.get("ok", True))
        session.execution_result = {
            "ok": success,
            "dry_run": dry_run,
            "auto_execute": AUTO_EXECUTE_ON_RESOLVE and not dry_run,
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


def create_session(request: StartSessionRequest) -> SessionState:
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
        else:
            clarifier.initialize_session(session)
        if session.resolved_target is not None:
            finalize_resolved_target(session)
            maybe_auto_execute(session)
        return session
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "ok": True,
        "sessions": len(SESSIONS),
        "vllm_base_url": VLLM_BASE_URL,
        "vllm_model": VLLM_MODEL,
        "stretch_transport_mode": stretch_transport.mode,
        "stretch_zmq_endpoint": STRETCH_ZMQ_ENDPOINT if stretch_transport.mode == "zmq" else None,
        "pipeline_mode": grasp_runtime.mode,
        "auto_execute_on_resolve": AUTO_EXECUTE_ON_RESOLVE,
        "stretch_observe_timeout_ms": STRETCH_OBSERVE_TIMEOUT_MS,
        "stretch_execute_timeout_ms": STRETCH_EXECUTE_TIMEOUT_MS,
    }


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
