from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    candidate_id: str
    display_id: Optional[int] = None
    label: str
    score: float
    bbox_xyxy: List[float]
    mask_rle: Optional[str] = None


class ScoredQuestion(BaseModel):
    id: int
    text: str
    count: Dict[str, int] = Field(default_factory=dict)
    score_py: float
    score_pn: float
    efe_score: float


class QuestionTurn(BaseModel):
    round_index: int
    question_id: int
    text: str
    answer: Literal["y", "n"]
    score_py: float
    score_pn: float
    efe_score: float


class ResolvedTarget(BaseModel):
    candidate_id: str
    display_id: Optional[int] = None
    label: str
    score: float
    bbox_xyxy: List[float]
    mask_rle: Optional[str] = None


class GraspPlanResult(BaseModel):
    pipeline_mode: str
    planner_backend: str
    target_bbox_xyxy: List[float]
    point_cloud_count: Optional[int] = None
    selected_grasp_score: Optional[float] = None
    trajectory_waypoint_count: Optional[int] = None
    pipeline_run_dir: Optional[str] = None
    success: Optional[bool] = None
    note: Optional[str] = None


class StartSessionRequest(BaseModel):
    instruction: str
    fetch_observation: bool = True
    observation_image_data_url: Optional[str] = None
    observation_id: Optional[str] = None


class StepSessionRequest(BaseModel):
    answer: Literal["y", "n"]


class ExecuteSessionRequest(BaseModel):
    dry_run: bool = False


class ConfirmSessionRequest(BaseModel):
    success: bool
    note: Optional[str] = None
    reset_ready: bool = True


@dataclass
class SessionState:
    session_id: str
    instruction: str
    instruction_phrases: List[str]
    detection_prompt: str
    observation_id: Optional[str]
    observation_source: str
    observation_image_bytes: bytes
    observation_image_data_url: str
    candidate_overlay_data_url: str
    candidates: List[Candidate]
    vlm_messages: List[Dict[str, Any]]
    observation_raw_response: Optional[Dict[str, Any]] = None
    current_round: int = 1
    current_questions: List[ScoredQuestion] = field(default_factory=list)
    current_question: Optional[ScoredQuestion] = None
    question_history: List[QuestionTurn] = field(default_factory=list)
    asked_history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    last_protocol_json: Optional[Dict[str, Any]] = None
    resolved_target: Optional[ResolvedTarget] = None
    grasp_plan_result: Optional[GraspPlanResult] = None
    final_image_data_url: Optional[str] = None
    status: str = "awaiting_answer"
    execution_result: Optional[Dict[str, Any]] = None
    confirmation_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
