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
    yes_candidate_ids: List[str] = Field(default_factory=list)
    no_candidate_ids: List[str] = Field(default_factory=list)


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


class OfflineExperimentRequest(BaseModel):
    experiment_id: Optional[str] = None
    name: str = ""
    experiment_type: str = "pilot"
    notes: Optional[str] = None


class OfflineSceneRequest(BaseModel):
    scene_id: str
    scene_type: str
    object_categories: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    fetch_observation: bool = True
    observation_image_data_url: Optional[str] = None
    observation_id: Optional[str] = None


class OfflineTrialStartRequest(BaseModel):
    scene_id: str
    prompt: str
    prompt_type: Literal["clear", "ambiguous", "partial"]
    method: Literal[
        "proposed_efe",
        "top_score",
        "random_candidate",
        "vlm_direct",
        "first_question",
        "random_question",
        "vlm_best_question",
    ] = "proposed_efe"
    expected_candidate_id: Optional[str] = None
    expected_display_id: Optional[int] = None
    notes: Optional[str] = None


class OfflineTrialStepRequest(BaseModel):
    answer: Literal["y", "n"]
    request_id: Optional[str] = None
    question_count: Optional[int] = None


class OfflineTrialFinishRequest(BaseModel):
    outcome: Literal["correct", "wrong", "unresolved", "target_pruned", "aborted"]
    expected_candidate_id: Optional[str] = None
    expected_display_id: Optional[int] = None
    note: Optional[str] = None


class OfflineTrialManualSelectRequest(BaseModel):
    candidate_id: Optional[str] = None
    display_id: Optional[int] = None
    note: Optional[str] = None


class OfflineAuditUpdateRequest(BaseModel):
    outcome: Optional[Literal["correct", "wrong", "unresolved"]] = None
    include_in_audit: Optional[bool] = None
    failure_reason: Optional[str] = None
    failure_reason_detail: Optional[str] = None
    audit_note: Optional[str] = None
    prompt_type: Optional[Literal["clear", "ambiguous", "partial"]] = None
    scene_type: Optional[str] = None
    reviewer: Optional[str] = None


class OnlineExperimentRequest(BaseModel):
    experiment_id: Optional[str] = None
    name: str = ""
    experiment_type: str = "online_main"
    notes: Optional[str] = None


class OnlineSceneRequest(BaseModel):
    scene_id: str
    scene_type: Literal["cup_only", "bottle_only", "utensil_only", "mixed", "pilot"] = "pilot"
    object_categories: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    fetch_observation: bool = True
    observation_image_data_url: Optional[str] = None
    observation_id: Optional[str] = None


class OnlineTrialStartRequest(BaseModel):
    scene_id: str
    prompt: str
    prompt_type: Literal["clear", "ambiguous", "partial"]
    method: Literal[
        "proposed_efe",
        "top_score",
        "random_candidate",
        "vlm_best_question",
    ] = "proposed_efe"
    expected_candidate_id: Optional[str] = None
    expected_display_id: Optional[int] = None
    notes: Optional[str] = None


class OnlineTrialStepRequest(BaseModel):
    answer: Literal["y", "n"]
    request_id: Optional[str] = None
    question_count: Optional[int] = None


class OnlineTrialExecuteRequest(BaseModel):
    expected_candidate_id: Optional[str] = None
    expected_display_id: Optional[int] = None
    dry_run: bool = False
    note: Optional[str] = None


class OnlineTrialConfirmRequest(BaseModel):
    physical_grasp_success: bool
    correct_object_grasp_success: Optional[bool] = None
    wrong_object_grasp: bool = False
    note: Optional[str] = None
    reset_ready: bool = True


class OnlineTrialFinishRequest(BaseModel):
    outcome: Literal[
        "correct",
        "wrong",
        "unresolved",
        "skipped_wrong_target",
        "grasp_failed",
        "execution_failed",
        "aborted",
    ]
    expected_candidate_id: Optional[str] = None
    expected_display_id: Optional[int] = None
    note: Optional[str] = None


class OnlineAuditUpdateRequest(BaseModel):
    target_selection_outcome: Optional[Literal["correct", "wrong", "unresolved"]] = None
    grasp_attempted: Optional[bool] = None
    physical_grasp_success: Optional[bool] = None
    correct_object_grasp_success: Optional[bool] = None
    wrong_object_grasp: Optional[bool] = None
    include_in_audit: Optional[bool] = None
    failure_reason: Optional[str] = None
    failure_reason_detail: Optional[str] = None
    audit_note: Optional[str] = None
    prompt_type: Optional[Literal["clear", "ambiguous", "partial"]] = None
    scene_type: Optional[str] = None
    reviewer: Optional[str] = None


class OnlineHeadPoseRequest(BaseModel):
    head_pan_rad: float
    head_tilt_rad: float
    persist: bool = True


class OnlineGraspTuningRequest(BaseModel):
    rubber_local_x_correction_m: Optional[float] = None
    rubber_local_y_correction_m: Optional[float] = None
    rubber_local_z_correction_m: Optional[float] = None
    slender_rubber_local_x_correction_m: Optional[float] = None
    slender_rubber_local_y_correction_m: Optional[float] = None
    slender_rubber_local_z_correction_m: Optional[float] = None
    approx_topdown_x_correction_m: Optional[float] = None
    approx_topdown_y_correction_m: Optional[float] = None
    side_x_bias_m: Optional[float] = None
    side_x_bias_deadband_m: Optional[float] = None
    right_extra_x_bias_m: Optional[float] = None
    left_center_y_bias_m: Optional[float] = None
    right_y_bias_m: Optional[float] = None
    slender_side_x_bias_m: Optional[float] = None
    slender_side_x_bias_deadband_m: Optional[float] = None
    slender_right_extra_x_bias_m: Optional[float] = None
    slender_left_center_y_bias_m: Optional[float] = None
    slender_right_y_bias_m: Optional[float] = None
    slender_long_axis_bias: Optional[float] = None
    max_top_grasp_delta_m: Optional[float] = None
    gripper_open_cmd_override: Optional[float] = None
    stretch_gripper_real_open_cmd: Optional[float] = None
    stretch_gripper_real_close_cmd: Optional[float] = None
    stretch_release_gripper_cmd: Optional[float] = None


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
    question_mode: Optional[str] = None
    prompt_type: Optional[str] = None
    single_candidate_audit: bool = False
    plausible_candidate_ids: List[str] = field(default_factory=list)
    eliminated_candidate_ids: List[str] = field(default_factory=list)
    last_removed_candidate_ids: List[str] = field(default_factory=list)
    candidate_state_history: List[Dict[str, Any]] = field(default_factory=list)
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
