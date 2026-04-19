from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


@dataclass
class TargetSelection:
    target_id: str
    label: str
    primitive_hint: str
    camera_name: str
    bbox_xyxy: list[int]
    mask_path: str | None
    confidence: float
    source: str

    @staticmethod
    def from_dict(payload: dict[str, Any], *, base_dir: Path | None = None) -> "TargetSelection":
        required = [
            "target_id",
            "label",
            "primitive_hint",
            "camera_name",
            "bbox_xyxy",
            "confidence",
            "source",
        ]
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"TargetSelection is missing required fields: {missing}")

        bbox = [int(v) for v in payload["bbox_xyxy"]]
        if len(bbox) != 4:
            raise ValueError("TargetSelection.bbox_xyxy must have exactly 4 integers.")

        mask_path = payload.get("mask_path")
        if mask_path is not None and base_dir is not None:
            candidate = Path(mask_path)
            if not candidate.is_absolute():
                mask_path = str((base_dir / candidate).resolve())

        return TargetSelection(
            target_id=str(payload["target_id"]),
            label=str(payload["label"]),
            primitive_hint=str(payload["primitive_hint"]),
            camera_name=str(payload["camera_name"]),
            bbox_xyxy=bbox,
            mask_path=mask_path,
            confidence=float(payload["confidence"]),
            source=str(payload["source"]),
        )


@dataclass
class GraspCandidate:
    candidate_id: str
    pose_head_4x4: list[list[float]]
    pose_world_4x4: list[list[float]]
    width_m: float
    score: float
    approach_dir_world: list[float]
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectedGraspPlan:
    target: TargetSelection
    best_candidate: GraspCandidate
    pregrasp_pose_world: list[list[float]]
    pregrasp_joint_targets: dict[str, float]
    primitive_mode: str
    clearance_summary: dict[str, float]


@dataclass
class WristRefinementDelta:
    dx_m: float
    dy_m: float
    dz_m: float
    dyaw_rad: float
    confidence: float
    target_visible: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class FailureReason(str, Enum):
    no_valid_candidate = "no_valid_candidate"
    pregrasp_timeout = "pregrasp_timeout"
    wrist_target_lost = "wrist_target_lost"
    approach_miss = "approach_miss"
    gripper_close_fail = "gripper_close_fail"


@dataclass
class CandidateEvaluation:
    candidate: GraspCandidate
    passed_hard_filters: bool
    hard_filter_reasons: list[str]
    soft_score: float
    total_score: float
    metrics: dict[str, float] = field(default_factory=dict)
