from __future__ import annotations

from typing import Any

import numpy as np

from pipeline_types import CandidateEvaluation, GraspCandidate, SelectedGraspPlan, TargetSelection
from grasp_proposal import LocalRegionGeometry


MAX_GRIPPER_WIDTH_M = 0.085
MIN_TABLE_CLEARANCE_M = 0.015
MIN_SAFE_GRASP_HEIGHT_M = 0.02
MIN_LIFT_CLEARANCE_M = 0.05
MAX_LIFT_TARGET_M = 1.05
WRIST_PITCH_RANGE_RAD = (-1.35, 0.30)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def evaluate_candidates(
    *,
    target: TargetSelection,
    candidates: list[GraspCandidate],
    geometry: LocalRegionGeometry,
) -> list[CandidateEvaluation]:
    evaluations: list[CandidateEvaluation] = []
    object_width = float(max(geometry.extent_xyz_m[0], geometry.extent_xyz_m[1]))
    primitive_hint = target.primitive_hint.lower()

    for candidate in candidates:
        candidate_pose = np.asarray(candidate.pose_world_4x4, dtype=float)
        candidate_xyz = candidate_pose[:3, 3]
        support_z = float(geometry.support_z_m)
        top_z = float(geometry.top_z_m)
        table_clearance = float(candidate_xyz[2] - support_z)
        lift_clearance = float(MAX_LIFT_TARGET_M - candidate_xyz[2])
        preferred_pitch = float(candidate.metadata.get("preferred_wrist_pitch_rad", -0.55))
        approach_dir = np.asarray(candidate.approach_dir_world, dtype=float)
        hard_filter_reasons: list[str] = []

        if candidate.width_m > MAX_GRIPPER_WIDTH_M:
            hard_filter_reasons.append("width_exceeds_gripper_limit")
        if not (WRIST_PITCH_RANGE_RAD[0] <= preferred_pitch <= WRIST_PITCH_RANGE_RAD[1]):
            hard_filter_reasons.append("wrist_pitch_out_of_range")
        if candidate_xyz[2] < support_z + MIN_SAFE_GRASP_HEIGHT_M:
            hard_filter_reasons.append("grasp_height_below_table_safety_margin")
        if lift_clearance < MIN_LIFT_CLEARANCE_M:
            hard_filter_reasons.append("insufficient_lift_clearance")
        if table_clearance < MIN_TABLE_CLEARANCE_M and approach_dir[2] < -0.4:
            hard_filter_reasons.append("approach_likely_to_hit_support_surface")

        geometry_match = _clip01(1.0 - abs(candidate.width_m - object_width) / max(object_width, 0.03))
        table_clearance_score = _clip01((table_clearance - MIN_TABLE_CLEARANCE_M) / 0.08)
        lift_clearance_score = _clip01((lift_clearance - MIN_LIFT_CLEARANCE_M) / 0.20)
        primitive_score = 0.1
        if primitive_hint in {"cup", "bowl", "box", "cup/bowl/box"}:
            primitive_score = 0.8 if "top_down" in str(candidate.metadata.get("candidate_style", "")) else 0.3
        primitive_score = _clip01(primitive_score + float(candidate.metadata.get("primitive_bonus", 0.0)))

        soft_score = float(
            0.45 * candidate.score
            + 0.20 * table_clearance_score
            + 0.15 * lift_clearance_score
            + 0.10 * geometry_match
            + 0.10 * primitive_score
        )
        passed = len(hard_filter_reasons) == 0
        total_score = soft_score if passed else -1.0
        evaluations.append(
            CandidateEvaluation(
                candidate=candidate,
                passed_hard_filters=passed,
                hard_filter_reasons=hard_filter_reasons,
                soft_score=soft_score,
                total_score=total_score,
                metrics={
                    "candidate_score": float(candidate.score),
                    "table_clearance_m": table_clearance,
                    "lift_clearance_m": lift_clearance,
                    "geometry_match": geometry_match,
                    "primitive_score": primitive_score,
                },
            )
        )

    return sorted(evaluations, key=lambda item: item.total_score, reverse=True)


def build_selected_grasp_plan(
    *,
    target: TargetSelection,
    evaluation: CandidateEvaluation,
    geometry: LocalRegionGeometry,
    baseline_params: dict[str, float],
) -> SelectedGraspPlan:
    candidate = evaluation.candidate
    candidate_pose_world = np.asarray(candidate.pose_world_4x4, dtype=float)
    candidate_xyz_world = candidate_pose_world[:3, 3]
    centroid_head = np.asarray(geometry.centroid_head_m, dtype=float)
    standoff_m = float(candidate.metadata.get("pregrasp_standoff_m", 0.09))

    pregrasp_pose_world = candidate_pose_world.copy()
    pregrasp_pose_world[:3, 3] = candidate_xyz_world + np.asarray([0.0, 0.0, standoff_m], dtype=float)

    arm_target = float(
        np.clip(
            baseline_params["pregrasp_arm_m"] + 0.05 * (centroid_head[2] - 0.75),
            0.16,
            0.40,
        )
    )
    lift_target = float(
        np.clip(
            baseline_params["pregrasp_lift_m"] + 0.15 * (candidate_xyz_world[2] - 0.78),
            0.68,
            0.95,
        )
    )
    base_translate_m = float(np.clip(0.40 * centroid_head[0], -0.05, 0.05))
    open_gripper_m = float(np.clip(candidate.width_m + 0.01, 0.02, 0.04))

    pregrasp_joint_targets = {
        "base_translate_m": base_translate_m,
        "pregrasp_lift_m": lift_target,
        "pregrasp_arm_m": arm_target,
        "pregrasp_wrist_yaw_rad": float(candidate.metadata.get("preferred_wrist_yaw_rad", 0.0)),
        "pregrasp_wrist_pitch_rad": float(candidate.metadata.get("preferred_wrist_pitch_rad", -0.55)),
        "pregrasp_wrist_roll_rad": float(baseline_params["pregrasp_wrist_roll_rad"]),
        "head_pan_rad": float(baseline_params["head_pan_rad"]),
        "head_tilt_rad": float(baseline_params["head_tilt_rad"]),
        "open_gripper_m": open_gripper_m,
        "close_gripper_m": float(baseline_params["close_gripper_m"]),
        "descend_delta_m": float(baseline_params["descend_delta_m"]),
        "min_descend_delta_m": float(baseline_params["min_descend_delta_m"]),
        "max_descend_delta_m": float(baseline_params["max_descend_delta_m"]),
        "lift_after_grasp_delta_m": float(baseline_params["lift_after_grasp_delta_m"]),
        "retract_arm_target_m": float(baseline_params["retract_arm_target_m"]),
        "nominal_target_depth_m": float(baseline_params["nominal_target_depth_m"]),
    }
    return SelectedGraspPlan(
        target=target,
        best_candidate=candidate,
        pregrasp_pose_world=pregrasp_pose_world.astype(float).tolist(),
        pregrasp_joint_targets=pregrasp_joint_targets,
        primitive_mode="top_down_v1",
        clearance_summary={
            "table_clearance_m": float(evaluation.metrics["table_clearance_m"]),
            "lift_clearance_m": float(evaluation.metrics["lift_clearance_m"]),
            "support_z_m": float(geometry.support_z_m),
            "top_z_m": float(geometry.top_z_m),
        },
    )


def serialize_evaluations(evaluations: list[CandidateEvaluation]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for evaluation in evaluations:
        serialized.append(
            {
                "candidate": evaluation.candidate,
                "passed_hard_filters": evaluation.passed_hard_filters,
                "hard_filter_reasons": evaluation.hard_filter_reasons,
                "soft_score": evaluation.soft_score,
                "total_score": evaluation.total_score,
                "metrics": evaluation.metrics,
            }
        )
    return serialized
