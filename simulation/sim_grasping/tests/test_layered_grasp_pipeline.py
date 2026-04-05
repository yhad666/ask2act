from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SIM_GRASPING_DIR = ROOT / "sim_grasping"
if str(SIM_GRASPING_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_GRASPING_DIR))

from candidate_filter import build_selected_grasp_plan, evaluate_candidates
from grasp_proposal import MockGraspProposalBackend, create_local_segmask, summarize_local_region
from layer0_target_selection import ManualJsonTargetProvider
from pipeline_types import TargetSelection
from wrist_refinement import compute_wrist_refinement_delta


def make_head_observation() -> dict:
    depth = np.full((8, 10), 0.9, dtype=np.float32)
    depth[2:6, 3:7] = 0.72
    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    rgb[2:6, 3:7] = 255
    return {
        "rgb": rgb,
        "depth": depth,
        "k_matrix": np.array(
            [
                [100.0, 0.0, 5.0],
                [0.0, 100.0, 4.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        "camera_pose_4x4": np.eye(4, dtype=float),
        "pose": {
            "ee_pose_4x4": np.eye(4, dtype=float).tolist(),
            "wrist_camera_pose_4x4": np.eye(4, dtype=float).tolist(),
            "base_pose": [0.0, 0.0, 0.0],
        },
        "camera_time": 0.0,
        "camera_fps": 15.0,
        "camera_name": "cam_d435i_rgb",
    }


def make_target(mask_path: str | None = None) -> TargetSelection:
    return TargetSelection(
        target_id="target-1",
        label="cup",
        primitive_hint="cup",
        camera_name="head_d435i",
        bbox_xyxy=[2, 1, 8, 7],
        mask_path=mask_path,
        confidence=0.95,
        source="manual_json",
    )


def test_manual_json_provider_resolves_relative_mask(tmp_path: Path):
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1:5, 2:4] = 255
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), mask)
    json_path = tmp_path / "target.json"
    json_path.write_text(
        json.dumps(
            {
                "target_id": "cup-1",
                "label": "cup",
                "primitive_hint": "cup",
                "camera_name": "head_d435i",
                "bbox_xyxy": [1, 1, 5, 5],
                "mask_path": "mask.png",
                "confidence": 0.9,
                "source": "manual_json",
            }
        ),
        encoding="utf-8",
    )
    target = ManualJsonTargetProvider(json_path).load()
    assert target.mask_path == str(mask_path.resolve())


def test_create_local_segmask_uses_bbox_when_no_mask():
    segmask = create_local_segmask(make_target(), image_shape=(8, 10))
    assert segmask.shape == (8, 10)
    assert int(segmask.sum()) == 255 * (6 * 6)


def test_summarize_geometry_and_mock_candidates():
    target = make_target()
    geometry = summarize_local_region(target=target, head_observation=make_head_observation())
    assert geometry.point_cloud_head.shape[0] > 0
    backend = MockGraspProposalBackend()
    candidates = backend.propose_grasps(target=target, geometry=geometry)
    assert len(candidates) == 3
    assert candidates[0].candidate_id.startswith("mock_candidate_")
    assert candidates[0].metadata["candidate_style"] == "top_down_center"


def test_candidate_filter_ranks_top_down_for_cup():
    target = make_target()
    geometry = summarize_local_region(target=target, head_observation=make_head_observation())
    candidates = MockGraspProposalBackend().propose_grasps(target=target, geometry=geometry)
    evaluations = evaluate_candidates(target=target, candidates=candidates, geometry=geometry)
    assert evaluations[0].passed_hard_filters is True
    assert evaluations[0].candidate.metadata["candidate_style"] == "top_down_center"
    selected = build_selected_grasp_plan(
        target=target,
        evaluation=evaluations[0],
        geometry=geometry,
        baseline_params={
            "pregrasp_lift_m": 0.80,
            "pregrasp_arm_m": 0.24,
            "pregrasp_wrist_yaw_rad": 0.0,
            "pregrasp_wrist_pitch_rad": -0.55,
            "pregrasp_wrist_roll_rad": 0.0,
            "head_pan_rad": 0.0,
            "head_tilt_rad": -1.0,
            "open_gripper_m": 0.03,
            "close_gripper_m": -0.02,
            "descend_delta_m": 0.06,
            "min_descend_delta_m": 0.03,
            "max_descend_delta_m": 0.09,
            "lift_after_grasp_delta_m": 0.08,
            "retract_arm_target_m": 0.08,
            "nominal_target_depth_m": 0.28,
        },
    )
    assert "base_translate_m" in selected.pregrasp_joint_targets
    assert selected.primitive_mode == "top_down_v1"


def test_wrist_refinement_clamps_large_deltas():
    depth = np.full((120, 160), 0.0, dtype=np.float32)
    depth[45:85, 55:95] = 0.34
    wrist_observation = {
        "rgb": np.zeros((120, 160, 3), dtype=np.uint8),
        "depth": depth,
        "k_matrix": np.array(
            [
                [120.0, 0.0, 80.0],
                [0.0, 120.0, 60.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        "camera_pose_4x4": np.eye(4, dtype=float),
        "pose": {
            "ee_pose_4x4": np.eye(4, dtype=float).tolist(),
            "wrist_camera_pose_4x4": np.eye(4, dtype=float).tolist(),
            "base_pose": [0.0, 0.0, 0.0],
        },
        "camera_time": 0.0,
        "camera_fps": 15.0,
        "camera_name": "cam_d405_rgb",
    }
    target = make_target()
    geometry = summarize_local_region(target=target, head_observation=make_head_observation())
    candidate = MockGraspProposalBackend().propose_grasps(target=target, geometry=geometry)[0]
    evaluations = evaluate_candidates(target=target, candidates=[candidate], geometry=geometry)
    selected = build_selected_grasp_plan(
        target=target,
        evaluation=evaluations[0],
        geometry=geometry,
        baseline_params={
            "pregrasp_lift_m": 0.80,
            "pregrasp_arm_m": 0.24,
            "pregrasp_wrist_yaw_rad": 0.0,
            "pregrasp_wrist_pitch_rad": -0.55,
            "pregrasp_wrist_roll_rad": 0.0,
            "head_pan_rad": 0.0,
            "head_tilt_rad": -1.0,
            "open_gripper_m": 0.03,
            "close_gripper_m": -0.02,
            "descend_delta_m": 0.06,
            "min_descend_delta_m": 0.03,
            "max_descend_delta_m": 0.09,
            "lift_after_grasp_delta_m": 0.08,
            "retract_arm_target_m": 0.08,
            "nominal_target_depth_m": 0.28,
        },
    )
    delta, pixel_result = compute_wrist_refinement_delta(
        wrist_observation=wrist_observation,
        selected_plan=selected,
    )
    assert pixel_result is not None
    assert abs(delta.dx_m) <= 0.03
    assert abs(delta.dy_m) <= 0.03
    assert abs(delta.dz_m) <= 0.05
    assert abs(delta.dyaw_rad) <= 0.2
