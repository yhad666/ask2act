from __future__ import annotations

import math
import os

import numpy as np

from ask2act_grasp.stretch3_specs import (
    CGN_GRIPPER_DEPTH_M,
    STRETCH3_JOINT_LIMITS,
    TOPDOWN_SIMPLEIK_EXECUTION_Z_BIAS_M,
    topdown_grasp_center_to_rubber_offset_m,
    rotate_topdown_offset_to_world_m,
    topdown_simpleik_wrist_model_error_m,
    topdown_wrist_to_rubber_offset_m,
    topdown_wrist_to_grasp_center_offset_m,
    WRIST_TO_FINGER_MID_LOCAL_M,
    cgn_frame_to_wrist_local_m,
    clip_to_joint_limits,
    gripper_close_command,
    gripper_width_to_command,
    wrist_to_cgn_grasp_frame_offset_m,
)
from ask2act_grasp.types import GraspCandidate, GraspConfig, MotionPlan, MotionWaypoint, SceneConfig

TOP_DOWN_Z_CORRECTION_M = -0.04
ANGLED_XY_DAMPING = 0.3
APPROX_GEOMETRIC_TOP_DOWN_X_CORRECTION_M = 0.045
APPROX_GEOMETRIC_TOP_DOWN_Y_CORRECTION_M = -0.065
APPROX_GEOMETRIC_TOP_DOWN_WRIST_Z_OFFSET_OVERRIDE = os.getenv("ASK2ACT_APPROX_GEOMETRIC_TOP_DOWN_WRIST_Z_OFFSET_M")
GEOMETRIC_TOP_DOWN_GRASP_Z_MODE = os.getenv("ASK2ACT_GEOMETRIC_TOP_DOWN_GRASP_Z_MODE", "center").strip().lower()
GEOMETRIC_TOP_DOWN_GRASP_TOP_CLEARANCE_M = float(
    os.getenv("ASK2ACT_GEOMETRIC_TOP_DOWN_GRASP_TOP_CLEARANCE_M", "0.0")
)
GEOMETRIC_TOP_DOWN_PREGRASP_CLEARANCE_M = float(
    os.getenv("ASK2ACT_GEOMETRIC_TOP_DOWN_PREGRASP_CLEARANCE_M", "0.12")
)
GEOMETRIC_TOP_DOWN_POSTGRASP_LIFT_M = float(os.getenv("ASK2ACT_GEOMETRIC_TOP_DOWN_POSTGRASP_LIFT_M", "0.12"))
GEOMETRIC_TOP_DOWN_ENABLE_BASE_REACH_TRANSLATE = (
    os.getenv("ASK2ACT_GEOMETRIC_TOP_DOWN_ENABLE_BASE_REACH_TRANSLATE", "1").strip().lower()
    in {"1", "true", "yes", "on"}
)
GEOMETRIC_TOP_DOWN_BASE_REACH_TRANSLATE_MAX_M = float(
    os.getenv("ASK2ACT_GEOMETRIC_TOP_DOWN_BASE_REACH_TRANSLATE_MAX_M", "0.16")
)
GEOMETRIC_TOP_DOWN_BASE_REACH_TRANSLATE_MARGIN_M = float(
    os.getenv("ASK2ACT_GEOMETRIC_TOP_DOWN_BASE_REACH_TRANSLATE_MARGIN_M", "0.02")
)


class MotionPlanner:
    """Convert grasp poses into Stretch joint-space waypoints."""

    BASE_ROTATE_LIMIT_RAD = 0.35

    def __init__(self, scene_config: SceneConfig, grasp_config: GraspConfig) -> None:
        self.scene_config = scene_config
        self.grasp_config = grasp_config
        self.simple_ik = None
        self.simple_ik_init_error: str | None = None
        if not self.grasp_config.use_simple_ik_for_topdown:
            self.simple_ik_init_error = "disabled by grasp_config.use_simple_ik_for_topdown"
            print(
                "SimpleIK disabled by config; using approximate world-to-joint mapping for top-down grasps",
                flush=True,
            )
            return
        try:
            from ask2act_grasp.ik.simple_ik import SimpleIK

            self.simple_ik = SimpleIK()
        except Exception as exc:
            self.simple_ik_init_error = str(exc)
            print(
                f"WARNING: SimpleIK unavailable, falling back to approximate world-to-joint mapping ({exc})",
                flush=True,
            )

    def _world_z_to_lift(self, z_world_m: float) -> float:
        return float(
            np.clip(
                z_world_m,
                STRETCH3_JOINT_LIMITS["lift"][0],
                STRETCH3_JOINT_LIMITS["lift"][1],
            )
        )

    @staticmethod
    def _normalize_angle(angle_rad: float) -> float:
        return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)

    def _select_wrist_yaw(
        self,
        raw_yaw: float,
        *,
        current_yaw: float,
        allow_pi_flip: bool,
    ) -> float:
        lower, upper = STRETCH3_JOINT_LIMITS["wrist_yaw"]
        candidate_set: set[float] = set()
        for wrap in (-2.0 * math.pi, 0.0, 2.0 * math.pi):
            candidate_set.add(raw_yaw + wrap)
            if allow_pi_flip:
                candidate_set.add(raw_yaw + math.pi + wrap)
                candidate_set.add(raw_yaw - math.pi + wrap)

        valid_candidates = [value for value in candidate_set if lower <= value <= upper]
        if not valid_candidates:
            return float(np.clip(raw_yaw, lower, upper))

        def rank(value: float) -> tuple[float, float]:
            margin = min(value - lower, upper - value)
            if allow_pi_flip:
                # For top-down grasps, yaw and yaw +/- pi are equivalent.
                # Prefer the solution with the healthiest joint-limit margin so
                # we do not stall near the wrist yaw boundary before reaching
                # the descent/close phases.
                return (-margin, abs(value - current_yaw))
            return (abs(value - current_yaw), -margin)

        return min(valid_candidates, key=rank)

    def _world_y_to_arm(self, y_world_m: float) -> float:
        desired = max(-float(y_world_m) - self.grasp_config.oracle_arm_backoff_m, 0.0)
        return float(np.clip(desired, STRETCH3_JOINT_LIMITS["arm"][0], STRETCH3_JOINT_LIMITS["arm"][1]))

    def _world_y_to_arm_unclipped(self, y_world_m: float) -> float:
        return float(max(-float(y_world_m) - self.grasp_config.oracle_arm_backoff_m, 0.0))

    @staticmethod
    def _geometric_topdown_pregrasp_clearance_m(grasp_config: GraspConfig) -> float:
        return max(0.10, float(grasp_config.pregrasp_offset_m), GEOMETRIC_TOP_DOWN_PREGRASP_CLEARANCE_M)

    @staticmethod
    def _geometric_topdown_postgrasp_lift_m(grasp_config: GraspConfig) -> float:
        return max(0.08, float(grasp_config.oracle_postgrasp_lift_delta_m), GEOMETRIC_TOP_DOWN_POSTGRASP_LIFT_M)

    @staticmethod
    def _geometric_topdown_grasp_z(geometric_grasp: dict[str, object]) -> float:
        center_z = float(geometric_grasp["grasp_z"])
        top_z = float(geometric_grasp.get("object_top_z", center_z))
        bottom_z = float(geometric_grasp.get("object_bottom_z", center_z))
        mode = GEOMETRIC_TOP_DOWN_GRASP_Z_MODE
        if mode in {"upper", "top", "rim"}:
            return float(top_z + GEOMETRIC_TOP_DOWN_GRASP_TOP_CLEARANCE_M)
        if mode in {"upper_inside", "below_top"}:
            return float(top_z - abs(GEOMETRIC_TOP_DOWN_GRASP_TOP_CLEARANCE_M))
        if mode in {"ratio", "height_ratio"}:
            ratio = float(os.getenv("ASK2ACT_GEOMETRIC_TOP_DOWN_GRASP_HEIGHT_RATIO", "0.75"))
            return float(bottom_z + np.clip(ratio, 0.0, 1.2) * max(top_z - bottom_z, 0.0))
        return center_z

    @staticmethod
    def _approx_geometric_topdown_wrist_z_offset_m(gripper_open_cmd: float) -> float:
        if APPROX_GEOMETRIC_TOP_DOWN_WRIST_Z_OFFSET_OVERRIDE is not None:
            return float(APPROX_GEOMETRIC_TOP_DOWN_WRIST_Z_OFFSET_OVERRIDE)
        wrist_to_rubber_local = np.asarray(topdown_wrist_to_rubber_offset_m(gripper_open_cmd), dtype=float)
        return float(-wrist_to_rubber_local[2])

    def plan_to_grasp(self, candidate: GraspCandidate, current_state: dict[str, float]) -> MotionPlan:
        if candidate.source == "scene_oracle_single_cup":
            return self._plan_single_cup_oracle(candidate, current_state)

        if candidate.source == "geometric_point_cloud":
            geometric_grasp = candidate.metadata.get("geometric_grasp")
            if not isinstance(geometric_grasp, dict):
                raise ValueError("geometric_point_cloud candidate is missing geometric_grasp metadata")
            numeric_targets = self.geometric_grasp_targets(geometric_grasp, current_state=current_state)
            return self._build_waypoint_plan(candidate, numeric_targets, current_state)

        if self.grasp_config.cgn_execution_mode == "guided_oracle":
            numeric_targets = self.cgn_guided_oracle_targets(candidate, current_state=current_state)
            if "ik_base_rotate" in numeric_targets:
                return self._build_waypoint_plan(candidate, numeric_targets, current_state)
            return self._plan_single_cup_oracle_from_targets(candidate, numeric_targets, current_state)

        numeric_targets = self.cgn_pose_to_targets(
            candidate.pose_4x4,
            candidate.width_m,
            approach_type=candidate.approach_type or candidate.metadata.get("approach_type", "angled"),
            current_state=current_state,
        )
        return self._build_waypoint_plan(candidate, numeric_targets, current_state)

    def oracle_targets(
        self,
        candidate: GraspCandidate,
        *,
        current_state: dict[str, float] | None = None,
    ) -> dict[str, float]:
        del candidate
        cup_x, cup_y, _ = self.scene_config.cup_position_m
        grasp_x = float(cup_x + self.grasp_config.oracle_lateral_offset_m)
        grasp_y = float(cup_y + self.grasp_config.oracle_forward_offset_m)
        grasp_z = float(
            self.scene_config.table_top_z_m
            + self.scene_config.cup_height_m * self.grasp_config.oracle_grasp_height_ratio
        )
        pregrasp_z = float(grasp_z + self.grasp_config.oracle_approach_height_offset_m)
        return {
            "grasp_x": grasp_x,
            "grasp_y": grasp_y,
            "grasp_z": grasp_z,
            "pregrasp_x": grasp_x,
            "pregrasp_y": grasp_y,
            "pregrasp_z": pregrasp_z,
            "wrist_yaw": 0.0,
            "wrist_pitch": float(self.grasp_config.oracle_side_grasp_wrist_pitch_rad),
            "wrist_roll": 0.0,
            "gripper_open_cmd": float(self.grasp_config.oracle_side_open_width_cmd),
            "gripper_close_cmd": float(gripper_close_command()),
            "approach_type": "side",
            "approach_direction": [0.0, 1.0, 0.0],
            "contact_point": [grasp_x, grasp_y, grasp_z],
            "cgn_grasp_frame": None,
            "finger_length_used": None,
            "wrist_to_cgn_frame_offset_m": None,
            "reachable": True,
        }

    def _plan_single_cup_oracle(self, candidate: GraspCandidate, current_state: dict[str, float]) -> MotionPlan:
        numeric_targets = self.oracle_targets(candidate, current_state=current_state)
        return self._plan_single_cup_oracle_from_targets(candidate, numeric_targets, current_state)

    def cgn_guided_oracle_targets(
        self,
        candidate: GraspCandidate,
        *,
        current_state: dict[str, float] | None = None,
    ) -> dict[str, float]:
        pose = np.asarray(candidate.pose_4x4, dtype=float)
        contact = candidate.metadata.get("contact_point")
        if contact is None:
            approach = np.asarray(pose[:3, 2], dtype=float)
            approach = approach / max(np.linalg.norm(approach), 1e-8)
            contact_point = pose[:3, 3] + CGN_GRIPPER_DEPTH_M * approach
        else:
            contact_point = np.asarray(contact, dtype=float)

        grasp_x = float(contact_point[0] + self.grasp_config.oracle_lateral_offset_m)
        grasp_y = float(contact_point[1] + self.grasp_config.oracle_forward_offset_m)
        grasp_z = float(contact_point[2])
        pregrasp_z = float(grasp_z + self.grasp_config.oracle_approach_height_offset_m)

        if self.simple_ik is not None:
            desired_rubber_xyz = np.array(
                [
                    float(contact_point[0]),
                    float(contact_point[1]),
                    float(self.scene_config.table_top_z_m + self.scene_config.cup_height_m * 0.35),
                ],
                dtype=float,
            )
            requested_open_width = min(
                float(self.scene_config.cup_radius_m * 2.0 + 0.015),
                float(self.grasp_config.max_gripper_width_m),
            )
            ik_targets = self._solve_topdown_simple_ik_targets(
                desired_rubber_xyz=desired_rubber_xyz,
                requested_open_width=requested_open_width,
                planning_mode="cgn_guided_oracle_simple_ik",
                grip_angle_rad=0.0,
                current_state=current_state,
                extra_metadata={
                    "contact_point": desired_rubber_xyz.tolist(),
                    "cgn_grasp_frame": pose[:3, 3].tolist(),
                    "finger_length_used": None,
                    "wrist_to_cgn_frame_offset_m": None,
                },
            )
            if ik_targets is not None:
                return ik_targets
            print("WARNING: SimpleIK could not solve cgn_guided_oracle target; falling back to oracle mapping", flush=True)

        return {
            "grasp_x": grasp_x,
            "grasp_y": grasp_y,
            "grasp_z": grasp_z,
            "pregrasp_x": grasp_x,
            "pregrasp_y": grasp_y,
            "pregrasp_z": pregrasp_z,
            "wrist_yaw": 0.0,
            "wrist_pitch": float(self.grasp_config.oracle_side_grasp_wrist_pitch_rad),
            "wrist_roll": 0.0,
            "gripper_open_cmd": float(self.grasp_config.oracle_side_open_width_cmd),
            "gripper_close_cmd": float(gripper_close_command()),
            "approach_type": "side",
            "approach_direction": [0.0, 1.0, 0.0],
            "contact_point": np.asarray(contact_point, dtype=float).tolist(),
            "cgn_grasp_frame": pose[:3, 3].tolist(),
            "finger_length_used": None,
            "wrist_to_cgn_frame_offset_m": None,
            "reachable": True,
            "planning_mode": "cgn_guided_oracle",
        }

    def _solve_topdown_simple_ik_targets(
        self,
        *,
        desired_rubber_xyz: np.ndarray,
        requested_open_width: float,
        planning_mode: str,
        grip_angle_rad: float,
        current_state: dict[str, float] | None,
        extra_metadata: dict[str, object] | None = None,
    ) -> dict[str, float] | None:
        if self.simple_ik is None:
            return None

        requested_rubber_xyz = np.asarray(desired_rubber_xyz, dtype=float).reshape(3)
        desired_rubber_xyz = requested_rubber_xyz.copy()
        desired_rubber_xyz[2] += TOPDOWN_SIMPLEIK_EXECUTION_Z_BIAS_M
        base_world_translation = np.zeros(3, dtype=float)
        if current_state is not None:
            base_world_translation[0] = float(current_state.get("base_x", 0.0))
            base_world_translation[1] = float(current_state.get("base_y", 0.0))
        gripper_open_cmd = float(gripper_width_to_command(requested_open_width))
        wrist_to_grasp_center_local = np.asarray(topdown_wrist_to_grasp_center_offset_m(), dtype=float)
        grasp_center_to_rubber_local = np.asarray(topdown_grasp_center_to_rubber_offset_m(gripper_open_cmd), dtype=float)

        desired_wrist_yaw = 0.0
        if requested_open_width < 0.04:
            current_yaw = 0.0 if current_state is None else float(current_state.get("wrist_yaw", 0.0))
            desired_wrist_yaw = self._select_wrist_yaw(
                grip_angle_rad,
                current_yaw=current_yaw,
                allow_pi_flip=True,
            )

        base_rotate_guess = float(
            np.clip(
                math.atan2(float(desired_rubber_xyz[0]), max(-float(desired_rubber_xyz[1]), 1e-3)),
                -self.BASE_ROTATE_LIMIT_RAD,
                self.BASE_ROTATE_LIMIT_RAD,
            )
        )
        ik_result = None
        wrist_model_error_local = np.zeros(3, dtype=float)
        desired_wrist_yaw_pos = np.zeros(3, dtype=float)
        wrist_to_grasp_center_world = np.zeros(3, dtype=float)
        grasp_center_to_rubber_world = np.zeros(3, dtype=float)
        wrist_model_error_world = np.zeros(3, dtype=float)
        desired_grasp_center_xyz = desired_rubber_xyz.copy()
        base_rotate = base_rotate_guess
        lift_val = 0.0
        arm_val = 0.0
        for _ in range(5):
            total_yaw = base_rotate_guess + desired_wrist_yaw
            wrist_to_grasp_center_world = rotate_topdown_offset_to_world_m(wrist_to_grasp_center_local, total_yaw)
            grasp_center_to_rubber_world = rotate_topdown_offset_to_world_m(grasp_center_to_rubber_local, total_yaw)
            wrist_model_error_world = rotate_topdown_offset_to_world_m(wrist_model_error_local, total_yaw)
            desired_grasp_center_xyz = desired_rubber_xyz - grasp_center_to_rubber_world
            desired_wrist_yaw_pos = (
                desired_grasp_center_xyz
                - base_world_translation
                - wrist_to_grasp_center_world
                - wrist_model_error_world
            )

            ik_result = self.simple_ik.ik_rotary_base(desired_wrist_yaw_pos.tolist())
            if ik_result is None:
                print(f"WARNING: SimpleIK returned None for target {desired_wrist_yaw_pos.tolist()}", flush=True)
                return None

            self.simple_ik.clip_with_joint_limits(ik_result)
            base_rotate = float(ik_result["joint_mobile_base_rotation"])
            lift_val = float(ik_result["joint_lift"])
            arm_val = float(ik_result["joint_arm_l0"])
            next_wrist_model_error_local = np.asarray(topdown_simpleik_wrist_model_error_m(lift_val, arm_val), dtype=float)
            if (
                abs(self._normalize_angle(base_rotate - base_rotate_guess)) < 1e-5
                and np.linalg.norm(next_wrist_model_error_local - wrist_model_error_local) < 1e-6
            ):
                wrist_model_error_local = next_wrist_model_error_local
                break
            wrist_model_error_local = next_wrist_model_error_local
            base_rotate_guess = base_rotate

        total_yaw = base_rotate + desired_wrist_yaw
        wrist_to_grasp_center_world = rotate_topdown_offset_to_world_m(wrist_to_grasp_center_local, total_yaw)
        grasp_center_to_rubber_world = rotate_topdown_offset_to_world_m(grasp_center_to_rubber_local, total_yaw)
        wrist_model_error_world = rotate_topdown_offset_to_world_m(wrist_model_error_local, total_yaw)
        desired_grasp_center_xyz = desired_rubber_xyz - grasp_center_to_rubber_world
        desired_wrist_yaw_pos = (
            desired_grasp_center_xyz
            - base_world_translation
            - wrist_to_grasp_center_world
            - wrist_model_error_world
        )

        pregrasp_clearance_m = self._geometric_topdown_pregrasp_clearance_m(self.grasp_config)
        pregrasp_rubber_xyz = desired_rubber_xyz.copy()
        pregrasp_rubber_xyz[2] += pregrasp_clearance_m
        pregrasp_grasp_center_xyz = pregrasp_rubber_xyz.copy()
        pregrasp_ik = None
        pregrasp_lift = min(
            STRETCH3_JOINT_LIMITS["lift"][1],
            lift_val + pregrasp_clearance_m,
        )
        pregrasp_arm = arm_val
        pregrasp_wrist_pos = np.zeros(3, dtype=float)
        pregrasp_error_local = wrist_model_error_local.copy()
        pregrasp_base_rotate_guess = base_rotate
        for _ in range(5):
            pregrasp_total_yaw = pregrasp_base_rotate_guess + desired_wrist_yaw
            pregrasp_wrist_to_grasp_center_world = rotate_topdown_offset_to_world_m(
                wrist_to_grasp_center_local,
                pregrasp_total_yaw,
            )
            pregrasp_grasp_center_to_rubber_world = rotate_topdown_offset_to_world_m(
                grasp_center_to_rubber_local,
                pregrasp_total_yaw,
            )
            pregrasp_error_world = rotate_topdown_offset_to_world_m(pregrasp_error_local, pregrasp_total_yaw)
            pregrasp_grasp_center_xyz = pregrasp_rubber_xyz - pregrasp_grasp_center_to_rubber_world
            pregrasp_wrist_pos = (
                pregrasp_grasp_center_xyz
                - base_world_translation
                - pregrasp_wrist_to_grasp_center_world
                - pregrasp_error_world
            )
            pregrasp_ik = self.simple_ik.ik_rotary_base(pregrasp_wrist_pos.tolist())
            if pregrasp_ik is None:
                break
            self.simple_ik.clip_with_joint_limits(pregrasp_ik)
            pregrasp_lift = float(pregrasp_ik["joint_lift"])
            pregrasp_arm = float(pregrasp_ik["joint_arm_l0"])
            pregrasp_base_rotate = float(pregrasp_ik["joint_mobile_base_rotation"])
            next_pregrasp_error_local = np.asarray(
                topdown_simpleik_wrist_model_error_m(pregrasp_lift, pregrasp_arm),
                dtype=float,
            )
            if (
                abs(self._normalize_angle(pregrasp_base_rotate - pregrasp_base_rotate_guess)) < 1e-5
                and np.linalg.norm(next_pregrasp_error_local - pregrasp_error_local) < 1e-6
            ):
                pregrasp_error_local = next_pregrasp_error_local
                break
            pregrasp_error_local = next_pregrasp_error_local
            pregrasp_base_rotate_guess = pregrasp_base_rotate
        if pregrasp_ik is not None:
            self.simple_ik.clip_with_joint_limits(pregrasp_ik)
            pregrasp_lift = float(pregrasp_ik["joint_lift"])
            pregrasp_arm = float(pregrasp_ik["joint_arm_l0"])

        fk_check = np.asarray(
            self.simple_ik.fk_rotary_base(
                {
                    "joint_mobile_base_rotation": base_rotate,
                    "joint_lift": lift_val,
                    "joint_arm_l0": arm_val,
                }
            ),
            dtype=float,
        )
        predicted_wrist_world = fk_check + base_world_translation + wrist_model_error_world
        predicted_grasp_center_world = predicted_wrist_world + wrist_to_grasp_center_world
        predicted_rubber_world = predicted_grasp_center_world + grasp_center_to_rubber_world
        fk_error = float(np.linalg.norm(predicted_rubber_world - desired_rubber_xyz))

        print("\n=== SimpleIK RESULT ===", flush=True)
        print(f"  Desired rubber tip: {desired_rubber_xyz.tolist()}", flush=True)
        print(f"  Desired grasp ctr:  {desired_grasp_center_xyz.tolist()}", flush=True)
        print(f"  Desired wrist_yaw:  {desired_wrist_yaw_pos.tolist()}", flush=True)
        print(
            f"  IK joints: base_rot={base_rotate:.4f} lift={lift_val:.4f} arm={arm_val:.4f}",
            flush=True,
        )
        print(f"  Wrist yaw:          {desired_wrist_yaw:.4f}", flush=True)
        print(f"  Wrist model error:  {wrist_model_error_world.tolist()}", flush=True)
        print(f"  Wrist->grasp ctr:   {wrist_to_grasp_center_world.tolist()}", flush=True)
        print(f"  Grasp->rubber cmd:  {grasp_center_to_rubber_world.tolist()}", flush=True)
        print(f"  FK grasp center:    {predicted_grasp_center_world.tolist()}", flush=True)
        print(f"  FK check rubber:    {predicted_rubber_world.tolist()}", flush=True)
        print(f"  FK error:           {fk_error:.6f} m", flush=True)
        print("=======================\n", flush=True)

        targets: dict[str, float] = {
            "grasp_x": float(desired_rubber_xyz[0]),
            "grasp_y": float(desired_rubber_xyz[1]),
            "grasp_z": float(desired_rubber_xyz[2]),
            "pregrasp_x": float(pregrasp_rubber_xyz[0]),
            "pregrasp_y": float(pregrasp_rubber_xyz[1]),
            "pregrasp_z": float(pregrasp_rubber_xyz[2]),
            "wrist_yaw": float(desired_wrist_yaw),
            "wrist_pitch": clip_to_joint_limits("wrist_pitch", -1.57),
            "wrist_roll": 0.0,
            "gripper_open_width": float(requested_open_width),
            "gripper_open_cmd": gripper_open_cmd,
            "gripper_close_cmd": float(gripper_close_command()),
            "approach_type": "top_down",
            "approach_direction": [0.0, 0.0, -1.0],
            "contact_point": desired_rubber_xyz.tolist(),
            "requested_contact_point": requested_rubber_xyz.tolist(),
            "planning_mode": planning_mode,
            "grip_angle_rad": float(grip_angle_rad),
            "ik_base_rotate": base_rotate,
            "ik_lift": lift_val,
            "ik_arm": arm_val,
            "ik_pregrasp_lift": float(pregrasp_lift),
            "ik_pregrasp_arm": float(pregrasp_arm),
            "fk_error_m": fk_error,
            "desired_grasp_center_world_xyz": desired_grasp_center_xyz.tolist(),
            "ik_target_wrist_yaw_xyz": desired_wrist_yaw_pos.tolist(),
            "wrist_to_grasp_center_offset_m": wrist_to_grasp_center_world.tolist(),
            "wrist_to_grasp_center_offset_local_m": wrist_to_grasp_center_local.tolist(),
            "grasp_center_to_rubber_offset_m": grasp_center_to_rubber_world.tolist(),
            "grasp_center_to_rubber_offset_local_m": grasp_center_to_rubber_local.tolist(),
            "wrist_to_rubber_offset_m": (wrist_to_grasp_center_world + grasp_center_to_rubber_world).tolist(),
            "wrist_to_rubber_offset_local_m": (wrist_to_grasp_center_local + grasp_center_to_rubber_local).tolist(),
            "wrist_model_error_m": wrist_model_error_world.tolist(),
            "wrist_model_error_local_m": wrist_model_error_local.tolist(),
            "base_world_translation_m": base_world_translation.tolist(),
            "predicted_grasp_center_world_xyz": predicted_grasp_center_world.tolist(),
            "predicted_rubber_world_xyz": predicted_rubber_world.tolist(),
            "execution_z_bias_m": float(TOPDOWN_SIMPLEIK_EXECUTION_Z_BIAS_M),
            "reachable": True,
        }
        if extra_metadata:
            targets.update(extra_metadata)
        return targets

    def _approximate_geometric_targets(
        self,
        geometric_grasp: dict[str, object],
        *,
        current_state: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Convert a geometry-only top-down grasp into wrist-space motion targets."""
        raw_grasp_x = float(geometric_grasp["grasp_x"])
        raw_grasp_y = float(geometric_grasp["grasp_y"])
        grasp_x_correction = APPROX_GEOMETRIC_TOP_DOWN_X_CORRECTION_M
        grasp_y_correction = APPROX_GEOMETRIC_TOP_DOWN_Y_CORRECTION_M
        grasp_x = float(raw_grasp_x + grasp_x_correction)
        grasp_y = float(raw_grasp_y + grasp_y_correction)
        contact_grasp_z = self._geometric_topdown_grasp_z(geometric_grasp)
        grip_angle = float(geometric_grasp.get("grip_angle_rad", 0.0))
        requested_open_width = min(
            float(geometric_grasp["gripper_open_width"]),
            float(self.grasp_config.max_gripper_width_m),
        )
        gripper_open_cmd = float(gripper_width_to_command(requested_open_width))

        # The approximate real fallback does not have SimpleIK, so explicitly
        # convert the rubber/contact target into the wrist/lift target using the
        # calibrated top-down gripper length measured from simulation.
        wrist_vertical_offset = self._approx_geometric_topdown_wrist_z_offset_m(gripper_open_cmd)
        wrist_grasp_z = float(contact_grasp_z + wrist_vertical_offset)
        pregrasp_z = float(wrist_grasp_z + self._geometric_topdown_pregrasp_clearance_m(self.grasp_config))
        arm_upper = float(STRETCH3_JOINT_LIMITS["arm"][1])
        desired_arm_before_base = self._world_y_to_arm_unclipped(grasp_y)
        base_translate_arm_axis_m = 0.0
        if GEOMETRIC_TOP_DOWN_ENABLE_BASE_REACH_TRANSLATE and desired_arm_before_base > arm_upper:
            base_translate_arm_axis_m = float(
                np.clip(
                    desired_arm_before_base - arm_upper + GEOMETRIC_TOP_DOWN_BASE_REACH_TRANSLATE_MARGIN_M,
                    0.0,
                    GEOMETRIC_TOP_DOWN_BASE_REACH_TRANSLATE_MAX_M,
                )
            )
        arm_planning_grasp_y = float(grasp_y + base_translate_arm_axis_m)
        desired_arm_after_base = self._world_y_to_arm_unclipped(arm_planning_grasp_y)

        wrist_yaw = 0.0
        if requested_open_width < 0.04:
            current_yaw = 0.0 if current_state is None else float(current_state.get("wrist_yaw", 0.0))
            wrist_yaw = self._select_wrist_yaw(
                grip_angle,
                current_yaw=current_yaw,
                allow_pi_flip=True,
            )

        targets = {
            "grasp_x": grasp_x,
            "grasp_y": grasp_y,
            "grasp_z": wrist_grasp_z,
            "pregrasp_x": grasp_x,
            "pregrasp_y": grasp_y,
            "pregrasp_z": pregrasp_z,
            "grasp_y_for_arm": arm_planning_grasp_y,
            "pregrasp_y_for_arm": arm_planning_grasp_y,
            "base_translate_arm_axis_m": base_translate_arm_axis_m,
            "arm_required_before_base_translate_m": desired_arm_before_base,
            "arm_required_after_base_translate_m": desired_arm_after_base,
            "arm_reach_shortfall_before_base_translate_m": max(0.0, desired_arm_before_base - arm_upper),
            "arm_reach_shortfall_after_base_translate_m": max(0.0, desired_arm_after_base - arm_upper),
            "wrist_yaw": float(wrist_yaw),
            "wrist_pitch": clip_to_joint_limits("wrist_pitch", -1.57),
            "wrist_roll": 0.0,
            "gripper_open_width": float(requested_open_width),
            "gripper_open_cmd": gripper_open_cmd,
            "gripper_close_cmd": float(gripper_close_command()),
            "approach_type": "top_down",
            "approach_direction": [0.0, 0.0, -1.0],
            "contact_point": [grasp_x, grasp_y, contact_grasp_z],
            "rubber_contact_target_world_xyz": [grasp_x, grasp_y, contact_grasp_z],
            "commanded_wrist_target_world_xyz": [grasp_x, grasp_y, wrist_grasp_z],
            "raw_grasp_z": float(geometric_grasp["grasp_z"]),
            "z_execution_mode": GEOMETRIC_TOP_DOWN_GRASP_Z_MODE,
            "top_clearance_m": GEOMETRIC_TOP_DOWN_GRASP_TOP_CLEARANCE_M,
            "planning_mode": "geometric_point_cloud",
            "grip_angle_rad": grip_angle,
            "raw_grasp_x": raw_grasp_x,
            "raw_grasp_y": raw_grasp_y,
            "xy_execution_correction_m": [
                grasp_x_correction,
                grasp_y_correction,
            ],
            "min_cross_section_width": float(geometric_grasp.get("min_cross_section_width", requested_open_width)),
            "object_center": geometric_grasp.get("object_center"),
            "object_height": float(geometric_grasp.get("object_height", 0.0)),
            "object_top_z": float(geometric_grasp.get("object_top_z", contact_grasp_z)),
            "object_bottom_z": float(geometric_grasp.get("object_bottom_z", contact_grasp_z)),
            "grasp_point_validated": bool(geometric_grasp.get("grasp_point_validated", True)),
            "width_near_limit": bool(geometric_grasp.get("width_near_limit", False)),
            "wrist_to_cgn_frame_offset_m": wrist_vertical_offset,
            "wrist_to_rubber_vertical_offset_m": wrist_vertical_offset,
            "wrist_z_offset_source": (
                "env_override"
                if APPROX_GEOMETRIC_TOP_DOWN_WRIST_Z_OFFSET_OVERRIDE is not None
                else "calibrated_topdown_wrist_to_rubber"
            ),
        }
        targets["reachable"] = self._check_reachable(targets)
        return targets

    def geometric_grasp_targets(
        self,
        geometric_grasp: dict[str, object],
        *,
        current_state: dict[str, float] | None = None,
    ) -> dict[str, float]:
        desired_rubber_xyz = np.array(
            [
                float(geometric_grasp["grasp_x"]),
                float(geometric_grasp["grasp_y"]),
                self._geometric_topdown_grasp_z(geometric_grasp),
            ],
            dtype=float,
        )
        requested_open_width = min(
            float(geometric_grasp["gripper_open_width"]),
            float(self.grasp_config.max_gripper_width_m),
        )
        ik_targets = self._solve_topdown_simple_ik_targets(
            desired_rubber_xyz=desired_rubber_xyz,
            requested_open_width=requested_open_width,
            planning_mode="geometric_simple_ik",
            grip_angle_rad=float(geometric_grasp.get("grip_angle_rad", 0.0)),
            current_state=current_state,
            extra_metadata={
                "min_cross_section_width": float(geometric_grasp.get("min_cross_section_width", requested_open_width)),
                "object_center": geometric_grasp.get("object_center"),
                "object_height": float(geometric_grasp.get("object_height", 0.0)),
                "object_top_z": float(geometric_grasp.get("object_top_z", desired_rubber_xyz[2])),
                "object_bottom_z": float(geometric_grasp.get("object_bottom_z", desired_rubber_xyz[2])),
                "raw_grasp_z": float(geometric_grasp["grasp_z"]),
                "z_execution_mode": GEOMETRIC_TOP_DOWN_GRASP_Z_MODE,
                "top_clearance_m": GEOMETRIC_TOP_DOWN_GRASP_TOP_CLEARANCE_M,
                "grasp_point_validated": bool(geometric_grasp.get("grasp_point_validated", True)),
                "width_near_limit": bool(geometric_grasp.get("width_near_limit", False)),
            },
        )
        if ik_targets is not None:
            return ik_targets

        if self.grasp_config.allow_approximate_topdown_fallback:
            print("WARNING: Falling back to approximate geometric top-down mapping", flush=True)
            approximate_targets = self._approximate_geometric_targets(geometric_grasp, current_state=current_state)
            if not bool(approximate_targets.get("reachable", False)):
                raise RuntimeError(
                    "Approximate geometric top-down target remains unreachable after base reach translation: "
                    f"arm_shortfall_after={float(approximate_targets.get('arm_reach_shortfall_after_base_translate_m', 0.0)):.3f} m, "
                    f"base_translate_arm_axis={float(approximate_targets.get('base_translate_arm_axis_m', 0.0)):.3f} m"
                )
            return approximate_targets

        raise RuntimeError(
            "SimpleIK could not solve the geometric top-down target and "
            "allow_approximate_topdown_fallback=false, so execution was aborted "
            "instead of silently switching to the old approximate mapping."
        )

    def _plan_single_cup_oracle_from_targets(
        self,
        candidate: GraspCandidate,
        numeric_targets: dict[str, float],
        current_state: dict[str, float],
    ) -> MotionPlan:
        grasp_x = float(numeric_targets["grasp_x"])
        grasp_y = float(numeric_targets["grasp_y"])
        grasp_z = float(numeric_targets["grasp_z"])
        cup_top_z = float(self.scene_config.table_top_z_m + self.scene_config.cup_height_m)

        base_rotate = float(
            np.clip(
                math.atan2(grasp_x, max(-grasp_y, 1e-3)),
                -self.BASE_ROTATE_LIMIT_RAD,
                self.BASE_ROTATE_LIMIT_RAD,
            )
        )
        tucked_wrist_yaw = clip_to_joint_limits("wrist_yaw", float(self.grasp_config.oracle_tucked_wrist_yaw_rad))
        wrist_yaw = 0.0
        tucked_pitch = 0.18
        side_grasp_pitch = float(
            np.clip(
                candidate.metadata.get("preferred_wrist_pitch_rad", self.grasp_config.oracle_side_grasp_wrist_pitch_rad),
                -0.05,
                0.12,
            )
        )

        arm_target = float(
            np.clip(
                max(-grasp_y - self.grasp_config.oracle_arm_backoff_m, 0.0),
                STRETCH3_JOINT_LIMITS["arm"][0],
                STRETCH3_JOINT_LIMITS["arm"][1],
            )
        )
        approach_lift = self._world_z_to_lift(grasp_z + min(self.grasp_config.oracle_approach_height_offset_m, 0.005))
        grasp_lift = self._world_z_to_lift(grasp_z)
        postgrasp_lift = self._world_z_to_lift(grasp_z + self.grasp_config.oracle_postgrasp_lift_delta_m)
        side_open_cmd = float(self.grasp_config.oracle_side_open_width_cmd)

        print("\n=== MOTION PLAN DEBUG ===", flush=True)
        print(f"  source: {candidate.source}", flush=True)
        print(f"  planning_mode: {numeric_targets.get('planning_mode', 'oracle')}", flush=True)
        print("  approach_type: side", flush=True)
        print(f"  world target: x={grasp_x:.3f} y={grasp_y:.3f} z={grasp_z:.3f}", flush=True)
        print(
            f"  lift: approach={approach_lift:.3f} grasp={grasp_lift:.3f} post={postgrasp_lift:.3f}",
            flush=True,
        )
        print(f"  arm: target={arm_target:.3f}", flush=True)
        print(f"  wrist: yaw={wrist_yaw:.3f} pitch={side_grasp_pitch:.3f}", flush=True)
        print(f"  gripper: open={side_open_cmd:.3f} close=0.100", flush=True)
        print(f"  base_rotate: {base_rotate:.3f}", flush=True)
        print(
            f"  current state: lift={float(current_state.get('lift', 0.0)):.3f} arm={float(current_state.get('arm', 0.0)):.3f} "
            f"wrist_yaw={float(current_state.get('wrist_yaw', 0.0)):.3f} wrist_pitch={float(current_state.get('wrist_pitch', 0.0)):.3f}",
            flush=True,
        )
        print("=========================\n", flush=True)

        retract_targets = {
            "base_rotate": base_rotate,
            "lift": max(float(current_state.get("lift", 0.0)), grasp_lift),
            "arm": 0.0,
            "wrist_yaw": tucked_wrist_yaw,
            "wrist_pitch": tucked_pitch,
            "wrist_roll": 0.0,
        }
        height_band_targets = {
            "base_rotate": base_rotate,
            "lift": approach_lift,
            "arm": 0.0,
            "wrist_yaw": tucked_wrist_yaw,
            "wrist_pitch": tucked_pitch,
            "wrist_roll": 0.0,
        }
        rotate_for_side_targets = {
            "base_rotate": base_rotate,
            "lift": approach_lift,
            "arm": 0.0,
            "wrist_yaw": wrist_yaw,
            "wrist_pitch": side_grasp_pitch,
            "wrist_roll": 0.0,
        }
        pre_open_targets = {
            "base_rotate": base_rotate,
            "lift": approach_lift,
            "arm": max(STRETCH3_JOINT_LIMITS["arm"][0], arm_target - 0.06),
            "wrist_yaw": wrist_yaw,
            "wrist_pitch": side_grasp_pitch,
            "wrist_roll": 0.0,
            "stretch_gripper": side_open_cmd,
        }
        extend_targets = {
            "base_rotate": base_rotate,
            "lift": approach_lift,
            "arm": max(STRETCH3_JOINT_LIMITS["arm"][0], arm_target - 0.02),
            "wrist_yaw": wrist_yaw,
            "wrist_pitch": side_grasp_pitch,
            "wrist_roll": 0.0,
            "stretch_gripper": side_open_cmd,
        }
        grasp_targets = {
            "base_rotate": base_rotate,
            "lift": grasp_lift,
            "arm": extend_targets["arm"],
            "wrist_yaw": wrist_yaw,
            "wrist_pitch": side_grasp_pitch,
            "wrist_roll": 0.0,
            "stretch_gripper": side_open_cmd,
        }
        close_targets = dict(grasp_targets)
        close_targets["stretch_gripper"] = 0.1
        lift_targets = dict(close_targets)
        lift_targets["lift"] = postgrasp_lift

        return MotionPlan(
            backend=self.grasp_config.planner_backend,
            waypoints=[
                MotionWaypoint(name="retract_and_tuck", joint_targets=retract_targets, settle_s=0.6),
                MotionWaypoint(name="move_to_grasp_height_band", joint_targets=height_band_targets, settle_s=0.3),
                MotionWaypoint(name="rotate_for_side_grasp", joint_targets=rotate_for_side_targets, settle_s=0.3),
                MotionWaypoint(name="pre_open_near_object", joint_targets=pre_open_targets, settle_s=0.4),
                MotionWaypoint(name="extend_toward_cup", joint_targets=extend_targets, settle_s=0.5),
                MotionWaypoint(name="grasp", joint_targets=grasp_targets, settle_s=0.5),
                MotionWaypoint(name="close_gripper", joint_targets=close_targets, settle_s=0.5),
                MotionWaypoint(name="postgrasp_lift", joint_targets=lift_targets, settle_s=0.5),
            ],
            metadata={
                "target_position_m": [grasp_x, grasp_y, grasp_z],
                "pregrasp_position_m": [grasp_x, grasp_y, grasp_z],
                "selected_grasp": {
                    "score": float(candidate.score),
                    "source": candidate.source,
                    "width_m": float(candidate.width_m),
                    "approach_type": "side",
                },
                "numeric_targets": numeric_targets,
                "planning_mode": str(numeric_targets.get("planning_mode", "oracle")),
                "joint_targets": {
                    "base_rotate": base_rotate,
                    "grasp_arm": extend_targets["arm"],
                    "pregrasp_arm": pre_open_targets["arm"],
                    "pregrasp_lift": approach_lift,
                    "grasp_lift": grasp_lift,
                    "postgrasp_lift": postgrasp_lift,
                    "wrist_yaw": wrist_yaw,
                    "wrist_pitch": side_grasp_pitch,
                    "gripper_open_cmd": side_open_cmd,
                    "gripper_close_cmd": 0.1,
                },
                "oracle_geometry": {
                    "cup_top_z_m": cup_top_z,
                    "cup_grasp_z_m": grasp_z,
                    "arm_target_m": arm_target,
                    "oracle_arm_backoff_m": self.grasp_config.oracle_arm_backoff_m,
                    "oracle_tucked_wrist_yaw_rad": tucked_wrist_yaw,
                    "oracle_lateral_offset_m": self.grasp_config.oracle_lateral_offset_m,
                    "oracle_forward_offset_m": self.grasp_config.oracle_forward_offset_m,
                    "side_open_width_m": side_open_cmd,
                    "approach_lift_cmd": approach_lift,
                    "grasp_lift_cmd": grasp_lift,
                    "postgrasp_lift_cmd": postgrasp_lift,
                },
            },
        )

    def cgn_pose_to_targets(
        self,
        grasp_pose_4x4: np.ndarray,
        grasp_width: float,
        *,
        approach_type: str = "angled",
        current_state: dict[str, float] | None = None,
    ) -> dict[str, float]:
        pose = np.asarray(grasp_pose_4x4, dtype=float)
        cgn_grasp_frame = pose[:3, 3]
        rot = pose[:3, :3]
        approach = np.asarray(rot[:, 2], dtype=float)
        approach = approach / max(np.linalg.norm(approach), 1e-8)
        grasp_x_axis = np.asarray(rot[:, 0], dtype=float)
        grasp_x_axis = grasp_x_axis / max(np.linalg.norm(grasp_x_axis), 1e-8)
        wrist_offset_m = wrist_to_cgn_grasp_frame_offset_m()
        contact_point = cgn_grasp_frame + rot @ np.array([0.0, 0.0, CGN_GRIPPER_DEPTH_M], dtype=float)

        if approach_type == "top_down":
            wrist_grasp_pos = np.asarray(contact_point, dtype=float).copy()
            wrist_grasp_pos[2] += wrist_offset_m + TOP_DOWN_Z_CORRECTION_M
            wrist_pregrasp_pos = wrist_grasp_pos.copy()
            wrist_pregrasp_pos[2] += float(self.grasp_config.pregrasp_offset_m)
            wrist_offset_local = np.asarray(wrist_grasp_pos - cgn_grasp_frame, dtype=float)
            wrist_yaw = 0.0
            wrist_pitch = clip_to_joint_limits("wrist_pitch", -1.57)
        elif approach_type == "angled":
            wrist_grasp_pos = np.asarray(contact_point, dtype=float).copy()
            wrist_grasp_pos[2] += wrist_offset_m * max(abs(float(approach[2])), 0.6) + TOP_DOWN_Z_CORRECTION_M
            wrist_pregrasp_pos = wrist_grasp_pos.copy()
            wrist_pregrasp_pos[2] += float(self.grasp_config.pregrasp_offset_m)
            wrist_offset_local = np.asarray(wrist_grasp_pos - cgn_grasp_frame, dtype=float)
            wrist_yaw = 0.0
            wrist_pitch = clip_to_joint_limits(
                "wrist_pitch",
                float(np.arcsin(np.clip(approach[2], -1.0, 1.0))),
            )
        else:
            wrist_from_cgn_local = cgn_frame_to_wrist_local_m()
            wrist_grasp_pos = cgn_grasp_frame + rot @ wrist_from_cgn_local
            wrist_pregrasp_pos = wrist_grasp_pos - float(self.grasp_config.pregrasp_offset_m) * approach
            wrist_offset_local = np.asarray(wrist_grasp_pos - cgn_grasp_frame, dtype=float)
            raw_wrist_yaw = float(np.arctan2(approach[1], approach[0]))
            current_yaw = 0.0 if current_state is None else float(current_state.get("wrist_yaw", 0.0))
            wrist_yaw = self._select_wrist_yaw(
                raw_wrist_yaw,
                current_yaw=current_yaw,
                allow_pi_flip=False,
            )
            wrist_pitch = clip_to_joint_limits("wrist_pitch", float(np.arcsin(np.clip(approach[2], -1.0, 1.0))))

        if approach_type in {"top_down", "angled"}:
            object_diameter = float(self.scene_config.cup_radius_m * 2.0)
            desired_open_width = float(
                min(
                    object_diameter + 0.015,
                    self.grasp_config.max_gripper_width_m,
                )
            )
        else:
            desired_open_width = float(
                min(
                    max(grasp_width + 0.01, 0.05),
                    self.grasp_config.max_gripper_width_m,
                )
            )
        gripper_open_cmd = gripper_width_to_command(desired_open_width)
        # Regression recovery: use the same cup-friendly close command scale as the
        # older working oracle path instead of always slamming fully shut.
        gripper_close_cmd = 0.1

        targets = {
            "grasp_x": float(wrist_grasp_pos[0]),
            "grasp_y": float(wrist_grasp_pos[1]),
            "grasp_z": float(wrist_grasp_pos[2]),
            "pregrasp_x": float(wrist_pregrasp_pos[0]),
            "pregrasp_y": float(wrist_pregrasp_pos[1]),
            "pregrasp_z": float(wrist_pregrasp_pos[2]),
            "wrist_yaw": float(wrist_yaw),
            "wrist_pitch": float(wrist_pitch),
            "wrist_roll": 0.0,
            "gripper_open_width": desired_open_width,
            "gripper_open_cmd": float(gripper_open_cmd),
            "gripper_close_cmd": float(gripper_close_cmd),
            "approach_type": approach_type,
            "approach_direction": approach.tolist(),
            "contact_point": np.asarray(contact_point, dtype=float).tolist(),
            "cgn_grasp_frame": np.asarray(cgn_grasp_frame, dtype=float).tolist(),
            "finger_length_used": float(np.linalg.norm(WRIST_TO_FINGER_MID_LOCAL_M)),
            "wrist_to_cgn_frame_offset_m": float(np.linalg.norm(wrist_offset_local)),
            "wrist_to_cgn_frame_local_m": wrist_offset_local.tolist(),
        }
        targets["reachable"] = self._check_reachable(targets)
        return targets

    def _check_reachable(self, targets: dict[str, float]) -> bool:
        if "ik_lift" in targets and "ik_arm" in targets:
            lift_val = float(targets["ik_lift"])
            arm_val = float(targets["ik_arm"])
            wrist_pitch = float(targets["wrist_pitch"])
            return (
                STRETCH3_JOINT_LIMITS["lift"][0] <= lift_val <= STRETCH3_JOINT_LIMITS["lift"][1]
                and STRETCH3_JOINT_LIMITS["arm"][0] <= arm_val <= STRETCH3_JOINT_LIMITS["arm"][1]
                and STRETCH3_JOINT_LIMITS["wrist_pitch"][0] <= wrist_pitch <= STRETCH3_JOINT_LIMITS["wrist_pitch"][1]
            )

        grasp_z = float(targets["grasp_z"])
        wrist_pitch = float(targets["wrist_pitch"])
        arm_y = float(targets.get("grasp_y_for_arm", targets["grasp_y"]))
        grasp_arm = self._world_y_to_arm_unclipped(arm_y)
        return (
            STRETCH3_JOINT_LIMITS["lift"][0] <= grasp_z <= STRETCH3_JOINT_LIMITS["lift"][1]
            and STRETCH3_JOINT_LIMITS["arm"][0] <= grasp_arm <= STRETCH3_JOINT_LIMITS["arm"][1]
            and STRETCH3_JOINT_LIMITS["wrist_pitch"][0] <= wrist_pitch <= STRETCH3_JOINT_LIMITS["wrist_pitch"][1]
        )

    def _build_waypoint_plan(
        self,
        candidate: GraspCandidate,
        numeric_targets: dict[str, float],
        current_state: dict[str, float],
    ) -> MotionPlan:
        grasp_x = float(numeric_targets["grasp_x"])
        grasp_y = float(numeric_targets["grasp_y"])
        grasp_z = float(numeric_targets["grasp_z"])
        pregrasp_y = float(numeric_targets["pregrasp_y"])
        pregrasp_z = float(numeric_targets["pregrasp_z"])
        arm_grasp_y = float(numeric_targets.get("grasp_y_for_arm", grasp_y))
        arm_pregrasp_y = float(numeric_targets.get("pregrasp_y_for_arm", pregrasp_y))
        base_translate_arm_axis_m = float(numeric_targets.get("base_translate_arm_axis_m", 0.0))
        approach_type = str(numeric_targets.get("approach_type", candidate.approach_type or "angled"))

        base_rotate = float(
            np.clip(
                candidate.metadata.get(
                    "preferred_base_rotate_rad",
                    math.atan2(grasp_x, max(-grasp_y, 1e-3)),
                ),
                -self.BASE_ROTATE_LIMIT_RAD,
                self.BASE_ROTATE_LIMIT_RAD,
            )
        )
        tucked_wrist_yaw = clip_to_joint_limits("wrist_yaw", float(self.grasp_config.oracle_tucked_wrist_yaw_rad))
        desired_wrist_yaw = clip_to_joint_limits("wrist_yaw", float(numeric_targets.get("wrist_yaw", 0.0)))
        desired_wrist_pitch = clip_to_joint_limits(
            "wrist_pitch",
            float(
                numeric_targets.get(
                    "wrist_pitch",
                    candidate.metadata.get("preferred_wrist_pitch_rad", self.grasp_config.oracle_side_grasp_wrist_pitch_rad),
                )
            ),
        )
        desired_wrist_roll = clip_to_joint_limits("wrist_roll", float(numeric_targets.get("wrist_roll", 0.0)))
        tucked_pitch = 0.18

        if "ik_base_rotate" in numeric_targets:
            base_rotate = float(numeric_targets["ik_base_rotate"])
            grasp_arm = float(np.clip(numeric_targets["ik_arm"], *STRETCH3_JOINT_LIMITS["arm"]))
            grasp_lift = float(np.clip(numeric_targets["ik_lift"], *STRETCH3_JOINT_LIMITS["lift"]))
            pregrasp_arm = float(
                np.clip(numeric_targets.get("ik_pregrasp_arm", grasp_arm), *STRETCH3_JOINT_LIMITS["arm"])
            )
            pregrasp_lift = float(
                np.clip(numeric_targets.get("ik_pregrasp_lift", grasp_lift + 0.08), *STRETCH3_JOINT_LIMITS["lift"])
            )
        else:
            grasp_arm = self._world_y_to_arm(arm_grasp_y)
            pregrasp_arm = self._world_y_to_arm(arm_pregrasp_y)
            if approach_type == "side":
                pregrasp_arm = max(STRETCH3_JOINT_LIMITS["arm"][0], grasp_arm - 0.06)
            elif approach_type == "angled":
                pregrasp_arm = max(STRETCH3_JOINT_LIMITS["arm"][0], grasp_arm - 0.04)
            grasp_lift = self._world_z_to_lift(grasp_z)
            pregrasp_lift = self._world_z_to_lift(max(pregrasp_z, grasp_z))
            if approach_type == "top_down":
                pregrasp_lift = min(
                    self._world_z_to_lift(
                        max(pregrasp_z, grasp_z + self._geometric_topdown_pregrasp_clearance_m(self.grasp_config))
                    ),
                    STRETCH3_JOINT_LIMITS["lift"][1] - 0.02,
                )

        postgrasp_lift = min(
            grasp_lift
            + (
                self._geometric_topdown_postgrasp_lift_m(self.grasp_config)
                if approach_type == "top_down"
                else max(self.grasp_config.oracle_postgrasp_lift_delta_m, 0.08)
            ),
            STRETCH3_JOINT_LIMITS["lift"][1],
        )
        min_lift_delta = 0.05
        if postgrasp_lift - grasp_lift < min_lift_delta:
            grasp_lift = min(
                grasp_lift,
                STRETCH3_JOINT_LIMITS["lift"][1] - min_lift_delta - 0.02,
            )
            postgrasp_lift = min(
                STRETCH3_JOINT_LIMITS["lift"][1],
                grasp_lift + min_lift_delta,
            )
            print(
                f"Adjusted lift for headroom: grasp_lift={grasp_lift:.3f}, postgrasp_lift={postgrasp_lift:.3f}",
                flush=True,
            )
        if approach_type == "top_down":
            pregrasp_lift = min(
                max(pregrasp_lift, grasp_lift),
                STRETCH3_JOINT_LIMITS["lift"][1] - 0.02,
            )
        gripper_open_cmd = float(numeric_targets.get("gripper_open_cmd", self.grasp_config.oracle_side_open_width_cmd))
        gripper_close_cmd = float(numeric_targets.get("gripper_close_cmd", gripper_close_command()))

        print("\n=== MOTION PLAN DEBUG ===", flush=True)
        print(f"  source: {candidate.source}", flush=True)
        print(f"  approach_type: {approach_type}", flush=True)
        print(f"  world target: x={grasp_x:.3f} y={grasp_y:.3f} z={grasp_z:.3f}", flush=True)
        contact_target = numeric_targets.get("rubber_contact_target_world_xyz") or numeric_targets.get("contact_point")
        if isinstance(contact_target, list) and len(contact_target) == 3:
            print(
                f"  contact/rubber target: x={float(contact_target[0]):.3f} "
                f"y={float(contact_target[1]):.3f} z={float(contact_target[2]):.3f}",
                flush=True,
            )
        if "wrist_to_rubber_vertical_offset_m" in numeric_targets:
            print(
                "  wrist/rubber vertical offset: "
                f"{float(numeric_targets['wrist_to_rubber_vertical_offset_m']):.3f} "
                f"({numeric_targets.get('wrist_z_offset_source', 'unknown')})",
                flush=True,
            )
        print(
            f"  lift: pregrasp={pregrasp_lift:.3f} grasp={grasp_lift:.3f} post={postgrasp_lift:.3f}",
            flush=True,
        )
        print(
            f"  arm:  pregrasp={pregrasp_arm:.3f} grasp={grasp_arm:.3f}",
            flush=True,
        )
        print(
            f"  wrist: yaw={desired_wrist_yaw:.3f} pitch={desired_wrist_pitch:.3f} roll={desired_wrist_roll:.3f}",
            flush=True,
        )
        print(
            f"  gripper: open={gripper_open_cmd:.3f} close={gripper_close_cmd:.3f}",
            flush=True,
        )
        print(f"  base_rotate: {base_rotate:.3f}", flush=True)
        if abs(base_translate_arm_axis_m) > 1e-4:
            print(f"  base_translate_arm_axis: {base_translate_arm_axis_m:.3f}", flush=True)
            print(
                "  arm reach shortfall: "
                f"before={float(numeric_targets.get('arm_reach_shortfall_before_base_translate_m', 0.0)):.3f} "
                f"after={float(numeric_targets.get('arm_reach_shortfall_after_base_translate_m', 0.0)):.3f}",
                flush=True,
            )
        print(
            f"  current state: lift={float(current_state.get('lift', 0.0)):.3f} arm={float(current_state.get('arm', 0.0)):.3f} "
            f"wrist_yaw={float(current_state.get('wrist_yaw', 0.0)):.3f} wrist_pitch={float(current_state.get('wrist_pitch', 0.0)):.3f}",
            flush=True,
        )
        print("=========================\n", flush=True)

        retract_wrist_yaw = tucked_wrist_yaw
        if approach_type == "top_down":
            # For top-down calibration/execution, avoid an unnecessary large
            # yaw sweep into a tucked pose and then back out again. That extra
            # rotation is what most often times out under the GUI viewer.
            retract_wrist_yaw = desired_wrist_yaw

        waypoints: list[MotionWaypoint] = [
            MotionWaypoint(
                name="retract_and_tuck",
                joint_targets={
                    "base_rotate": base_rotate,
                    "lift": max(float(current_state.get("lift", 0.0)), min(pregrasp_lift, postgrasp_lift)),
                    "arm": 0.0,
                    "wrist_yaw": retract_wrist_yaw,
                    "wrist_pitch": tucked_pitch,
                    "wrist_roll": 0.0,
                },
                settle_s=0.6,
            ),
        ]
        if abs(base_translate_arm_axis_m) > 1e-4:
            waypoints.append(
                MotionWaypoint(
                    name="base_translate_for_reach",
                    joint_targets={"base_translate_arm_axis": base_translate_arm_axis_m},
                    settle_s=0.4,
                )
            )
        waypoints.extend(
            [
                MotionWaypoint(
                    name="move_lift_to_pregrasp",
                    joint_targets={"lift": pregrasp_lift, "base_rotate": base_rotate},
                    settle_s=0.3,
                ),
                MotionWaypoint(
                    name="orient_wrist",
                    joint_targets={
                        "wrist_yaw": desired_wrist_yaw,
                        "wrist_pitch": desired_wrist_pitch,
                        "wrist_roll": desired_wrist_roll,
                    },
                    settle_s=0.8 if approach_type == "top_down" else 0.3,
                ),
                MotionWaypoint(
                    name="open_gripper",
                    joint_targets={"stretch_gripper": gripper_open_cmd},
                    settle_s=0.3,
                ),
                MotionWaypoint(
                    name="move_to_pregrasp",
                    joint_targets={"arm": pregrasp_arm, "base_rotate": base_rotate},
                    settle_s=0.4,
                ),
            ]
        )

        if approach_type == "top_down":
            waypoints.append(
                MotionWaypoint(
                    name="descend_to_grasp",
                    joint_targets={"lift": grasp_lift, "arm": grasp_arm},
                    settle_s=0.5,
                )
            )
        elif approach_type == "side":
            waypoints.append(
                MotionWaypoint(
                    name="extend_to_grasp",
                    joint_targets={"arm": grasp_arm, "lift": grasp_lift},
                    settle_s=0.5,
                )
            )
        else:
            waypoints.append(
                MotionWaypoint(
                    name="angled_approach",
                    joint_targets={"arm": grasp_arm, "lift": grasp_lift},
                    settle_s=0.5,
                )
            )

        waypoints.extend(
            [
                MotionWaypoint(
                    name="close_gripper",
                    joint_targets={"stretch_gripper": gripper_close_cmd},
                    settle_s=1.2,
                ),
                MotionWaypoint(
                    name="secure_grasp",
                    joint_targets={"stretch_gripper": gripper_close_cmd},
                    settle_s=1.0,
                ),
                MotionWaypoint(
                    name="postgrasp_lift",
                    joint_targets={"lift": postgrasp_lift, "stretch_gripper": gripper_close_cmd},
                    settle_s=0.5,
                ),
            ]
        )

        return MotionPlan(
            backend=self.grasp_config.planner_backend,
            waypoints=waypoints,
            metadata={
                "target_position_m": [grasp_x, grasp_y, grasp_z],
                "rubber_contact_target_position_m": (
                    contact_target if isinstance(contact_target, list) and len(contact_target) == 3 else None
                ),
                "pregrasp_position_m": [float(numeric_targets["pregrasp_x"]), pregrasp_y, pregrasp_z],
                "selected_grasp": {
                    "score": float(candidate.score),
                    "source": candidate.source,
                    "width_m": float(candidate.width_m),
                    "approach_type": approach_type,
                },
                "numeric_targets": numeric_targets,
                "planning_mode": str(numeric_targets.get("planning_mode", "direct_pose")),
                "joint_targets": {
                    "base_rotate": base_rotate,
                    "grasp_arm": grasp_arm,
                    "pregrasp_arm": pregrasp_arm,
                    "pregrasp_lift": pregrasp_lift,
                    "grasp_lift": grasp_lift,
                    "postgrasp_lift": postgrasp_lift,
                    "wrist_yaw": desired_wrist_yaw,
                    "wrist_pitch": desired_wrist_pitch,
                    "gripper_open_cmd": gripper_open_cmd,
                    "gripper_close_cmd": gripper_close_cmd,
                    "base_translate_arm_axis_m": base_translate_arm_axis_m,
                },
            },
        )
