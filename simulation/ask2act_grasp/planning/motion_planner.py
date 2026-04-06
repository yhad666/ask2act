from __future__ import annotations

import math

import numpy as np

from ask2act_grasp.types import GraspCandidate, GraspConfig, MotionPlan, MotionWaypoint, SceneConfig


class MotionPlanner:
    BASE_ROTATE_LIMIT_RAD = 0.35
    WRIST_YAW_LIMIT_RAD = 1.30
    WRIST_PITCH_LIMITS_RAD = (-1.20, 0.55)

    def __init__(self, scene_config: SceneConfig, grasp_config: GraspConfig) -> None:
        self.scene_config = scene_config
        self.grasp_config = grasp_config

    def _world_z_to_lift(self, z_world_m: float) -> float:
        return float(
            np.clip(
                z_world_m - 0.08,
                self.scene_config.table_top_z_m - 0.46 + 0.10,
                1.05,
            )
        )

    def plan_to_grasp(self, candidate: GraspCandidate, current_state: dict[str, float]) -> MotionPlan:
        if candidate.source == "scene_oracle_single_cup":
            return self._plan_single_cup_oracle(candidate, current_state)

        position = np.asarray(candidate.position_m, dtype=float)
        pregrasp_position = np.asarray(position - candidate.approach_axis_world * self.grasp_config.pregrasp_offset_m, dtype=float)
        pregrasp_position[2] = max(
            float(pregrasp_position[2]),
            self.scene_config.table_top_z_m + 0.18,
        )
        position[2] = max(
            float(position[2]),
            self.scene_config.table_top_z_m + 0.10,
        )
        if self._intersects_table(position, pregrasp_position):
            raise RuntimeError("Pregrasp segment intersects table geometry.")

        pregrasp_targets = self._solve_joint_targets(pregrasp_position, current_state, candidate)
        grasp_targets = self._solve_joint_targets(position, current_state, candidate)
        grasp_targets["stretch_gripper"] = 0.06
        close_targets = dict(grasp_targets)
        close_targets["stretch_gripper"] = -0.2
        lift_targets = dict(close_targets)
        lift_targets["lift"] = min(1.08, max(lift_targets["lift"] + 0.10, self.scene_config.table_top_z_m - 0.46 + 0.20))

        approach_targets = dict(pregrasp_targets)
        approach_targets["lift"] = max(pregrasp_targets["lift"], self.scene_config.table_top_z_m - 0.46 + 0.18)
        approach_targets["arm"] = min(max(grasp_targets["arm"] - 0.06, 0.0), 0.52)

        waypoints = [
            MotionWaypoint(name="pregrasp", joint_targets=pregrasp_targets),
            MotionWaypoint(name="approach_above_table", joint_targets=approach_targets),
            MotionWaypoint(name="grasp", joint_targets=grasp_targets),
            MotionWaypoint(name="close_gripper", joint_targets=close_targets),
            MotionWaypoint(name="postgrasp_lift", joint_targets=lift_targets),
        ]
        return MotionPlan(
            backend=self.grasp_config.planner_backend,
            waypoints=waypoints,
            metadata={
                "target_position_m": position.tolist(),
                "pregrasp_position_m": pregrasp_position.tolist(),
                "selected_grasp": {
                    "score": candidate.score,
                    "source": candidate.source,
                    "width_m": candidate.width_m,
                },
            },
        )

    def _plan_single_cup_oracle(self, candidate: GraspCandidate, current_state: dict[str, float]) -> MotionPlan:
        scene = self.scene_config
        cup_x, cup_y, _ = scene.cup_position_m
        target_x = cup_x + self.grasp_config.oracle_lateral_offset_m
        target_y = cup_y + self.grasp_config.oracle_forward_offset_m
        cup_top_z = scene.table_top_z_m + scene.cup_height_m
        cup_grasp_z = scene.table_top_z_m + scene.cup_height_m * self.grasp_config.oracle_grasp_height_ratio

        # Side grasp recipe for the current known single-cup stage:
        # 1. retract/tuck so the wrist clears the table edge
        # 2. move to the cup-height band before extending
        # 3. rotate the wrist back to a straight side-grasp pose
        # 4. extend in-plane and close around the cup
        base_rotate = float(
            np.clip(
                math.atan2(target_x, max(-target_y, 1e-3)),
                -self.BASE_ROTATE_LIMIT_RAD,
                self.BASE_ROTATE_LIMIT_RAD,
            )
        )
        tucked_wrist_yaw = float(np.clip(self.grasp_config.oracle_tucked_wrist_yaw_rad, -self.WRIST_YAW_LIMIT_RAD, self.WRIST_YAW_LIMIT_RAD))
        wrist_yaw = 0.0
        tucked_pitch = 0.18
        side_grasp_pitch = float(
            np.clip(
                candidate.metadata.get("preferred_wrist_pitch_rad", self.grasp_config.oracle_side_grasp_wrist_pitch_rad),
                -0.05,
                0.12,
            )
        )

        arm_target = float(np.clip(max(-target_y - max(self.grasp_config.oracle_arm_backoff_m, 0.24), 0.0), 0.24, 0.40))
        approach_lift = self._world_z_to_lift(cup_grasp_z + min(self.grasp_config.oracle_approach_height_offset_m, 0.005))
        grasp_lift = self._world_z_to_lift(cup_grasp_z)
        postgrasp_lift = self._world_z_to_lift(cup_grasp_z + self.grasp_config.oracle_postgrasp_lift_delta_m)

        retract_targets = {
            "base_rotate": base_rotate,
            "lift": max(float(current_state.get("lift", 0.0)), grasp_lift),
            "arm": 0.0,
            "wrist_yaw": tucked_wrist_yaw,
            "wrist_pitch": tucked_pitch,
            "wrist_roll": 0.0,
            "stretch_gripper": 0.045,
        }
        height_band_targets = {
            "base_rotate": base_rotate,
            "lift": approach_lift,
            "arm": 0.0,
            "wrist_yaw": tucked_wrist_yaw,
            "wrist_pitch": tucked_pitch,
            "wrist_roll": 0.0,
            "stretch_gripper": 0.045,
        }
        side_open_width = float(self.grasp_config.oracle_side_open_width_cmd)

        rotate_for_side_targets = {
            "base_rotate": base_rotate,
            "lift": approach_lift,
            "arm": 0.0,
            "wrist_yaw": wrist_yaw,
            "wrist_pitch": side_grasp_pitch,
            "wrist_roll": 0.0,
            "stretch_gripper": side_open_width,
        }
        pre_open_targets = {
            "base_rotate": base_rotate,
            "lift": approach_lift,
            "arm": max(0.22, arm_target - 0.06),
            "wrist_yaw": wrist_yaw,
            "wrist_pitch": side_grasp_pitch,
            "wrist_roll": 0.0,
            "stretch_gripper": side_open_width,
        }
        extend_targets = {
            "base_rotate": base_rotate,
            "lift": approach_lift,
            "arm": max(0.24, arm_target - 0.02),
            "wrist_yaw": wrist_yaw,
            "wrist_pitch": side_grasp_pitch,
            "wrist_roll": 0.0,
            "stretch_gripper": side_open_width,
        }
        grasp_targets = {
            "base_rotate": base_rotate,
            "lift": grasp_lift,
            "arm": extend_targets["arm"],
            "wrist_yaw": wrist_yaw,
            "wrist_pitch": side_grasp_pitch,
            "wrist_roll": 0.0,
            "stretch_gripper": side_open_width,
        }
        close_targets = dict(grasp_targets)
        close_targets["stretch_gripper"] = 0.1
        lift_targets = dict(close_targets)
        lift_targets["lift"] = postgrasp_lift

        return MotionPlan(
            backend=self.grasp_config.planner_backend,
            waypoints=[
                MotionWaypoint(name="retract_and_tuck", joint_targets=retract_targets),
                MotionWaypoint(name="move_to_grasp_height_band", joint_targets=height_band_targets),
                MotionWaypoint(name="rotate_for_side_grasp", joint_targets=rotate_for_side_targets),
                MotionWaypoint(name="pre_open_near_object", joint_targets=pre_open_targets),
                MotionWaypoint(name="extend_toward_cup", joint_targets=extend_targets),
                MotionWaypoint(name="grasp", joint_targets=grasp_targets),
                MotionWaypoint(name="close_gripper", joint_targets=close_targets),
                MotionWaypoint(name="postgrasp_lift", joint_targets=lift_targets),
            ],
            metadata={
                "target_position_m": [target_x, target_y, cup_grasp_z],
                "pregrasp_position_m": [target_x, target_y, cup_grasp_z],
                "selected_grasp": {
                    "score": candidate.score,
                    "source": candidate.source,
                    "width_m": candidate.width_m,
                },
                "oracle_geometry": {
                    "cup_top_z_m": cup_top_z,
                    "cup_grasp_z_m": cup_grasp_z,
                    "oracle_grasp_height_ratio": self.grasp_config.oracle_grasp_height_ratio,
                    "arm_target_m": arm_target,
                    "oracle_arm_backoff_m": self.grasp_config.oracle_arm_backoff_m,
                    "oracle_final_arm_delta_m": self.grasp_config.oracle_final_arm_delta_m,
                    "oracle_tucked_wrist_yaw_rad": tucked_wrist_yaw,
                    "oracle_lateral_offset_m": self.grasp_config.oracle_lateral_offset_m,
                    "oracle_forward_offset_m": self.grasp_config.oracle_forward_offset_m,
                    "side_open_width_m": side_open_width,
                    "approach_lift_cmd": approach_lift,
                    "grasp_lift_cmd": grasp_lift,
                    "postgrasp_lift_cmd": postgrasp_lift,
                },
            },
        )

    def _solve_joint_targets(self, target_xyz: np.ndarray, current_state: dict[str, float], candidate: GraspCandidate) -> dict[str, float]:
        x, y, z = map(float, target_xyz)
        forward_distance = max(-y, 0.0)
        lateral_offset = x

        # Keep the mobile base mostly fixed for now; only allow a mild correction.
        theta = float(np.clip(math.atan2(lateral_offset, max(forward_distance, 1e-3)), -self.BASE_ROTATE_LIMIT_RAD, self.BASE_ROTATE_LIMIT_RAD))
        theta = float(np.clip(candidate.metadata.get("preferred_base_rotate_rad", theta), -self.BASE_ROTATE_LIMIT_RAD, self.BASE_ROTATE_LIMIT_RAD))

        # Approximate tabletop reaching geometry for the current simple scene.
        arm_extension = float(np.clip(forward_distance - 0.42, 0.02, 0.40))
        lift = float(np.clip(z - 0.44, self.scene_config.table_top_z_m - 0.46 + 0.10, 1.05))
        wrist_pitch = float(np.clip(candidate.metadata.get("preferred_wrist_pitch_rad", -0.85), *self.WRIST_PITCH_LIMITS_RAD))
        wrist_yaw = float(
            np.clip(candidate.metadata.get("preferred_wrist_yaw_rad", -theta), -self.WRIST_YAW_LIMIT_RAD, self.WRIST_YAW_LIMIT_RAD)
        )

        # Never drive the wrist under the table on the current heuristic backend.
        if z <= self.scene_config.table_top_z_m + 0.05:
            lift = max(lift, self.scene_config.table_top_z_m - 0.46 + 0.12)

        return {
            "base_rotate": theta,
            "lift": lift,
            "arm": arm_extension,
            "wrist_yaw": wrist_yaw,
            "wrist_pitch": wrist_pitch,
            "wrist_roll": 0.0,
            "stretch_gripper": 0.06,
        }

    def _intersects_table(self, grasp_position_m: np.ndarray, pregrasp_position_m: np.ndarray) -> bool:
        table_x, table_y, _ = self.scene_config.table_position_m
        table_sx, table_sy, _ = self.scene_config.table_size_m
        xy_inside = (
            abs(float(grasp_position_m[0]) - table_x) <= table_sx + 0.02
            and abs(float(grasp_position_m[1]) - table_y) <= table_sy + 0.02
        )
        low_enough = float(pregrasp_position_m[2]) <= self.scene_config.table_top_z_m + 0.01
        return xy_inside and low_enough
