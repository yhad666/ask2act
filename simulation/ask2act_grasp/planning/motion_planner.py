from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from ask2act_grasp.types import GraspCandidate, GraspConfig, MotionPlan, MotionWaypoint, SceneConfig


class MotionPlanner:
    def __init__(self, scene_config: SceneConfig, grasp_config: GraspConfig) -> None:
        self.scene_config = scene_config
        self.grasp_config = grasp_config

    def plan_to_grasp(self, candidate: GraspCandidate, current_state: dict[str, float]) -> MotionPlan:
        position = candidate.position_m
        pregrasp_position = position - candidate.approach_axis_world * self.grasp_config.pregrasp_offset_m
        if self._intersects_table(position, pregrasp_position):
            raise RuntimeError("Pregrasp segment intersects table geometry.")

        pregrasp_targets = self._solve_joint_targets(pregrasp_position, current_state)
        grasp_targets = self._solve_joint_targets(position, current_state)
        grasp_targets["stretch_gripper"] = float(current_state["stretch_gripper"])
        close_targets = dict(grasp_targets)
        close_targets["stretch_gripper"] = -0.2
        lift_targets = dict(close_targets)
        lift_targets["lift"] = min(1.08, lift_targets["lift"] + 0.10)

        waypoints = [
            MotionWaypoint(name="pregrasp", joint_targets=pregrasp_targets),
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

    def _solve_joint_targets(self, target_xyz: np.ndarray, current_state: dict[str, float]) -> dict[str, float]:
        del current_state
        x, y, z = map(float, target_xyz)
        theta = math.atan2(x, -y)
        forward_distance = float(np.hypot(x, y))
        arm_extension = float(np.clip(forward_distance - 0.33, 0.0, 0.52))
        lift = float(np.clip(z - 0.46, 0.0, 1.10))
        wrist_pitch = self.grasp_config.top_down_grasp_pitch_rad
        return {
            "base_rotate": theta,
            "lift": lift,
            "arm": arm_extension,
            "wrist_yaw": -theta,
            "wrist_pitch": wrist_pitch,
            "wrist_roll": 0.0,
            "stretch_gripper": 0.045,
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

