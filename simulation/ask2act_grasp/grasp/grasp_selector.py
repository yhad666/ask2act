from __future__ import annotations

import numpy as np

from ask2act_grasp.types import GraspCandidate, GraspConfig, SceneConfig


class GraspSelector:
    def __init__(self, scene_config: SceneConfig, grasp_config: GraspConfig) -> None:
        self.scene_config = scene_config
        self.grasp_config = grasp_config

    def select_best(
        self,
        candidates: list[GraspCandidate],
        *,
        robot_state: dict[str, float],
        target_object_mask: np.ndarray | None = None,
    ) -> GraspCandidate | None:
        del target_object_mask
        feasible: list[GraspCandidate] = []
        for candidate in candidates:
            if candidate.score < self.grasp_config.score_threshold:
                continue
            if candidate.width_m > self.grasp_config.max_gripper_width_m:
                continue
            if candidate.position_m[2] <= self.scene_config.table_top_z_m + self.scene_config.table_clearance_margin_m:
                continue
            if float(np.dot(candidate.approach_axis_world, np.array([0.0, 0.0, -1.0]))) < self.grasp_config.approach_preference_cosine:
                continue
            if not self._is_reachable(candidate.position_m, robot_state):
                continue
            feasible.append(candidate)
        feasible.sort(key=lambda item: item.score, reverse=True)
        return feasible[0] if feasible else None

    def _is_reachable(self, position_m: np.ndarray, robot_state: dict[str, float]) -> bool:
        del robot_state
        x, y, z = map(float, position_m)
        forward_distance = float(np.hypot(x, y))
        return 0.15 <= forward_distance <= 0.75 and 0.65 <= z <= 1.15

