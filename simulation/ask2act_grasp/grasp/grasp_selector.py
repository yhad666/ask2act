from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ask2act_grasp.stretch3_specs import CGN_GRIPPER_DEPTH_M, STRETCH3_JOINT_LIMITS
from ask2act_grasp.types import GraspCandidate, GraspConfig, SceneConfig


def classify_cup_approach(approach_axis_world: np.ndarray) -> str:
    approach_z = float(np.asarray(approach_axis_world, dtype=float)[2])
    if approach_z < -0.85:
        return "top_down"
    if approach_z < -0.3:
        return "angled"
    if abs(approach_z) < 0.3:
        return "side"
    return "angled"


class GraspStrategy(ABC):
    """Per-object-class grasp filtering strategy."""

    @abstractmethod
    def filter_candidates(
        self,
        candidates: list[GraspCandidate],
        table_z: float,
        object_center: np.ndarray | None = None,
        object_radius: float | None = None,
    ) -> list[GraspCandidate]:
        """Apply object-specific filters to grasp candidates."""

    @property
    def preferred_approach(self) -> str:
        """Return the preferred grasp family for downstream planning."""
        return "any"

    @property
    def needs_wrist_refinement(self) -> bool:
        """Whether the selected object class should trigger wrist-camera refinement."""
        return False

    @property
    def approach_tolerance_deg(self) -> float:
        """Expose a future tuning hook for approach-angle tolerance."""
        return 45.0


class CupStrategy(GraspStrategy):
    """Cup/mug: allow both top-down rim grasps and side grasps around the body."""

    @property
    def preferred_approach(self) -> str:
        return "any"

    def filter_candidates(
        self,
        candidates: list[GraspCandidate],
        table_z: float,
        object_center: np.ndarray | None = None,
        object_radius: float | None = None,
    ) -> list[GraspCandidate]:
        filtered: list[GraspCandidate] = []
        max_dist = float(object_radius if object_radius is not None else 0.035) * 2.0
        cup_height = 0.10
        for candidate in candidates:
            pos = np.asarray(candidate.position_m, dtype=float)
            approach = np.asarray(candidate.approach_axis_world, dtype=float)
            approach_z = float(approach[2])
            contact_pos = pos + CGN_GRIPPER_DEPTH_M * approach
            candidate.approach_type = classify_cup_approach(approach)

            if approach_z > 0.5:
                continue

            wrist_pitch = float(np.arcsin(np.clip(approach_z, -1.0, 1.0)))
            if wrist_pitch < STRETCH3_JOINT_LIMITS["wrist_pitch"][0] or wrist_pitch > STRETCH3_JOINT_LIMITS["wrist_pitch"][1]:
                continue

            if contact_pos[2] < table_z + 0.005:
                continue
            if contact_pos[2] > table_z + cup_height + 0.05:
                continue

            is_top_down = candidate.approach_type == "top_down"
            if is_top_down and contact_pos[2] < table_z + cup_height * 0.4:
                continue

            if float(candidate.width_m) > 0.085:
                continue
            if object_center is not None:
                xy_dist = float(np.linalg.norm(pos[:2] - np.asarray(object_center, dtype=float).reshape(2)))
                if xy_dist > max_dist:
                    continue
                candidate.metadata["distance_to_object_center_xy_m"] = xy_dist
            candidate.metadata["approach_type"] = candidate.approach_type
            filtered.append(candidate)

        filtered.sort(key=lambda item: float(item.score), reverse=True)
        return filtered


class BottleStrategy(GraspStrategy):
    """Bottle placeholder: keep only broadly confident candidates for now."""

    @property
    def preferred_approach(self) -> str:
        return "side"

    def filter_candidates(
        self,
        candidates: list[GraspCandidate],
        table_z: float,
        object_center: np.ndarray | None = None,
        object_radius: float | None = None,
    ) -> list[GraspCandidate]:
        del table_z, object_center, object_radius
        return [candidate for candidate in candidates if float(candidate.score) > 0.3]


class PlateStrategy(GraspStrategy):
    """Plate placeholder: edge pinch will eventually need precise wrist refinement."""

    @property
    def preferred_approach(self) -> str:
        return "side"

    @property
    def needs_wrist_refinement(self) -> bool:
        return True

    def filter_candidates(
        self,
        candidates: list[GraspCandidate],
        table_z: float,
        object_center: np.ndarray | None = None,
        object_radius: float | None = None,
    ) -> list[GraspCandidate]:
        del table_z, object_center, object_radius
        return [candidate for candidate in candidates if float(candidate.score) > 0.3]


class SpoonStrategy(GraspStrategy):
    """Spoon placeholder: handle grasp will later rely on wrist-camera precision."""

    @property
    def preferred_approach(self) -> str:
        return "top"

    @property
    def needs_wrist_refinement(self) -> bool:
        return True

    def filter_candidates(
        self,
        candidates: list[GraspCandidate],
        table_z: float,
        object_center: np.ndarray | None = None,
        object_radius: float | None = None,
    ) -> list[GraspCandidate]:
        del table_z, object_center, object_radius
        return [
            candidate
            for candidate in candidates
            if float(candidate.score) > 0.3 and float(candidate.width_m) < 0.03
        ]


class ForkStrategy(SpoonStrategy):
    """Fork behaves like spoon for the current placeholder logic."""


STRATEGY_MAP: dict[str, GraspStrategy] = {
    "cup": CupStrategy(),
    "bottle": BottleStrategy(),
    "plate": PlateStrategy(),
    "spoon": SpoonStrategy(),
    "fork": ForkStrategy(),
}


class GraspSelector:
    """Common grasp filtering plus per-object strategy dispatch."""

    def __init__(self, scene_config: SceneConfig, grasp_config: GraspConfig) -> None:
        self.scene_config = scene_config
        self.grasp_config = grasp_config
        self.strategies = STRATEGY_MAP

    def select(
        self,
        candidates: list[GraspCandidate],
        table_z: float,
        robot_state: dict[str, float],
        target_object_mask: np.ndarray | None = None,
        object_class: str = "cup",
        object_center: np.ndarray | None = None,
        object_radius: float | None = None,
    ) -> GraspCandidate | None:
        """Run common filtering, then object-specific filtering, and return the best grasp."""
        del target_object_mask
        common_filtered = self._common_filter(
            candidates,
            table_z=table_z,
            robot_state=robot_state,
            object_center=object_center,
            object_radius=object_radius,
        )

        strategy = self.strategies.get(object_class, self.strategies["cup"])
        specific_filtered = strategy.filter_candidates(
            common_filtered,
            table_z=table_z,
            object_center=object_center,
            object_radius=object_radius,
        )
        selection_pool = specific_filtered
        if not specific_filtered and common_filtered:
            print(
                f"WARNING: No valid grasps for {object_class}; "
                "falling back to object-aware common-filtered ranking."
            )
            selection_pool = common_filtered

        if not selection_pool:
            return None

        best = self._choose_best_candidate(
            selection_pool,
            object_class=object_class,
            strategy_filtered=bool(specific_filtered),
            table_z=table_z,
            object_center=object_center,
        )
        best.preferred_approach = strategy.preferred_approach
        best.needs_wrist_refinement = strategy.needs_wrist_refinement
        if object_class == "cup" and best.approach_type == "unknown":
            best.approach_type = classify_cup_approach(best.approach_axis_world)
        best.metadata = {
            **best.metadata,
            "object_class": object_class,
            "preferred_approach": strategy.preferred_approach,
            "needs_wrist_refinement": strategy.needs_wrist_refinement,
            "strategy_name": strategy.__class__.__name__,
            "used_strategy_filtered_pool": bool(specific_filtered),
            "approach_type": best.approach_type,
        }
        if object_center is not None:
            center_xy = np.asarray(object_center, dtype=float).reshape(2)
            best.metadata["distance_to_object_center_xy_m"] = float(np.linalg.norm(best.position_m[:2] - center_xy))
        return best

    def select_best(
        self,
        candidates: list[GraspCandidate],
        *,
        robot_state: dict[str, float],
        target_object_mask: np.ndarray | None = None,
        object_class: str = "cup",
        object_center: np.ndarray | None = None,
        object_radius: float | None = None,
    ) -> GraspCandidate | None:
        """Backward-compatible alias used by existing tests and executor code."""
        return self.select(
            candidates,
            table_z=self.scene_config.table_top_z_m,
            robot_state=robot_state,
            target_object_mask=target_object_mask,
            object_class=object_class,
            object_center=object_center,
            object_radius=object_radius,
        )

    def _common_filter(
        self,
        candidates: list[GraspCandidate],
        *,
        table_z: float,
        robot_state: dict[str, float],
        object_center: np.ndarray | None = None,
        object_radius: float | None = None,
        max_distance_multiplier: float = 3.0,
    ) -> list[GraspCandidate]:
        """Filters that apply to every object class before strategy dispatch."""
        min_score = min(float(self.grasp_config.score_threshold), 0.15)
        center_xy = None if object_center is None else np.asarray(object_center, dtype=float).reshape(2)
        max_dist = (
            float(object_radius) * float(max_distance_multiplier)
            if object_radius is not None
            else 0.08
        )
        filtered: list[GraspCandidate] = []
        for candidate in candidates:
            pos = np.asarray(candidate.position_m, dtype=float)
            if float(candidate.score) <= min_score:
                continue
            if pos[2] <= table_z - 0.02:
                continue
            if pos[2] >= 1.05:
                continue
            if float(candidate.width_m) >= 0.09:
                continue
            if not self._is_reachable(pos, robot_state):
                continue
            if center_xy is not None:
                xy_dist = float(np.linalg.norm(pos[:2] - center_xy))
                if xy_dist > max_dist:
                    continue
            filtered.append(candidate)

        filtered.sort(key=lambda item: float(item.score), reverse=True)
        return filtered

    def _is_reachable(self, position_m: np.ndarray, robot_state: dict[str, float]) -> bool:
        """Apply a rough Stretch reachability test for lift and arm workspace."""
        del robot_state
        x, y, z = map(float, position_m)
        forward_distance = max(-y, 0.0)
        lateral_offset = abs(x)
        return (
            0.10 <= forward_distance <= 0.80
            and lateral_offset <= 0.25
            and 0.0 <= z <= 1.05
        )

    def _choose_best_candidate(
        self,
        candidates: list[GraspCandidate],
        *,
        object_class: str,
        strategy_filtered: bool,
        table_z: float,
        object_center: np.ndarray | None,
    ) -> GraspCandidate:
        """Choose the final grasp from either the strict or fallback candidate pool."""
        if strategy_filtered or object_class != "cup":
            return max(candidates, key=lambda item: float(item.score))
        return min(
            candidates,
            key=lambda item: self._cup_fallback_rank(
                item,
                table_z=table_z,
                object_center=object_center,
            ),
        )

    @staticmethod
    def _cup_fallback_rank(
        candidate: GraspCandidate,
        *,
        table_z: float,
        object_center: np.ndarray | None,
    ) -> tuple[float, float, float, float]:
        """Prefer near-center, side-like, mid-height cup grasps when strict cup filters are empty."""
        pos = np.asarray(candidate.position_m, dtype=float)
        approach = np.asarray(candidate.approach_axis_world, dtype=float)
        center_xy = pos[:2] if object_center is None else np.asarray(object_center, dtype=float).reshape(2)
        xy_dist = float(np.linalg.norm(pos[:2] - center_xy))
        horizontal_penalty = abs(float(approach[2]))
        target_height = table_z + 0.065
        height_penalty = abs(float(pos[2]) - target_height)
        score_penalty = -float(candidate.score)
        return (xy_dist, horizontal_penalty, height_penalty, score_penalty)
