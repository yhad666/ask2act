from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SceneConfig:
    table_position_m: tuple[float, float, float]
    table_size_m: tuple[float, float, float]
    cup_position_m: tuple[float, float, float]
    cup_radius_m: float
    cup_height_m: float
    cup_mass_kg: float
    cup_friction: tuple[float, float, float]
    robot_start_translation_m: tuple[float, float, float]
    robot_start_rotation_quat_xyzw: tuple[float, float, float, float]
    cup_randomization_range_m: tuple[float, float]
    table_clearance_margin_m: float

    @property
    def table_top_z_m(self) -> float:
        return float(self.table_position_m[2] + self.table_size_m[2])


@dataclass
class HeadAlignmentConfig:
    head_pan_rad: float
    head_tilt_rad: float
    settle_seconds: float
    camera_hz: float


@dataclass
class GraspConfig:
    score_threshold: float
    forward_passes: int
    z_min_m: float
    z_max_m: float
    pregrasp_offset_m: float
    top_down_grasp_pitch_rad: float
    max_gripper_width_m: float
    approach_preference_cosine: float
    contact_graspnet_repo: str
    contact_graspnet_checkpoint: str | None
    enable_contact_graspnet: bool
    enable_fallback_generator: bool
    planner_backend: str


@dataclass
class HeadObservation:
    rgb_image: np.ndarray
    depth_image: np.ndarray
    camera_intrinsics: np.ndarray
    camera_extrinsics: np.ndarray
    camera_source: str
    capture_time_s: float


@dataclass
class PointCloudResult:
    world_points_xyz: np.ndarray
    camera_points_xyz: np.ndarray
    applied_bbox_2d: tuple[int, int, int, int] | None
    filtered_point_count: int


@dataclass
class GraspCandidate:
    pose_4x4: np.ndarray
    score: float
    width_m: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def position_m(self) -> np.ndarray:
        return np.asarray(self.pose_4x4[:3, 3], dtype=float)

    @property
    def approach_axis_world(self) -> np.ndarray:
        return np.asarray(self.pose_4x4[:3, 2], dtype=float)


@dataclass
class MotionWaypoint:
    name: str
    joint_targets: dict[str, float]


@dataclass
class MotionPlan:
    backend: str
    waypoints: list[MotionWaypoint]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    workspace_root: Path
    simulation_root: Path
    run_dir: Path
    scene_xml_path: Path
    scene_config: SceneConfig
    grasp_config: GraspConfig
    head_config: HeadAlignmentConfig


@dataclass
class PipelineResult:
    success: bool
    scene_xml_path: str
    point_cloud_count: int
    selected_grasp_score: float | None
    planner_backend: str
    trajectory: list[dict[str, Any]]
    intermediate: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

