from __future__ import annotations

from pathlib import Path

import yaml

from ask2act_grasp.types import GraspConfig, HeadAlignmentConfig, SceneConfig


def _as_tuple(data, length: int) -> tuple[float, ...]:
    values = tuple(float(v) for v in data)
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {len(values)}")
    return values


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_scene_config(path: str | Path) -> tuple[SceneConfig, HeadAlignmentConfig]:
    payload = load_yaml(path)
    scene = payload["scene"]
    head = payload["head_alignment"]
    return (
        SceneConfig(
            table_position_m=_as_tuple(scene["table_position_m"], 3),
            table_size_m=_as_tuple(scene["table_size_m"], 3),
            cup_position_m=_as_tuple(scene["cup_position_m"], 3),
            cup_radius_m=float(scene["cup_radius_m"]),
            cup_height_m=float(scene["cup_height_m"]),
            cup_mass_kg=float(scene["cup_mass_kg"]),
            cup_friction=_as_tuple(scene["cup_friction"], 3),
            robot_start_translation_m=_as_tuple(scene["robot_start_translation_m"], 3),
            robot_start_rotation_quat_xyzw=_as_tuple(scene["robot_start_rotation_quat_xyzw"], 4),
            cup_randomization_range_m=_as_tuple(scene["cup_randomization_range_m"], 2),
            table_clearance_margin_m=float(scene["table_clearance_margin_m"]),
        ),
        HeadAlignmentConfig(
            head_pan_rad=float(head["head_pan_rad"]),
            head_tilt_rad=float(head["head_tilt_rad"]),
            settle_seconds=float(head["settle_seconds"]),
            camera_hz=float(head["camera_hz"]),
        ),
    )


def load_grasp_config(path: str | Path) -> GraspConfig:
    payload = load_yaml(path)
    grasp = payload["grasp"]
    planner = payload["planner"]
    return GraspConfig(
        score_threshold=float(grasp["score_threshold"]),
        forward_passes=int(grasp["forward_passes"]),
        z_min_m=float(grasp["z_range_m"][0]),
        z_max_m=float(grasp["z_range_m"][1]),
        pregrasp_offset_m=float(grasp["pregrasp_offset_m"]),
        top_down_grasp_pitch_rad=float(grasp["top_down_grasp_pitch_rad"]),
        max_gripper_width_m=float(grasp["max_gripper_width_m"]),
        approach_preference_cosine=float(grasp["approach_preference_cosine"]),
        contact_graspnet_repo=str(grasp["contact_graspnet_repo"]),
        contact_graspnet_checkpoint=grasp.get("contact_graspnet_checkpoint"),
        enable_contact_graspnet=bool(grasp["enable_contact_graspnet"]),
        enable_fallback_generator=bool(grasp["enable_fallback_generator"]),
        planner_backend=str(planner["backend"]),
    )

