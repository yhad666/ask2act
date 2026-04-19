from __future__ import annotations

from pathlib import Path

import yaml

from ask2act_grasp.types import GraspConfig, HeadAlignmentConfig, SceneConfig


def _as_tuple(data, length: int) -> tuple[float, ...]:
    values = tuple(float(v) for v in data)
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {len(values)}")
    return values


def _as_int_tuple(data, length: int) -> tuple[int, ...]:
    values = tuple(int(v) for v in data)
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
            grasp_method=str(payload.get("grasp_method", scene.get("grasp_method", "oracle"))).lower(),
            table_position_m=_as_tuple(scene["table_position_m"], 3),
            table_size_m=_as_tuple(scene["table_size_m"], 3),
            cup_position_m=_as_tuple(scene["cup_position_m"], 3),
            cup_body_name=str(scene.get("cup_body_name", "target_cup")),
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
        d435i_render_resolution_px=_as_int_tuple(grasp.get("d435i_render_resolution_px", [424, 240]), 2),
        d435i_sensor_resolution_px=_as_int_tuple(grasp.get("d435i_sensor_resolution_px", [1920, 1080]), 2),
        z_min_m=float(grasp["z_range_m"][0]),
        z_max_m=float(grasp["z_range_m"][1]),
        cgn_crop_radius_m=float(grasp.get("cgn_crop_radius", 0.15)),
        cgn_subsample_n=int(grasp.get("cgn_subsample_n", 2048)),
        add_depth_noise=bool(grasp.get("add_depth_noise", True)),
        depth_noise_sigma_m=float(grasp.get("depth_noise_sigma", 0.001)),
        depth_dropout_ratio=float(grasp.get("depth_dropout_ratio", 0.01)),
        pregrasp_offset_m=float(grasp["pregrasp_offset_m"]),
        top_down_grasp_pitch_rad=float(grasp["top_down_grasp_pitch_rad"]),
        max_gripper_width_m=float(grasp["max_gripper_width_m"]),
        approach_preference_cosine=float(grasp["approach_preference_cosine"]),
        contact_graspnet_repo=str(grasp["contact_graspnet_repo"]),
        contact_graspnet_checkpoint=grasp.get("contact_graspnet_checkpoint"),
        contact_graspnet_python=grasp.get("contact_graspnet_python"),
        cgn_device=str(grasp.get("cgn_device", "cuda")),
        enable_contact_graspnet=bool(grasp["enable_contact_graspnet"]),
        enable_fallback_generator=bool(grasp["enable_fallback_generator"]),
        oracle_grasp_height_ratio=float(grasp.get("oracle_grasp_height_ratio", 0.42)),
        oracle_approach_height_offset_m=float(grasp.get("oracle_approach_height_offset_m", 0.02)),
        oracle_postgrasp_lift_delta_m=float(grasp.get("oracle_postgrasp_lift_delta_m", 0.10)),
        oracle_side_grasp_wrist_pitch_rad=float(grasp.get("oracle_side_grasp_wrist_pitch_rad", 0.02)),
        oracle_arm_backoff_m=float(grasp.get("oracle_arm_backoff_m", 0.22)),
        oracle_final_arm_delta_m=float(grasp.get("oracle_final_arm_delta_m", 0.02)),
        oracle_tucked_wrist_yaw_rad=float(grasp.get("oracle_tucked_wrist_yaw_rad", 1.20)),
        oracle_side_open_width_cmd=float(grasp.get("oracle_side_open_width_cmd", 0.52)),
        oracle_lateral_offset_m=float(grasp.get("oracle_lateral_offset_m", 0.0)),
        oracle_forward_offset_m=float(grasp.get("oracle_forward_offset_m", 0.0)),
        cgn_execution_mode=str(grasp.get("cgn_execution_mode", "guided_oracle")),
        use_simple_ik_for_topdown=bool(grasp.get("use_simple_ik_for_topdown", True)),
        allow_approximate_topdown_fallback=bool(grasp.get("allow_approximate_topdown_fallback", False)),
        enable_base_preposition=bool(grasp.get("enable_base_preposition", True)),
        base_preposition_trigger_distance_m=float(grasp.get("base_preposition_trigger_distance_m", 0.62)),
        base_preposition_goal_distance_m=float(grasp.get("base_preposition_goal_distance_m", 0.54)),
        base_preposition_max_translate_m=float(grasp.get("base_preposition_max_translate_m", 0.40)),
        base_preposition_heading_tolerance_rad=float(grasp.get("base_preposition_heading_tolerance_rad", 0.10)),
        base_preposition_longitudinal_extra_m=float(grasp.get("base_preposition_longitudinal_extra_m", 0.07)),
        base_preposition_table_clearance_m=float(grasp.get("base_preposition_table_clearance_m", 0.12)),
        base_preposition_rotate_clearance_m=float(grasp.get("base_preposition_rotate_clearance_m", 0.26)),
        base_preposition_lidar_backoff_step_m=float(grasp.get("base_preposition_lidar_backoff_step_m", 0.04)),
        base_preposition_lidar_max_backoff_m=float(grasp.get("base_preposition_lidar_max_backoff_m", 0.16)),
        planner_backend=str(planner["backend"]),
    )
