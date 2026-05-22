from __future__ import annotations

import importlib.util
from pathlib import Path
import types
import sys
import numpy as np
import pytest

from ask2act_grasp.grasp.geometric_grasp import (
    compute_geometric_grasp,
    estimate_object_center,
    fit_circle_2d,
    fit_circle_with_confidence,
    find_minimum_grip_direction,
    save_geometric_grasp_debug,
    validate_grasp_point,
)
from ask2act_grasp.grasp.grasp_generator import ContactGraspNetWrapper
from ask2act_grasp.grasp.grasp_selector import GraspSelector, classify_cup_approach
from ask2act_grasp.stretch3_specs import (
    CGN_GRIPPER_DEPTH_M,
    TOPDOWN_SIMPLEIK_EXECUTION_Z_BIAS_M,
    cgn_frame_to_wrist_local_m,
    gripper_width_to_command,
    rotate_topdown_offset_to_world_m,
    topdown_grasp_center_to_rubber_offset_m,
    topdown_simpleik_wrist_model_error_m,
    topdown_wrist_to_grasp_center_offset_m,
    topdown_wrist_to_rubber_offset_m,
    wrist_to_cgn_grasp_frame_offset_m,
)
from ask2act_grasp.perception.point_cloud_gen import PointCloudGenerator
from ask2act_grasp.perception.tabletop_geometry import (
    estimate_tabletop_object_geometry,
    intersect_pixels_with_plane,
    solve_depth_scale_shift_from_table,
)
from ask2act_grasp.planning.motion_planner import MotionPlanner
from ask2act_grasp.scene.scene_setup import SceneSetup
from ask2act_grasp.types import GraspCandidate
from ask2act_grasp.utils.config_loader import load_grasp_config, load_scene_config
from ask2act_grasp.utils.head_camera_orientation import rotate_head_observation_clockwise
from ask2act_grasp.utils.pipeline_overrides import apply_scene_overrides
from ask2act_grasp.utils.tf_utils import rotation_matrix_from_rpy


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SIMULATION_ROOT = PACKAGE_ROOT.parent
STRETCH_MUJOCO_SRC = SIMULATION_ROOT / "stretch_mujoco"


def _ensure_stretch_mujoco_on_path() -> None:
    stretch_path = str(STRETCH_MUJOCO_SRC)
    if stretch_path not in sys.path:
        sys.path.append(stretch_path)


def _install_fake_stretch_mujoco_modules(monkeypatch) -> None:
    fake_pkg = types.ModuleType("stretch_mujoco")
    fake_enums_pkg = types.ModuleType("stretch_mujoco.enums")
    fake_actuators = types.ModuleType("stretch_mujoco.enums.actuators")
    fake_cameras = types.ModuleType("stretch_mujoco.enums.stretch_cameras")
    fake_sensors = types.ModuleType("stretch_mujoco.enums.stretch_sensors")

    fake_actuators.Actuators = types.SimpleNamespace(
        head_pan="head_pan",
        head_tilt="head_tilt",
        base_rotate="base_rotate",
        base_translate="base_translate",
        lift="lift",
        arm="arm",
        wrist_yaw="wrist_yaw",
        wrist_pitch="wrist_pitch",
        wrist_roll="wrist_roll",
        gripper="gripper",
    )
    fake_cameras.StretchCameras = types.SimpleNamespace(
        cam_d435i_rgb=types.SimpleNamespace(
            camera_name_in_mjcf="cam_d435i_rgb",
            initial_camera_settings=types.SimpleNamespace(field_of_view_vertical_in_degrees=42),
        )
    )
    fake_sensors.StretchSensors = types.SimpleNamespace(base_lidar="base_lidar")

    monkeypatch.setitem(sys.modules, "stretch_mujoco", fake_pkg)
    monkeypatch.setitem(sys.modules, "stretch_mujoco.enums", fake_enums_pkg)
    monkeypatch.setitem(sys.modules, "stretch_mujoco.enums.actuators", fake_actuators)
    monkeypatch.setitem(sys.modules, "stretch_mujoco.enums.stretch_cameras", fake_cameras)
    monkeypatch.setitem(sys.modules, "stretch_mujoco.enums.stretch_sensors", fake_sensors)


def _make_synthetic_tilted_camera_extrinsics() -> np.ndarray:
    extrinsics = np.eye(4, dtype=float)
    extrinsics[:3, :3] = rotation_matrix_from_rpy(np.deg2rad(120.0), 0.0, 0.0)
    extrinsics[:3, 3] = np.array([0.0, -0.15, 1.10], dtype=float)
    return extrinsics


def _plane_depth_for_pixels(
    pixels_uv: np.ndarray,
    *,
    camera_intrinsics: np.ndarray,
    camera_extrinsics: np.ndarray,
    table_top_z_m: float,
    height_above_table_m: float = 0.0,
) -> np.ndarray:
    plane_normal = np.array([0.0, 0.0, 1.0], dtype=float)
    plane_offset = -(float(table_top_z_m) + float(height_above_table_m))
    pixels = np.asarray(pixels_uv, dtype=float).reshape(-1, 2)
    k_inv = np.linalg.inv(np.asarray(camera_intrinsics, dtype=float))
    rays_camera = (k_inv @ np.concatenate([pixels, np.ones((len(pixels), 1), dtype=float)], axis=1).T).T
    rays_world = (np.asarray(camera_extrinsics[:3, :3], dtype=float) @ rays_camera.T).T
    camera_center = np.asarray(camera_extrinsics[:3, 3], dtype=float)
    lambdas = -((camera_center @ plane_normal) + plane_offset) / (rays_world @ plane_normal)
    return lambdas.astype(np.float64, copy=False)


def test_scene_setup_writes_runtime_xml(tmp_path):
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    scene_path = SceneSetup(scene_config).write_scene(tmp_path / "scene.xml")
    text = scene_path.read_text()
    assert "target_cup" in text
    assert "freejoint" in text


def test_scene_setup_writes_multi_cup_runtime_xml(tmp_path):
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    scene_path = SceneSetup(scene_config).write_multi_cup_scene(
        tmp_path / "multi_scene.xml",
        cup_configs=[
            {"name": "target_cup", "position": [0.0, -0.60, 0.86], "color": [0.82, 0.2, 0.16, 1.0], "is_target": True},
            {"name": "cup_01", "position": [0.12, -0.70, 0.86], "color": [0.18, 0.38, 0.82, 1.0], "is_target": False},
        ],
    )
    text = scene_path.read_text()
    assert "target_cup" in text
    assert "cup_01" in text
    assert text.count("freejoint") >= 2


def test_scene_config_loads_cup_body_name():
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    assert scene_config.cup_body_name == "target_cup"


def test_grasp_config_loads_simple_ik_toggle():
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    assert grasp_config.use_simple_ik_for_topdown is True


def test_grasp_config_loads_d435i_resolutions():
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    assert grasp_config.d435i_render_resolution_px == (1280, 720)
    assert grasp_config.d435i_sensor_resolution_px == (1280, 720)


def test_d435i_camera_settings_respect_env_resolution_overrides(monkeypatch):
    fake_pkg = types.ModuleType("stretch_mujoco")
    fake_config = types.ModuleType("stretch_mujoco.config")
    fake_utils = types.ModuleType("stretch_mujoco.utils")
    fake_config.depth_limits = {"d405": (0.0, 1.0), "d435i": (0.0, 1.0)}
    fake_utils.limit_depth_distance = lambda render, _limits: render
    monkeypatch.setitem(sys.modules, "stretch_mujoco", fake_pkg)
    monkeypatch.setitem(sys.modules, "stretch_mujoco.config", fake_config)
    monkeypatch.setitem(sys.modules, "stretch_mujoco.utils", fake_utils)

    module_path = (
        STRETCH_MUJOCO_SRC / "stretch_mujoco" / "enums" / "stretch_cameras.py"
    )
    spec = importlib.util.spec_from_file_location("stretch_cameras_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    StretchCameras = module.StretchCameras

    monkeypatch.setenv("ASK2ACT_D435I_RENDER_WIDTH", "1280")
    monkeypatch.setenv("ASK2ACT_D435I_RENDER_HEIGHT", "720")
    monkeypatch.setenv("ASK2ACT_D435I_SENSOR_WIDTH", "1280")
    monkeypatch.setenv("ASK2ACT_D435I_SENSOR_HEIGHT", "720")

    settings = StretchCameras.cam_d435i_rgb.initial_camera_settings
    assert settings.width == 1280
    assert settings.height == 720
    assert settings.sensor_resolution == (1280, 720)
    assert settings.focal[0] == pytest.approx(304.24 * (1280.0 / 424.0), rel=1e-6)
    assert settings.focal[1] == pytest.approx(304.07 * (720.0 / 240.0), rel=1e-6)


def test_point_cloud_generation_filters_below_table():
    generator = PointCloudGenerator()
    depth = np.array([[0.82, 0.84], [0.88, 0.0]], dtype=float)
    k = np.array([[100.0, 0.0, 0.5], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]])
    result = generator.generate(
        depth_image=depth,
        camera_intrinsics=k,
        camera_extrinsics=np.eye(4),
        table_top_z_m=0.81,
        table_margin_m=0.01,
        z_min_m=0.2,
        z_max_m=1.1,
    )
    assert result.filtered_point_count == 2


def test_point_cloud_crop_to_object_region_reduces_scene():
    generator = PointCloudGenerator()
    points = np.array(
        [
            [0.00, -0.60, 0.82],
            [0.01, -0.61, 0.83],
            [-0.01, -0.59, 0.81],
            [0.40, -0.20, 0.84],
        ],
        dtype=float,
    )
    cropped, mask, center_xy = generator.crop_to_object_region(
        points,
        object_center_xy=np.array([0.0, -0.60], dtype=float),
        crop_radius=0.08,
        table_z=0.78,
        return_mask=True,
    )
    assert cropped.shape[0] == 3
    assert mask.tolist() == [True, True, True, False]
    assert np.allclose(center_xy, np.array([0.0, -0.60]), atol=0.02)


def test_point_cloud_crop_to_object_region_uses_elevated_points_for_center():
    generator = PointCloudGenerator()
    points = np.array(
        [
            [0.00, -0.60, 0.86],
            [0.01, -0.59, 0.87],
            [-0.01, -0.61, 0.85],
            [0.25, -0.85, 0.79],
            [0.28, -0.88, 0.79],
            [0.22, -0.83, 0.79],
        ],
        dtype=float,
    )
    cropped, mask, center_xy = generator.crop_to_object_region(
        points,
        crop_radius=0.08,
        table_z=0.76,
        return_mask=True,
    )
    assert cropped.shape[0] == 3
    assert mask.tolist() == [True, True, True, False, False, False]
    assert np.allclose(center_xy, np.array([0.0, -0.60]), atol=0.03)


def test_point_cloud_crop_to_object_region_applies_height_band():
    generator = PointCloudGenerator()
    points = np.array(
        [
            [0.00, -0.60, 0.86],
            [0.01, -0.59, 0.85],
            [-0.01, -0.61, 0.78],
            [0.00, -0.62, 0.905],
        ],
        dtype=float,
    )
    cropped, mask, _ = generator.crop_to_object_region(
        points,
        object_center_xy=np.array([0.0, -0.60], dtype=float),
        crop_radius=0.08,
        table_z=0.76,
        height_min=0.80,
        height_max=0.89,
        return_mask=True,
    )
    assert cropped.shape[0] == 2
    assert mask.tolist() == [True, True, False, False]


def test_ray_plane_intersection_lands_on_table_plane():
    intrinsics = np.array([[120.0, 0.0, 80.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]], dtype=float)
    extrinsics = _make_synthetic_tilted_camera_extrinsics()
    plane_normal = np.array([0.0, 0.0, 1.0], dtype=float)
    plane_offset = -0.76
    pixels = np.array([[80.0, 70.0], [92.0, 74.0]], dtype=float)

    intersections, valid = intersect_pixels_with_plane(
        pixels,
        camera_intrinsics=intrinsics,
        camera_extrinsics=extrinsics,
        plane_normal=plane_normal,
        plane_offset=plane_offset,
    )

    assert valid.tolist() == [True, True]
    assert np.allclose(intersections[:, 2], 0.76, atol=1e-6)


def test_depth_scale_shift_is_recovered_from_synthetic_table_pixels():
    intrinsics = np.array([[120.0, 0.0, 80.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]], dtype=float)
    extrinsics = _make_synthetic_tilted_camera_extrinsics()
    table_top_z_m = 0.76
    image_shape = (120, 160)

    v_coords, u_coords = np.meshgrid(np.arange(image_shape[0]), np.arange(image_shape[1]), indexing="ij")
    pixels = np.stack([u_coords.reshape(-1), v_coords.reshape(-1)], axis=1)
    true_depth = _plane_depth_for_pixels(
        pixels,
        camera_intrinsics=intrinsics,
        camera_extrinsics=extrinsics,
        table_top_z_m=table_top_z_m,
    ).reshape(image_shape)
    depth_scale = 1.18
    depth_shift = -0.14
    predicted_depth = (true_depth - depth_shift) / depth_scale

    solution = solve_depth_scale_shift_from_table(
        depth_image=predicted_depth,
        camera_intrinsics=intrinsics,
        camera_extrinsics=extrinsics,
        plane_normal=np.array([0.0, 0.0, 1.0], dtype=float),
        plane_offset=-table_top_z_m,
    )

    assert solution["valid"] is True
    assert solution["model"] == "scale_shift"
    assert solution["a"] == pytest.approx(depth_scale, abs=5e-3)
    assert solution["b"] == pytest.approx(depth_shift, abs=5e-3)


def test_tabletop_geometry_estimates_height_from_bbox_support_pixels():
    intrinsics = np.array([[120.0, 0.0, 80.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]], dtype=float)
    extrinsics = _make_synthetic_tilted_camera_extrinsics()
    table_top_z_m = 0.76
    image_shape = (120, 160)
    depth = np.full(image_shape, np.nan, dtype=np.float64)

    all_pixels = np.stack(np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]), indexing="xy"), axis=-1)
    flat_pixels = all_pixels.reshape(-1, 2)
    depth[:] = _plane_depth_for_pixels(
        flat_pixels,
        camera_intrinsics=intrinsics,
        camera_extrinsics=extrinsics,
        table_top_z_m=table_top_z_m,
    ).reshape(image_shape)

    bbox = (72, 38, 90, 74)
    object_height_m = 0.11
    for v in range(bbox[1], bbox[3]):
        row_ratio = (bbox[3] - 1 - v) / max(1, (bbox[3] - bbox[1] - 1))
        row_height = max(0.0, object_height_m * row_ratio)
        row_pixels = np.stack(
            [np.arange(bbox[0], bbox[2], dtype=np.float64), np.full((bbox[2] - bbox[0],), float(v), dtype=np.float64)],
            axis=1,
        )
        depth[v, bbox[0]:bbox[2]] = _plane_depth_for_pixels(
            row_pixels,
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
            table_top_z_m=table_top_z_m,
            height_above_table_m=row_height,
        )

    geometry = estimate_tabletop_object_geometry(
        depth_image=depth,
        camera_intrinsics=intrinsics,
        camera_extrinsics=extrinsics,
        table_top_z_m=table_top_z_m,
        object_bbox=bbox,
    )

    assert geometry["quality"]["need_reobserve"] is False
    assert geometry["height_m"] == pytest.approx(object_height_m, abs=0.02)
    assert geometry["base_center_world"][2] == pytest.approx(table_top_z_m)
    assert geometry["object_center_world"][2] == pytest.approx(table_top_z_m + 0.5 * geometry["height_m"], abs=1e-6)


def test_tabletop_geometry_primary_estimator_does_not_depend_on_elevated_point_median():
    intrinsics = np.array([[120.0, 0.0, 80.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]], dtype=float)
    extrinsics = _make_synthetic_tilted_camera_extrinsics()
    table_top_z_m = 0.76
    image_shape = (120, 160)
    depth = np.full(image_shape, np.nan, dtype=np.float64)
    all_pixels = np.stack(np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]), indexing="xy"), axis=-1)
    depth[:] = _plane_depth_for_pixels(
        all_pixels.reshape(-1, 2),
        camera_intrinsics=intrinsics,
        camera_extrinsics=extrinsics,
        table_top_z_m=table_top_z_m,
    ).reshape(image_shape)
    bbox = (70, 42, 92, 78)
    depth[bbox[1]:bbox[3], bbox[0]:bbox[2]] *= 0.97

    geometry = estimate_tabletop_object_geometry(
        depth_image=depth,
        camera_intrinsics=intrinsics,
        camera_extrinsics=extrinsics,
        table_top_z_m=table_top_z_m,
        object_bbox=bbox,
    )

    misleading_elevated_points = np.array(
        [
            [0.35, -0.25, 0.92],
            [0.33, -0.22, 0.91],
            [0.36, -0.21, 0.93],
        ],
        dtype=float,
    )
    misleading_center = np.median(misleading_elevated_points[:, :2], axis=0)

    assert np.linalg.norm(np.asarray(geometry["base_center_world"][:2], dtype=float) - misleading_center) > 0.10


def test_point_cloud_subsample_matches_target_count():
    generator = PointCloudGenerator()
    points = np.array([[0.0, -0.6, 0.82], [0.01, -0.61, 0.83]], dtype=float)
    subsampled, indices = generator.subsample(points, target_n=5, return_indices=True)
    assert subsampled.shape == (5, 3)
    assert indices.shape == (5,)


def test_point_cloud_sensor_noise_preserves_shape():
    generator = PointCloudGenerator()
    depth = np.full((4, 4), 0.8, dtype=np.float32)
    noisy = generator.add_sensor_noise(depth, noise_sigma=0.001, dropout_ratio=0.0)
    assert noisy.shape == depth.shape
    assert noisy.dtype == np.float32
    assert np.any(np.abs(noisy - depth) > 0.0)


def test_head_observation_rotation_clockwise_updates_shapes_and_intrinsics():
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    depth = np.arange(6, dtype=np.float32).reshape(2, 3)
    intrinsics = np.array([[10.0, 0.0, 1.5], [0.0, 20.0, 0.5], [0.0, 0.0, 1.0]], dtype=float)
    extrinsics = np.eye(4, dtype=float)

    rotated_rgb, rotated_depth, rotated_k, rotated_extrinsics = rotate_head_observation_clockwise(
        rgb,
        depth,
        intrinsics,
        extrinsics,
    )

    assert rotated_rgb.shape == (3, 2, 3)
    assert rotated_depth.shape == (3, 2)
    assert np.allclose(rotated_k, np.array([[20.0, 0.0, 0.5], [0.0, 10.0, 1.5], [0.0, 0.0, 1.0]]))
    assert np.allclose(
        rotated_extrinsics[:3, :3],
        np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
    )


def test_head_observation_rotation_preserves_world_points():
    generator = PointCloudGenerator()
    depth = np.array([[0.80, 0.82, 0.0], [0.78, 0.81, 0.79]], dtype=np.float32)
    intrinsics = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]], dtype=float)
    extrinsics = np.eye(4, dtype=float)
    base_points = generator.generate(
        depth_image=depth,
        camera_intrinsics=intrinsics,
        camera_extrinsics=extrinsics,
        table_top_z_m=0.0,
        table_margin_m=-1.0,
        z_min_m=0.0,
        z_max_m=2.0,
    ).world_points_xyz

    _, rotated_depth, rotated_k, rotated_extrinsics = rotate_head_observation_clockwise(
        np.zeros((2, 3, 3), dtype=np.uint8),
        depth,
        intrinsics,
        extrinsics,
    )
    rotated_points = generator.generate(
        depth_image=rotated_depth,
        camera_intrinsics=rotated_k,
        camera_extrinsics=rotated_extrinsics,
        table_top_z_m=0.0,
        table_margin_m=-1.0,
        z_min_m=0.0,
        z_max_m=2.0,
    ).world_points_xyz

    base_sorted = base_points[np.lexsort((base_points[:, 2], base_points[:, 1], base_points[:, 0]))]
    rotated_sorted = rotated_points[np.lexsort((rotated_points[:, 2], rotated_points[:, 1], rotated_points[:, 0]))]
    assert np.allclose(base_sorted, rotated_sorted, atol=1e-6)


def test_head_aligner_retries_until_settle_before_accepting_frame(monkeypatch):
    _install_fake_stretch_mujoco_modules(monkeypatch)
    from ask2act_grasp.perception.head_alignment import HeadAligner

    scene_config, head_config = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    del scene_config
    aligner = HeadAligner(head_config)

    settle_attempts: list[int] = []
    capture_attempts: list[int] = []

    def fake_command_head_pose(sim, *, head_pan_rad, head_tilt_rad) -> None:
        del sim
        del head_pan_rad
        del head_tilt_rad

    def fake_validate_head_settle(sim, *, head_pan_rad, head_tilt_rad) -> None:
        del sim
        del head_pan_rad
        del head_tilt_rad
        settle_attempts.append(len(settle_attempts))
        if len(settle_attempts) == 1:
            raise RuntimeError("head_tilt did not settle near target")

    class FakeCameraStatus:
        time = 1.23
        cam_d435i_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        cam_d435i_depth = np.ones((2, 2), dtype=np.float32)

    def fake_poll_valid_head_camera_data(sim):
        del sim
        capture_attempts.append(len(capture_attempts))
        return FakeCameraStatus()

    monkeypatch.setattr(aligner, "_command_head_pose", fake_command_head_pose)
    monkeypatch.setattr(aligner, "_validate_head_settle", fake_validate_head_settle)
    monkeypatch.setattr(aligner, "_poll_valid_head_camera_data", fake_poll_valid_head_camera_data)
    monkeypatch.setattr(
        aligner,
        "_resolve_head_camera_intrinsics",
        lambda *, depth_shape, fallback_k: (np.eye(3, dtype=float), "test_intrinsics"),
    )
    monkeypatch.setattr(
        aligner,
        "_resolve_head_camera_pose",
        lambda sim: (np.eye(4, dtype=float), "test_extrinsics"),
    )
    monkeypatch.setattr(
        "ask2act_grasp.perception.head_alignment.rotate_head_observation_clockwise",
        lambda rgb, depth, camera_intrinsics, camera_extrinsics: (rgb, depth, camera_intrinsics, camera_extrinsics),
    )
    monkeypatch.setattr("ask2act_grasp.perception.head_alignment.time.sleep", lambda *_args, **_kwargs: None)

    observation = aligner.align_and_capture(sim=object())

    assert len(settle_attempts) == 2
    assert len(capture_attempts) == 1
    assert observation.capture_time_s == pytest.approx(1.23)


def test_topdown_alignment_keeps_iterating_after_partial_move(monkeypatch):
    _install_fake_stretch_mujoco_modules(monkeypatch)
    from ask2act_grasp.execution.grasp_executor import GraspExecutor

    executor = GraspExecutor.__new__(GraspExecutor)
    executor.TOPDOWN_CONTACT_ALIGNMENT_ITERS = 3
    executor.TOPDOWN_CONTACT_ALIGNMENT_GAIN = 1.0
    executor.TOPDOWN_CONTACT_ALIGNMENT_XY_TOL_M = 0.008
    executor.TOPDOWN_CONTACT_ALIGNMENT_Z_TOL_M = 0.008
    executor.TOPDOWN_CONTACT_ALIGNMENT_MAX_XY_STEP_M = 0.03
    executor.TOPDOWN_CONTACT_ALIGNMENT_MAX_Z_STEP_M = 0.04
    executor.context = types.SimpleNamespace(
        grasp_config=types.SimpleNamespace(max_gripper_width_m=0.09),
    )

    class FakePlanner:
        simple_ik = object()

        @staticmethod
        def _solve_topdown_simple_ik_targets(**kwargs):
            corrected = np.asarray(kwargs["desired_rubber_xyz"], dtype=float)
            return {
                "ik_base_rotate": float(corrected[0]),
                "ik_arm": float(-corrected[1]),
                "ik_lift": float(corrected[2]),
            }

    executor.motion_planner = FakePlanner()

    measurements = iter(
        [
            {
                "target_contact_point": [0.0, -0.60, 0.84],
                "rubber_contact_center_delta_to_target": [0.03, 0.0, 0.04],
                "rigid_grasp_center_delta_to_target_grasp_center": [0.03, 0.0, 0.04],
                "rigid_grasp_center": [0.03, -0.60, 0.88],
                "target_grasp_center": [0.0, -0.60, 0.84],
            },
            {
                "target_contact_point": [0.0, -0.60, 0.84],
                "rubber_contact_center_delta_to_target": [0.005, 0.0, 0.003],
                "rigid_grasp_center_delta_to_target_grasp_center": [0.005, 0.0, 0.003],
                "rigid_grasp_center": [0.005, -0.60, 0.843],
                "target_grasp_center": [0.0, -0.60, 0.84],
            },
            {
                "target_contact_point": [0.0, -0.60, 0.84],
                "rubber_contact_center_delta_to_target": [0.005, 0.0, 0.003],
                "rigid_grasp_center_delta_to_target_grasp_center": [0.005, 0.0, 0.003],
                "rigid_grasp_center": [0.005, -0.60, 0.843],
                "target_grasp_center": [0.0, -0.60, 0.84],
            },
        ]
    )

    move_calls: list[tuple[float, float, float]] = []

    monkeypatch.setattr(
        executor,
        "_measure_topdown_contact_error",
        lambda sim, numeric_targets: next(measurements, None),
    )
    monkeypatch.setattr(executor, "_read_robot_state", lambda sim: {"lift": 0.9, "arm": 0.4, "wrist_yaw": 0.0})

    def fake_apply(sim, *, base_rotate, arm, lift):
        del sim
        move_calls.append((base_rotate, arm, lift))
        return len(move_calls) > 1

    monkeypatch.setattr(executor, "_apply_topdown_alignment_joint_targets", fake_apply)

    history = executor._run_topdown_contact_alignment(
        sim=object(),
        numeric_targets={
            "approach_type": "top_down",
            "ik_base_rotate": 0.1,
            "gripper_open_width": 0.08,
            "grip_angle_rad": 0.0,
            "planning_mode": "geometric_simple_ik",
            "target_contact_point": [0.0, -0.60, 0.84],
        },
    )

    assert len(move_calls) == 1
    assert history[0]["move_ok"] is False
    assert "move_warning" in history[0]
    assert history[1]["iteration"] == 1


def test_geometric_pipeline_defers_base_preposition_until_after_initial_observation(monkeypatch):
    _install_fake_stretch_mujoco_modules(monkeypatch)
    from ask2act_grasp.execution.grasp_executor import GraspExecutor

    executor = GraspExecutor.__new__(GraspExecutor)
    executor.context = types.SimpleNamespace(
        scene_config=types.SimpleNamespace(grasp_method="geometric"),
        grasp_config=types.SimpleNamespace(enable_base_preposition=True),
    )

    diagnostic = executor._maybe_preposition_base_for_target(sim=object())

    assert diagnostic == {
        "enabled": True,
        "performed": False,
        "reason": "deferred_until_after_initial_geometric_observation",
    }


def test_geometric_base_refine_uses_simple_lateral_then_longitudinal_policy(monkeypatch):
    _install_fake_stretch_mujoco_modules(monkeypatch)
    from ask2act_grasp.execution.grasp_executor import GraspExecutor

    executor = GraspExecutor.__new__(GraspExecutor)
    executor.motion_planner = types.SimpleNamespace(
        simple_ik=object(),
        geometric_grasp_targets=lambda *_args, **_kwargs: {"ik_base_rotate": 0.0},
    )
    executor.context = types.SimpleNamespace(
        scene_config=types.SimpleNamespace(grasp_method="geometric"),
        grasp_config=types.SimpleNamespace(
            enable_base_preposition=True,
            base_preposition_goal_distance_m=0.54,
            base_preposition_max_translate_m=0.40,
            base_preposition_longitudinal_extra_m=0.07,
            base_preposition_rotate_clearance_m=0.26,
            base_preposition_lidar_backoff_step_m=0.04,
            base_preposition_lidar_max_backoff_m=0.16,
        ),
    )

    sim = types.SimpleNamespace(get_base_pose=lambda: np.array([0.0, 0.0, 0.0], dtype=float))

    reach_checks = iter([False, False, True])
    monkeypatch.setattr(executor, "_read_robot_state", lambda _sim: {"base_x": 0.0, "base_y": 0.0})
    monkeypatch.setattr(executor, "_drive_base_distance", lambda _sim, *, distance_m, timeout_s, **_kwargs: {
        "requested_distance_m": float(distance_m),
        "actual_distance_m": float(distance_m),
        "start_xy": [0.0, 0.0],
        "final_xy": [float(distance_m), 0.0],
        "stop_reason": "within_tolerance",
        "performed": abs(float(distance_m)) > 0.005,
        "ok": True,
    })
    monkeypatch.setattr(executor, "_rotate_base_to_heading", lambda _sim, *, target_theta, timeout_s: {
        "target_theta_rad": float(target_theta),
        "start_theta_rad": 0.0,
        "final_theta_rad": float(target_theta),
        "actual_delta_rad": float(target_theta),
        "stop_reason": "within_tolerance",
        "ok": True,
    })
    monkeypatch.setattr(executor, "_normalize_angle", GraspExecutor._normalize_angle)
    monkeypatch.setattr(executor, "_nearest_lidar_clearance", lambda _sim: {
        "available": True,
        "nearest_distance_m": 0.8,
        "percentile_5_m": 0.8,
        "num_valid_returns": 360,
    })
    monkeypatch.setattr(executor, "_ensure_lidar_rotate_clearance", lambda _sim, *, min_clearance_m, timeout_s: {
        "required_clearance_m": float(min_clearance_m),
        "before": {"available": True, "nearest_distance_m": 0.4},
        "after": {"available": True, "nearest_distance_m": 0.4},
        "performed": False,
        "total_backoff_m": 0.0,
        "backoff_attempts": [],
        "stop_reason": "within_clearance",
        "ok": True,
    })
    monkeypatch.setattr(
        executor,
        "_base_forward_axis",
        lambda theta_rad: np.array([np.cos(theta_rad), np.sin(theta_rad)], dtype=float),
    )
    monkeypatch.setattr(
        executor,
        "_arm_reach_axis",
        lambda theta_rad: np.array([np.sin(theta_rad), -np.cos(theta_rad)], dtype=float),
    )

    def fake_exact_targets(_geometric_grasp, current_state=None):
        del current_state
        if not next(reach_checks):
            raise RuntimeError("unreachable")
        return {"ik_base_rotate": 0.0}

    monkeypatch.setattr(executor.motion_planner, "geometric_grasp_targets", fake_exact_targets)

    diagnostic = executor._maybe_refine_base_for_geometric_grasp(
        sim,
        {
            "grasp_x": 0.10,
            "grasp_y": -0.80,
            "tabletop_geometry": {"base_center_world": [0.10, -0.80, 0.78]},
        },
    )

    assert diagnostic["performed"] is True
    assert diagnostic["lateral_centering"]["requested_distance_m"] == pytest.approx(0.10)
    assert diagnostic["longitudinal_reach_adjustment"]["performed"] is True
    assert diagnostic["longitudinal_reach_adjustment"]["requested_move_m"] > 0.20
    assert diagnostic["longitudinal_reach_adjustment"]["target_heading_rad"] == pytest.approx(-np.pi / 2.0)


def test_lidar_rotate_clearance_backs_off_until_safe(monkeypatch):
    _install_fake_stretch_mujoco_modules(monkeypatch)
    from ask2act_grasp.execution.grasp_executor import GraspExecutor

    executor = GraspExecutor.__new__(GraspExecutor)
    executor.context = types.SimpleNamespace(
        grasp_config=types.SimpleNamespace(
            base_preposition_lidar_backoff_step_m=0.04,
            base_preposition_lidar_max_backoff_m=0.16,
        )
    )

    clearances = iter(
        [
            {"available": True, "nearest_distance_m": 0.18, "percentile_5_m": 0.18, "num_valid_returns": 360},
            {"available": True, "nearest_distance_m": 0.23, "percentile_5_m": 0.23, "num_valid_returns": 360},
            {"available": True, "nearest_distance_m": 0.29, "percentile_5_m": 0.29, "num_valid_returns": 360},
        ]
    )
    monkeypatch.setattr(executor, "_nearest_lidar_clearance", lambda _sim: next(clearances))
    monkeypatch.setattr(executor, "_get_sim_time", lambda _sim: 0.0)
    monkeypatch.setattr(executor, "_drive_base_distance", lambda _sim, *, distance_m, timeout_s, **_kwargs: {
        "requested_distance_m": float(distance_m),
        "actual_distance_m": float(distance_m),
        "start_xy": [0.0, 0.0],
        "final_xy": [0.0, float(distance_m)],
        "stop_reason": "within_tolerance",
        "performed": True,
        "ok": True,
    })

    diagnostic = executor._ensure_lidar_rotate_clearance(
        sim=object(),
        min_clearance_m=0.26,
        timeout_s=5.0,
    )

    assert diagnostic["performed"] is True
    assert diagnostic["ok"] is True
    assert diagnostic["total_backoff_m"] == pytest.approx(0.08)


def test_reobserve_rejects_simpleik_unreachable_candidate(monkeypatch, tmp_path):
    _install_fake_stretch_mujoco_modules(monkeypatch)
    from ask2act_grasp.execution.grasp_executor import GraspExecutor
    from ask2act_grasp.types import GraspCandidate

    executor = GraspExecutor.__new__(GraspExecutor)
    executor.context = types.SimpleNamespace(
        scene_config=types.SimpleNamespace(table_top_z_m=0.78, cup_height_m=0.10, table_clearance_margin_m=0.005),
        grasp_config=types.SimpleNamespace(z_min_m=0.0, z_max_m=1.5),
    )
    executor.motion_planner = types.SimpleNamespace(
        geometric_grasp_targets=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ik none"))
    )
    fake_observation = types.SimpleNamespace(
        rgb_image=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_image=np.ones((2, 2), dtype=float),
        camera_intrinsics=np.eye(3, dtype=float),
        camera_extrinsics=np.eye(4, dtype=float),
    )
    executor.head_aligner = types.SimpleNamespace(align_and_capture=lambda _sim, target_center_xy=None: fake_observation)
    executor.point_cloud_gen = types.SimpleNamespace(
        generate=lambda **_kwargs: types.SimpleNamespace(
            world_points_xyz=np.ones((10, 3), dtype=float),
            camera_points_xyz=np.ones((10, 3), dtype=float),
        )
    )
    monkeypatch.setattr("ask2act_grasp.execution.grasp_executor.save_point_cloud", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor, "_save_head_artifacts", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor, "_read_robot_state", lambda _sim: {})
    monkeypatch.setattr(executor, "_compute_geometric_candidate_from_point_cloud", lambda *_args, **_kwargs: (
        {
            "grasp_x": 0.1,
            "grasp_y": -0.8,
            "grasp_z": 0.84,
            "object_center": [0.1, -0.8, 0.84],
        },
        GraspCandidate(np.eye(4), 1.0, 0.07, "geometric"),
        {
            "object_center_xy": [0.1, -0.8],
            "cropped_cloud_points": 100,
            "object_height": 0.10,
            "object_bottom_z": 0.78,
        },
    ))

    previous_grasp = {
        "grasp_x": 0.0,
        "grasp_y": -0.6,
        "grasp_z": 0.83,
        "object_center": [0.0, -0.6, 0.83],
    }
    previous_selected = GraspCandidate(np.eye(4), 0.9, 0.07, "geometric")
    previous_diag = {
        "object_center_xy": [0.0, -0.6],
        "cropped_cloud_points": 120,
        "object_height": 0.10,
        "object_bottom_z": 0.78,
    }

    _, _, returned_grasp, returned_selected, returned_diag, meta = executor._reobserve_geometric_after_base_motion(
        sim=object(),
        head_observation=fake_observation,
        point_cloud=types.SimpleNamespace(
            world_points_xyz=np.ones((10, 3), dtype=float),
            camera_points_xyz=np.ones((10, 3), dtype=float),
        ),
        geometric_grasp=previous_grasp,
        selected=previous_selected,
        geometric_diagnostics=previous_diag,
        target_bbox_2d=None,
        artifacts_dir=tmp_path,
        cgn_debug_dir=tmp_path,
        stage_times_s={},
        stage_prefix="test",
    )

    assert meta["accepted"] is False
    assert meta["accept_reason"] == "reobserved_candidate_simpleik_unreachable"
    assert returned_grasp == previous_grasp
    assert returned_selected == previous_selected
    assert returned_diag == previous_diag


def test_fallback_grasp_generator_produces_candidates():
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    grasp_config.enable_contact_graspnet = False
    generator = ContactGraspNetWrapper(grasp_config)
    points = np.array(
        [
            [0.00, -0.34, 0.87],
            [0.01, -0.33, 0.86],
            [-0.01, -0.35, 0.88],
        ],
        dtype=float,
    )
    candidates = generator.generate(points)
    assert len(candidates) >= 1
    assert candidates[0].score >= 0.5


def test_fallback_grasp_generator_applies_frame_transform():
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    grasp_config.enable_contact_graspnet = False
    generator = ContactGraspNetWrapper(grasp_config)
    points = np.array(
        [
            [0.00, 0.00, 0.40],
            [0.01, 0.00, 0.41],
            [-0.01, 0.00, 0.39],
        ],
        dtype=float,
    )
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = np.array([0.1, -0.2, 0.3], dtype=float)
    candidates = generator.generate(points, frame_transform=transform)
    assert len(candidates) >= 1
    expected = np.median(points, axis=0) + transform[:3, 3]
    assert np.allclose(candidates[0].position_m, expected)


def test_grasp_selector_allows_top_down_cup_candidate_when_it_scores_best():
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    grasp_config.score_threshold = 0.2
    selector = GraspSelector(scene_config, grasp_config)
    good_pose = np.eye(4)
    good_pose[:3, 2] = np.array([0.0, 1.0, 0.0])
    good_pose[:3, 3] = np.array([0.0, -0.34, 0.86])
    bad_pose = np.eye(4)
    bad_pose[:3, 2] = np.array([0.0, 0.0, -1.0])
    bad_pose[:3, 3] = np.array([0.0, -0.34, 0.99])
    selected = selector.select_best(
        [
            GraspCandidate(good_pose, 0.7, 0.06, "test"),
            GraspCandidate(bad_pose, 0.9, 0.06, "test"),
        ],
        robot_state={"stretch_gripper": 0.04},
    )
    assert selected is not None
    assert selected.score == 0.9
    assert selected.preferred_approach == "any"
    assert selected.needs_wrist_refinement is False
    assert selected.approach_type == "top_down"


@pytest.mark.parametrize(
    ("approach", "expected"),
    [
        (np.array([0.0, 0.0, -0.95], dtype=float), "top_down"),
        (np.array([0.37, -0.65, -0.67], dtype=float), "angled"),
        (np.array([0.0, 1.0, 0.0], dtype=float), "side"),
    ],
)
def test_classify_cup_approach_uses_three_band_thresholds(approach, expected):
    assert classify_cup_approach(approach) == expected


def test_grasp_selector_rejects_high_score_grasp_far_from_object_center():
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    grasp_config.score_threshold = 0.2
    selector = GraspSelector(scene_config, grasp_config)

    near_pose = np.eye(4)
    near_pose[:3, 2] = np.array([0.0, 1.0, 0.0])
    near_pose[:3, 3] = np.array([0.01, -0.60, 0.86])

    far_pose = np.eye(4)
    far_pose[:3, 2] = np.array([0.0, 1.0, 0.0])
    far_pose[:3, 3] = np.array([0.14, -0.60, 0.86])

    selected = selector.select_best(
        [
            GraspCandidate(far_pose, 0.9, 0.06, "test"),
            GraspCandidate(near_pose, 0.6, 0.06, "test"),
        ],
        robot_state={"stretch_gripper": 0.04},
        object_center=np.array([0.0, -0.60], dtype=float),
        object_radius=0.035,
    )

    assert selected is not None
    assert selected.score == 0.6
    assert np.allclose(selected.position_m[:2], np.array([0.01, -0.60]), atol=1e-6)
    assert selected.metadata["distance_to_object_center_xy_m"] < 0.053


def test_grasp_selector_requests_fallback_when_nearby_cup_grasps_are_only_top_down():
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    grasp_config.score_threshold = 0.2
    selector = GraspSelector(scene_config, grasp_config)

    top_down_pose = np.eye(4)
    top_down_pose[:3, 2] = np.array([0.0, 0.0, -1.0])
    top_down_pose[:3, 3] = np.array([0.01, -0.60, 0.97])

    selected = selector.select_best(
        [GraspCandidate(top_down_pose, 0.8, 0.03, "test")],
        robot_state={"stretch_gripper": 0.04},
        object_center=np.array([0.0, -0.60], dtype=float),
        object_radius=0.035,
    )

    assert selected is not None
    assert selected.score == 0.8
    assert selected.preferred_approach == "any"


def test_scene_config_loads_grasp_method():
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    assert scene_config.grasp_method in {"oracle", "cgn", "geometric"}


def test_pipeline_scene_overrides_update_grasp_method_and_cup_xy_only():
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    updated = apply_scene_overrides(scene_config, grasp_method="geometric", cup_position_xy=(0.12, -0.48))
    assert updated.grasp_method == "geometric"
    assert np.allclose(updated.cup_position_m[:2], np.array([0.12, -0.48]))
    assert np.isclose(updated.cup_position_m[2], scene_config.cup_position_m[2])
    assert updated.table_position_m == scene_config.table_position_m


def test_grasp_config_loads_cgn_preprocessing_defaults():
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    assert grasp_config.cgn_crop_radius_m == 0.08
    assert grasp_config.cgn_subsample_n == 2048
    assert grasp_config.add_depth_noise is True
    assert grasp_config.cgn_execution_mode == "guided_oracle"
    assert grasp_config.use_simple_ik_for_topdown is True


def test_motion_planner_converts_cgn_pose_to_targets():
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    planner = MotionPlanner(scene_config, grasp_config)
    pose = np.eye(4)
    pose[:3, 3] = np.array([0.02, -0.60, 0.83], dtype=float)
    pose[:3, 2] = np.array([0.0, 1.0, 0.0], dtype=float)
    pose[:3, 0] = np.array([1.0, 0.0, 0.0], dtype=float)
    pose[:3, 1] = np.array([0.0, 0.0, -1.0], dtype=float)
    targets = planner.cgn_pose_to_targets(pose, 0.05, approach_type="side", current_state={"wrist_yaw": 0.0})
    expected_grasp = pose[:3, 3] + pose[:3, :3] @ cgn_frame_to_wrist_local_m()
    expected_contact = pose[:3, 3] + pose[:3, :3] @ np.array([0.0, 0.0, CGN_GRIPPER_DEPTH_M], dtype=float)
    assert np.allclose(
        np.array([targets["grasp_x"], targets["grasp_y"], targets["grasp_z"]]),
        expected_grasp,
    )
    assert np.allclose(np.array(targets["contact_point"]), expected_contact)
    assert np.isclose(targets["gripper_open_width"], 0.06)
    assert targets["gripper_open_cmd"] != targets["gripper_open_width"]


def test_motion_planner_maps_top_down_cgn_pose_to_negative_pitch():
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    planner = MotionPlanner(scene_config, grasp_config)
    pose = np.eye(4)
    pose[:3, 0] = np.array([1.0, 0.0, 0.0], dtype=float)
    pose[:3, 1] = np.array([0.0, -1.0, 0.0], dtype=float)
    pose[:3, 2] = np.array([0.0, 0.0, -1.0], dtype=float)
    pose[:3, 3] = np.array([0.0, -0.60, 0.96], dtype=float)
    targets = planner.cgn_pose_to_targets(pose, 0.04, approach_type="top_down", current_state={"wrist_yaw": 2.5})
    assert np.isclose(targets["wrist_yaw"], 0.0)
    assert np.isclose(targets["wrist_pitch"], -1.57)
    assert np.isclose(targets["grasp_x"], targets["contact_point"][0])
    assert np.isclose(targets["grasp_y"], targets["contact_point"][1])
    assert targets["grasp_z"] > targets["contact_point"][2]


def test_motion_planner_top_down_uses_equivalent_yaw_when_raw_solution_is_near_limit():
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    planner = MotionPlanner(scene_config, grasp_config)
    pose = np.eye(4)
    pose[:3, 0] = np.array([-0.95, -0.31, 0.0], dtype=float)
    pose[:3, 1] = np.array([-0.31, 0.95, 0.0], dtype=float)
    pose[:3, 2] = np.array([0.0, 0.0, -1.0], dtype=float)
    pose[:3, 3] = np.array([0.0, -0.60, 0.96], dtype=float)

    targets = planner.cgn_pose_to_targets(pose, 0.04, approach_type="top_down", current_state={"wrist_yaw": 2.4})

    assert -1.39 <= targets["wrist_yaw"] <= 4.42
    assert np.isclose(targets["wrist_yaw"], 0.0)


def test_motion_planner_routes_cgn_candidate_through_guided_oracle_strategy():
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    planner = MotionPlanner(scene_config, grasp_config)

    pose = np.eye(4)
    pose[:3, 2] = np.array([0.0, 1.0, 0.0], dtype=float)
    pose[:3, 3] = np.array([0.01, -0.60, 0.86], dtype=float)
    candidate = GraspCandidate(pose, 0.7, 0.05, "contact_graspnet")
    candidate.approach_type = "top_down"
    candidate.metadata["contact_point"] = [0.02, -0.61, 0.84]

    plan = planner.plan_to_grasp(candidate, current_state={"lift": 0.6, "arm": 0.0, "wrist_yaw": 2.5, "wrist_pitch": 0.0})

    assert plan.metadata["planning_mode"] == "cgn_guided_oracle"
    assert plan.metadata["selected_grasp"]["source"] == "contact_graspnet"
    assert plan.metadata["numeric_targets"]["approach_type"] == "side"
    assert plan.waypoints[0].name == "retract_and_tuck"


def test_estimate_object_center_uses_robust_xy_and_vertical_midpoint():
    points = np.array(
        [
            [0.00, -0.60, 0.80],
            [0.01, -0.59, 0.81],
            [-0.01, -0.61, 0.90],
            [0.20, -0.20, 1.50],
        ],
        dtype=float,
    )

    center = estimate_object_center(points)

    assert np.isclose(center[0], np.median(points[:, 0]))
    assert np.isclose(center[1], np.median(points[:, 1]))
    assert np.isclose(center[2], (np.percentile(points[:, 2], 95) + np.percentile(points[:, 2], 5)) / 2.0)


def test_find_minimum_grip_direction_detects_narrow_axis_in_center_slice():
    x_vals = np.linspace(-0.01, 0.01, 6)
    y_vals = np.linspace(-0.06, 0.06, 12)
    z_vals = np.linspace(0.81, 0.89, 5)
    grid = np.array([[x, y, z] for x in x_vals for y in y_vals for z in z_vals], dtype=float)
    center = np.array([0.0, 0.0, 0.85], dtype=float)

    angle_rad, min_width = find_minimum_grip_direction(grid, center)

    assert min_width < 0.03
    assert min(abs(angle_rad), abs(angle_rad - np.pi)) < np.radians(6.0)


def test_fit_circle_2d_recovers_center_from_partial_arc():
    true_center = np.array([0.02, -0.58], dtype=float)
    true_radius = 0.035
    theta = np.linspace(-0.8, 0.8, 80, dtype=float)
    arc = np.stack(
        [
            true_center[0] + true_radius * np.cos(theta),
            true_center[1] + true_radius * np.sin(theta),
        ],
        axis=1,
    )

    fitted_cx, fitted_cy, fitted_radius = fit_circle_2d(arc)

    assert np.allclose([fitted_cx, fitted_cy], true_center, atol=2e-3)
    assert np.isclose(fitted_radius, true_radius, atol=2e-3)


def test_fit_circle_with_confidence_reports_small_error_for_clean_arc():
    true_center = np.array([0.02, -0.58], dtype=float)
    true_radius = 0.035
    theta = np.linspace(-1.2, 1.2, 120, dtype=float)
    arc = np.stack(
        [
            true_center[0] + true_radius * np.cos(theta),
            true_center[1] + true_radius * np.sin(theta),
        ],
        axis=1,
    )

    fitted_cx, fitted_cy, fitted_radius, estimated_error, residual_std, arc_coverage = fit_circle_with_confidence(arc)

    assert np.allclose([fitted_cx, fitted_cy], true_center, atol=2e-3)
    assert np.isclose(fitted_radius, true_radius, atol=2e-3)
    assert estimated_error < 1e-3
    assert residual_std < 1e-3
    assert arc_coverage > 2.3


def test_validate_grasp_point_accepts_points_under_vertical_ray():
    points = np.array(
        [
            [0.00, -0.60, 0.80],
            [0.00, -0.60, 0.82],
            [0.00, -0.60, 0.84],
            [0.01, -0.60, 0.83],
            [-0.01, -0.60, 0.83],
            [0.00, -0.59, 0.83],
            [0.00, -0.61, 0.83],
        ],
        dtype=float,
    )
    assert validate_grasp_point(points, np.array([0.0, -0.60], dtype=float))


def test_compute_geometric_grasp_returns_top_down_geometry_diagnostics():
    true_center = np.array([0.01, -0.60], dtype=float)
    true_radius = 0.035
    theta = np.linspace(-0.9, 0.9, 80, dtype=float)
    z_vals = np.linspace(0.80, 0.90, 10)
    points = np.array(
        [
            [
                true_center[0] + true_radius * np.cos(angle),
                true_center[1] + true_radius * np.sin(angle),
                z,
            ]
            for angle in theta
            for z in z_vals
        ],
        dtype=float,
    )

    result = compute_geometric_grasp(points, table_z=0.76, max_gripper_width_m=0.09)

    assert result["method"] == "geometric_point_cloud"
    assert result["grasp_point_validated"] is True
    assert np.allclose(result["object_center"][:2], true_center, atol=2e-3)
    assert np.isclose(result["fitted_circle_radius"], true_radius, atol=2e-3)
    assert np.isclose(result["min_cross_section_width"], true_radius * 2.0, atol=3e-3)
    assert result["estimated_error"] < 1e-3
    assert result["arc_coverage_deg"] > 100.0
    assert np.isclose(
        result["gripper_open_width"],
        result["min_cross_section_width"] + result["estimated_error"] * 4.0 + result["fixed_clearance_margin"],
        atol=2e-3,
    )
    assert np.isclose(result["grasp_z"], (result["object_top_z"] + result["object_bottom_z"]) / 2.0)


def test_save_geometric_grasp_debug_writes_single_overlay_image(tmp_path):
    x_vals = np.linspace(-0.03, 0.03, 7)
    y_vals = np.linspace(-0.02, 0.02, 5)
    z_vals = np.linspace(0.80, 0.90, 5)
    points = np.array([[x, y, z] for x in x_vals for y in y_vals for z in z_vals], dtype=float)
    result = compute_geometric_grasp(points, table_z=0.76, max_gripper_width_m=0.09)
    output_path = tmp_path / "geometric_grasp_debug.png"

    save_geometric_grasp_debug(points, result, output_path)

    assert output_path.exists() or output_path.with_suffix(".npy").exists()


def test_motion_planner_converts_geometric_grasp_to_top_down_targets():
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    grasp_config.allow_approximate_topdown_fallback = True
    planner = MotionPlanner(scene_config, grasp_config)
    planner.simple_ik = None

    geometric_grasp = {
        "grasp_x": 0.01,
        "grasp_y": -0.60,
        "grasp_z": 0.83,
        "grip_angle_rad": 0.0,
        "gripper_open_width": 0.08,
        "min_cross_section_width": 0.06,
        "object_center": [0.01, -0.60, 0.85],
        "object_height": 0.10,
        "object_top_z": 0.90,
        "object_bottom_z": 0.80,
        "grasp_point_validated": True,
        "width_near_limit": False,
    }

    targets = planner.geometric_grasp_targets(geometric_grasp, current_state={"wrist_yaw": 2.5})

    expected_wrist_z = 0.83 + 0.080
    assert np.isclose(targets["grasp_z"], expected_wrist_z)
    assert np.isclose(targets["grasp_x"], 0.01 + 0.045)
    assert np.isclose(targets["grasp_y"], -0.60 - 0.065)
    assert np.isclose(targets["contact_point"][2], 0.83)
    assert np.isclose(targets["wrist_pitch"], -1.57)
    assert targets["approach_type"] == "top_down"
    assert targets["planning_mode"] == "geometric_point_cloud"
    assert np.isclose(targets["wrist_yaw"], 0.0)


def test_motion_planner_uses_simple_ik_when_available():
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    planner = MotionPlanner(scene_config, grasp_config)

    class FakeSimpleIK:
        def ik_rotary_base(self, wrist_position):
            target_z = float(wrist_position[2])
            if target_z > 1.15:
                lift = 0.99
                arm = 0.33
            else:
                lift = 0.91
                arm = 0.31
            return {
                "joint_mobile_base_rotation": 0.12,
                "joint_lift": lift,
                "joint_arm_l0": arm,
            }

        def clip_with_joint_limits(self, robot_configuration):
            return None

        def fk_rotary_base(self, robot_configuration):
            gripper_cmd = gripper_width_to_command(0.08)
            base_rotate = float(robot_configuration["joint_mobile_base_rotation"])
            rubber_offset = rotate_topdown_offset_to_world_m(
                topdown_wrist_to_rubber_offset_m(gripper_cmd),
                base_rotate,
            )
            wrist_error = rotate_topdown_offset_to_world_m(
                topdown_simpleik_wrist_model_error_m(
                    robot_configuration["joint_lift"],
                    robot_configuration["joint_arm_l0"],
                ),
                base_rotate,
            )
            target = np.array([0.01, -0.60, 0.83 + TOPDOWN_SIMPLEIK_EXECUTION_Z_BIAS_M])
            return target - wrist_error - rubber_offset

    planner.simple_ik = FakeSimpleIK()
    targets = planner.geometric_grasp_targets(
        {
            "grasp_x": 0.01,
            "grasp_y": -0.60,
            "grasp_z": 0.83,
            "gripper_open_width": 0.08,
            "grip_angle_rad": 0.0,
            "min_cross_section_width": 0.06,
            "object_center": [0.01, -0.60, 0.85],
            "object_height": 0.10,
            "object_top_z": 0.90,
            "object_bottom_z": 0.80,
            "grasp_point_validated": True,
            "width_near_limit": False,
        }
    )

    assert targets["planning_mode"] == "geometric_simple_ik"
    assert np.isclose(targets["ik_base_rotate"], 0.12)
    assert np.isclose(targets["ik_lift"], 0.91)
    assert np.isclose(targets["ik_arm"], 0.31)
    assert np.isclose(targets["ik_pregrasp_lift"], 0.99)
    assert np.isclose(targets["ik_pregrasp_arm"], 0.33)
    assert targets["fk_error_m"] < 1e-6


def test_motion_planner_applies_side_x_bias_for_lateral_targets():
    from ask2act_grasp.planning import motion_planner as motion_planner_module

    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    planner = MotionPlanner(scene_config, grasp_config)
    original_bias = motion_planner_module.GEOMETRIC_TOP_DOWN_SIDE_X_BIAS_M
    original_deadband = motion_planner_module.GEOMETRIC_TOP_DOWN_SIDE_X_BIAS_DEADBAND_M
    original_right_extra = motion_planner_module.GEOMETRIC_TOP_DOWN_RIGHT_EXTRA_X_BIAS_M
    original_left_center_y = motion_planner_module.GEOMETRIC_TOP_DOWN_LEFT_CENTER_Y_BIAS_M
    original_right_y = motion_planner_module.GEOMETRIC_TOP_DOWN_RIGHT_Y_BIAS_M
    original_fk_threshold = motion_planner_module.GEOMETRIC_TOP_DOWN_SIMPLEIK_MAX_FK_ERROR_M

    class FakeSimpleIK:
        def ik_rotary_base(self, _wrist_position):
            return {
                "joint_mobile_base_rotation": 0.0,
                "joint_lift": 0.91,
                "joint_arm_l0": 0.31,
            }

        def clip_with_joint_limits(self, _robot_configuration):
            return None

        def fk_rotary_base(self, _robot_configuration):
            return [0.0, 0.0, 0.0]

    try:
        motion_planner_module.GEOMETRIC_TOP_DOWN_SIDE_X_BIAS_M = 0.02
        motion_planner_module.GEOMETRIC_TOP_DOWN_SIDE_X_BIAS_DEADBAND_M = 0.05
        motion_planner_module.GEOMETRIC_TOP_DOWN_RIGHT_EXTRA_X_BIAS_M = 0.01
        motion_planner_module.GEOMETRIC_TOP_DOWN_LEFT_CENTER_Y_BIAS_M = 0.005
        motion_planner_module.GEOMETRIC_TOP_DOWN_RIGHT_Y_BIAS_M = -0.005
        motion_planner_module.GEOMETRIC_TOP_DOWN_SIMPLEIK_MAX_FK_ERROR_M = 0.0
        planner.simple_ik = FakeSimpleIK()
        base_grasp = {
            "grasp_y": -0.60,
            "grasp_z": 0.83,
            "gripper_open_width": 0.08,
            "grip_angle_rad": 0.0,
            "min_cross_section_width": 0.06,
            "object_height": 0.10,
            "object_top_z": 0.90,
            "object_bottom_z": 0.80,
            "grasp_point_validated": True,
            "width_near_limit": False,
        }

        left_targets = planner.geometric_grasp_targets({**base_grasp, "grasp_x": 0.12})
        right_targets = planner.geometric_grasp_targets({**base_grasp, "grasp_x": -0.12})
        center_targets = planner.geometric_grasp_targets({**base_grasp, "grasp_x": 0.02})

        assert left_targets["contact_point"][0] == pytest.approx(0.14)
        assert left_targets["contact_point"][1] == pytest.approx(-0.595)
        assert left_targets["side_x_bias_applied_m"] == pytest.approx(0.02)
        assert left_targets["side_y_bias_applied_m"] == pytest.approx(0.005)
        assert left_targets["side_bias_region"] == "left"
        assert right_targets["contact_point"][0] == pytest.approx(-0.15)
        assert right_targets["contact_point"][1] == pytest.approx(-0.605)
        assert right_targets["side_x_bias_applied_m"] == pytest.approx(-0.03)
        assert right_targets["side_y_bias_applied_m"] == pytest.approx(-0.005)
        assert right_targets["side_bias_region"] == "right"
        assert center_targets["contact_point"][0] == pytest.approx(0.02)
        assert center_targets["contact_point"][1] == pytest.approx(-0.595)
        assert center_targets["side_x_bias_applied_m"] == pytest.approx(0.0)
        assert center_targets["side_y_bias_applied_m"] == pytest.approx(0.005)
        assert center_targets["side_bias_region"] == "center"
    finally:
        motion_planner_module.GEOMETRIC_TOP_DOWN_SIDE_X_BIAS_M = original_bias
        motion_planner_module.GEOMETRIC_TOP_DOWN_SIDE_X_BIAS_DEADBAND_M = original_deadband
        motion_planner_module.GEOMETRIC_TOP_DOWN_RIGHT_EXTRA_X_BIAS_M = original_right_extra
        motion_planner_module.GEOMETRIC_TOP_DOWN_LEFT_CENTER_Y_BIAS_M = original_left_center_y
        motion_planner_module.GEOMETRIC_TOP_DOWN_RIGHT_Y_BIAS_M = original_right_y
        motion_planner_module.GEOMETRIC_TOP_DOWN_SIMPLEIK_MAX_FK_ERROR_M = original_fk_threshold


def test_motion_planner_caps_center_grasp_depth_from_object_top():
    from ask2act_grasp.planning import motion_planner as motion_planner_module

    original_mode = motion_planner_module.GEOMETRIC_TOP_DOWN_GRASP_Z_MODE
    original_delta = motion_planner_module.GEOMETRIC_TOP_DOWN_MAX_TOP_GRASP_DELTA_M
    try:
        motion_planner_module.GEOMETRIC_TOP_DOWN_GRASP_Z_MODE = "center"
        motion_planner_module.GEOMETRIC_TOP_DOWN_MAX_TOP_GRASP_DELTA_M = 0.07
        assert MotionPlanner._geometric_topdown_grasp_z(
            {
                "grasp_z": 0.50,
                "object_top_z": 0.62,
                "object_bottom_z": 0.38,
            }
        ) == pytest.approx(0.55)
        assert MotionPlanner._geometric_topdown_grasp_z(
            {
                "grasp_z": 0.57,
                "object_top_z": 0.62,
                "object_bottom_z": 0.52,
            }
        ) == pytest.approx(0.57)
    finally:
        motion_planner_module.GEOMETRIC_TOP_DOWN_GRASP_Z_MODE = original_mode
        motion_planner_module.GEOMETRIC_TOP_DOWN_MAX_TOP_GRASP_DELTA_M = original_delta


def test_motion_planner_simple_ik_accounts_for_base_translation():
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    planner = MotionPlanner(scene_config, grasp_config)
    captured: dict[str, list[list[float]]] = {"ik_wrist_positions": []}

    class FakeSimpleIK:
        def ik_rotary_base(self, wrist_position):
            captured["ik_wrist_positions"].append(list(map(float, wrist_position)))
            return {
                "joint_mobile_base_rotation": 0.12,
                "joint_lift": 0.91,
                "joint_arm_l0": 0.31,
            }

        def clip_with_joint_limits(self, robot_configuration):
            return None

        def fk_rotary_base(self, robot_configuration):
            return np.asarray(captured["ik_wrist_positions"][-1], dtype=float)

    planner.simple_ik = FakeSimpleIK()
    geometric_grasp = {
        "grasp_x": 0.01,
        "grasp_y": -0.60,
        "grasp_z": 0.83,
        "gripper_open_width": 0.08,
        "grip_angle_rad": 0.0,
        "min_cross_section_width": 0.06,
        "object_center": [0.01, -0.60, 0.85],
        "object_height": 0.10,
        "object_top_z": 0.90,
        "object_bottom_z": 0.80,
        "grasp_point_validated": True,
        "width_near_limit": False,
    }

    planner.geometric_grasp_targets(
        geometric_grasp,
        current_state={"wrist_yaw": 0.0},
    )
    wrist_without_base_translation = np.asarray(captured["ik_wrist_positions"][0], dtype=float)

    captured["ik_wrist_positions"].clear()
    targets = planner.geometric_grasp_targets(
        geometric_grasp,
        current_state={"wrist_yaw": 0.0, "base_x": 0.12, "base_y": -0.05},
    )
    wrist_with_base_translation = np.asarray(captured["ik_wrist_positions"][0], dtype=float)

    assert np.allclose(
        wrist_with_base_translation - wrist_without_base_translation,
        np.array([-0.12, 0.05, 0.0], dtype=float),
    )
    assert np.allclose(targets["base_world_translation_m"], [0.12, -0.05, 0.0])


def test_motion_planner_disables_simple_ik_when_config_requests_fallback():
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    grasp_config.use_simple_ik_for_topdown = False
    planner = MotionPlanner(scene_config, grasp_config)

    assert planner.simple_ik is None
    assert planner.simple_ik_init_error == "disabled by grasp_config.use_simple_ik_for_topdown"


def test_motion_planner_uses_geometric_yaw_for_narrow_objects():
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    grasp_config.allow_approximate_topdown_fallback = True
    planner = MotionPlanner(scene_config, grasp_config)
    planner.simple_ik = None

    targets = planner.geometric_grasp_targets(
        {
            "grasp_x": 0.0,
            "grasp_y": -0.60,
            "grasp_z": 0.83,
            "grip_angle_rad": np.pi / 4.0,
            "gripper_open_width": 0.03,
            "min_cross_section_width": 0.01,
            "object_center": [0.0, -0.60, 0.85],
            "object_height": 0.10,
            "object_top_z": 0.90,
            "object_bottom_z": 0.80,
            "grasp_point_validated": True,
            "width_near_limit": False,
        },
        current_state={"wrist_yaw": 0.7},
    )

    assert np.isclose(targets["wrist_yaw"], np.pi / 4.0)


def test_motion_planner_routes_geometric_candidate_through_top_down_waypoints():
    scene_config, _ = load_scene_config(PACKAGE_ROOT / "config" / "scene_config.yaml")
    grasp_config = load_grasp_config(PACKAGE_ROOT / "config" / "grasp_config.yaml")
    grasp_config.allow_approximate_topdown_fallback = True
    planner = MotionPlanner(scene_config, grasp_config)
    planner.simple_ik = None

    candidate = GraspCandidate(
        pose_4x4=np.eye(4, dtype=float),
        score=0.95,
        width_m=0.08,
        source="geometric_point_cloud",
        approach_type="top_down",
        preferred_approach="top_down",
        metadata={
            "geometric_grasp": {
                "grasp_x": 0.01,
                "grasp_y": -0.60,
                "grasp_z": 0.83,
                "grip_angle_rad": 0.0,
                "gripper_open_width": 0.08,
                "min_cross_section_width": 0.06,
                "object_center": [0.01, -0.60, 0.85],
                "object_height": 0.10,
                "object_top_z": 0.90,
                "object_bottom_z": 0.80,
                "grasp_point_validated": True,
                "width_near_limit": False,
            }
        },
    )

    plan = planner.plan_to_grasp(
        candidate,
        current_state={"lift": 0.6, "arm": 0.0, "wrist_yaw": 2.5, "wrist_pitch": 0.0},
    )

    assert plan.metadata["planning_mode"] == "geometric_point_cloud"
    assert plan.metadata["selected_grasp"]["source"] == "geometric_point_cloud"
    assert plan.metadata["selected_grasp"]["approach_type"] == "top_down"
    assert any(waypoint.name == "descend_to_grasp" for waypoint in plan.waypoints)
    assert any(waypoint.name == "secure_grasp" for waypoint in plan.waypoints)
