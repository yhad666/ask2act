from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stretch_mujoco import StretchMujocoSimulator, utils
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras

from tabletop_scene_builder import resolve_scene_xml_path
from validation_common import (
    LOG_ROOT,
    aligned_intrinsics_for_saved_image,
    depth_to_color,
    ensure_dir,
    matrix_to_rows,
    save_json,
    save_rgb_image,
    utc_timestamp,
)


MINIMAL_GRASP_ROOT = LOG_ROOT / "minimal_grasping"
DEFAULT_NOMINAL_TARGET_DEPTH_M = 0.28


def homogeneous_from_translation_euler(
    translation_xyz: tuple[float, float, float],
    euler_xyz_rad: tuple[float, float, float],
) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(
        utils.Rx(euler_xyz_rad[0]) @ utils.Ry(euler_xyz_rad[1]) @ utils.Rz(euler_xyz_rad[2]),
        dtype=float,
    )
    transform[:3, 3] = np.asarray(translation_xyz, dtype=float)
    return transform


def camera_pose_world(sim: StretchMujocoSimulator, camera: StretchCameras) -> np.ndarray | None:
    if camera in [StretchCameras.cam_d405_rgb, StretchCameras.cam_d405_depth]:
        try:
            d405_body = sim.get_link_pose("d405_cam")
        except Exception:
            try:
                d405_body = sim.get_link_pose("link_d405")
            except Exception:
                return None
        d405_camera_extrinsic = homogeneous_from_translation_euler((0.0, 0.0, 0.0), (-3.14, 0.0, -3.14))
        return np.asarray(d405_body @ d405_camera_extrinsic, dtype=float)

    if camera in [StretchCameras.cam_d435i_rgb, StretchCameras.cam_d435i_depth]:
        head_tilt_body = sim.get_link_pose("link_head_tilt")
        realsense_offset = homogeneous_from_translation_euler((0.0406, 0.0053, 0.0307), (0.0, 0.0, 0.0))
        d435_camera_extrinsic = homogeneous_from_translation_euler((0.0, 0.015, 0.0), (1.57, -1.57, 0.0))
        return np.asarray(head_tilt_body @ realsense_offset @ d435_camera_extrinsic, dtype=float)

    if camera == StretchCameras.cam_nav_rgb:
        nav_body = sim.get_link_pose("head_nav_cam")
        nav_camera_extrinsic = homogeneous_from_translation_euler((0.0, 0.0, 0.0), (-3.1415926, 0.0, -1.5707963))
        return np.asarray(nav_body @ nav_camera_extrinsic, dtype=float)

    return None


def add_scene_and_output_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--scene-xml-path",
        default=str(resolve_scene_xml_path(Path(__file__).resolve().parent / "tabletop_minimal_scene.xml")),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--camera-hz", type=float, default=15.0)
    return parser


def ensure_minimal_grasp_root() -> Path:
    return ensure_dir(MINIMAL_GRASP_ROOT)


def create_run_dir(output_dir: str | Path | None, run_tag: str | None, prefix: str) -> Path:
    root = ensure_minimal_grasp_root()
    if output_dir is not None:
        return ensure_dir(Path(output_dir))

    tag = run_tag or f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    return ensure_dir(root / tag)


def wait_for_cameras(
    sim: StretchMujocoSimulator,
    required_cameras: list[StretchCameras],
    timeout_s: float = 10.0,
):
    start = time.time()
    last_error = "camera data not ready"
    while time.time() - start < timeout_s:
        camera_data = sim.pull_camera_data()
        try:
            for camera in required_cameras:
                kwargs = {"auto_correct_rgb": False} if not camera.is_depth else {}
                camera_data.get_camera_data(camera, **kwargs)
            return camera_data
        except Exception as exc:  # pragma: no cover - defensive polling
            last_error = str(exc)
            time.sleep(0.2)
    raise TimeoutError(last_error)


def sim_pose_snapshot(sim: StretchMujocoSimulator) -> dict[str, Any]:
    status = sim.pull_status()
    wrist_camera_pose = None
    head_camera_pose = None
    for link_name in ["link_d405", "d405_cam"]:
        try:
            wrist_camera_pose = matrix_to_rows(sim.get_link_pose(link_name))
            break
        except Exception:
            continue
    for link_name in ["realsense", "link_head_tilt"]:
        try:
            head_camera_pose = matrix_to_rows(sim.get_link_pose(link_name))
            break
        except Exception:
            continue
    return {
        "wall_time_utc": utc_timestamp(),
        "sim_time": float(status.time),
        "base_pose": [float(v) for v in sim.get_base_pose()],
        "ee_pose_4x4": matrix_to_rows(sim.get_ee_pose()),
        "wrist_camera_pose_4x4": wrist_camera_pose,
        "head_camera_pose_4x4": head_camera_pose,
        "status": status.to_dict(),
    }


def actuator_tolerance(actuator: Actuators) -> float:
    if actuator == Actuators.gripper:
        return 0.01
    if actuator in [Actuators.arm, Actuators.head_pan, Actuators.head_tilt]:
        return 0.02
    return 0.03


def move_to_and_record(
    sim: StretchMujocoSimulator,
    actuator: Actuators,
    target: float,
    *,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    before_status = sim.pull_status()
    before = actuator.get_position(before_status)
    sim.move_to(actuator, target)
    tolerance = actuator_tolerance(actuator)
    reached = sim.wait_until_at_setpoint(actuator, timeout=timeout_s, position_tolerance=tolerance)
    after_status = sim.pull_status()
    after = actuator.get_position(after_status)
    return {
        "actuator": actuator.name,
        "target": float(target),
        "before": float(before),
        "after": float(after),
        "reached": bool(reached),
        "tolerance": float(tolerance),
        "abs_error": float(abs(after - target)),
        "sim_time_after": float(after_status.time),
    }


def start_wrist_sim(
    *,
    scene_xml_path: Path,
    camera_hz: float,
    settle_seconds: float,
) -> StretchMujocoSimulator:
    return start_sim(
        scene_xml_path=scene_xml_path,
        cameras_to_use=[StretchCameras.cam_d405_rgb, StretchCameras.cam_d405_depth],
        camera_hz=camera_hz,
        settle_seconds=settle_seconds,
    )


def start_sim(
    *,
    scene_xml_path: Path,
    cameras_to_use: list[StretchCameras],
    camera_hz: float,
    settle_seconds: float,
) -> StretchMujocoSimulator:
    sim = StretchMujocoSimulator(
        scene_xml_path=str(resolve_scene_xml_path(scene_xml_path)),
        cameras_to_use=cameras_to_use,
        camera_hz=camera_hz,
    )
    sim.start(headless=True)
    time.sleep(settle_seconds)
    return sim


def capture_camera_observation(
    sim: StretchMujocoSimulator,
    *,
    rgb_camera: StretchCameras,
    depth_camera: StretchCameras,
) -> dict[str, Any]:
    camera_data = wait_for_cameras(
        sim,
        [rgb_camera, depth_camera],
    )
    rgb = camera_data.get_camera_data(rgb_camera, auto_correct_rgb=False)
    depth = camera_data.get_camera_data(depth_camera)
    k_matrix = np.asarray(aligned_intrinsics_for_saved_image(rgb_camera), dtype=float)
    camera_pose = camera_pose_world(sim, rgb_camera)
    pose = sim_pose_snapshot(sim)
    return {
        "wall_time_utc": utc_timestamp(),
        "camera_time": float(camera_data.time),
        "camera_fps": float(camera_data.fps),
        "rgb": rgb,
        "depth": depth,
        "k_matrix": k_matrix,
        "camera_name": rgb_camera.name,
        "camera_pose_4x4": camera_pose,
        "pose": pose,
    }


def capture_rgb_frame(sim: StretchMujocoSimulator, *, camera: StretchCameras) -> np.ndarray:
    camera_data = wait_for_cameras(sim, [camera])
    return camera_data.get_camera_data(camera, auto_correct_rgb=False)


def capture_wrist_observation(sim: StretchMujocoSimulator) -> dict[str, Any]:
    return capture_camera_observation(
        sim,
        rgb_camera=StretchCameras.cam_d405_rgb,
        depth_camera=StretchCameras.cam_d405_depth,
    )


def capture_head_observation(sim: StretchMujocoSimulator) -> dict[str, Any]:
    return capture_camera_observation(
        sim,
        rgb_camera=StretchCameras.cam_d435i_rgb,
        depth_camera=StretchCameras.cam_d435i_depth,
    )


def pixel_to_camera_xyz(u: int, v: int, depth_m: float, k_matrix: np.ndarray) -> list[float]:
    fx = float(k_matrix[0, 0])
    fy = float(k_matrix[1, 1])
    cx = float(k_matrix[0, 2])
    cy = float(k_matrix[1, 2])
    x = (float(u) - cx) * depth_m / fx
    y = (float(v) - cy) * depth_m / fy
    z = float(depth_m)
    return [x, y, z]


def validate_pixel_coordinates(u: int, v: int, image_shape: tuple[int, ...]) -> None:
    height, width = image_shape[:2]
    if not (0 <= u < width and 0 <= v < height):
        raise ValueError(f"Pixel ({u}, {v}) is outside image bounds width={width}, height={height}.")


def save_observation_bundle(output_dir: Path, prefix: str, observation: dict[str, Any]) -> dict[str, str]:
    return save_camera_observation_bundle(output_dir, prefix, observation, camera_label="wrist")


def save_camera_observation_bundle(
    output_dir: Path,
    prefix: str,
    observation: dict[str, Any],
    *,
    camera_label: str,
) -> dict[str, str]:
    rgb_path = output_dir / f"{prefix}_wrist_rgb.png"
    depth_npy_path = output_dir / f"{prefix}_wrist_depth.npy"
    depth_color_path = output_dir / f"{prefix}_wrist_depth_color.png"
    pose_path = output_dir / f"{prefix}_pose_snapshot.json"
    if camera_label != "wrist":
        rgb_path = output_dir / f"{prefix}_{camera_label}_rgb.png"
        depth_npy_path = output_dir / f"{prefix}_{camera_label}_depth.npy"
        depth_color_path = output_dir / f"{prefix}_{camera_label}_depth_color.png"
        pose_path = output_dir / f"{prefix}_{camera_label}_pose.json"

    save_rgb_image(rgb_path, observation["rgb"])
    np.save(str(depth_npy_path), observation["depth"])
    cv2.imwrite(str(depth_color_path), depth_to_color(observation["depth"]))
    save_json(
        pose_path,
        {
            "wall_time_utc": observation["wall_time_utc"],
            "camera_time": observation["camera_time"],
            "camera_fps": observation["camera_fps"],
            "k_matrix": observation["k_matrix"],
            "camera_name": observation.get("camera_name"),
            "camera_pose_4x4": observation.get("camera_pose_4x4"),
            "pose": observation["pose"],
        },
    )
    return {
        "rgb_path": str(rgb_path),
        "depth_npy_path": str(depth_npy_path),
        "depth_color_path": str(depth_color_path),
        "pose_path": str(pose_path),
    }


def save_pixel_overlay(
    image_rgb: np.ndarray,
    pixel_u: int,
    pixel_v: int,
    output_path: Path,
    text_lines: list[str] | None = None,
) -> None:
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.circle(canvas, (pixel_u, pixel_v), 6, (0, 0, 255), thickness=2)
    cv2.drawMarker(canvas, (pixel_u, pixel_v), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
    if text_lines:
        y = 24
        for line in text_lines:
            cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
            y += 22
    cv2.imwrite(str(output_path), canvas)


def pick_pixel_with_opencv(image_rgb: np.ndarray, window_name: str = "wrist_d405_rgb") -> tuple[int, int]:
    selection: dict[str, int] = {}
    display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            selection["u"] = int(x)
            selection["v"] = int(y)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while True:
            frame = display.copy()
            if "u" in selection and "v" in selection:
                cv2.circle(frame, (selection["u"], selection["v"]), 6, (0, 0, 255), 2)
                cv2.drawMarker(
                    frame,
                    (selection["u"], selection["v"]),
                    (255, 255, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=16,
                    thickness=2,
                )
            cv2.putText(
                frame,
                "Left click to select pixel, Enter to confirm, q to abort",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10) and "u" in selection and "v" in selection:
                return selection["u"], selection["v"]
            if key == ord("q"):
                raise RuntimeError("Pixel selection aborted by user.")
    finally:
        cv2.destroyWindow(window_name)


def choose_pixel(
    *,
    image_rgb: np.ndarray,
    u: int | None,
    v: int | None,
    interactive: bool,
) -> tuple[int, int]:
    if u is not None or v is not None:
        if u is None or v is None:
            raise ValueError("Both --u and --v must be provided together.")
        validate_pixel_coordinates(int(u), int(v), image_rgb.shape)
        return int(u), int(v)

    if interactive:
        return pick_pixel_with_opencv(image_rgb=image_rgb)

    raise ValueError("Provide --u and --v, or pass --interactive for mouse selection.")


def build_pixel_result(
    *,
    observation: dict[str, Any],
    pixel_u: int,
    pixel_v: int,
) -> dict[str, Any]:
    validate_pixel_coordinates(pixel_u, pixel_v, observation["depth"].shape)
    depth_value = float(observation["depth"][pixel_v, pixel_u])
    if not np.isfinite(depth_value) or depth_value <= 0.0:
        raise ValueError(
            f"Depth at pixel ({pixel_u}, {pixel_v}) is invalid: {depth_value}. Pick a pixel with valid positive depth."
        )

    xyz_camera = pixel_to_camera_xyz(pixel_u, pixel_v, depth_value, observation["k_matrix"])
    return {
        "wall_time_utc": utc_timestamp(),
        "pixel_u": int(pixel_u),
        "pixel_v": int(pixel_v),
        "depth_m": depth_value,
        "camera_xyz_m": xyz_camera,
        "k_matrix": matrix_to_rows(observation["k_matrix"]),
        "camera_name": observation.get("camera_name"),
        "camera_pose_4x4": matrix_to_rows(np.asarray(observation["camera_pose_4x4"], dtype=float))
        if observation.get("camera_pose_4x4") is not None
        else None,
        "ee_pose_4x4": observation["pose"]["ee_pose_4x4"],
        "wrist_camera_pose_4x4": observation["pose"]["wrist_camera_pose_4x4"],
        "base_pose": observation["pose"]["base_pose"],
        "camera_time": observation["camera_time"],
        "camera_fps": observation["camera_fps"],
    }


def heuristic_descend_delta_m(
    depth_m: float,
    *,
    nominal_target_depth_m: float = DEFAULT_NOMINAL_TARGET_DEPTH_M,
    base_delta_m: float = 0.06,
    gain: float = 0.20,
    min_delta_m: float = 0.03,
    max_delta_m: float = 0.09,
) -> float:
    adjusted = base_delta_m + gain * (depth_m - nominal_target_depth_m)
    return float(np.clip(adjusted, min_delta_m, max_delta_m))
