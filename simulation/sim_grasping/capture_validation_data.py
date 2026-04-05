from __future__ import annotations

import argparse
import time
from pathlib import Path

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras

from tabletop_scene_builder import resolve_scene_xml_path
from validation_common import (
    DATA_ROOT,
    DEFAULT_SCENE_PATH,
    SCREENSHOT_ROOT,
    aligned_intrinsics_for_saved_image,
    depth_to_color,
    ensure_dir,
    ensure_validation_dirs,
    matrix_to_rows,
    save_depth_outputs,
    save_json,
    save_rgb_image,
    summarize_array,
    utc_timestamp,
)


CAPTURE_SESSIONS = [
    {
        "name": "head",
        "required": True,
        "cameras": [StretchCameras.cam_d435i_rgb, StretchCameras.cam_d435i_depth],
    },
    {
        "name": "wrist",
        "required": True,
        "cameras": [StretchCameras.cam_d405_rgb, StretchCameras.cam_d405_depth],
    },
    {
        "name": "nav",
        "required": False,
        "cameras": [StretchCameras.cam_nav_rgb],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture sample Stretch MuJoCo camera and status data.")
    parser.add_argument("--scene-xml-path", default=str(DEFAULT_SCENE_PATH))
    parser.add_argument("--tag", default="default_scene")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--camera-hz", type=float, default=15.0)
    return parser.parse_args()


def wait_for_camera_data(
    sim: StretchMujocoSimulator, required_cameras: list[StretchCameras], timeout_s: float = 10.0
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


def capture_session(
    *,
    scene_xml_path: Path,
    output_dir: Path,
    screenshot_dir: Path,
    tag: str,
    session_name: str,
    cameras_to_use: list[StretchCameras],
    camera_hz: float,
    settle_seconds: float,
) -> dict[str, object]:
    sim = StretchMujocoSimulator(
        scene_xml_path=str(scene_xml_path),
        cameras_to_use=cameras_to_use,
        camera_hz=camera_hz,
    )

    sim.start(headless=True)
    try:
        time.sleep(settle_seconds)

        camera_data = wait_for_camera_data(sim, cameras_to_use)
        status = sim.pull_status()
        sensors = sim.pull_sensor_data()
        ee_pose = sim.get_ee_pose()
        base_pose = sim.get_base_pose()

        session_payload: dict[str, object] = {
            "session_name": session_name,
            "cameras": [camera.name for camera in cameras_to_use],
            "capture_pose": "home",
            "status_fps": float(status.fps),
            "camera_fps": float(camera_data.fps),
            "sim_time": float(status.time),
            "camera_time": float(camera_data.time),
            "base_pose": [float(v) for v in base_pose],
            "ee_pose_4x4": matrix_to_rows(ee_pose),
            "sensors": {
                "base_gyro": summarize_array(sensors.base_gyro),
                "base_imu": summarize_array(sensors.base_imu),
                "lidar": summarize_array(sensors.lidar),
            },
            "intrinsics": {},
        }

        if StretchCameras.cam_d435i_rgb in cameras_to_use:
            d435_rgb = camera_data.get_camera_data(
                StretchCameras.cam_d435i_rgb, auto_correct_rgb=False
            )
            d435_depth = camera_data.get_camera_data(StretchCameras.cam_d435i_depth)
            save_rgb_image(output_dir / "head_d435i_rgb.png", d435_rgb)
            save_rgb_image(screenshot_dir / f"{tag}_head_rgb.png", d435_rgb)
            d435_depth_stats = save_depth_outputs(output_dir / "head_d435i_depth", d435_depth)
            import cv2

            cv2.imwrite(str(screenshot_dir / f"{tag}_head_depth.png"), depth_to_color(d435_depth))
            session_payload["head_rgb_shape"] = list(d435_rgb.shape)
            session_payload["head_depth_shape"] = list(d435_depth.shape)
            session_payload["head_depth_stats"] = d435_depth_stats
            session_payload["intrinsics"] = {
                **session_payload["intrinsics"],
                "cam_d435i_K_raw": matrix_to_rows(camera_data.cam_d435i_K),
                "cam_d435i_K_saved_image": matrix_to_rows(
                    aligned_intrinsics_for_saved_image(StretchCameras.cam_d435i_rgb)
                ),
            }

        if StretchCameras.cam_d405_rgb in cameras_to_use:
            d405_rgb = camera_data.get_camera_data(
                StretchCameras.cam_d405_rgb, auto_correct_rgb=False
            )
            d405_depth = camera_data.get_camera_data(StretchCameras.cam_d405_depth)
            save_rgb_image(output_dir / "wrist_d405_rgb.png", d405_rgb)
            save_rgb_image(screenshot_dir / f"{tag}_wrist_rgb.png", d405_rgb)
            d405_depth_stats = save_depth_outputs(output_dir / "wrist_d405_depth", d405_depth)
            import cv2

            cv2.imwrite(str(screenshot_dir / f"{tag}_wrist_depth.png"), depth_to_color(d405_depth))
            session_payload["wrist_rgb_shape"] = list(d405_rgb.shape)
            session_payload["wrist_depth_shape"] = list(d405_depth.shape)
            session_payload["wrist_depth_stats"] = d405_depth_stats
            session_payload["intrinsics"] = {
                **session_payload["intrinsics"],
                "cam_d405_K_raw": matrix_to_rows(camera_data.cam_d405_K),
                "cam_d405_K_saved_image": matrix_to_rows(
                    aligned_intrinsics_for_saved_image(StretchCameras.cam_d405_rgb)
                ),
            }

        if StretchCameras.cam_nav_rgb in cameras_to_use:
            nav_rgb = camera_data.get_camera_data(StretchCameras.cam_nav_rgb, auto_correct_rgb=False)
            save_rgb_image(output_dir / "nav_rgb.png", nav_rgb)
            save_rgb_image(screenshot_dir / f"{tag}_nav_rgb.png", nav_rgb)
            session_payload["nav_rgb_shape"] = list(nav_rgb.shape)

        session_payload["status"] = status.to_dict()
        return session_payload
    finally:
        sim.stop()


def capture_status_session(scene_xml_path: Path) -> dict[str, object]:
    sim = StretchMujocoSimulator(scene_xml_path=str(scene_xml_path), cameras_to_use=[], camera_hz=5.0)
    sim.start(headless=True)
    try:
        time.sleep(0.5)
        sim.wait_while_is_moving(Actuators.lift, timeout=5.0)
        sim.wait_while_is_moving(Actuators.wrist_pitch, timeout=5.0)
        status = sim.pull_status()
        sensors = sim.pull_sensor_data()
        return {
            "session_name": "status_only",
            "scene_xml_path": str(scene_xml_path),
            "sim_time": float(status.time),
            "status_fps": float(status.fps),
            "sim_to_real_time_ratio_msg": status.sim_to_real_time_ratio_msg,
            "status": status.to_dict(),
            "base_pose": [float(v) for v in sim.get_base_pose()],
            "ee_pose_4x4": matrix_to_rows(sim.get_ee_pose()),
            "sensors": {
                "base_gyro": summarize_array(sensors.base_gyro),
                "base_imu": summarize_array(sensors.base_imu),
                "lidar": summarize_array(sensors.lidar),
            },
        }
    finally:
        sim.stop()


def main() -> int:
    args = parse_args()
    ensure_validation_dirs()

    output_dir = Path(args.output_dir) if args.output_dir else DATA_ROOT / args.tag
    output_dir = ensure_dir(output_dir)
    screenshot_dir = ensure_dir(SCREENSHOT_ROOT)
    scene_xml_path = resolve_scene_xml_path(Path(args.scene_xml_path))

    save_json(
        output_dir / "run_context.json",
        {
            "tag": args.tag,
            "scene_xml_path": str(scene_xml_path),
            "wall_time_utc": utc_timestamp(),
            "settle_seconds": args.settle_seconds,
            "camera_hz": args.camera_hz,
            "capture_sessions": CAPTURE_SESSIONS,
        },
    )

    session_summaries: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    intrinsics_payload: dict[str, object] = {}
    status_summary = capture_status_session(scene_xml_path)
    save_json(
        output_dir / "status_snapshot.json",
        {
            "wall_time_utc": utc_timestamp(),
            "scene_xml_path": str(scene_xml_path),
            "primary_session": "status_only",
            "session_summary": status_summary,
        },
    )

    for session in CAPTURE_SESSIONS:
        try:
            session_summary = capture_session(
                scene_xml_path=scene_xml_path,
                output_dir=output_dir,
                screenshot_dir=screenshot_dir,
                tag=args.tag,
                session_name=session["name"],
                cameras_to_use=session["cameras"],
                camera_hz=args.camera_hz,
                settle_seconds=args.settle_seconds,
            )
            session_summaries.append(session_summary)
            intrinsics_payload.update(session_summary.get("intrinsics", {}))
        except Exception as exc:
            error_payload = {
                "session_name": session["name"],
                "required": session["required"],
                "cameras": [camera.name for camera in session["cameras"]],
                "error": str(exc),
            }
            errors.append(error_payload)
            if session["required"]:
                save_json(output_dir / "capture_errors.json", errors)
                raise

    if errors:
        save_json(output_dir / "capture_errors.json", errors)

    save_json(output_dir / "camera_intrinsics.json", intrinsics_payload)
    save_json(
        output_dir / "sample_manifest.json",
        {
            "scene_xml_path": str(scene_xml_path),
            "wall_time_utc": utc_timestamp(),
            "status_session": status_summary,
            "sessions": session_summaries,
            "errors": errors,
            "intrinsics": intrinsics_payload,
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
