from __future__ import annotations

import argparse
import time
from pathlib import Path

from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras

from minimal_grasping_common import (
    create_run_dir,
    move_to_and_record,
    save_json,
    save_observation_bundle,
    sim_pose_snapshot,
    start_wrist_sim,
    wait_for_cameras,
)
from tabletop_scene_builder import resolve_scene_xml_path
from validation_common import ensure_dir
from stretch_mujoco import StretchMujocoSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture corrected head or wrist images at a specified robot pose.")
    parser.add_argument("--scene-xml-path", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--camera", choices=["head", "wrist", "both"], default="head")
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--camera-hz", type=float, default=15.0)
    parser.add_argument("--move-timeout-s", type=float, default=60.0)
    parser.add_argument("--home-first", action="store_true")
    parser.add_argument("--lift", type=float, default=None)
    parser.add_argument("--arm", type=float, default=None)
    parser.add_argument("--wrist-yaw", type=float, default=None)
    parser.add_argument("--wrist-pitch", type=float, default=None)
    parser.add_argument("--wrist-roll", type=float, default=None)
    parser.add_argument("--head-pan", type=float, default=None)
    parser.add_argument("--head-tilt", type=float, default=None)
    parser.add_argument("--gripper", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = create_run_dir(args.output_dir, args.run_tag, prefix="capture_pose")
    ensure_dir(output_dir)

    cameras = []
    if args.camera in ["head", "both"]:
        cameras.extend([StretchCameras.cam_d435i_rgb, StretchCameras.cam_d435i_depth])
    if args.camera in ["wrist", "both"]:
        cameras.extend([StretchCameras.cam_d405_rgb, StretchCameras.cam_d405_depth])

    sim = StretchMujocoSimulator(
        scene_xml_path=str(resolve_scene_xml_path(Path(args.scene_xml_path))),
        cameras_to_use=cameras,
        camera_hz=float(args.camera_hz),
    )
    sim.start(headless=True)
    try:
        if args.home_first:
            sim.home()
            time.sleep(float(args.settle_seconds))

        target_moves = [
            (Actuators.lift, args.lift),
            (Actuators.arm, args.arm),
            (Actuators.wrist_yaw, args.wrist_yaw),
            (Actuators.wrist_pitch, args.wrist_pitch),
            (Actuators.wrist_roll, args.wrist_roll),
            (Actuators.head_pan, args.head_pan),
            (Actuators.head_tilt, args.head_tilt),
            (Actuators.gripper, args.gripper),
        ]
        motion_log = []
        for actuator, target in target_moves:
            if target is None:
                continue
            before = actuator.get_position(sim.pull_status())
            sim.move_to(actuator, float(target))
            tolerance = 0.01 if actuator == Actuators.gripper else 0.02 if actuator in [Actuators.arm, Actuators.head_pan, Actuators.head_tilt] else 0.03
            reached = sim.wait_until_at_setpoint(
                actuator,
                timeout=float(args.move_timeout_s),
                position_tolerance=tolerance,
            )
            after = actuator.get_position(sim.pull_status())
            motion_log.append(
                {
                    "actuator": actuator.name,
                    "target": float(target),
                    "before": float(before),
                    "after": float(after),
                    "reached": bool(reached),
                    "tolerance": float(tolerance),
                    "abs_error": float(abs(after - float(target))),
                }
            )

        time.sleep(float(args.settle_seconds))
        camera_data = wait_for_cameras(sim, cameras)
        status = sim_pose_snapshot(sim)

        payload = {
            "scene_xml_path": str(Path(args.scene_xml_path).resolve()),
            "camera": args.camera,
            "motion_log": motion_log,
            "status": status,
        }
        save_json(output_dir / "capture_pose_context.json", payload)

        if args.camera in ["wrist", "both"]:
            observation = {
                "wall_time_utc": status["wall_time_utc"],
                "camera_time": float(camera_data.time),
                "camera_fps": float(camera_data.fps),
                "rgb": camera_data.get_camera_data(StretchCameras.cam_d405_rgb, auto_correct_rgb=False),
                "depth": camera_data.get_camera_data(StretchCameras.cam_d405_depth),
                "k_matrix": camera_data.cam_d405_K,
                "pose": status,
            }
            save_observation_bundle(output_dir, "pose", observation)

        if args.camera in ["head", "both"]:
            from validation_common import aligned_intrinsics_for_saved_image, save_depth_outputs, save_rgb_image, depth_to_color
            import cv2
            head_rgb = camera_data.get_camera_data(StretchCameras.cam_d435i_rgb, auto_correct_rgb=False)
            head_depth = camera_data.get_camera_data(StretchCameras.cam_d435i_depth)
            save_rgb_image(output_dir / "pose_head_rgb.png", head_rgb)
            save_depth_outputs(output_dir / "pose_head_depth", head_depth)
            cv2.imwrite(str(output_dir / "pose_head_depth_color.png"), depth_to_color(head_depth))
            save_json(
                output_dir / "pose_head_pose.json",
                {
                    "wall_time_utc": status["wall_time_utc"],
                    "camera_time": float(camera_data.time),
                    "camera_fps": float(camera_data.fps),
                    "k_matrix": aligned_intrinsics_for_saved_image(StretchCameras.cam_d435i_rgb),
                    "pose": status,
                },
            )
    finally:
        sim.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
