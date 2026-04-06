from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


SIMULATION_ROOT = Path(__file__).resolve().parents[1]
STRETCH_VENV_PYTHON = SIMULATION_ROOT / "stretch_mujoco" / ".venv" / "bin" / "python"
if STRETCH_VENV_PYTHON.exists():
    current = Path(sys.executable)
    if current != STRETCH_VENV_PYTHON:
        os.execv(str(STRETCH_VENV_PYTHON), [str(STRETCH_VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
if str(SIMULATION_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATION_ROOT))
if str(SIMULATION_ROOT / "stretch_mujoco") not in sys.path:
    sys.path.insert(0, str(SIMULATION_ROOT / "stretch_mujoco"))

from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator

from ask2act_grasp.scene.scene_setup import SceneSetup
from ask2act_grasp.utils.config_loader import load_scene_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose head pan/tilt motion in stretch_mujoco.")
    parser.add_argument("--camera-profile", choices=["none", "head", "all"], default="none")
    parser.add_argument("--head-pan-rad", type=float, default=-1.57)
    parser.add_argument("--head-tilt-rad", type=float, default=-0.55)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", os.environ.get("STRETCH_SIM_HEADLESS_MUJOCO_GL", "egl"))

    scene_config, head_config = load_scene_config(SIMULATION_ROOT / "ask2act_grasp" / "config" / "scene_config.yaml")
    generated_scene_dir = SIMULATION_ROOT / "stretch_mujoco" / "stretch_mujoco" / "models" / "generated"
    generated_scene_dir.mkdir(parents=True, exist_ok=True)
    assets_link = generated_scene_dir / "assets"
    if not assets_link.exists():
        assets_link.symlink_to("../assets")
    scene_path = generated_scene_dir / "diag_head_motion_scene.xml"
    SceneSetup(scene_config).write_scene(scene_path)

    cameras_to_use = []
    if args.camera_profile == "head":
        cameras_to_use = [
            StretchCameras.cam_d435i_rgb,
            StretchCameras.cam_d435i_depth,
        ]
    elif args.camera_profile == "all":
        cameras_to_use = [
            StretchCameras.cam_d435i_rgb,
            StretchCameras.cam_d435i_depth,
            StretchCameras.cam_d405_rgb,
            StretchCameras.cam_d405_depth,
        ]

    sim = StretchMujocoSimulator(
        scene_xml_path=str(scene_path),
        camera_hz=head_config.camera_hz,
        cameras_to_use=cameras_to_use,
        start_translation=list(scene_config.robot_start_translation_m),
        start_rotation_quat=list(scene_config.robot_start_rotation_quat_xyzw),
    )

    try:
        sim.start(headless=True)
        start_status = sim.pull_status()
        sim.move_to(Actuators.head_pan, float(args.head_pan_rad))
        sim.move_to(Actuators.head_tilt, float(args.head_tilt_rad))
        pan_ok = sim.wait_until_at_setpoint(Actuators.head_pan, timeout=args.timeout_s)
        tilt_ok = sim.wait_until_at_setpoint(Actuators.head_tilt, timeout=args.timeout_s)
        end_status = sim.pull_status()
        payload = {
            "camera_profile": args.camera_profile,
            "requested": {
                "head_pan_rad": float(args.head_pan_rad),
                "head_tilt_rad": float(args.head_tilt_rad),
                "timeout_s": float(args.timeout_s),
            },
            "start": {
                "head_pan_rad": float(start_status.head_pan.pos),
                "head_tilt_rad": float(start_status.head_tilt.pos),
                "sim_time_s": float(start_status.time),
            },
            "end": {
                "head_pan_rad": float(end_status.head_pan.pos),
                "head_tilt_rad": float(end_status.head_tilt.pos),
                "sim_time_s": float(end_status.time),
            },
            "reached": {
                "head_pan": bool(pan_ok),
                "head_tilt": bool(tilt_ok),
            },
        }
        print(json.dumps(payload, indent=2))
        return 0
    finally:
        time.sleep(0.2)
        sim.stop()


if __name__ == "__main__":
    raise SystemExit(main())
