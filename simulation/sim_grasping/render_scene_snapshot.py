from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

from tabletop_scene_builder import resolve_scene_xml_path
from validation_common import DEFAULT_SCENE_PATH, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an offscreen scene snapshot from MuJoCo XML.")
    parser.add_argument("--scene-xml-path", default=str(DEFAULT_SCENE_PATH))
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-name", default=None)
    parser.add_argument("--azimuth", type=float, default=145.0)
    parser.add_argument("--elevation", type=float, default=-20.0)
    parser.add_argument("--distance", type=float, default=2.2)
    parser.add_argument("--lookat", nargs=3, type=float, default=[0.0, -0.6, 0.55])
    parser.add_argument("--keyframe", default=None)
    parser.add_argument("--ctrl", action="append", default=[])
    parser.add_argument("--settle-steps", type=int, default=300)
    return parser.parse_args()


def parse_ctrl_assignments(assignments: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for assignment in assignments:
        name, value = assignment.split("=", 1)
        parsed[name.strip()] = float(value)
    return parsed


def main() -> int:
    args = parse_args()
    scene_xml_path = resolve_scene_xml_path(Path(args.scene_xml_path))
    output_path = ensure_dir(Path(args.output_path).resolve().parent) / Path(args.output_path).name

    model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
    data = mujoco.MjData(model)

    if args.keyframe:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
        if key_id < 0:
            raise ValueError(f"Keyframe not found: {args.keyframe}")
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    if args.width > model.vis.global_.offwidth:
        model.vis.global_.offwidth = args.width
    if args.height > model.vis.global_.offheight:
        model.vis.global_.offheight = args.height

    for actuator_name, target in parse_ctrl_assignments(args.ctrl).items():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise ValueError(f"Actuator not found: {actuator_name}")
        data.ctrl[actuator_id] = target

    for _ in range(args.settle_steps):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, width=args.width, height=args.height)
    try:
        if args.camera_name:
            renderer.update_scene(data, camera=args.camera_name)
        else:
            free_camera = mujoco.MjvCamera()
            free_camera.azimuth = args.azimuth
            free_camera.elevation = args.elevation
            free_camera.distance = args.distance
            free_camera.lookat = np.array(args.lookat, dtype=float)
            renderer.update_scene(data, camera=free_camera)

        image = renderer.render()
    finally:
        renderer.close()

    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
