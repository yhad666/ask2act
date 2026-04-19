from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import urchin as urdf_loader


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def _matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation)
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation[2, 1] - rotation[1, 2]) * s
        y = (rotation[0, 2] - rotation[2, 0]) * s
        z = (rotation[1, 0] - rotation[0, 1]) * s
    else:
        if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            s = 2.0 * math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif rotation[1, 1] > rotation[2, 2]:
            s = 2.0 * math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
    return np.array([x, y, z, w], dtype=float)


def _format_transform(name: str, transform: np.ndarray) -> str:
    xyz = transform[:3, 3]
    rpy = _matrix_to_rpy(transform[:3, :3])
    quat_xyzw = _matrix_to_quat_xyzw(transform[:3, :3])
    return (
        f"{name}\n"
        f"  xyz_m        = {np.array2string(xyz, precision=6, suppress_small=False)}\n"
        f"  rpy_rad      = {np.array2string(rpy, precision=6, suppress_small=False)}\n"
        f"  quat_xyzw    = {np.array2string(quat_xyzw, precision=6, suppress_small=False)}\n"
        f"  transform_4x4=\n{transform}"
    )


def _find_link_fk(urdf: urdf_loader.URDF, cfg: dict[str, float], links: Iterable[str]) -> dict[str, np.ndarray]:
    filtered_cfg = {joint.name: cfg[joint.name] for joint in urdf.actuated_joints if joint.name in cfg}
    link_fk = urdf.link_fk(cfg=filtered_cfg, links=list(links))
    result: dict[str, np.ndarray] = {}
    for link_name in links:
        result[link_name] = np.asarray(link_fk[urdf.link_map[link_name]], dtype=float)
    return result


def default_urdf_path() -> Path:
    return Path(
        "/home/yhad/robot/ask2act/simulation/stretch_mujoco/.venv/lib/python3.10/site-packages/stretch_urdf/SE3/"
        "stretch_description_SE3_eoa_wrist_dw3_tool_sg3.urdf"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, default=default_urdf_path())
    parser.add_argument("--wrist-pitch", type=float, default=-1.57)
    parser.add_argument("--wrist-roll", type=float, default=0.0)
    parser.add_argument("--wrist-yaw", type=float, default=0.0)
    parser.add_argument("--lift", type=float, default=0.9)
    parser.add_argument("--arm", type=float, default=0.3)
    args = parser.parse_args()

    urdf = urdf_loader.URDF.load(str(args.urdf), lazy_load_meshes=True)
    link_names = {
        "link_wrist_yaw",
        "link_wrist_yaw_bottom",
        "link_wrist_pitch",
        "link_wrist_roll",
        "link_wrist_quick_connect",
        "link_gripper_s3_body",
        "link_grasp_center",
        "link_gripper_finger_left",
        "link_gripper_fingertip_left",
        "link_gripper_finger_right",
        "link_gripper_fingertip_right",
    }
    missing = sorted(name for name in link_names if name not in urdf.link_map)
    if missing:
        raise RuntimeError(f"Missing links in URDF: {missing}")

    cfg = {
        "joint_lift": float(args.lift),
        "joint_arm_l0": float(args.arm),
        "joint_mobile_base_rotation": 0.0,
        "joint_wrist_yaw": float(args.wrist_yaw),
        "joint_wrist_pitch": float(args.wrist_pitch),
        "joint_wrist_roll": float(args.wrist_roll),
    }
    world = _find_link_fk(urdf, cfg, link_names)
    wrist_world = world["link_wrist_yaw"]
    wrist_inv = np.linalg.inv(wrist_world)

    print(f"URDF: {args.urdf}")
    print("rubber_tip_center exists in URDF:", "rubber_tip_center" in urdf.link_map)
    for name in [
        "link_wrist_yaw_bottom",
        "link_wrist_pitch",
        "link_wrist_roll",
        "link_wrist_quick_connect",
        "link_gripper_s3_body",
        "link_grasp_center",
        "link_gripper_finger_left",
        "link_gripper_fingertip_left",
        "link_gripper_finger_right",
        "link_gripper_fingertip_right",
    ]:
        rel = wrist_inv @ world[name]
        print()
        print(_format_transform(f"T(link_wrist_yaw -> {name})", rel))

    left_rel = wrist_inv @ world["link_gripper_fingertip_left"]
    right_rel = wrist_inv @ world["link_gripper_fingertip_right"]
    midpoint = np.eye(4, dtype=float)
    midpoint[:3, :3] = wrist_inv[:3, :3] @ world["link_grasp_center"][:3, :3]
    midpoint[:3, 3] = 0.5 * (left_rel[:3, 3] + right_rel[:3, 3])
    print()
    print(_format_transform("Midpoint estimate from left/right fingertip frames", midpoint))

    grasp_center_rel = wrist_inv @ world["link_grasp_center"]
    midpoint_from_grasp_center = np.linalg.inv(grasp_center_rel) @ midpoint
    print()
    print(_format_transform("T(link_grasp_center -> fingertip midpoint)", midpoint_from_grasp_center))


if __name__ == "__main__":
    main()
