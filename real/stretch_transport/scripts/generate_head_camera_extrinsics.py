from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
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


def make_transform(xyz: Iterable[float], rpy: Iterable[float]) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_matrix_from_rpy(*[float(value) for value in rpy])
    transform[:3, 3] = np.asarray([float(value) for value in xyz], dtype=float)
    return transform


def z_axis_rotation(angle_rad: float) -> np.ndarray:
    cos_v = math.cos(angle_rad)
    sin_v = math.sin(angle_rad)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.array(
        [
            [cos_v, -sin_v, 0.0],
            [sin_v, cos_v, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return transform


def compute_camera_color_optical_transform(head_pan_rad: float, head_tilt_rad: float) -> np.ndarray:
    """Compute base_link -> camera_color_optical_frame from the Stretch SE3 URDF chain."""
    chain = [
        make_transform(
            (-0.06494455220452494, 0.13287700932848, 0.018903732885073965),
            (1.5771833238322903, -0.023672995513785322, -0.009349444571737697),
        ),
        make_transform((0.0, 1.33, 0.0), (1.5707963267949, -1.5707963267949, 3.1416)),
        make_transform(
            (0.12901223313352317, 0.06909628156918923, -0.008195432278471772),
            (0.00014688273627883053, 0.009462252869463672, 1.527436382729067),
        ),
        z_axis_rotation(head_pan_rad),
        make_transform(
            (-0.004009625698851536, 0.030834696914169, -0.06156785564304229),
            (1.5711269454428733, -0.08081421283520851, -0.008956447574033688),
        ),
        z_axis_rotation(head_tilt_rad),
        make_transform(
            (0.02568974707618361, -0.012249988586924835, 0.020336962630221625),
            (0.01958615052546083, 0.00228014272541599, 0.03534366391968014),
        ),
        make_transform((0.010600000000000002, 0.0175, 0.0125), (0.0, 0.0, 0.0)),
        make_transform((0.0, 0.015, 0.0), (0.0, 0.0, 0.0)),
        make_transform((0.0, 0.0, 0.0), (-1.5707963267948966, 0.0, -1.5707963267948966)),
    ]
    transform = np.eye(4, dtype=float)
    for step in chain:
        transform = transform @ step
    return transform


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parents[1] / "head_camera_extrinsics.json"
    parser = argparse.ArgumentParser(description="Generate head D435i camera extrinsics from the Stretch SE3 URDF chain.")
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument(
        "--head-pan-rad",
        type=float,
        default=float(os.getenv("ASK2ACT_STRETCH_INIT_HEAD_PAN_RAD", "-1.57")),
    )
    parser.add_argument(
        "--head-tilt-rad",
        type=float,
        default=float(os.getenv("ASK2ACT_STRETCH_INIT_HEAD_TILT_RAD", "-0.55")),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    transform = compute_camera_color_optical_transform(args.head_pan_rad, args.head_tilt_rad)
    payload = {
        "source": "stretch_se3_urdf_chain",
        "parent_frame": "base_link",
        "child_frame": "camera_color_optical_frame",
        "head_pan_rad": float(args.head_pan_rad),
        "head_tilt_rad": float(args.head_tilt_rad),
        "note": "Camera-to-base transform derived from the robot URDF for the configured tabletop head pose.",
        "camera_to_world": transform.tolist(),
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
