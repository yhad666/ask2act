from __future__ import annotations

import numpy as np

"""
Stretch 3 robot specifications used by the CGN grasp pipeline.

Measured values come from:
  simulation/scripts/measure_gripper.py

Important nuance:
  CGN poses are not fingertip contact points. Their grasp frame already sits
  behind the contact region by Contact-GraspNet's internal gripper depth
  (0.1034 m). We therefore keep the measured wrist->fingertip distance here,
  and derive the wrist->CGN-frame offset from it.
"""

STRETCH3_JOINT_LIMITS = {
    "lift": (0.0, 1.1),
    "arm": (0.0, 0.52),
    "wrist_yaw": (-1.39, 4.42),
    "wrist_pitch": (-1.571, 0.56),
    "wrist_roll": (-3.14, 3.14),
    "head_pan": (-4.04, 1.73),
    "head_tilt": (-1.53, 0.79),
}

STRETCH3_VELOCITIES = {
    "lift": 0.15,
    "arm": 0.4,
    "base_linear": 0.3,
    "base_rotation": 1.90,
    "wrist_yaw": 2.98,
}

STRETCH3_FORCES = {
    "lift_up": 38.1,
    "lift_down": 55.8,
    "arm_extend": 56.3,
    "arm_retract": 39.5,
    "wrist_yaw_torque": 4.56,
    "wrist_pitch_torque": 10.6,
    "wrist_roll_torque": 4.1,
}

STRETCH3_GRIPPER = {
    "finger_length_m": 0.257404,
    "wrist_to_grasp_center_m": 0.271673,
    "grasp_center_to_fingertip_contact_center_m": 0.014348,
    "max_aperture_m": 0.09,
    "min_aperture_m": 0.0,
    "sim_command_min": -0.376,
    "sim_command_max": 0.56,
    "sim_joint_ctrl_min": -0.02,
    "sim_joint_ctrl_max": 0.04,
    "weight_with_gripper_kg": 0.695,
}

CGN_GRIPPER_DEPTH_M = 0.1034
WRIST_TO_FINGER_MID_LOCAL_M = np.array(
    [
        -0.01507953,
        -0.10251649,
        0.04029956,
    ],
    dtype=float,
)
TOPDOWN_WRIST_TO_RUBBER_OFFSET = np.array(
    [-0.000560, 0.023975, -0.283433],
    dtype=float,
)
TOPDOWN_SIMPLEIK_WRIST_MODEL_ERROR_COEFFS = np.array(
    [
        [-0.017609620889587427, 0.001582897248282331, 0.025065181129398574],
        [0.02773911295735246, 0.006048439231415709, 0.0002945037038626372],
        [-0.04985861500033883, 9.744875202815197e-05, 0.009264418110996119],
    ],
    dtype=float,
)
TOPDOWN_WRIST_TO_GRASP_CENTER_OFFSET = np.array(
    [-0.0004421260608655532, 0.02406606210412511, -0.30040947920693084],
    dtype=float,
)
TOPDOWN_WRIST_TO_RUBBER_OFFSET_COEFFS = np.array(
    [
        [-0.0005581459276297099, 0.024011680765833352, -0.28823440667573147],
        [-0.0002597871736280639, -9.689045612677711e-05, -0.04942800810490048],
    ],
    dtype=float,
)
TOPDOWN_GRASP_CENTER_TO_RUBBER_OFFSET_COEFFS = np.array(
    [
        [-0.00011582920447485673, -5.435895584092837e-05, 0.01217507408878828],
        [-0.00026127200618380824, -9.706476532965171e-05, -0.04942802023503287],
    ],
    dtype=float,
)
TOPDOWN_SIMPLEIK_EXECUTION_Z_BIAS_M = 0.0


def clip_to_joint_limits(joint_name: str, value: float) -> float:
    lower, upper = STRETCH3_JOINT_LIMITS[joint_name]
    return float(np.clip(value, lower, upper))


def gripper_width_to_command(width_m: float) -> float:
    width = float(np.clip(width_m, STRETCH3_GRIPPER["min_aperture_m"], STRETCH3_GRIPPER["max_aperture_m"]))
    ratio = width / max(STRETCH3_GRIPPER["max_aperture_m"], 1e-8)
    cmd_span = STRETCH3_GRIPPER["sim_command_max"] - STRETCH3_GRIPPER["sim_command_min"]
    return float(STRETCH3_GRIPPER["sim_command_min"] + ratio * cmd_span)


def gripper_close_command() -> float:
    return float(STRETCH3_GRIPPER["sim_command_min"])


def wrist_to_cgn_grasp_frame_offset_m() -> float:
    return float(max(STRETCH3_GRIPPER["finger_length_m"] - CGN_GRIPPER_DEPTH_M, 0.0))


def cgn_frame_to_wrist_local_m() -> np.ndarray:
    contact_local = np.array([0.0, 0.0, CGN_GRIPPER_DEPTH_M], dtype=float)
    return contact_local - WRIST_TO_FINGER_MID_LOCAL_M


def topdown_simpleik_wrist_model_error_m(lift_m: float, arm_m: float) -> np.ndarray:
    features = np.array([1.0, float(lift_m), float(arm_m)], dtype=float)
    return features @ TOPDOWN_SIMPLEIK_WRIST_MODEL_ERROR_COEFFS


def topdown_wrist_to_rubber_offset_m(gripper_cmd: float) -> np.ndarray:
    features = np.array([1.0, float(gripper_cmd)], dtype=float)
    return features @ TOPDOWN_WRIST_TO_RUBBER_OFFSET_COEFFS


def topdown_wrist_to_grasp_center_offset_m() -> np.ndarray:
    return np.asarray(TOPDOWN_WRIST_TO_GRASP_CENTER_OFFSET, dtype=float).copy()


def topdown_grasp_center_to_rubber_offset_m(gripper_cmd: float) -> np.ndarray:
    features = np.array([1.0, float(gripper_cmd)], dtype=float)
    return features @ TOPDOWN_GRASP_CENTER_TO_RUBBER_OFFSET_COEFFS


def rotate_topdown_offset_to_world_m(offset_m: np.ndarray | list[float], yaw_rad: float) -> np.ndarray:
    """Rotate a canonical top-down local offset into the world frame.

    The calibrated top-down offsets are measured in a canonical pose where the
    gripper points straight down and its in-plane heading is zero. When the
    base rotates, or when we add a wrist yaw for a thin object, the horizontal
    components of that offset must rotate with the gripper instead of staying
    fixed in world coordinates.
    """

    offset = np.asarray(offset_m, dtype=float).reshape(3)
    cos_yaw = float(np.cos(yaw_rad))
    sin_yaw = float(np.sin(yaw_rad))
    return np.array(
        [
            cos_yaw * offset[0] - sin_yaw * offset[1],
            sin_yaw * offset[0] + cos_yaw * offset[1],
            offset[2],
        ],
        dtype=float,
    )
