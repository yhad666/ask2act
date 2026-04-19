from __future__ import annotations

import math

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


def make_transform(translation_xyz: tuple[float, float, float], rpy_rad: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_matrix_from_rpy(*rpy_rad)
    transform[:3, 3] = np.asarray(translation_xyz, dtype=float)
    return transform


def transform_points(transform: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xyz.shape[0], 1), dtype=float)
    homogeneous = np.concatenate([points_xyz, ones], axis=1)
    world = (transform @ homogeneous.T).T
    return world[:, :3]


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return vector.copy()
    return vector / norm


def pose_from_axes(position_m: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 0] = normalize(x_axis)
    pose[:3, 1] = normalize(y_axis)
    pose[:3, 2] = normalize(z_axis)
    pose[:3, 3] = np.asarray(position_m, dtype=float)
    return pose

