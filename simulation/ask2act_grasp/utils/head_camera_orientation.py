from __future__ import annotations

import numpy as np


CW90_CAMERA_FRAME_ROTATION = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def rotate_intrinsics_clockwise_90(
    camera_intrinsics: np.ndarray,
    *,
    original_depth_shape: tuple[int, int],
) -> np.ndarray:
    """Adjust camera intrinsics after rotating the image clockwise by 90 degrees."""
    height, _width = original_depth_shape
    fx = float(camera_intrinsics[0, 0])
    fy = float(camera_intrinsics[1, 1])
    cx = float(camera_intrinsics[0, 2])
    cy = float(camera_intrinsics[1, 2])
    return np.array(
        [
            [fy, 0.0, height - 1.0 - cy],
            [0.0, fx, cx],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def rotate_camera_extrinsics_clockwise_90(camera_extrinsics: np.ndarray) -> np.ndarray:
    """Rotate the camera frame so an upright image backprojects to the same world points."""
    rotated = np.array(camera_extrinsics, dtype=float, copy=True)
    rotated[:3, :3] = rotated[:3, :3] @ CW90_CAMERA_FRAME_ROTATION
    return rotated


def rotate_head_observation_clockwise(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_extrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rotate head-camera RGB-D and calibration clockwise so saved images are upright."""
    rotated_rgb = np.rot90(np.asarray(rgb_image), k=-1)
    rotated_depth = np.rot90(np.asarray(depth_image), k=-1)
    rotated_intrinsics = rotate_intrinsics_clockwise_90(
        np.asarray(camera_intrinsics, dtype=float),
        original_depth_shape=np.asarray(depth_image).shape,
    )
    rotated_extrinsics = rotate_camera_extrinsics_clockwise_90(np.asarray(camera_extrinsics, dtype=float))
    return rotated_rgb, rotated_depth, rotated_intrinsics, rotated_extrinsics
