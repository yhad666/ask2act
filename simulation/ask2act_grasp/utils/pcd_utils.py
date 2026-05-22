from __future__ import annotations

import numpy as np


def backproject_depth(depth_image: np.ndarray, camera_intrinsics: np.ndarray) -> np.ndarray:
    height, width = depth_image.shape
    v_coords, u_coords = np.indices((height, width))
    z = depth_image.astype(float)
    valid = np.isfinite(z) & (z > 0.0)
    fx, fy = float(camera_intrinsics[0, 0]), float(camera_intrinsics[1, 1])
    cx, cy = float(camera_intrinsics[0, 2]), float(camera_intrinsics[1, 2])

    x = ((u_coords - cx) * z) / fx
    y = ((v_coords - cy) * z) / fy

    return np.stack([x[valid], y[valid], z[valid]], axis=1)


def crop_depth_to_bbox(depth_image: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox_xyxy
    cropped = np.zeros_like(depth_image)
    cropped[y0:y1, x0:x1] = depth_image[y0:y1, x0:x1]
    return cropped


def crop_depth_to_mask(depth_image: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_image)
    mask = np.asarray(mask_2d, dtype=bool)
    if depth.shape != mask.shape:
        raise ValueError(f"Depth/mask shape mismatch: depth={depth.shape}, mask={mask.shape}")
    cropped = np.zeros_like(depth)
    cropped[mask] = depth[mask]
    return cropped
