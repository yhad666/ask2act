from __future__ import annotations

from pathlib import Path

import numpy as np

from ask2act_grasp.types import GraspCandidate


def save_point_cloud(points_xyz: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    try:
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points_xyz)
        o3d.io.write_point_cloud(str(output_path), cloud)
        return
    except ModuleNotFoundError:
        pass

    fallback_path = output_path.with_suffix(".xyz")
    np.savetxt(fallback_path, points_xyz, fmt="%.6f")


def save_grasp_debug_scene(points_xyz: np.ndarray, candidates: list[GraspCandidate], output_path: str | Path) -> None:
    overlay_points = [np.asarray(points_xyz, dtype=float)]
    for candidate in candidates:
        overlay_points.append(_grasp_wireframe_points(candidate.pose_4x4, candidate.width_m))
    merged = np.concatenate(overlay_points, axis=0) if overlay_points else np.empty((0, 3), dtype=float)
    save_point_cloud(merged, output_path)


def _grasp_wireframe_points(pose_4x4: np.ndarray, width_m: float) -> np.ndarray:
    """Create a coarse point-sampled wireframe for a parallel-jaw grasp."""
    pose = np.asarray(pose_4x4, dtype=float)
    position = pose[:3, 3]
    x_axis = pose[:3, 0]
    y_axis = pose[:3, 1]
    z_axis = pose[:3, 2]
    jaw_half_width = max(float(width_m) / 2.0, 0.015)
    finger_length = 0.04
    palm_half_width = 0.01

    left_base = position - x_axis * jaw_half_width
    right_base = position + x_axis * jaw_half_width
    left_tip = left_base - z_axis * finger_length
    right_tip = right_base - z_axis * finger_length
    palm_left = position - y_axis * palm_half_width
    palm_right = position + y_axis * palm_half_width
    segments = [
        (left_base, left_tip),
        (right_base, right_tip),
        (palm_left, palm_right),
        (left_base, palm_left),
        (right_base, palm_right),
    ]
    samples = [_sample_segment(start, end) for start, end in segments]
    return np.concatenate(samples, axis=0)


def _sample_segment(start_xyz: np.ndarray, end_xyz: np.ndarray, num_points: int = 12) -> np.ndarray:
    """Uniformly sample a 3D segment into a small point cloud."""
    t_values = np.linspace(0.0, 1.0, num=num_points, dtype=float)[:, None]
    return (1.0 - t_values) * start_xyz[None, :] + t_values * end_xyz[None, :]
