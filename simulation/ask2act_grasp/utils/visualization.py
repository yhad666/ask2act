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
    del candidates
    save_point_cloud(points_xyz, output_path)
