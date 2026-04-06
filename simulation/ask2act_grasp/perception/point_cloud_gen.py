from __future__ import annotations

import numpy as np

from ask2act_grasp.types import PointCloudResult
from ask2act_grasp.utils.pcd_utils import backproject_depth, crop_depth_to_bbox
from ask2act_grasp.utils.tf_utils import transform_points


class PointCloudGenerator:
    def generate(
        self,
        *,
        depth_image: np.ndarray,
        camera_intrinsics: np.ndarray,
        camera_extrinsics: np.ndarray,
        table_top_z_m: float,
        table_margin_m: float,
        z_min_m: float,
        z_max_m: float,
        target_bbox_2d: tuple[int, int, int, int] | None = None,
    ) -> PointCloudResult:
        depth_for_projection = depth_image
        if target_bbox_2d is not None:
            depth_for_projection = crop_depth_to_bbox(depth_image, target_bbox_2d)

        camera_points = backproject_depth(depth_for_projection, camera_intrinsics)
        world_points = transform_points(camera_extrinsics, camera_points)
        valid = (
            np.isfinite(world_points[:, 2])
            & (world_points[:, 2] >= table_top_z_m + table_margin_m)
            & (world_points[:, 2] >= z_min_m)
            & (world_points[:, 2] <= z_max_m)
        )
        filtered_camera = camera_points[valid]
        filtered_world = world_points[valid]
        if filtered_world.size == 0 and world_points.size > 0:
            depth_valid = (
                np.isfinite(camera_points[:, 2])
                & (camera_points[:, 2] >= z_min_m)
                & (camera_points[:, 2] <= z_max_m)
            )
            filtered_camera = camera_points[depth_valid]
            filtered_world = world_points[depth_valid]
        return PointCloudResult(
            world_points_xyz=filtered_world,
            camera_points_xyz=filtered_camera,
            applied_bbox_2d=target_bbox_2d,
            filtered_point_count=int(filtered_world.shape[0]),
        )
