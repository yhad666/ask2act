from __future__ import annotations

import numpy as np

from ask2act_grasp.types import PointCloudResult
from ask2act_grasp.utils.pcd_utils import crop_depth_to_bbox, crop_depth_to_mask
from ask2act_grasp.utils.tf_utils import transform_points


class PointCloudGenerator:
    """Backproject a depth image into a filtered world-frame point cloud."""

    def add_sensor_noise(
        self,
        depth_image: np.ndarray,
        noise_sigma: float = 0.001,
        dropout_ratio: float = 0.01,
    ) -> np.ndarray:
        """Add light RealSense-like depth noise and sparse dropout to sim depth."""
        noisy = np.asarray(depth_image, dtype=np.float32).copy()
        valid_mask = noisy > 0.01
        if not np.any(valid_mask):
            return noisy

        valid_depth = noisy[valid_mask]
        mean_depth = float(np.mean(valid_depth)) if valid_depth.size else 1.0
        gaussian = np.random.normal(0.0, noise_sigma, size=noisy.shape).astype(np.float32)
        scaled_noise = np.where(valid_mask, gaussian * (noisy / max(mean_depth, 1e-6)), 0.0)
        noisy = np.where(valid_mask, noisy + scaled_noise, noisy)

        dropout_mask = np.random.random(size=noisy.shape) < float(dropout_ratio)
        noisy[dropout_mask & valid_mask] = 0.0
        return np.maximum(noisy, 0.0).astype(np.float32, copy=False)

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
        target_mask_2d: np.ndarray | None = None,
    ) -> PointCloudResult:
        """Generate a world-frame point cloud from depth and camera calibration."""
        depth_for_projection = np.asarray(depth_image, dtype=np.float32)
        if target_mask_2d is not None:
            depth_for_projection = crop_depth_to_mask(depth_for_projection, target_mask_2d)
        elif target_bbox_2d is not None:
            depth_for_projection = crop_depth_to_bbox(depth_for_projection, target_bbox_2d)

        camera_points = self._backproject_depth(depth_for_projection, np.asarray(camera_intrinsics, dtype=float))
        world_points = transform_points(np.asarray(camera_extrinsics, dtype=float), camera_points)
        valid_mask = (
            np.isfinite(world_points[:, 2])
            & (world_points[:, 2] > max(table_top_z_m + table_margin_m, z_min_m))
            & (world_points[:, 2] < z_max_m)
        )

        filtered_camera = camera_points[valid_mask]
        filtered_world = world_points[valid_mask]
        return PointCloudResult(
            world_points_xyz=filtered_world.astype(np.float64, copy=False),
            camera_points_xyz=filtered_camera.astype(np.float64, copy=False),
            applied_bbox_2d=target_bbox_2d,
            filtered_point_count=int(filtered_world.shape[0]),
        )

    def crop_to_object_region(
        self,
        pcd_world: np.ndarray,
        object_center_xy: np.ndarray | None = None,
        crop_radius: float = 0.08,
        table_z: float | None = None,
        height_min: float | None = None,
        height_max: float | None = None,
        return_mask: bool = False,
    ):
        """Crop a scene cloud to a fixed XY radius around the estimated object center.

        When ``object_center_xy`` is omitted this falls back to the legacy
        elevated-point median seed. The main geometric path now supplies an
        explicit center estimate upstream.
        """
        world_points = np.asarray(pcd_world, dtype=np.float64)
        if world_points.size == 0:
            if return_mask:
                return world_points, np.zeros((0,), dtype=bool), np.zeros((2,), dtype=np.float64)
            return world_points

        if object_center_xy is None:
            if table_z is None:
                raise ValueError("table_z is required when object_center_xy is not provided")
            table_height = float(table_z)
            above_table = world_points[world_points[:, 2] > table_height + 0.005]
            if len(above_table) < 3:
                print("WARNING: too few above-table points, returning full cloud")
                crop_mask = np.ones((len(world_points),), dtype=bool)
                estimated_center = np.median(world_points[:, :2], axis=0)
                if return_mask:
                    return world_points, crop_mask, estimated_center
                return world_points

            # Prefer a seed set that is clearly above the table plane so we center on
            # the object body rather than the entire visible scene.
            elevated_points = above_table[above_table[:, 2] > table_height + 0.04]
            if len(above_table) >= 100 and len(elevated_points) < 50:
                top_quantile = np.percentile(above_table[:, 2], 85.0)
                elevated_points = above_table[above_table[:, 2] >= top_quantile]
            if len(elevated_points) < 3:
                elevated_points = above_table

            estimated_center = np.median(elevated_points[:, :2], axis=0)
        else:
            estimated_center = np.asarray(object_center_xy, dtype=np.float64).reshape(2)

        xy_dist = np.linalg.norm(world_points[:, :2] - estimated_center[None, :], axis=1)
        xy_mask = xy_dist < float(crop_radius)
        xy_cropped = world_points[xy_mask]
        print(
            f"Crop: {len(world_points)} -> {len(xy_cropped)} points "
            f"(center={np.round(estimated_center, 3)}, r={crop_radius}m)"
        )

        crop_mask = xy_mask.copy()
        cropped = xy_cropped
        if len(xy_cropped) and table_z is not None:
            effective_height_min = float(height_min) if height_min is not None else float(table_z) + 0.01
            effective_height_max = float(height_max) if height_max is not None else float(table_z) + 0.12
            z_mask = (xy_cropped[:, 2] > effective_height_min) & (xy_cropped[:, 2] < effective_height_max)
            print(
                f"Height filter: {int(z_mask.sum())}/{len(z_mask)} points kept "
                f"(z={effective_height_min:.3f} to {effective_height_max:.3f})"
            )
            cropped = xy_cropped[z_mask]
            crop_mask[xy_mask] = z_mask
        if return_mask:
            return cropped, crop_mask, estimated_center
        return cropped

    def subsample(
        self,
        pcd: np.ndarray,
        target_n: int = 2048,
        return_indices: bool = False,
    ):
        """Match CGN's expected point density via uniform random down/oversampling."""
        points = np.asarray(pcd, dtype=np.float64)
        point_count = len(points)
        if point_count == 0:
            empty_idx = np.zeros((0,), dtype=np.int64)
            if return_indices:
                return points, empty_idx
            return points

        if point_count > target_n:
            indices = np.random.choice(point_count, target_n, replace=False)
        elif point_count < target_n:
            indices = np.concatenate(
                [
                    np.arange(point_count, dtype=np.int64),
                    np.random.choice(point_count, target_n - point_count, replace=True),
                ]
            )
        else:
            indices = np.arange(point_count, dtype=np.int64)

        sampled = points[indices]
        if return_indices:
            return sampled, indices
        return sampled

    @staticmethod
    def _backproject_depth(depth_image: np.ndarray, camera_intrinsics: np.ndarray) -> np.ndarray:
        """Backproject valid depth pixels into camera coordinates."""
        height, width = depth_image.shape
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
        valid_mask = np.isfinite(depth_image) & (depth_image > 0.01)
        if not np.any(valid_mask):
            return np.empty((0, 3), dtype=np.float64)

        z_cam = depth_image[valid_mask].astype(np.float64)
        fx = float(camera_intrinsics[0, 0])
        fy = float(camera_intrinsics[1, 1])
        cx = float(camera_intrinsics[0, 2])
        cy = float(camera_intrinsics[1, 2])
        x_cam = (u_coords[valid_mask] - cx) * z_cam / fx
        y_cam = (v_coords[valid_mask] - cy) * z_cam / fy
        return np.stack([x_cam, y_cam, z_cam], axis=-1)
