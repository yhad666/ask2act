from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline_types import GraspCandidate, TargetSelection
from validation_common import ensure_dir, matrix_to_rows, save_json


@dataclass
class LocalRegionGeometry:
    segmask: np.ndarray
    point_cloud_head: np.ndarray
    point_cloud_world: np.ndarray
    centroid_head_m: list[float]
    centroid_world_m: list[float]
    support_z_m: float
    top_z_m: float
    extent_xyz_m: list[float]
    bbox_xyxy: list[int]
    metadata: dict[str, Any]


def create_local_segmask(
    target: TargetSelection,
    *,
    image_shape: tuple[int, ...],
) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    x0, y0, x1, y1 = [int(v) for v in target.bbox_xyxy]
    x0 = max(0, min(width - 1, x0))
    x1 = max(x0 + 1, min(width, x1))
    y0 = max(0, min(height - 1, y0))
    y1 = max(y0 + 1, min(height, y1))
    mask[y0:y1, x0:x1] = 255

    if target.mask_path is None:
        return mask

    loaded = cv2.imread(str(Path(target.mask_path)), cv2.IMREAD_GRAYSCALE)
    if loaded is None:
        raise ValueError(f"Failed to read target mask image: {target.mask_path}")

    if loaded.shape != mask.shape:
        loaded = cv2.resize(loaded, (width, height), interpolation=cv2.INTER_NEAREST)

    return np.where(loaded > 0, 255, 0).astype(np.uint8)


def depth_mask_to_point_cloud(
    depth_m: np.ndarray,
    k_matrix: np.ndarray,
    segmask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (segmask > 0) & np.isfinite(depth_m) & (depth_m > 0.0)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 2), dtype=int)

    depths = depth_m[ys, xs].astype(float)
    fx = float(k_matrix[0, 0])
    fy = float(k_matrix[1, 1])
    cx = float(k_matrix[0, 2])
    cy = float(k_matrix[1, 2])
    x = (xs.astype(float) - cx) * depths / fx
    y = (ys.astype(float) - cy) * depths / fy
    points = np.stack([x, y, depths], axis=1)
    pixels = np.stack([xs, ys], axis=1)
    return points, pixels


def build_depth_fallback_segmask(depth_m: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    if not np.any(valid):
        return np.zeros_like(depth_m, dtype=np.uint8)

    depth_values = depth_m[valid]
    near_depth = float(np.percentile(depth_values, 15))
    near_mask = valid & (depth_m <= near_depth + 0.06)
    lower_half_mask = np.zeros_like(near_mask, dtype=bool)
    lower_half_mask[depth_m.shape[0] // 3 :, :] = True
    candidate_mask = near_mask & lower_half_mask
    if candidate_mask.sum() < 30:
        candidate_mask = near_mask
    if candidate_mask.sum() < 30:
        return np.zeros_like(depth_m, dtype=np.uint8)

    components = cv2.connectedComponentsWithStats(candidate_mask.astype(np.uint8), connectivity=8)
    _, labels, stats, _ = components
    if stats.shape[0] <= 1:
        return (candidate_mask.astype(np.uint8) * 255).astype(np.uint8)

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(component_areas.argmax())
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def transform_points(points_xyz: np.ndarray, pose_4x4: np.ndarray) -> np.ndarray:
    if points_xyz.size == 0:
        return np.empty((0, 3), dtype=float)
    points_h = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=float)], axis=1)
    transformed = (pose_4x4 @ points_h.T).T
    return transformed[:, :3]


def summarize_local_region(
    *,
    target: TargetSelection,
    head_observation: dict[str, Any],
) -> LocalRegionGeometry:
    segmask = create_local_segmask(target, image_shape=head_observation["depth"].shape)
    cloud_head, pixels = depth_mask_to_point_cloud(
        head_observation["depth"],
        np.asarray(head_observation["k_matrix"], dtype=float),
        segmask,
    )
    if cloud_head.shape[0] == 0:
        segmask = build_depth_fallback_segmask(np.asarray(head_observation["depth"], dtype=float))
        cloud_head, pixels = depth_mask_to_point_cloud(
            head_observation["depth"],
            np.asarray(head_observation["k_matrix"], dtype=float),
            segmask,
        )
        if cloud_head.shape[0] == 0:
            raise ValueError("Local target region has no valid depth pixels.")
        fallback_mode = "global_depth_fallback"
    else:
        fallback_mode = "manual_target"

    camera_pose = np.asarray(head_observation["camera_pose_4x4"], dtype=float)
    cloud_world = transform_points(cloud_head, camera_pose)
    centroid_head = cloud_head.mean(axis=0)
    centroid_world = cloud_world.mean(axis=0)
    extent = cloud_world.max(axis=0) - cloud_world.min(axis=0)
    support_z = float(np.percentile(cloud_world[:, 2], 5))
    top_z = float(np.percentile(cloud_world[:, 2], 95))
    bbox = [
        int(pixels[:, 0].min()),
        int(pixels[:, 1].min()),
        int(pixels[:, 0].max()) + 1,
        int(pixels[:, 1].max()) + 1,
    ]
    return LocalRegionGeometry(
        segmask=segmask,
        point_cloud_head=cloud_head,
        point_cloud_world=cloud_world,
        centroid_head_m=centroid_head.astype(float).tolist(),
        centroid_world_m=centroid_world.astype(float).tolist(),
        support_z_m=support_z,
        top_z_m=top_z,
        extent_xyz_m=extent.astype(float).tolist(),
        bbox_xyxy=bbox,
        metadata={
            "point_count": int(cloud_head.shape[0]),
            "head_camera_pose_4x4": matrix_to_rows(camera_pose),
            "segmentation_mode": fallback_mode,
        },
    )


def save_local_region_artifacts(output_dir: Path, geometry: LocalRegionGeometry) -> dict[str, str]:
    ensure_dir(output_dir)
    segmask_path = output_dir / "local_segmask.png"
    point_cloud_path = output_dir / "local_region_geometry.json"
    cv2.imwrite(str(segmask_path), geometry.segmask)
    save_json(
        point_cloud_path,
        {
            "centroid_head_m": geometry.centroid_head_m,
            "centroid_world_m": geometry.centroid_world_m,
            "support_z_m": geometry.support_z_m,
            "top_z_m": geometry.top_z_m,
            "extent_xyz_m": geometry.extent_xyz_m,
            "bbox_xyxy": geometry.bbox_xyxy,
            "metadata": geometry.metadata,
            "point_cloud_head_sample": geometry.point_cloud_head[:200].tolist(),
            "point_cloud_world_sample": geometry.point_cloud_world[:200].tolist(),
        },
    )
    return {
        "segmask_path": str(segmask_path),
        "geometry_path": str(point_cloud_path),
    }


class GraspProposalBackend:
    name = "base_backend"

    def propose_grasps(
        self,
        *,
        target: TargetSelection,
        geometry: LocalRegionGeometry,
    ) -> list[GraspCandidate]:
        raise NotImplementedError


class MockGraspProposalBackend(GraspProposalBackend):
    name = "mock_contact_graspnet"

    @staticmethod
    def _pose_from_translation_yaw(translation_xyz: np.ndarray, yaw_rad: float) -> np.ndarray:
        c = float(np.cos(yaw_rad))
        s = float(np.sin(yaw_rad))
        rotation = np.array(
            [
                [c, s, 0.0],
                [s, -c, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        )
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation_xyz
        return pose

    def propose_grasps(
        self,
        *,
        target: TargetSelection,
        geometry: LocalRegionGeometry,
    ) -> list[GraspCandidate]:
        centroid_head = np.asarray(geometry.centroid_head_m, dtype=float)
        centroid_world = np.asarray(geometry.centroid_world_m, dtype=float)
        extents = np.asarray(geometry.extent_xyz_m, dtype=float)
        width_guess = float(np.clip(max(extents[0], extents[1]) + 0.01, 0.025, 0.095))
        standoff = 0.09

        variants = [
            {
                "name": "top_down_center",
                "yaw": 0.0,
                "world_offset": np.array([0.0, 0.0, 0.0], dtype=float),
                "score": 0.92,
                "primitive_bonus": 0.18,
                "preferred_wrist_pitch_rad": -0.55,
            },
            {
                "name": "top_down_offset",
                "yaw": 0.35,
                "world_offset": np.array([0.015, -0.01, 0.005], dtype=float),
                "score": 0.83,
                "primitive_bonus": 0.12,
                "preferred_wrist_pitch_rad": -0.60,
            },
            {
                "name": "edge_bias",
                "yaw": -0.45,
                "world_offset": np.array([-0.02, 0.015, -0.01], dtype=float),
                "score": 0.58,
                "primitive_bonus": -0.04,
                "preferred_wrist_pitch_rad": -0.25,
            },
        ]

        candidates: list[GraspCandidate] = []
        for index, variant in enumerate(variants):
            world_translation = centroid_world + variant["world_offset"]
            head_translation = centroid_head.copy()
            head_translation[:2] += variant["world_offset"][:2]
            pose_head = self._pose_from_translation_yaw(head_translation, variant["yaw"])
            pose_world = self._pose_from_translation_yaw(world_translation, variant["yaw"])
            candidates.append(
                GraspCandidate(
                    candidate_id=f"mock_candidate_{index}",
                    pose_head_4x4=matrix_to_rows(pose_head),
                    pose_world_4x4=matrix_to_rows(pose_world),
                    width_m=float(np.clip(width_guess + 0.005 * index, 0.025, 0.095)),
                    score=float(variant["score"]),
                    approach_dir_world=[0.0, 0.0, -1.0],
                    source=self.name,
                    metadata={
                        "candidate_style": variant["name"],
                        "preferred_wrist_pitch_rad": float(variant["preferred_wrist_pitch_rad"]),
                        "preferred_wrist_yaw_rad": float(variant["yaw"]),
                        "pregrasp_standoff_m": float(standoff),
                        "primitive_bonus": float(variant["primitive_bonus"]),
                        "wrist_desired_depth_m": 0.22,
                        "extent_xyz_m": geometry.extent_xyz_m,
                    },
                )
            )
        return candidates
