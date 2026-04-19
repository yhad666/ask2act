from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ask2act_grasp.utils.tf_utils import transform_points


@dataclass(frozen=True)
class TablePlaneEstimate:
    normal: np.ndarray
    offset: float
    used: str
    residual: float


def estimate_table_plane(
    *,
    table_top_z_m: float,
    world_points: np.ndarray | None = None,
    prefer_known_plane: bool = True,
) -> TablePlaneEstimate:
    """Estimate the tabletop plane in world coordinates.

    By default we use the known horizontal tabletop plane:
        n = [0, 0, 1], d = -table_top_z_m
    This is the stable path for the current simulation setup.
    """
    table_top_z_m = float(table_top_z_m)
    if prefer_known_plane or world_points is None or len(world_points) < 3:
        return TablePlaneEstimate(
            normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            offset=-table_top_z_m,
            used="known_plane",
            residual=0.0,
        )

    points = np.asarray(world_points, dtype=np.float64)
    centroid = np.median(points, axis=0)
    centered = points - centroid[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = np.asarray(vh[-1], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return TablePlaneEstimate(
            normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            offset=-table_top_z_m,
            used="known_plane",
            residual=0.0,
        )
    normal /= norm
    if normal[2] < 0.0:
        normal *= -1.0
    offset = -float(normal @ centroid)
    residual = float(np.sqrt(np.mean((points @ normal + offset) ** 2)))
    return TablePlaneEstimate(
        normal=normal,
        offset=offset,
        used="svd_fit",
        residual=residual,
    )


def select_support_pixels_from_mask(
    mask: np.ndarray,
    *,
    support_band_ratio: float = 0.10,
    support_middle_ratio: float = 0.60,
) -> np.ndarray:
    """Select pixels from the bottom-middle support band of a binary mask."""
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.ndim != 2 or not np.any(mask_array):
        return np.zeros((0, 2), dtype=np.int32)

    v_coords, u_coords = np.nonzero(mask_array)
    u_min = int(u_coords.min())
    u_max = int(u_coords.max())
    v_min = int(v_coords.min())
    v_max = int(v_coords.max())
    width = max(1, u_max - u_min + 1)
    height = max(1, v_max - v_min + 1)

    band_height = max(1, int(np.ceil(height * float(support_band_ratio))))
    v_lower = v_max - band_height + 1
    middle_margin = max(0.0, 0.5 * (1.0 - float(support_middle_ratio)) * width)
    u_lower = int(np.floor(u_min + middle_margin))
    u_upper = int(np.ceil(u_max - middle_margin))

    support_mask = (
        mask_array
        & (np.arange(mask_array.shape[0])[:, None] >= v_lower)
        & (np.arange(mask_array.shape[1])[None, :] >= u_lower)
        & (np.arange(mask_array.shape[1])[None, :] <= u_upper)
    )
    support_v, support_u = np.nonzero(support_mask)
    if len(support_u) == 0:
        support_mask = mask_array & (np.arange(mask_array.shape[0])[:, None] >= v_lower)
        support_v, support_u = np.nonzero(support_mask)
    if len(support_u) == 0:
        support_v, support_u = np.nonzero(mask_array)
    return np.stack([support_u, support_v], axis=1).astype(np.int32, copy=False)


def select_support_pixels_from_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    *,
    image_shape: tuple[int, int],
    bbox_bottom_v_ratio: tuple[float, float] = (0.88, 0.98),
    bbox_middle_u_ratio: tuple[float, float] = (0.20, 0.80),
) -> np.ndarray:
    """Select bottom-strip pixels from a bbox when no mask is available."""
    if len(image_shape) != 2:
        raise ValueError("image_shape must be (height, width)")
    height, width = image_shape
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1 = int(np.clip(x1, 0, width - 1))
    x2 = int(np.clip(x2, x1 + 1, width))
    y1 = int(np.clip(y1, 0, height - 1))
    y2 = int(np.clip(y2, y1 + 1, height))
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    u_lower = int(np.floor(x1 + box_w * float(bbox_middle_u_ratio[0])))
    u_upper = int(np.ceil(x1 + box_w * float(bbox_middle_u_ratio[1])))
    v_lower = int(np.floor(y1 + box_h * float(bbox_bottom_v_ratio[0])))
    v_upper = int(np.ceil(y1 + box_h * float(bbox_bottom_v_ratio[1])))
    u_lower = int(np.clip(u_lower, x1, x2 - 1))
    u_upper = int(np.clip(max(u_lower + 1, u_upper), u_lower + 1, x2))
    v_lower = int(np.clip(v_lower, y1, y2 - 1))
    v_upper = int(np.clip(max(v_lower + 1, v_upper), v_lower + 1, y2))

    u_grid, v_grid = np.meshgrid(
        np.arange(u_lower, u_upper, dtype=np.int32),
        np.arange(v_lower, v_upper, dtype=np.int32),
    )
    if u_grid.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.stack([u_grid.reshape(-1), v_grid.reshape(-1)], axis=1)


def intersect_pixels_with_plane(
    pixels_uv: np.ndarray,
    *,
    camera_intrinsics: np.ndarray,
    camera_extrinsics: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
    parallel_epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect image rays with a world-frame plane."""
    pixels = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    if len(pixels) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=bool)

    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)
    extrinsics = np.asarray(camera_extrinsics, dtype=np.float64).reshape(4, 4)
    plane_normal = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    plane_offset = float(plane_offset)

    homogeneous_pixels = np.concatenate(
        [pixels, np.ones((len(pixels), 1), dtype=np.float64)],
        axis=1,
    )
    rays_camera = (np.linalg.inv(intrinsics) @ homogeneous_pixels.T).T
    rays_world = (extrinsics[:3, :3] @ rays_camera.T).T
    camera_center_world = extrinsics[:3, 3]

    denominators = rays_world @ plane_normal
    valid = np.abs(denominators) > float(parallel_epsilon)
    intersections = np.zeros((len(pixels), 3), dtype=np.float64)
    if not np.any(valid):
        return intersections, valid

    lambdas = np.full((len(pixels),), np.nan, dtype=np.float64)
    lambdas[valid] = -((camera_center_world @ plane_normal) + plane_offset) / denominators[valid]
    positive = valid & np.isfinite(lambdas) & (lambdas > 0.0)
    intersections[positive] = camera_center_world[None, :] + lambdas[positive, None] * rays_world[positive]
    return intersections, positive


def solve_depth_scale_shift_from_table(
    *,
    depth_image: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_extrinsics: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
    object_mask: np.ndarray | None = None,
    object_bbox: tuple[int, int, int, int] | None = None,
    min_pixels: int = 40,
    near_plane_threshold_m: float = 0.03,
) -> dict[str, float | bool | int | str]:
    """Solve z_corr = a * z_pred + b using table pixels and a known plane."""
    depth = np.asarray(depth_image, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth_image must be a 2D array")

    valid = np.isfinite(depth) & (depth > 1e-6)
    if object_mask is not None:
        valid &= ~np.asarray(object_mask, dtype=bool)
    if object_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in object_bbox]
        exclusion = np.zeros_like(valid, dtype=bool)
        exclusion[max(0, y1):min(depth.shape[0], y2), max(0, x1):min(depth.shape[1], x2)] = True
        valid &= ~exclusion

    if int(np.count_nonzero(valid)) < int(min_pixels):
        return {
            "valid": False,
            "a": 1.0,
            "b": 0.0,
            "residual": np.inf,
            "num_table_pixels": int(np.count_nonzero(valid)),
            "model": "identity_fallback",
        }

    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)
    extrinsics = np.asarray(camera_extrinsics, dtype=np.float64).reshape(4, 4)
    plane_normal = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    plane_offset = float(plane_offset)

    v_coords, u_coords = np.nonzero(valid)
    z_values = depth[valid]
    homogeneous = np.stack(
        [u_coords.astype(np.float64), v_coords.astype(np.float64), np.ones_like(z_values)],
        axis=1,
    )
    rays_camera = (np.linalg.inv(intrinsics) @ homogeneous.T).T
    raw_camera = z_values[:, None] * rays_camera
    raw_world = transform_points(extrinsics, raw_camera)
    signed_distance = raw_world @ plane_normal + plane_offset
    table_like = np.abs(signed_distance) < float(near_plane_threshold_m)
    if int(np.count_nonzero(table_like)) < int(min_pixels):
        return {
            "valid": False,
            "a": 1.0,
            "b": 0.0,
            "residual": np.inf,
            "num_table_pixels": int(np.count_nonzero(table_like)),
            "model": "identity_fallback",
        }

    rays_camera = rays_camera[table_like]
    z_values = z_values[table_like]
    alpha = (extrinsics[:3, :3] @ rays_camera.T).T @ plane_normal
    usable = np.isfinite(alpha) & (np.abs(alpha) > 1e-8)
    if int(np.count_nonzero(usable)) < int(min_pixels):
        return {
            "valid": False,
            "a": 1.0,
            "b": 0.0,
            "residual": np.inf,
            "num_table_pixels": int(np.count_nonzero(usable)),
            "model": "identity_fallback",
        }

    alpha = alpha[usable]
    z_values = z_values[usable]
    rhs_value = -((plane_normal @ extrinsics[:3, 3]) + plane_offset)
    system = np.stack([z_values * alpha, alpha], axis=1)
    rhs = np.full((len(system),), rhs_value, dtype=np.float64)

    try:
        solution, residuals, rank, singular_values = np.linalg.lstsq(system, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return {
            "valid": False,
            "a": 1.0,
            "b": 0.0,
            "residual": np.inf,
            "num_table_pixels": int(len(system)),
            "model": "identity_fallback",
        }

    a = float(solution[0])
    b = float(solution[1])
    if not np.isfinite(a) or not np.isfinite(b) or rank < 2:
        return {
            "valid": False,
            "a": 1.0,
            "b": 0.0,
            "residual": np.inf,
            "num_table_pixels": int(len(system)),
            "model": "identity_fallback",
        }

    residual = float(np.sqrt(residuals[0] / len(system))) if residuals.size else 0.0
    condition_number = float(singular_values[0] / max(singular_values[-1], 1e-12))
    if condition_number > 1e8:
        scale_only = np.linalg.lstsq((z_values * alpha)[:, None], rhs, rcond=None)[0]
        a = float(scale_only[0])
        b = 0.0
        residual = float(np.sqrt(np.mean((a * z_values * alpha - rhs) ** 2)))
        model = "scale_only"
    else:
        model = "scale_shift"

    return {
        "valid": True,
        "a": a,
        "b": b,
        "residual": residual,
        "num_table_pixels": int(len(system)),
        "model": model,
    }


def estimate_object_height_from_mask_or_bbox(
    *,
    depth_image: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_extrinsics: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
    object_mask: np.ndarray | None = None,
    object_bbox: tuple[int, int, int, int] | None = None,
    depth_scale_a: float = 1.0,
    depth_shift_b: float = 0.0,
    height_percentile: float = 95.0,
    max_valid_height_m: float = 0.5,
) -> dict[str, object]:
    """Estimate object height above the plane from corrected depth."""
    object_pixels = _select_object_pixels(
        depth_image=np.asarray(depth_image, dtype=np.float64),
        object_mask=object_mask,
        object_bbox=object_bbox,
    )
    if len(object_pixels) == 0:
        return {
            "height_m": 0.0,
            "num_valid_object_points": 0,
            "valid_object_points_world": np.zeros((0, 3), dtype=np.float64),
        }

    world_points, heights = _backproject_object_pixels(
        pixels_uv=object_pixels,
        depth_image=depth_image,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        plane_normal=plane_normal,
        plane_offset=plane_offset,
        depth_scale_a=depth_scale_a,
        depth_shift_b=depth_shift_b,
        max_valid_height_m=max_valid_height_m,
    )
    if len(heights) == 0:
        return {
            "height_m": 0.0,
            "num_valid_object_points": 0,
            "valid_object_points_world": np.zeros((0, 3), dtype=np.float64),
        }
    return {
        "height_m": float(np.percentile(heights, float(height_percentile))),
        "num_valid_object_points": int(len(heights)),
        "valid_object_points_world": world_points,
    }


def estimate_object_footprint_on_table(
    *,
    object_world_points: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray | None:
    """Estimate robust planar footprint dimensions on the tabletop plane."""
    points = np.asarray(object_world_points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return None

    normal = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-9:
        return None
    normal /= normal_norm
    e1 = np.cross(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    e2 /= max(np.linalg.norm(e2), 1e-9)

    signed_height = points @ normal
    projected = points - signed_height[:, None] * normal[None, :]
    coords_1 = projected @ e1
    coords_2 = projected @ e2
    footprint_w = float(np.percentile(coords_1, 95.0) - np.percentile(coords_1, 5.0))
    footprint_l = float(np.percentile(coords_2, 95.0) - np.percentile(coords_2, 5.0))
    return np.array([max(0.0, footprint_w), max(0.0, footprint_l)], dtype=np.float64)


def estimate_tabletop_object_geometry(
    *,
    depth_image: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_extrinsics: np.ndarray,
    table_top_z_m: float,
    object_mask: np.ndarray | None = None,
    object_bbox: tuple[int, int, int, int] | None = None,
    support_band_ratio: float = 0.10,
    support_middle_ratio: float = 0.60,
    bbox_bottom_v_ratio: tuple[float, float] = (0.88, 0.98),
    bbox_middle_u_ratio: tuple[float, float] = (0.20, 0.80),
    height_percentile: float = 95.0,
    max_valid_height_m: float = 0.5,
    min_support_pixels: int = 10,
    min_valid_object_points: int = 20,
) -> dict[str, object]:
    """Estimate tabletop object base center and height using plane-constrained support pixels."""
    plane = estimate_table_plane(table_top_z_m=table_top_z_m)

    working_mask = None if object_mask is None else np.asarray(object_mask, dtype=bool)
    mask_source = "provided_mask"
    if working_mask is None and object_bbox is None:
        working_mask = _derive_object_mask_from_depth(
            depth_image=np.asarray(depth_image, dtype=np.float64),
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            plane_normal=plane.normal,
            plane_offset=plane.offset,
            min_height_m=0.004,
            max_valid_height_m=max_valid_height_m,
        )
        mask_source = "derived_from_depth"
    elif working_mask is None:
        mask_source = "bbox_only"

    depth_correction = solve_depth_scale_shift_from_table(
        depth_image=depth_image,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        plane_normal=plane.normal,
        plane_offset=plane.offset,
        object_mask=working_mask,
        object_bbox=object_bbox,
    )
    depth_scale_a = float(depth_correction["a"])
    depth_shift_b = float(depth_correction["b"])

    if working_mask is not None:
        support_pixels = select_support_pixels_from_mask(
            working_mask,
            support_band_ratio=support_band_ratio,
            support_middle_ratio=support_middle_ratio,
        )
    elif object_bbox is not None:
        support_pixels = select_support_pixels_from_bbox(
            object_bbox,
            image_shape=np.asarray(depth_image).shape,
            bbox_bottom_v_ratio=bbox_bottom_v_ratio,
            bbox_middle_u_ratio=bbox_middle_u_ratio,
        )
    else:
        support_pixels = np.zeros((0, 2), dtype=np.int32)

    support_world, support_valid = intersect_pixels_with_plane(
        support_pixels,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        plane_normal=plane.normal,
        plane_offset=plane.offset,
    )
    valid_support_world = support_world[support_valid]
    if len(valid_support_world):
        base_center = np.median(valid_support_world, axis=0)
        base_center[2] = float(table_top_z_m)
    else:
        base_center = np.array([np.nan, np.nan, float(table_top_z_m)], dtype=np.float64)

    height_result = estimate_object_height_from_mask_or_bbox(
        depth_image=depth_image,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        plane_normal=plane.normal,
        plane_offset=plane.offset,
        object_mask=working_mask,
        object_bbox=object_bbox,
        depth_scale_a=depth_scale_a,
        depth_shift_b=depth_shift_b,
        height_percentile=height_percentile,
        max_valid_height_m=max_valid_height_m,
    )
    height_m = float(height_result["height_m"])
    object_world_points = np.asarray(height_result["valid_object_points_world"], dtype=np.float64)
    footprint_xy = estimate_object_footprint_on_table(
        object_world_points=object_world_points,
        plane_normal=plane.normal,
    )

    need_reobserve_reasons: list[str] = []
    if int(np.count_nonzero(support_valid)) < int(min_support_pixels):
        need_reobserve_reasons.append("too_few_support_pixels")
    if int(height_result["num_valid_object_points"]) < int(min_valid_object_points):
        need_reobserve_reasons.append("too_few_valid_object_points")
    if not depth_correction["valid"]:
        need_reobserve_reasons.append("invalid_depth_table_fit")
    if not np.isfinite(height_m) or height_m <= 0.0 or height_m > float(max_valid_height_m):
        need_reobserve_reasons.append("implausible_height")
    if not np.all(np.isfinite(base_center[:2])):
        need_reobserve_reasons.append("invalid_base_center")

    top_center_world = np.array(
        [base_center[0], base_center[1], float(table_top_z_m) + height_m],
        dtype=np.float64,
    )
    object_center_world = np.array(
        [base_center[0], base_center[1], float(table_top_z_m) + 0.5 * height_m],
        dtype=np.float64,
    )

    support_pixel_median = (
        np.median(support_pixels.astype(np.float64), axis=0).tolist() if len(support_pixels) else None
    )

    quality = {
        "num_support_pixels": int(np.count_nonzero(support_valid)),
        "num_valid_object_points": int(height_result["num_valid_object_points"]),
        "depth_scale_a": depth_scale_a,
        "depth_shift_b": depth_shift_b,
        "table_fit_used": plane.used,
        "table_ls_residual": float(depth_correction["residual"]),
        "is_sparse": bool(
            int(np.count_nonzero(support_valid)) < int(min_support_pixels)
            or int(height_result["num_valid_object_points"]) < int(min_valid_object_points)
        ),
        "need_reobserve": bool(need_reobserve_reasons),
        "need_reobserve_reasons": need_reobserve_reasons,
        "mask_source": mask_source,
        "depth_model": str(depth_correction["model"]),
    }

    return {
        "base_center_world": base_center.tolist(),
        "height_m": height_m,
        "footprint_xy": None if footprint_xy is None else footprint_xy.tolist(),
        "top_center_world": top_center_world.tolist(),
        "object_center_world": object_center_world.tolist(),
        "support_pixels_uv": support_pixels.tolist(),
        "support_pixel_median_uv": support_pixel_median,
        "quality": quality,
        "table_plane": {
            "normal": plane.normal.tolist(),
            "offset": float(plane.offset),
            "used": plane.used,
            "residual": float(plane.residual),
        },
        "derived_object_mask": working_mask,
    }


def _select_object_pixels(
    *,
    depth_image: np.ndarray,
    object_mask: np.ndarray | None,
    object_bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    if object_mask is not None:
        valid_mask = np.asarray(object_mask, dtype=bool) & np.isfinite(depth_image) & (depth_image > 1e-6)
        v_coords, u_coords = np.nonzero(valid_mask)
        return np.stack([u_coords, v_coords], axis=1).astype(np.int32, copy=False)

    if object_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in object_bbox]
        x1 = int(np.clip(x1, 0, depth_image.shape[1] - 1))
        x2 = int(np.clip(x2, x1 + 1, depth_image.shape[1]))
        y1 = int(np.clip(y1, 0, depth_image.shape[0] - 1))
        y2 = int(np.clip(y2, y1 + 1, depth_image.shape[0]))
        bbox_mask = np.zeros_like(depth_image, dtype=bool)
        bbox_mask[y1:y2, x1:x2] = True
        valid_mask = bbox_mask & np.isfinite(depth_image) & (depth_image > 1e-6)
        v_coords, u_coords = np.nonzero(valid_mask)
        return np.stack([u_coords, v_coords], axis=1).astype(np.int32, copy=False)

    return np.zeros((0, 2), dtype=np.int32)


def _backproject_object_pixels(
    *,
    pixels_uv: np.ndarray,
    depth_image: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_extrinsics: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
    depth_scale_a: float,
    depth_shift_b: float,
    max_valid_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(pixels_uv, dtype=np.int32).reshape(-1, 2)
    if len(pixels) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    depth = np.asarray(depth_image, dtype=np.float64)
    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)
    extrinsics = np.asarray(camera_extrinsics, dtype=np.float64).reshape(4, 4)
    plane_normal = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    plane_offset = float(plane_offset)

    u_coords = pixels[:, 0]
    v_coords = pixels[:, 1]
    raw_depth = depth[v_coords, u_coords]
    corrected_depth = depth_scale_a * raw_depth + depth_shift_b
    valid = np.isfinite(corrected_depth) & (corrected_depth > 1e-6)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    homogeneous = np.stack(
        [u_coords[valid].astype(np.float64), v_coords[valid].astype(np.float64), np.ones(int(np.count_nonzero(valid)))],
        axis=1,
    )
    rays_camera = (np.linalg.inv(intrinsics) @ homogeneous.T).T
    camera_points = corrected_depth[valid, None] * rays_camera
    world_points = transform_points(extrinsics, camera_points)
    heights = world_points @ plane_normal + plane_offset
    height_valid = np.isfinite(heights) & (heights > 0.0) & (heights < float(max_valid_height_m))
    return world_points[height_valid], heights[height_valid]


def _derive_object_mask_from_depth(
    *,
    depth_image: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_extrinsics: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
    min_height_m: float,
    max_valid_height_m: float,
) -> np.ndarray:
    mask = np.zeros_like(depth_image, dtype=bool)
    depth = np.asarray(depth_image, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 1e-6)
    if not np.any(valid):
        return mask

    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)
    extrinsics = np.asarray(camera_extrinsics, dtype=np.float64).reshape(4, 4)
    plane_normal = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    plane_offset = float(plane_offset)

    v_coords, u_coords = np.nonzero(valid)
    homogeneous = np.stack(
        [u_coords.astype(np.float64), v_coords.astype(np.float64), np.ones(int(np.count_nonzero(valid)))],
        axis=1,
    )
    rays_camera = (np.linalg.inv(intrinsics) @ homogeneous.T).T
    camera_points = depth[valid, None] * rays_camera
    world_points = transform_points(extrinsics, camera_points)
    heights = world_points @ plane_normal + plane_offset
    keep = np.isfinite(heights) & (heights > float(min_height_m)) & (heights < float(max_valid_height_m))
    if np.any(keep):
        mask[v_coords[keep], u_coords[keep]] = True
    return mask
