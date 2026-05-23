from __future__ import annotations

from pathlib import Path
import os

import numpy as np
from scipy.optimize import least_squares


def estimate_object_center(pcd: np.ndarray) -> np.ndarray:
    """Estimate a robust object center from a mostly-object point cloud."""
    points = _require_points(pcd)
    center_x = float(np.median(points[:, 0]))
    center_y = float(np.median(points[:, 1]))
    z_top = float(np.percentile(points[:, 2], 95.0))
    z_bottom = float(np.percentile(points[:, 2], 5.0))
    center_z = float((z_top + z_bottom) / 2.0)
    return np.array([center_x, center_y, center_z], dtype=float)


def fit_circle_2d(points_xy: np.ndarray) -> np.ndarray:
    """Fit a circle to a 2D point set, including partial arcs."""
    xy = np.asarray(points_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3:
        raise ValueError("Expected at least three 2D points for circle fitting")

    x = xy[:, 0]
    y = xy[:, 1]

    def residuals(params: np.ndarray) -> np.ndarray:
        cx, cy, radius = params
        return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius

    radius0 = float(max((x.max() - x.min() + y.max() - y.min()) / 4.0, 1e-4))
    result = least_squares(
        residuals,
        np.array([float(x.mean()), float(y.mean()), radius0], dtype=float),
    )
    fitted = np.asarray(result.x, dtype=float)
    fitted[2] = abs(float(fitted[2]))
    return fitted


def fit_circle_with_confidence(points_xy: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Fit a circle and estimate uncertainty from residuals and arc coverage."""
    xy = np.asarray(points_xy, dtype=float)
    cx, cy, radius = fit_circle_2d(xy)

    distances = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
    residuals = np.abs(distances - radius)
    residual_std = float(np.std(residuals))

    raw_angles = np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx)
    wrapped_angles = np.sort(np.mod(raw_angles, 2.0 * np.pi))
    if len(wrapped_angles) >= 2:
        cyclic_gaps = np.diff(np.concatenate([wrapped_angles, wrapped_angles[:1] + 2.0 * np.pi]))
        largest_gap = float(np.max(cyclic_gaps))
        arc_coverage = float(max(2.0 * np.pi - largest_gap, 0.0))
    else:
        arc_coverage = 0.0

    coverage_factor = float(max(1.0, np.pi / max(arc_coverage, 0.3)))
    estimated_error = float(residual_std * coverage_factor)
    return float(cx), float(cy), float(radius), estimated_error, residual_std, arc_coverage


def find_minimum_grip_direction(
    pcd: np.ndarray,
    center: np.ndarray,
    slice_thickness: float = 0.015,
) -> tuple[float, float]:
    """Find the minimum-width grasp direction through a center-height slice."""
    points = _require_points(pcd)
    center_xyz = np.asarray(center, dtype=float).reshape(3)

    z_mask = np.abs(points[:, 2] - center_xyz[2]) < float(slice_thickness)
    slice_pts = points[z_mask]
    if len(slice_pts) < 10:
        slice_pts = points

    xy = np.asarray(slice_pts[:, :2] - center_xyz[:2], dtype=float)
    if len(xy) == 0:
        return 0.0, 0.0

    best_angle = 0.0
    min_width = float("inf")
    for angle_deg in range(0, 180, 5):
        angle_rad = float(np.radians(angle_deg))
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=float)
        projections = xy @ direction
        span = float(projections.max() - projections.min())
        if span < min_width:
            min_width = span
            best_angle = angle_rad

    return best_angle, min_width


def validate_grasp_point(
    pcd: np.ndarray,
    grasp_xy: np.ndarray,
    tolerance: float = 0.01,
) -> bool:
    """Check whether a vertical ray through grasp_xy intersects enough object points."""
    points = _require_points(pcd)
    xy = np.asarray(grasp_xy, dtype=float).reshape(2)
    xy_dist = np.linalg.norm(points[:, :2] - xy[None, :], axis=1)
    nearby = points[xy_dist < float(tolerance) + 0.02]
    return bool(len(nearby) > 5)


def _pca_xy(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.asarray(points_xy, dtype=float)
    center = np.median(xy, axis=0)
    centered = xy - center[None, :]
    if len(centered) < 3:
        axes = np.eye(2, dtype=float)
        spans = np.zeros(2, dtype=float)
        return center, axes, spans
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    projections = centered @ axes
    spans = np.percentile(projections, 95.0, axis=0) - np.percentile(projections, 5.0, axis=0)
    return center, axes, spans


def _slender_object_grasp(points: np.ndarray, table_z: float, *, slice_thickness: float, max_gripper_width_m: float | None) -> dict[str, object]:
    z_top = float(np.percentile(points[:, 2], 95.0))
    z_bottom = float(np.percentile(points[:, 2], 5.0))
    object_height = float(z_top - z_bottom)
    center_z = float((z_top + z_bottom) / 2.0)
    slice_pts = _select_slice_points(points, center_z=center_z, slice_thickness=max(slice_thickness, 0.02))
    center_xy, axes, spans = _pca_xy(slice_pts[:, :2])
    major_width = float(max(spans[0], 1e-4))
    minor_width = float(max(spans[1], 1e-4))
    major_axis = axes[:, 0]
    minor_axis = axes[:, 1]
    grip_angle_rad = float(np.arctan2(minor_axis[1], minor_axis[0]))
    centered_xy = np.asarray(slice_pts[:, :2] - center_xy[None, :], dtype=float)
    projections = centered_xy @ axes
    major_lo, major_hi = np.percentile(projections[:, 0], [5.0, 95.0])
    minor_center = float(np.median(projections[:, 1]))
    long_axis_bias = float(os.getenv("ASK2ACT_GEOMETRIC_SLENDER_GRASP_LONG_AXIS_BIAS", "0.0"))
    long_axis_bias = float(np.clip(long_axis_bias, -0.45, 0.45))
    grasp_major = float((major_lo + major_hi) / 2.0 + long_axis_bias * max(major_hi - major_lo, 0.0))
    # For nonuniform tools such as forks, using the point-density median can
    # pull the grasp toward the wider head/tines. Use the physical length
    # midpoint along the PCA major axis instead.
    grasp_xy = np.asarray(center_xy + major_axis * grasp_major + minor_axis * minor_center, dtype=float)
    clearance_margin = float(os.getenv("ASK2ACT_GEOMETRIC_SLENDER_GRIP_CLEARANCE_M", "0.018"))
    min_open_width = float(os.getenv("ASK2ACT_GEOMETRIC_SLENDER_MIN_OPEN_WIDTH_M", "0.032"))
    gripper_open_width = float(max(min_open_width, minor_width + clearance_margin))
    if max_gripper_width_m is not None:
        gripper_open_width = float(min(gripper_open_width, float(max_gripper_width_m)))
    return {
        "grasp_x": float(grasp_xy[0]),
        "grasp_y": float(grasp_xy[1]),
        "grasp_z": center_z,
        "grip_angle_rad": grip_angle_rad,
        "gripper_open_width": gripper_open_width,
        "min_cross_section_width": minor_width,
        "object_center": [float(grasp_xy[0]), float(grasp_xy[1]), center_z],
        "object_height": object_height,
        "object_top_z": z_top,
        "object_bottom_z": z_bottom,
        "slice_center_z": center_z,
        "slice_point_count": int(len(slice_pts)),
        "fitted_circle_center_xy": [float(grasp_xy[0]), float(grasp_xy[1])],
        "fitted_circle_radius": float(minor_width / 2.0),
        "estimated_error": 0.0,
        "residual_std": 0.0,
        "arc_coverage_rad": 0.0,
        "arc_coverage_deg": 0.0,
        "conservative_diameter": minor_width,
        "uncertainty_margin": 0.0,
        "fixed_clearance_margin": clearance_margin,
        "grasp_point_validated": bool(validate_grasp_point(points, grasp_xy, tolerance=0.02)),
        "slice_thickness_m": float(slice_thickness),
        "validation_tolerance_m": 0.02,
        "table_z": float(table_z),
        "width_near_limit": bool(
            max_gripper_width_m is not None and gripper_open_width > float(max_gripper_width_m) - 0.01
        ),
        "open_width_exceeds_max": False,
        "max_gripper_width_m": None if max_gripper_width_m is None else float(max_gripper_width_m),
        "method": "geometric_point_cloud_pca_slender",
        "major_axis_width": major_width,
        "minor_axis_width": minor_width,
        "xy_aspect_ratio": float(major_width / max(minor_width, 1e-4)),
        "pca_density_center_xy": [float(center_xy[0]), float(center_xy[1])],
        "slender_grasp_center_mode": "major_axis_extent_midpoint",
        "slender_major_axis_xy": [float(major_axis[0]), float(major_axis[1])],
        "slender_minor_axis_xy": [float(minor_axis[0]), float(minor_axis[1])],
        "slender_major_axis_percentile_5_95": [float(major_lo), float(major_hi)],
        "slender_minor_axis_center_projection": minor_center,
        "slender_long_axis_bias": long_axis_bias,
    }


def compute_geometric_grasp(
    pcd: np.ndarray,
    table_z: float,
    *,
    slice_thickness: float = 0.015,
    validation_tolerance: float = 0.01,
    max_gripper_width_m: float | None = None,
) -> dict[str, object]:
    """Compute a top-down grasp directly from point-cloud geometry."""
    points = _require_points(pcd)
    z_top = float(np.percentile(points[:, 2], 95.0))
    z_bottom = float(np.percentile(points[:, 2], 5.0))
    object_height = float(z_top - z_bottom)
    center_z = float((z_top + z_bottom) / 2.0)
    slice_pts = _select_slice_points(points, center_z=center_z, slice_thickness=slice_thickness)
    center_xy, _axes, spans = _pca_xy(slice_pts[:, :2])
    major_width = float(max(spans[0], 1e-4))
    minor_width = float(max(spans[1], 1e-4))
    slender_aspect_threshold = float(os.getenv("ASK2ACT_GEOMETRIC_SLENDER_OBJECT_ASPECT_RATIO", "2.5"))
    slender_max_height_m = float(os.getenv("ASK2ACT_GEOMETRIC_SLENDER_OBJECT_MAX_HEIGHT_M", "0.065"))
    if object_height <= slender_max_height_m and major_width / max(minor_width, 1e-4) >= slender_aspect_threshold:
        return _slender_object_grasp(
            points,
            table_z,
            slice_thickness=slice_thickness,
            max_gripper_width_m=max_gripper_width_m,
        )

    (
        fitted_cx,
        fitted_cy,
        fitted_radius,
        estimated_error,
        residual_std,
        arc_coverage,
    ) = fit_circle_with_confidence(slice_pts[:, :2])
    center = np.array([float(fitted_cx), float(fitted_cy), center_z], dtype=float)
    grip_angle_rad, _ = find_minimum_grip_direction(
        slice_pts,
        center,
        slice_thickness=max(slice_thickness, 0.03),
    )
    # The geometric grasp point itself is the object's vertical midpoint.
    # Any hardware-specific correction should happen later in motion planning
    # via the wrist target offset, not by distorting the object estimate here.
    grasp_z = float(center_z)

    grasp_xy = np.asarray(center[:2], dtype=float).copy()
    grasp_point_validated = (
        validate_grasp_point(points, grasp_xy, tolerance=validation_tolerance)
        or _validate_grasp_point_against_fitted_circle(
            grasp_xy,
            circle_center_xy=np.array([fitted_cx, fitted_cy], dtype=float),
            radius=float(fitted_radius),
            tolerance=validation_tolerance,
        )
    )
    if not grasp_point_validated:
        grasp_xy = np.asarray([fitted_cx, fitted_cy], dtype=float)

    object_diameter = float(fitted_radius * 2.0)
    conservative_diameter = float((fitted_radius + estimated_error) * 2.0)
    uncertainty_margin = float(estimated_error * 2.0)
    fixed_clearance_margin = 0.02
    gripper_open_width = float(conservative_diameter + uncertainty_margin + fixed_clearance_margin)
    open_width_exceeds_max = bool(
        max_gripper_width_m is not None and gripper_open_width > float(max_gripper_width_m)
    )
    width_near_limit = bool(
        max_gripper_width_m is not None and gripper_open_width > float(max_gripper_width_m) - 0.01
    )

    return {
        "grasp_x": float(grasp_xy[0]),
        "grasp_y": float(grasp_xy[1]),
        "grasp_z": grasp_z,
        "grip_angle_rad": float(grip_angle_rad),
        "gripper_open_width": gripper_open_width,
        "min_cross_section_width": object_diameter,
        "object_center": center.tolist(),
        "object_height": object_height,
        "object_top_z": z_top,
        "object_bottom_z": z_bottom,
        "slice_center_z": center_z,
        "slice_point_count": int(len(slice_pts)),
        "fitted_circle_center_xy": [float(fitted_cx), float(fitted_cy)],
        "fitted_circle_radius": float(fitted_radius),
        "estimated_error": float(estimated_error),
        "residual_std": float(residual_std),
        "arc_coverage_rad": float(arc_coverage),
        "arc_coverage_deg": float(np.degrees(arc_coverage)),
        "conservative_diameter": float(conservative_diameter),
        "uncertainty_margin": float(uncertainty_margin),
        "fixed_clearance_margin": float(fixed_clearance_margin),
        "grasp_point_validated": bool(grasp_point_validated),
        "slice_thickness_m": float(slice_thickness),
        "validation_tolerance_m": float(validation_tolerance),
        "table_z": float(table_z),
        "width_near_limit": width_near_limit,
        "open_width_exceeds_max": open_width_exceeds_max,
        "max_gripper_width_m": None if max_gripper_width_m is None else float(max_gripper_width_m),
        "method": "geometric_point_cloud",
    }


def save_geometric_grasp_debug(
    pcd: np.ndarray,
    geometric_grasp: dict[str, object],
    output_path: str | Path,
) -> None:
    """Save a single debug figure with geometry overlays on the point cloud."""
    points = np.asarray(pcd, dtype=float)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    center = np.asarray(geometric_grasp["object_center"], dtype=float)
    grasp_xy = np.array(
        [float(geometric_grasp["grasp_x"]), float(geometric_grasp["grasp_y"])],
        dtype=float,
    )
    grasp_z = float(geometric_grasp["grasp_z"])
    grip_angle = float(geometric_grasp["grip_angle_rad"])
    grasp_width = float(geometric_grasp["min_cross_section_width"])
    slice_pts = _select_slice_points(
        points,
        center_z=float(geometric_grasp.get("slice_center_z", center[2])),
        slice_thickness=float(geometric_grasp.get("slice_thickness_m", 0.015)),
    )
    fitted_circle_center_xy = np.asarray(
        geometric_grasp.get("fitted_circle_center_xy", center[:2]),
        dtype=float,
    )
    fitted_radius = float(geometric_grasp.get("fitted_circle_radius", grasp_width / 2.0))
    estimated_error = float(geometric_grasp.get("estimated_error", 0.0))
    fixed_clearance_margin = float(geometric_grasp.get("fixed_clearance_margin", 0.0))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(13, 9))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=5,
            c="0.55",
            alpha=0.28,
            depthshade=False,
            label="point cloud",
        )
        ax.scatter(
            slice_pts[:, 0],
            slice_pts[:, 1],
            slice_pts[:, 2],
            s=8,
            c="#1f77b4",
            alpha=0.55,
            depthshade=False,
            label="center slice",
        )

        x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
        y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())
        x_pad = max(0.03, (x_max - x_min) * 0.25)
        y_pad = max(0.03, (y_max - y_min) * 0.25)
        x0, x1 = x_min - x_pad, x_max + x_pad
        y0, y1 = y_min - y_pad, y_max + y_pad
        table_z = float(geometric_grasp.get("table_z", np.percentile(points[:, 2], 1.0)))
        slice_z = float(center[2])

        table_plane = [[(x0, y0, table_z), (x1, y0, table_z), (x1, y1, table_z), (x0, y1, table_z)]]
        slice_plane = [[(x0, y0, slice_z), (x1, y0, slice_z), (x1, y1, slice_z), (x0, y1, slice_z)]]
        ax.add_collection3d(
            Poly3DCollection(table_plane, facecolors="#2ca02c", edgecolors="none", alpha=0.10)
        )
        ax.add_collection3d(
            Poly3DCollection(slice_plane, facecolors="#1f77b4", edgecolors="none", alpha=0.08)
        )

        x_dir = np.array([np.cos(grip_angle), np.sin(grip_angle), 0.0], dtype=float)
        y_dir = np.array([np.sin(grip_angle), -np.cos(grip_angle), 0.0], dtype=float)
        jaw_half = max(grasp_width / 2.0, 0.01)
        finger_len = max(float(geometric_grasp.get("gripper_open_width", grasp_width)) * 0.35, 0.03)
        line_half = max(grasp_width, 0.02) * 0.9

        min_width_start = center + np.array([-x_dir[0] * line_half, -x_dir[1] * line_half, 0.0], dtype=float)
        min_width_end = center + np.array([x_dir[0] * line_half, x_dir[1] * line_half, 0.0], dtype=float)
        ax.plot(
            [min_width_start[0], min_width_end[0]],
            [min_width_start[1], min_width_end[1]],
            [min_width_start[2], min_width_end[2]],
            color="#1f77b4",
            linewidth=2.6,
            label="grip axis",
        )

        theta = np.linspace(0.0, 2.0 * np.pi, 200, dtype=float)
        circle_x = fitted_circle_center_xy[0] + fitted_radius * np.cos(theta)
        circle_y = fitted_circle_center_xy[1] + fitted_radius * np.sin(theta)
        circle_z = np.full_like(circle_x, slice_z)
        ax.plot(circle_x, circle_y, circle_z, color="#9467bd", linewidth=2.0, label="fitted circle")

        center_marker = center.copy()
        grasp_marker = np.array([grasp_xy[0], grasp_xy[1], grasp_z], dtype=float)
        ax.scatter(
            [center_marker[0]],
            [center_marker[1]],
            [center_marker[2]],
            s=70,
            c="#003f5c",
            depthshade=False,
            label="object center",
        )
        ax.scatter(
            [grasp_marker[0]],
            [grasp_marker[1]],
            [grasp_marker[2]],
            s=90,
            c="#d62728",
            depthshade=False,
            label="grasp point",
        )

        _draw_gripper_wireframe_3d(
            ax,
            grasp_marker,
            x_dir=x_dir,
            y_dir=y_dir,
            jaw_half=jaw_half,
            finger_len=finger_len,
        )
        ax.quiver(
            grasp_marker[0],
            grasp_marker[1],
            grasp_marker[2] + 0.01,
            0.0,
            0.0,
            -0.06,
            color="#d62728",
            linewidth=2.0,
            arrow_length_ratio=0.18,
        )

        open_width = float(geometric_grasp.get("gripper_open_width", grasp_width))
        open_width_start = grasp_marker - x_dir * jaw_half
        open_width_end = grasp_marker + x_dir * jaw_half
        ax.plot(
            [open_width_start[0], open_width_end[0]],
            [open_width_start[1], open_width_end[1]],
            [open_width_start[2], open_width_end[2]],
            color="#ff7f0e",
            linewidth=3.0,
            linestyle="--",
            label="gripper opening",
        )

        text_x = x0
        text_y = y1 + y_pad * 0.35
        text_z = float(points[:, 2].max()) + 0.02
        ax.text(
            text_x,
            text_y,
            text_z,
            (
                f"angle={float(np.degrees(grip_angle)):.1f} deg\n"
                f"diameter={grasp_width:.3f} m\n"
                f"open_width={open_width:.3f} m\n"
                f"radius={fitted_radius:.3f} m\n"
                f"fit_err={estimated_error:.4f} m\n"
                f"fixed_margin={fixed_clearance_margin:.3f} m\n"
                f"coverage={float(geometric_grasp.get('arc_coverage_deg', 0.0)):.1f} deg\n"
                f"grasp_z={grasp_z:.3f} m\n"
                f"validated={bool(geometric_grasp.get('grasp_point_validated', False))}"
            ),
            fontsize=10,
            color="black",
        )

        ax.set_title("Geometric Grasp Debug Overlay")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=24, azim=-55)
        ax.set_box_aspect(
            (
                max(x1 - x0, 1e-3),
                max(y1 - y0, 1e-3),
                max(float(points[:, 2].max()) - table_z, 1e-3),
            )
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_zlim(table_z - 0.01, float(points[:, 2].max()) + 0.05)
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)
        return
    except Exception:
        pass

    try:
        import cv2

        canvas = np.full((720, 1280, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(output), canvas)
    except Exception:
        np.save(output.with_suffix(".npy"), points)


def _draw_xy_grasp_overlay(
    ax,
    center_xy: np.ndarray,
    grasp_xy: np.ndarray,
    grip_angle: float,
    grasp_width: float,
) -> None:
    x_dir = np.array([np.cos(grip_angle), np.sin(grip_angle)], dtype=float)
    y_dir = np.array([np.sin(grip_angle), -np.cos(grip_angle)], dtype=float)
    line_half = max(float(grasp_width), 0.02) * 0.75
    finger_len = 0.03

    line_start = center_xy - x_dir * line_half
    line_end = center_xy + x_dir * line_half
    ax.plot(
        [line_start[0], line_end[0]],
        [line_start[1], line_end[1]],
        color="#1f77b4",
        linewidth=2.0,
        label="min-width axis",
    )

    jaw_half = max(float(grasp_width) / 2.0, 0.01)
    left_base = grasp_xy - x_dir * jaw_half
    right_base = grasp_xy + x_dir * jaw_half
    left_tip = left_base + y_dir * finger_len
    right_tip = right_base + y_dir * finger_len
    palm_a = left_tip
    palm_b = right_tip

    for start, end in (
        (left_base, left_tip),
        (right_base, right_tip),
        (palm_a, palm_b),
    ):
        ax.plot([start[0], end[0]], [start[1], end[1]], color="#d62728", linewidth=2.0)


def _draw_gripper_wireframe_3d(
    ax,
    grasp_point: np.ndarray,
    *,
    x_dir: np.ndarray,
    y_dir: np.ndarray,
    jaw_half: float,
    finger_len: float,
) -> None:
    left_base = grasp_point - x_dir * jaw_half
    right_base = grasp_point + x_dir * jaw_half
    left_tip = left_base + y_dir * finger_len
    right_tip = right_base + y_dir * finger_len
    segments = (
        (left_base, left_tip),
        (right_base, right_tip),
        (left_tip, right_tip),
    )
    for start, end in segments:
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color="#d62728",
            linewidth=2.6,
        )


def _require_points(pcd: np.ndarray) -> np.ndarray:
    points = np.asarray(pcd, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Expected a non-empty point cloud with shape (N, 3)")
    return points


def _select_slice_points(
    points_xyz: np.ndarray,
    *,
    center_z: float,
    slice_thickness: float,
) -> np.ndarray:
    points = _require_points(points_xyz)
    z_mask = np.abs(points[:, 2] - float(center_z)) < float(slice_thickness)
    slice_pts = points[z_mask]
    if len(slice_pts) >= 10:
        return slice_pts

    wider_z_mask = np.abs(points[:, 2] - float(center_z)) < float(slice_thickness) * 2.0
    wider_slice_pts = points[wider_z_mask]
    if len(wider_slice_pts) >= 10:
        return wider_slice_pts
    return points


def _validate_grasp_point_against_fitted_circle(
    grasp_xy: np.ndarray,
    *,
    circle_center_xy: np.ndarray,
    radius: float,
    tolerance: float,
) -> bool:
    center_dist = float(
        np.linalg.norm(np.asarray(grasp_xy, dtype=float) - np.asarray(circle_center_xy, dtype=float))
    )
    return bool(radius > 0.0 and center_dist <= float(radius) + float(tolerance))
