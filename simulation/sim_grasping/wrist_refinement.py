from __future__ import annotations

from typing import Any

import numpy as np

from minimal_grasping_common import build_pixel_result
from pipeline_types import SelectedGraspPlan, WristRefinementDelta


MAX_DX_DY_M = 0.03
MAX_DZ_M = 0.05
MAX_DYAW_RAD = 0.2


def _clip(value: float, limit: float) -> float:
    return float(np.clip(value, -limit, limit))


def compute_wrist_refinement_delta(
    *,
    wrist_observation: dict[str, Any],
    selected_plan: SelectedGraspPlan,
) -> tuple[WristRefinementDelta, dict[str, Any] | None]:
    depth = np.asarray(wrist_observation["depth"], dtype=float)
    h, w = depth.shape[:2]
    cx = w // 2
    cy = h // 2
    patch_half = 45
    patch = depth[max(0, cy - patch_half):min(h, cy + patch_half), max(0, cx - patch_half):min(w, cx + patch_half)]
    valid = np.isfinite(patch) & (patch > 0.0)
    if valid.sum() < 30:
        return (
            WristRefinementDelta(
                dx_m=0.0,
                dy_m=0.0,
                dz_m=0.0,
                dyaw_rad=0.0,
                confidence=0.0,
                target_visible=False,
                metadata={"reason": "insufficient_valid_depth_in_center_patch"},
            ),
            None,
        )

    near_depth = float(np.percentile(patch[valid], 20))
    object_mask = valid & (patch <= near_depth + 0.03)
    if object_mask.sum() < 20:
        return (
            WristRefinementDelta(
                dx_m=0.0,
                dy_m=0.0,
                dz_m=0.0,
                dyaw_rad=0.0,
                confidence=0.0,
                target_visible=False,
                metadata={"reason": "target_patch_not_segmentable"},
            ),
            None,
        )

    ys, xs = np.nonzero(object_mask)
    pixel_u = int(xs.mean()) + max(0, cx - patch_half)
    pixel_v = int(ys.mean()) + max(0, cy - patch_half)
    pixel_result = build_pixel_result(
        observation=wrist_observation,
        pixel_u=pixel_u,
        pixel_v=pixel_v,
    )

    camera_xyz = np.asarray(pixel_result["camera_xyz_m"], dtype=float)
    desired_depth = float(selected_plan.best_candidate.metadata.get("wrist_desired_depth_m", 0.22))
    raw_dx = -float(camera_xyz[0])
    raw_dy = -float(camera_xyz[1])
    raw_dz = float(camera_xyz[2] - desired_depth)
    raw_dyaw = float(np.arctan2(camera_xyz[0], max(camera_xyz[2], 1e-6)) * 0.5)

    clipped = {
        "dx_m": _clip(raw_dx, MAX_DX_DY_M),
        "dy_m": _clip(raw_dy, MAX_DX_DY_M),
        "dz_m": _clip(raw_dz, MAX_DZ_M),
        "dyaw_rad": _clip(raw_dyaw, MAX_DYAW_RAD),
    }
    within_limits = (
        abs(raw_dx) <= MAX_DX_DY_M
        and abs(raw_dy) <= MAX_DX_DY_M
        and abs(raw_dz) <= MAX_DZ_M
        and abs(raw_dyaw) <= MAX_DYAW_RAD
    )
    visibility_confidence = float(np.clip(object_mask.sum() / 400.0, 0.0, 1.0))
    delta = WristRefinementDelta(
        dx_m=clipped["dx_m"],
        dy_m=clipped["dy_m"],
        dz_m=clipped["dz_m"],
        dyaw_rad=clipped["dyaw_rad"],
        confidence=visibility_confidence,
        target_visible=within_limits,
        metadata={
            "raw_delta": {
                "dx_m": raw_dx,
                "dy_m": raw_dy,
                "dz_m": raw_dz,
                "dyaw_rad": raw_dyaw,
            },
            "within_limits": within_limits,
            "pixel_u": pixel_u,
            "pixel_v": pixel_v,
            "desired_depth_m": desired_depth,
        },
    )
    return delta, pixel_result
