from __future__ import annotations

import base64
import io
import json
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .schemas import GraspPlanResult, ResolvedTarget


class LocalGraspRuntime:
    def __init__(
        self,
        mode: str = "mock",
        scene_config_path: str = "",
        grasp_config_path: str = "",
        run_root: str = "",
        headless: bool = True,
        show_viewer_ui: bool = False,
    ) -> None:
        self.mode = (mode or "mock").strip().lower()
        self.scene_config_path = scene_config_path.strip()
        self.grasp_config_path = grasp_config_path.strip()
        self.run_root = run_root.strip()
        self.headless = bool(headless)
        self.show_viewer_ui = bool(show_viewer_ui)

    def _bbox_to_int_tuple(self, bbox_xyxy: list[float]) -> tuple[int, int, int, int]:
        rounded = [int(round(float(value))) for value in bbox_xyxy]
        if len(rounded) != 4:
            raise ValueError("Expected resolved_target.bbox_xyxy to contain 4 values")
        return rounded[0], rounded[1], rounded[2], rounded[3]

    @staticmethod
    def _clamp_bbox_to_shape(
        bbox_xyxy: tuple[int, int, int, int],
        image_shape_hw: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        height, width = int(image_shape_hw[0]), int(image_shape_hw[1])
        x0, y0, x1, y1 = [int(value) for value in bbox_xyxy]
        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return x0, y0, x1, y1

    @staticmethod
    def _expand_bbox(
        bbox_xyxy: tuple[int, int, int, int],
        image_shape_hw: tuple[int, int],
        *,
        pixels: int,
        ratio: float,
    ) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = [int(value) for value in bbox_xyxy]
        pad = max(int(pixels), int(round(max(x1 - x0, y1 - y0) * float(ratio))))
        return LocalGraspRuntime._clamp_bbox_to_shape((x0 - pad, y0 - pad, x1 + pad, y1 + pad), image_shape_hw)

    @staticmethod
    def _map_bbox_from_cw_rotated_to_original(
        bbox_xyxy: tuple[int, int, int, int],
        original_shape_hw: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        height, _width = int(original_shape_hw[0]), int(original_shape_hw[1])
        x0, y0, x1, y1 = [int(value) for value in bbox_xyxy]
        mapped = (y0, height - x1, y1, height - x0)
        return LocalGraspRuntime._clamp_bbox_to_shape(mapped, original_shape_hw)

    @staticmethod
    def _depth_crop_stats(depth_m, bbox_xyxy: tuple[int, int, int, int]) -> Dict[str, Any]:
        import numpy as np

        x0, y0, x1, y1 = [int(value) for value in bbox_xyxy]
        crop = np.asarray(depth_m)[y0:y1, x0:x1]
        valid = crop[np.isfinite(crop) & (crop > 1e-6)]
        stats: Dict[str, Any] = {
            "crop_shape_hw": [int(crop.shape[0]), int(crop.shape[1])] if crop.ndim == 2 else [],
            "crop_area_px": int(crop.size),
            "valid_depth_pixels": int(valid.size),
        }
        if valid.size:
            stats.update(
                {
                    "depth_min_m": float(np.min(valid)),
                    "depth_median_m": float(np.median(valid)),
                    "depth_max_m": float(np.max(valid)),
                }
            )
        return stats

    @staticmethod
    def _load_rgb_image_from_metadata(observation_metadata: Dict[str, Any] | None):
        from PIL import Image, ImageOps

        metadata = observation_metadata or {}
        detail = LocalGraspRuntime._metadata_detail(metadata)
        encoded = metadata.get("image_base64") or detail.get("image_base64")
        data_url = metadata.get("image_data_url") or detail.get("image_data_url")
        if data_url and "," in str(data_url):
            encoded = str(data_url).split(",", 1)[1]
        if encoded:
            return ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(str(encoded))))).convert("RGB")
        path_raw = metadata.get("image_path") or detail.get("image_path")
        if path_raw and Path(str(path_raw)).expanduser().exists():
            return ImageOps.exif_transpose(Image.open(Path(str(path_raw)).expanduser())).convert("RGB")
        return None

    @staticmethod
    def _save_depth_debug_image(depth_m, bbox_xyxy: tuple[int, int, int, int], path: Path) -> None:
        import numpy as np
        from PIL import Image, ImageDraw

        depth = np.asarray(depth_m, dtype=np.float32)
        valid = depth[np.isfinite(depth) & (depth > 1e-6)]
        if valid.size:
            lo, hi = np.percentile(valid, [2.0, 98.0])
            scaled = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
        else:
            scaled = np.zeros_like(depth, dtype=np.float32)
        image = Image.fromarray((scaled * 255.0).astype(np.uint8), mode="L").convert("RGB")
        ImageDraw.Draw(image).rectangle(list(bbox_xyxy), outline=(255, 64, 64), width=4)
        image.save(path)

    @staticmethod
    def _save_rgb_debug_image(rgb_image, bbox_xyxy: tuple[int, int, int, int], path: Path) -> None:
        from PIL import ImageDraw

        image = rgb_image.copy()
        ImageDraw.Draw(image).rectangle(list(bbox_xyxy), outline=(54, 193, 255), width=5)
        image.save(path)

    @staticmethod
    def _pointcloud_filter_stats(
        *,
        depth_m,
        intrinsics,
        extrinsics,
        bbox_xyxy: tuple[int, int, int, int],
        table_top_z_m: float,
        table_margin_m: float,
        z_min_m: float,
        z_max_m: float,
    ) -> Dict[str, Any]:
        import numpy as np
        from ask2act_grasp.utils.pcd_utils import backproject_depth, crop_depth_to_bbox
        from ask2act_grasp.utils.tf_utils import transform_points

        cropped_depth = crop_depth_to_bbox(np.asarray(depth_m, dtype=np.float32), bbox_xyxy)
        camera_points = backproject_depth(cropped_depth, np.asarray(intrinsics, dtype=float))
        if camera_points.size == 0:
            return {"camera_point_count": 0, "world_point_count": 0}
        world_points = transform_points(np.asarray(extrinsics, dtype=float), camera_points)
        world_z = world_points[:, 2]
        finite = np.isfinite(world_z)
        lower = max(float(table_top_z_m) + float(table_margin_m), float(z_min_m))
        upper = float(z_max_m)
        within = finite & (world_z > lower) & (world_z < upper)
        return {
            "camera_point_count": int(camera_points.shape[0]),
            "world_point_count": int(world_points.shape[0]),
            "world_z_min_m": float(np.min(world_z[finite])) if np.any(finite) else None,
            "world_z_p05_m": float(np.percentile(world_z[finite], 5.0)) if np.any(finite) else None,
            "world_z_median_m": float(np.median(world_z[finite])) if np.any(finite) else None,
            "world_z_p95_m": float(np.percentile(world_z[finite], 95.0)) if np.any(finite) else None,
            "world_z_max_m": float(np.max(world_z[finite])) if np.any(finite) else None,
            "z_filter_lower_m": lower,
            "z_filter_upper_m": upper,
            "below_lower_count": int(np.count_nonzero(finite & (world_z <= lower))),
            "above_upper_count": int(np.count_nonzero(finite & (world_z >= upper))),
            "within_z_filter_count": int(np.count_nonzero(within)),
        }

    @staticmethod
    def _estimate_table_top_from_bbox_world(
        *,
        depth_m,
        intrinsics,
        extrinsics,
        bbox_xyxy: tuple[int, int, int, int],
        fallback_table_top_z_m: float,
        table_margin_m: float,
    ) -> Dict[str, Any]:
        import numpy as np
        from ask2act_grasp.utils.pcd_utils import backproject_depth, crop_depth_to_bbox
        from ask2act_grasp.utils.tf_utils import transform_points

        cropped_depth = crop_depth_to_bbox(np.asarray(depth_m, dtype=np.float32), bbox_xyxy)
        camera_points = backproject_depth(cropped_depth, np.asarray(intrinsics, dtype=float))
        if camera_points.size == 0:
            return {
                "table_top_z_m": float(fallback_table_top_z_m),
                "source": "fallback_no_camera_points",
                "world_point_count": 0,
            }
        world_points = transform_points(np.asarray(extrinsics, dtype=float), camera_points)
        world_z = world_points[:, 2]
        finite_z = world_z[np.isfinite(world_z)]
        if finite_z.size < 10:
            return {
                "table_top_z_m": float(fallback_table_top_z_m),
                "source": "fallback_too_few_world_points",
                "world_point_count": int(finite_z.size),
            }

        percentile = float(os.getenv("ASK2ACT_REAL_AUTO_TABLE_Z_PERCENTILE", "5.0"))
        support_z = float(np.percentile(finite_z, percentile))
        clearance = float(os.getenv("ASK2ACT_REAL_AUTO_TABLE_Z_CLEARANCE_M", "0.010"))
        table_top_z = support_z - float(table_margin_m) - clearance
        return {
            "table_top_z_m": table_top_z,
            "source": "bbox_world_z_percentile",
            "percentile": percentile,
            "support_z_m": support_z,
            "clearance_m": clearance,
            "world_point_count": int(finite_z.size),
            "world_z_min_m": float(np.min(finite_z)),
            "world_z_p05_m": float(np.percentile(finite_z, 5.0)),
            "world_z_median_m": float(np.median(finite_z)),
            "world_z_p95_m": float(np.percentile(finite_z, 95.0)),
            "world_z_max_m": float(np.max(finite_z)),
        }

    def _default_run_dir(self) -> Path:
        root = Path(self.run_root).expanduser() if self.run_root else (Path(__file__).resolve().parents[2] / "logs" / "a6000_runs")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = root / f"run_{stamp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _default_scene_config_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / "simulation" / "ask2act_grasp" / "config" / "scene_config.yaml"

    def _default_grasp_config_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / "simulation" / "ask2act_grasp" / "config" / "grasp_config.yaml"

    def _ensure_grasp_import_path(self) -> None:
        simulation_root = Path(__file__).resolve().parents[2] / "simulation"
        if str(simulation_root) not in sys.path:
            sys.path.insert(0, str(simulation_root))

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        try:
            import numpy as np
        except Exception:
            np = None
        if np is not None:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
        if isinstance(value, dict):
            return {str(key): LocalGraspRuntime._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [LocalGraspRuntime._to_jsonable(item) for item in value]
        return value

    @staticmethod
    def _metadata_detail(observation_metadata: Dict[str, Any] | None) -> Dict[str, Any]:
        if not observation_metadata:
            return {}
        detail = observation_metadata.get("detail")
        return detail if isinstance(detail, dict) else {}

    def _load_depth_image_m(self, observation_metadata: Dict[str, Any] | None):
        import numpy as np

        metadata = observation_metadata or {}
        detail = self._metadata_detail(metadata)
        encoded = metadata.get("depth_npy_base64") or detail.get("depth_npy_base64")
        if encoded:
            depth = np.load(io.BytesIO(base64.b64decode(str(encoded))))
        else:
            raw_path = metadata.get("depth_npy_path") or detail.get("depth_npy_path")
            if not raw_path:
                raise RuntimeError("Stretch observation did not include depth_npy_base64/depth_npy_path")
            path = Path(str(raw_path)).expanduser()
            if not path.exists():
                raise RuntimeError(
                    f"Depth npy path is not readable on the A6000: {path}. "
                    "Update the Stretch server so it sends depth_npy_base64, then git pull/restart on the robot."
                )
            depth = np.load(path)

        depth_scale = metadata.get("depth_scale_m_per_unit", detail.get("depth_scale_m_per_unit"))
        if depth_scale is not None and not np.issubdtype(depth.dtype, np.floating):
            return depth.astype(np.float32) * float(depth_scale)
        if not np.issubdtype(depth.dtype, np.floating):
            return depth.astype(np.float32) * 0.001
        return depth.astype(np.float32, copy=False)

    def _load_intrinsics_matrix(self, observation_metadata: Dict[str, Any] | None):
        import numpy as np

        metadata = observation_metadata or {}
        detail = self._metadata_detail(metadata)
        payload = metadata.get("camera_intrinsics") or detail.get("camera_intrinsics")
        if payload is None:
            path_raw = metadata.get("camera_intrinsics_path") or detail.get("camera_intrinsics_path")
            if path_raw and Path(str(path_raw)).expanduser().exists():
                payload = json.loads(Path(str(path_raw)).expanduser().read_text(encoding="utf-8"))
        if payload is None:
            raise RuntimeError("Stretch observation did not include camera_intrinsics")

        if isinstance(payload, dict):
            if "matrix" in payload:
                return np.asarray(payload["matrix"], dtype=np.float64).reshape(3, 3)
            fx = float(payload["fx"])
            fy = float(payload["fy"])
            cx = float(payload.get("cx", payload.get("ppx")))
            cy = float(payload.get("cy", payload.get("ppy")))
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        return np.asarray(payload, dtype=np.float64).reshape(3, 3)

    def _load_camera_extrinsics(self, observation_metadata: Dict[str, Any] | None):
        import numpy as np

        metadata = observation_metadata or {}
        detail = self._metadata_detail(metadata)
        payload: Any = metadata.get("camera_extrinsics") or detail.get("camera_extrinsics")
        if payload is None:
            raw_json = os.getenv("ASK2ACT_HEAD_CAMERA_EXTRINSICS_JSON", "").strip()
            if raw_json:
                payload = json.loads(raw_json)
        if payload is None:
            path_raw = (
                os.getenv("ASK2ACT_HEAD_CAMERA_EXTRINSICS_PATH", "").strip()
                or os.getenv("ASK2ACT_REAL_CAMERA_EXTRINSICS_PATH", "").strip()
            )
            if not path_raw:
                raise RuntimeError(
                    "Missing head camera extrinsics. Set ASK2ACT_HEAD_CAMERA_EXTRINSICS_PATH "
                    "to a 4x4 camera-to-world JSON file, or set ASK2ACT_HEAD_CAMERA_EXTRINSICS_JSON."
                )
            path = Path(path_raw).expanduser()
            if not path.exists():
                raise RuntimeError(f"Head camera extrinsics file not found: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(payload, dict):
            for key in ("camera_extrinsics", "camera_to_world", "matrix", "T_camera_world"):
                if key in payload:
                    payload = payload[key]
                    break
        return np.asarray(payload, dtype=np.float64).reshape(4, 4)

    @staticmethod
    def _default_robot_state() -> Dict[str, float]:
        raw = os.getenv("ASK2ACT_REAL_ROBOT_STATE_JSON", "").strip()
        if raw:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise RuntimeError("ASK2ACT_REAL_ROBOT_STATE_JSON must be a JSON object")
            return {str(key): float(value) for key, value in payload.items()}
        return {
            "base_x": 0.0,
            "base_y": 0.0,
            "lift": 0.0,
            "arm": 0.0,
            "wrist_yaw": 0.0,
            "wrist_pitch": 0.0,
            "wrist_roll": 0.0,
        }

    @staticmethod
    def _base_reach_preposition_distance_m(geometric_grasp: Dict[str, Any], grasp_config: Any) -> tuple[float, Dict[str, Any]]:
        import math
        import numpy as np

        target_x = float(geometric_grasp["grasp_x"])
        target_y = float(geometric_grasp["grasp_y"])
        # In the base-relative world frame the Stretch arm reaches along -Y.
        # The robot-side base_translate_arm_axis primitive rotates the base 90
        # deg, translates along that reach axis, then rotates back before the
        # A6000 reobserves and replans.
        arm_axis_distance = float(-target_y)
        goal_distance = float(
            os.getenv("ASK2ACT_REAL_BASE_PREPOSITION_GOAL_DISTANCE_M", str(grasp_config.base_preposition_goal_distance_m))
        )
        extra_margin = float(
            os.getenv(
                "ASK2ACT_REAL_BASE_PREPOSITION_LONGITUDINAL_EXTRA_M",
                str(grasp_config.base_preposition_longitudinal_extra_m),
            )
        )
        cfg_max_translate = float(getattr(grasp_config, "base_preposition_max_translate_m", 0.40))
        env_max_translate = float(os.getenv("ASK2ACT_GEOMETRIC_TOP_DOWN_BASE_REACH_TRANSLATE_MAX_M", "0.16"))
        max_translate = max(0.0, min(cfg_max_translate, env_max_translate))
        reach_error = arm_axis_distance - goal_distance
        deadband = float(os.getenv("ASK2ACT_REAL_BASE_PREPOSITION_DEADBAND_M", "0.015"))
        if abs(reach_error) <= deadband or max_translate <= 0.0:
            requested = 0.0
        else:
            requested = float(
                np.clip(
                    reach_error + math.copysign(extra_margin, reach_error),
                    -max_translate,
                    max_translate,
                )
            )
        reason = "target_beyond_comfort_reach_hand_short" if reach_error > 0.0 else "target_inside_comfort_reach_hand_long"
        diagnostics = {
            "target_grasp_xy_m": [target_x, target_y],
            "arm_axis_distance_m": arm_axis_distance,
            "goal_distance_m": goal_distance,
            "reach_error_m": reach_error,
            "extra_margin_m": extra_margin,
            "max_translate_m": max_translate,
            "deadband_m": deadband,
            "requested_base_translate_arm_axis_m": requested,
            "reason": reason,
        }
        return requested, diagnostics

    def _build_base_reach_preposition_plan(
        self,
        *,
        resolved_target: ResolvedTarget,
        geometric_grasp: Dict[str, Any],
        grasp_config: Any,
        planning_error: Exception,
        run_dir: Path,
        point_count: int,
        selected_option_name: str,
        diagnostics: Dict[str, Any],
        applied_bbox_tuple: tuple[int, int, int, int],
        raw_bbox_tuple: tuple[int, int, int, int],
        rotate_clockwise_90: bool,
        intrinsics: Any,
        extrinsics: Any,
        current_state: Dict[str, float],
        simple_ik_status: Dict[str, Any],
        dry_run: bool,
    ) -> Dict[str, Any]:
        from ask2act_grasp.types import MotionPlan, MotionWaypoint

        requested_move, base_preposition = self._base_reach_preposition_distance_m(geometric_grasp, grasp_config)
        if abs(requested_move) <= 0.01:
            raise RuntimeError(
                "SimpleIK could not produce an accurate top-down grasp, but the target is already near the "
                "base preposition comfort distance; refusing to move on stale coordinates. "
                f"IK/planning error: {planning_error}"
            ) from planning_error

        waypoint = MotionWaypoint(
            name="base_translate_for_reach",
            joint_targets={"base_translate_arm_axis": float(requested_move)},
            settle_s=0.4,
        )
        motion_plan = MotionPlan(
            backend="real_base_reach_preposition_required",
            waypoints=[waypoint],
            metadata={
                "planning_mode": "base_reach_preposition_then_reobserve",
                "numeric_targets": {
                    "base_translate_arm_axis_m": float(requested_move),
                    "planning_error": str(planning_error),
                    "simple_ik": simple_ik_status,
                    **base_preposition,
                },
                "joint_targets": {"base_translate_arm_axis_m": float(requested_move)},
                "base_preposition": base_preposition,
            },
        )
        trajectory = [
            {
                "name": waypoint.name,
                "joint_targets": {str(key): float(value) for key, value in waypoint.joint_targets.items()},
                "settle_s": float(waypoint.settle_s),
            }
            for waypoint in motion_plan.waypoints
        ]
        pipeline_result = {
            "success": True,
            "scene_xml_path": "",
            "point_cloud_count": int(point_count),
            "selected_grasp_score": float(resolved_target.score),
            "planner_backend": motion_plan.backend,
            "trajectory": trajectory,
            "intermediate": {
                "geometric_grasp": geometric_grasp,
                "motion_plan_metadata": motion_plan.metadata,
                "simple_ik": simple_ik_status,
                "base_preposition": base_preposition,
                "target_bbox_2d": list(applied_bbox_tuple),
                "raw_detection_bbox_2d": list(raw_bbox_tuple),
                "selected_pointcloud_option": selected_option_name,
                "pointcloud_options": diagnostics["options"],
                "rotated_observation_clockwise_90": bool(rotate_clockwise_90),
                "camera_intrinsics": intrinsics,
                "camera_extrinsics": extrinsics,
                "current_state_for_planning": current_state,
            },
            "error": None,
            "note": "Base preposition only; reobserve and replan before grasping the same target.",
        }
        (run_dir / "real_pointcloud_plan.json").write_text(
            json.dumps(self._to_jsonable(pipeline_result), indent=2),
            encoding="utf-8",
        )
        plan_summary = GraspPlanResult(
            pipeline_mode="real_pointcloud",
            planner_backend=motion_plan.backend,
            target_bbox_xyxy=list(resolved_target.bbox_xyxy),
            point_cloud_count=int(point_count),
            selected_grasp_score=float(resolved_target.score),
            trajectory_waypoint_count=len(trajectory),
            pipeline_run_dir=str(run_dir),
            success=True,
            note="SimpleIK requested base preposition; execute this move, reobserve, then replan the grasp.",
        )
        dispatch_payload = {
            "pipeline_mode": "real_pointcloud",
            "planner_backend": motion_plan.backend,
            "target_bbox_2d": list(applied_bbox_tuple),
            "target_bbox_xyxy": list(resolved_target.bbox_xyxy),
            "trajectory": trajectory,
            "run_dir": str(run_dir),
            "resolved_target": resolved_target.model_dump(),
            "geometric_grasp": self._to_jsonable(geometric_grasp),
            "motion_plan_metadata": self._to_jsonable(motion_plan.metadata),
            "simple_ik": simple_ik_status,
            "selected_pointcloud_option": selected_option_name,
            "pointcloud_diagnostics_path": str(run_dir / "real_pointcloud_diagnostics.json"),
            "preposition_only": True,
        }
        return {
            "ok": True,
            "dry_run": bool(dry_run),
            "pipeline_mode": "real_pointcloud",
            "plan_summary": plan_summary.model_dump(),
            "pipeline_result": self._to_jsonable(pipeline_result),
            "dispatch_payload": dispatch_payload,
        }

    @staticmethod
    def _fallback_geometric_grasp(points_xyz, table_top_z_m: float, max_gripper_width_m: float) -> Dict[str, Any]:
        import numpy as np

        points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        if len(points) < 3:
            raise RuntimeError("Need at least three target points for geometric grasp fallback")
        z_top = float(np.percentile(points[:, 2], 95.0))
        z_bottom = float(np.percentile(points[:, 2], 5.0))
        center_z = float((z_top + z_bottom) / 2.0)
        slice_mask = np.abs(points[:, 2] - center_z) < 0.015
        slice_pts = points[slice_mask] if int(np.count_nonzero(slice_mask)) >= 10 else points
        xy = np.asarray(slice_pts[:, :2], dtype=np.float64)
        center_xy = np.median(xy, axis=0)
        centered = xy - center_xy[None, :]
        if len(centered) >= 3:
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            minor_axis = eigvecs[:, int(np.argmin(eigvals))]
        else:
            minor_axis = np.array([1.0, 0.0], dtype=np.float64)
        projections = centered @ minor_axis
        min_cross_section_width = float(np.percentile(projections, 95.0) - np.percentile(projections, 5.0))
        grip_angle_rad = float(np.arctan2(minor_axis[1], minor_axis[0]))
        object_height = float(max(0.0, z_top - z_bottom))
        gripper_open_width = float(max(0.035, min_cross_section_width + 0.03))
        return {
            "grasp_x": float(center_xy[0]),
            "grasp_y": float(center_xy[1]),
            "grasp_z": center_z,
            "grip_angle_rad": grip_angle_rad,
            "gripper_open_width": gripper_open_width,
            "min_cross_section_width": min_cross_section_width,
            "object_center": [float(center_xy[0]), float(center_xy[1]), center_z],
            "object_height": object_height,
            "object_top_z": z_top,
            "object_bottom_z": z_bottom,
            "slice_center_z": center_z,
            "slice_point_count": int(len(slice_pts)),
            "fitted_circle_center_xy": [float(center_xy[0]), float(center_xy[1])],
            "fitted_circle_radius": float(max(min_cross_section_width / 2.0, 0.0)),
            "estimated_error": 0.0,
            "residual_std": 0.0,
            "arc_coverage_rad": 0.0,
            "arc_coverage_deg": 0.0,
            "conservative_diameter": min_cross_section_width,
            "uncertainty_margin": 0.0,
            "fixed_clearance_margin": 0.03,
            "grasp_point_validated": True,
            "slice_thickness_m": 0.015,
            "validation_tolerance_m": 0.01,
            "table_z": float(table_top_z_m),
            "width_near_limit": bool(gripper_open_width > float(max_gripper_width_m) - 0.01),
            "open_width_exceeds_max": bool(gripper_open_width > float(max_gripper_width_m)),
            "max_gripper_width_m": float(max_gripper_width_m),
            "method": "geometric_point_cloud_fallback",
        }

    def _compute_geometric_grasp(self, points_xyz, table_top_z_m: float, max_gripper_width_m: float) -> Dict[str, Any]:
        try:
            from ask2act_grasp.grasp.geometric_grasp import compute_geometric_grasp
        except ModuleNotFoundError as exc:
            if exc.name != "scipy":
                raise
            return self._fallback_geometric_grasp(points_xyz, table_top_z_m, max_gripper_width_m)
        return compute_geometric_grasp(
            points_xyz,
            table_top_z_m,
            max_gripper_width_m=max_gripper_width_m,
        )

    def _mock_plan(self, resolved_target: ResolvedTarget, dry_run: bool) -> Dict[str, Any]:
        bbox_tuple = self._bbox_to_int_tuple(resolved_target.bbox_xyxy)
        plan_summary = GraspPlanResult(
            pipeline_mode="mock",
            planner_backend="mock-preview",
            target_bbox_xyxy=list(resolved_target.bbox_xyxy),
            point_cloud_count=None,
            selected_grasp_score=resolved_target.score,
            trajectory_waypoint_count=0,
            pipeline_run_dir=None,
            success=True,
            note="Preview-only local plan. Set ASK2ACT_PIPELINE_MODE=local_sim to invoke the current Ask2Act grasp stack.",
        )
        dispatch_payload = {
            "pipeline_mode": "mock",
            "planner_backend": "mock-preview",
            "target_bbox_2d": list(bbox_tuple),
            "target_bbox_xyxy": list(resolved_target.bbox_xyxy),
            "resolved_target": resolved_target.model_dump(),
            "dry_run": dry_run,
        }
        return {
            "ok": True,
            "dry_run": dry_run,
            "pipeline_mode": "mock",
            "plan_summary": plan_summary.model_dump(),
            "pipeline_result": None,
            "dispatch_payload": dispatch_payload,
        }

    def _local_sim_plan(self, resolved_target: ResolvedTarget, dry_run: bool) -> Dict[str, Any]:
        if dry_run:
            return self._mock_plan(resolved_target, dry_run=True)

        from simulation.ask2act_grasp.pipeline import run_pipeline

        bbox_tuple = self._bbox_to_int_tuple(resolved_target.bbox_xyxy)
        run_dir = self._default_run_dir()
        result = run_pipeline(
            scene_config_path=self.scene_config_path or None,
            grasp_config_path=self.grasp_config_path or None,
            run_dir=run_dir,
            target_bbox_2d=bbox_tuple,
            headless=self.headless,
            show_viewer_ui=self.show_viewer_ui,
        )
        plan_summary = GraspPlanResult(
            pipeline_mode="local_sim",
            planner_backend=result.planner_backend,
            target_bbox_xyxy=list(resolved_target.bbox_xyxy),
            point_cloud_count=result.point_cloud_count,
            selected_grasp_score=result.selected_grasp_score,
            trajectory_waypoint_count=len(result.trajectory or []),
            pipeline_run_dir=str(run_dir),
            success=result.success,
            note=result.error,
        )
        dispatch_payload = {
            "pipeline_mode": "local_sim",
            "planner_backend": result.planner_backend,
            "target_bbox_2d": list(bbox_tuple),
            "target_bbox_xyxy": list(resolved_target.bbox_xyxy),
            "trajectory": result.trajectory,
            "run_dir": str(run_dir),
            "resolved_target": resolved_target.model_dump(),
        }
        pipeline_result = {
            "success": result.success,
            "scene_xml_path": result.scene_xml_path,
            "point_cloud_count": result.point_cloud_count,
            "selected_grasp_score": result.selected_grasp_score,
            "planner_backend": result.planner_backend,
            "trajectory": result.trajectory,
            "intermediate": result.intermediate,
            "error": result.error,
        }
        return {
            "ok": bool(result.success),
            "dry_run": False,
            "pipeline_mode": "local_sim",
            "plan_summary": plan_summary.model_dump(),
            "pipeline_result": pipeline_result,
            "dispatch_payload": dispatch_payload,
        }

    def _real_pointcloud_plan(
        self,
        resolved_target: ResolvedTarget,
        observation_metadata: Dict[str, Any] | None,
        dry_run: bool,
        rotate_clockwise_90: bool,
    ) -> Dict[str, Any]:
        self._ensure_grasp_import_path()

        import numpy as np
        from ask2act_grasp.perception.point_cloud_gen import PointCloudGenerator
        from ask2act_grasp.planning.motion_planner import MotionPlanner
        from ask2act_grasp.types import GraspCandidate
        from ask2act_grasp.utils.config_loader import load_grasp_config, load_scene_config
        from ask2act_grasp.utils.head_camera_orientation import (
            rotate_camera_extrinsics_clockwise_90,
            rotate_intrinsics_clockwise_90,
        )
        from ask2act_grasp.utils.tf_utils import pose_from_axes
        from ask2act_grasp.utils.visualization import save_point_cloud

        if not observation_metadata:
            raise RuntimeError("real_pointcloud mode requires a Stretch observation with RGB-D metadata")

        bbox_tuple = self._bbox_to_int_tuple(resolved_target.bbox_xyxy)
        original_depth_m = self._load_depth_image_m(observation_metadata)
        original_intrinsics = self._load_intrinsics_matrix(observation_metadata)
        original_extrinsics = self._load_camera_extrinsics(observation_metadata)
        scene_config, _head_config = load_scene_config(self.scene_config_path or self._default_scene_config_path())
        grasp_config = load_grasp_config(self.grasp_config_path or self._default_grasp_config_path())
        real_table_top_raw = os.getenv("ASK2ACT_REAL_TABLE_TOP_Z_M", "").strip()
        auto_table_top = real_table_top_raw.lower() in {"", "auto", "infer", "estimate"}
        if real_table_top_raw and not auto_table_top:
            table_top_z = float(real_table_top_raw)
            scene_config = replace(
                scene_config,
                table_position_m=(
                    scene_config.table_position_m[0],
                    scene_config.table_position_m[1],
                    table_top_z - scene_config.table_size_m[2],
                ),
            )
        real_table_margin_raw = os.getenv("ASK2ACT_REAL_TABLE_CLEARANCE_MARGIN_M", "").strip()
        if real_table_margin_raw:
            scene_config = replace(scene_config, table_clearance_margin_m=float(real_table_margin_raw))
        if os.getenv("ASK2ACT_REAL_ALLOW_APPROXIMATE_TOPDOWN_FALLBACK", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            grasp_config = replace(grasp_config, allow_approximate_topdown_fallback=True)

        run_dir = self._default_run_dir()
        rgb_image = self._load_rgb_image_from_metadata(observation_metadata)
        min_points = int(os.getenv("ASK2ACT_REAL_MIN_POINT_CLOUD_COUNT", "30"))
        object_z_max_above_table_m = float(os.getenv("ASK2ACT_REAL_OBJECT_Z_MAX_ABOVE_TABLE_M", "0.20"))
        expand_pixels = int(os.getenv("ASK2ACT_REAL_BBOX_EXPAND_PX", "12"))
        expand_ratio = float(os.getenv("ASK2ACT_REAL_BBOX_EXPAND_RATIO", "0.12"))
        options: list[dict[str, Any]] = []
        original_shape = tuple(int(v) for v in original_depth_m.shape)
        if rotate_clockwise_90:
            rotated_depth_m = np.rot90(original_depth_m, k=-1)
            rotated_intrinsics = rotate_intrinsics_clockwise_90(original_intrinsics, original_depth_shape=original_shape)
            rotated_extrinsics = rotate_camera_extrinsics_clockwise_90(original_extrinsics)
            rotated_shape = tuple(int(v) for v in rotated_depth_m.shape)
            primary_bbox = self._clamp_bbox_to_shape(bbox_tuple, rotated_shape)
            mapped_bbox = self._map_bbox_from_cw_rotated_to_original(bbox_tuple, original_shape)
            options.extend(
                [
                    {
                        "name": "rotated_cw_depth_with_detection_bbox",
                        "depth_m": rotated_depth_m,
                        "intrinsics": rotated_intrinsics,
                        "extrinsics": rotated_extrinsics,
                        "bbox": primary_bbox,
                        "rotated_observation_clockwise_90": True,
                    },
                    {
                        "name": "original_depth_bbox_mapped_from_rotated_detection",
                        "depth_m": original_depth_m,
                        "intrinsics": original_intrinsics,
                        "extrinsics": original_extrinsics,
                        "bbox": mapped_bbox,
                        "rotated_observation_clockwise_90": False,
                    },
                ]
            )
        else:
            options.append(
                {
                    "name": "original_depth_with_detection_bbox",
                    "depth_m": original_depth_m,
                    "intrinsics": original_intrinsics,
                    "extrinsics": original_extrinsics,
                    "bbox": self._clamp_bbox_to_shape(bbox_tuple, original_shape),
                    "rotated_observation_clockwise_90": False,
                }
            )

        expanded_options: list[dict[str, Any]] = []
        for option in options:
            expanded_options.append(option)
            expanded_bbox = self._expand_bbox(
                option["bbox"],
                tuple(int(v) for v in option["depth_m"].shape),
                pixels=expand_pixels,
                ratio=expand_ratio,
            )
            if expanded_bbox != option["bbox"]:
                expanded = dict(option)
                expanded["name"] = f"{option['name']}_expanded"
                expanded["bbox"] = expanded_bbox
                expanded["bbox_expanded"] = True
                expanded_options.append(expanded)
        options = expanded_options

        point_cloud_gen = PointCloudGenerator()
        option_results: list[dict[str, Any]] = []
        selected_result: dict[str, Any] | None = None
        for option in options:
            table_estimate = (
                self._estimate_table_top_from_bbox_world(
                    depth_m=option["depth_m"],
                    intrinsics=option["intrinsics"],
                    extrinsics=option["extrinsics"],
                    bbox_xyxy=option["bbox"],
                    fallback_table_top_z_m=scene_config.table_top_z_m,
                    table_margin_m=scene_config.table_clearance_margin_m,
                )
                if auto_table_top
                else {
                    "table_top_z_m": float(scene_config.table_top_z_m),
                    "source": "ASK2ACT_REAL_TABLE_TOP_Z_M" if real_table_top_raw else "scene_config",
                }
            )
            option_table_top_z = float(table_estimate["table_top_z_m"])
            point_cloud = point_cloud_gen.generate(
                depth_image=option["depth_m"],
                camera_intrinsics=option["intrinsics"],
                camera_extrinsics=option["extrinsics"],
                table_top_z_m=option_table_top_z,
                table_margin_m=scene_config.table_clearance_margin_m,
                z_min_m=grasp_config.z_min_m,
                z_max_m=min(grasp_config.z_max_m, option_table_top_z + object_z_max_above_table_m),
                target_bbox_2d=option["bbox"],
            )
            result = {
                "name": option["name"],
                "bbox": list(option["bbox"]),
                "bbox_expanded": bool(option.get("bbox_expanded", False)),
                "rotated_observation_clockwise_90": bool(option["rotated_observation_clockwise_90"]),
                "depth_shape_hw": [int(option["depth_m"].shape[0]), int(option["depth_m"].shape[1])],
                "table_top_estimate": table_estimate,
                "point_cloud_count": int(point_cloud.filtered_point_count),
                "depth_crop_stats": self._depth_crop_stats(option["depth_m"], option["bbox"]),
                "world_filter_stats": self._pointcloud_filter_stats(
                    depth_m=option["depth_m"],
                    intrinsics=option["intrinsics"],
                    extrinsics=option["extrinsics"],
                    bbox_xyxy=option["bbox"],
                    table_top_z_m=option_table_top_z,
                    table_margin_m=scene_config.table_clearance_margin_m,
                    z_min_m=grasp_config.z_min_m,
                    z_max_m=min(grasp_config.z_max_m, option_table_top_z + object_z_max_above_table_m),
                ),
                "point_cloud": point_cloud,
                "depth_m": option["depth_m"],
                "intrinsics": option["intrinsics"],
                "extrinsics": option["extrinsics"],
            }
            option_results.append(result)
            safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in option["name"])
            try:
                self._save_depth_debug_image(option["depth_m"], option["bbox"], run_dir / f"{safe_name}_depth_bbox.png")
                if rgb_image is not None:
                    option_rgb = rgb_image.rotate(-90, expand=True) if option["rotated_observation_clockwise_90"] else rgb_image
                    self._save_rgb_debug_image(option_rgb, option["bbox"], run_dir / f"{safe_name}_rgb_bbox.png")
            except Exception:
                pass
            if point_cloud.filtered_point_count >= min_points:
                selected_result = result
                break

        if selected_result is None and option_results:
            selected_result = max(option_results, key=lambda item: int(item["point_cloud_count"]))

        diagnostics = {
            "resolved_target": resolved_target.model_dump(),
            "raw_detection_bbox_xyxy": list(resolved_target.bbox_xyxy),
            "raw_detection_bbox_int": list(bbox_tuple),
            "dino_rotated_clockwise_90": bool(rotate_clockwise_90),
            "original_depth_shape_hw": [int(original_depth_m.shape[0]), int(original_depth_m.shape[1])],
            "min_required_points": min_points,
            "configured_table_top_z_m": float(scene_config.table_top_z_m),
            "selected_table_top_z_m": float(selected_result["table_top_estimate"]["table_top_z_m"]) if selected_result else None,
            "table_top_z_m": float(selected_result["table_top_estimate"]["table_top_z_m"]) if selected_result else float(scene_config.table_top_z_m),
            "table_top_source": (
                "auto_bbox_world_z" if auto_table_top else ("ASK2ACT_REAL_TABLE_TOP_Z_M" if real_table_top_raw else "scene_config")
            ),
            "table_clearance_margin_m": float(scene_config.table_clearance_margin_m),
            "z_filter_range_m": [
                float(grasp_config.z_min_m),
                float(
                    min(
                        grasp_config.z_max_m,
                        (
                            float(selected_result["table_top_estimate"]["table_top_z_m"])
                            if selected_result
                            else scene_config.table_top_z_m
                        )
                        + object_z_max_above_table_m,
                    )
                ),
            ],
            "object_z_max_above_table_m": object_z_max_above_table_m,
            "depth_aligned_to_color": bool(
                (observation_metadata or {}).get("depth_aligned_to_color")
                or self._metadata_detail(observation_metadata).get("depth_aligned_to_color")
            ),
            "options": [
                {key: self._to_jsonable(value) for key, value in item.items() if key not in {"point_cloud", "depth_m", "intrinsics", "extrinsics"}}
                for item in option_results
            ],
            "selected_option": selected_result["name"] if selected_result is not None else None,
        }
        (run_dir / "real_pointcloud_diagnostics.json").write_text(
            json.dumps(self._to_jsonable(diagnostics), indent=2),
            encoding="utf-8",
        )

        point_cloud = selected_result["point_cloud"] if selected_result is not None else None
        if point_cloud is None or point_cloud.filtered_point_count < min_points:
            raise RuntimeError(
                f"Target point cloud has too few points: {0 if point_cloud is None else point_cloud.filtered_point_count} < {min_points}. "
                f"Diagnostics saved to {run_dir / 'real_pointcloud_diagnostics.json'}. "
                "Check bbox/depth alignment, head pose, and camera extrinsics."
            )
        depth_m = selected_result["depth_m"]
        intrinsics = selected_result["intrinsics"]
        extrinsics = selected_result["extrinsics"]
        applied_bbox_tuple = tuple(int(v) for v in selected_result["bbox"])
        selected_table_top_z = float(selected_result["table_top_estimate"]["table_top_z_m"])
        scene_config = replace(
            scene_config,
            table_position_m=(
                scene_config.table_position_m[0],
                scene_config.table_position_m[1],
                selected_table_top_z - scene_config.table_size_m[2],
            ),
        )

        geometric_grasp = self._compute_geometric_grasp(
            point_cloud.world_points_xyz,
            scene_config.table_top_z_m,
            grasp_config.max_gripper_width_m,
        )
        pose = pose_from_axes(
            np.array(
                [
                    float(geometric_grasp["grasp_x"]),
                    float(geometric_grasp["grasp_y"]),
                    float(geometric_grasp["grasp_z"]),
                ],
                dtype=float,
            ),
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([0.0, 0.0, -1.0], dtype=float),
        )
        candidate = GraspCandidate(
            pose_4x4=pose,
            score=float(resolved_target.score),
            width_m=float(geometric_grasp["gripper_open_width"]),
            source="geometric_point_cloud",
            metadata={"geometric_grasp": geometric_grasp, "resolved_target": resolved_target.model_dump()},
            preferred_approach="top_down",
            approach_type="top_down",
        )
        np.save(run_dir / "target_depth_m.npy", depth_m)
        save_point_cloud(point_cloud.world_points_xyz, run_dir / "target_cloud_world.ply")
        save_point_cloud(point_cloud.camera_points_xyz, run_dir / "target_cloud_camera.ply")

        current_state = self._default_robot_state()
        motion_planner = MotionPlanner(scene_config, grasp_config)
        simple_ik_status = {
            "available": bool(motion_planner.simple_ik is not None),
            "init_error": motion_planner.simple_ik_init_error,
        }
        try:
            motion_plan = motion_planner.plan_to_grasp(candidate, current_state)
        except RuntimeError as exc:
            if motion_planner.simple_ik is not None and bool(getattr(grasp_config, "enable_base_preposition", False)):
                return self._build_base_reach_preposition_plan(
                    resolved_target=resolved_target,
                    geometric_grasp=geometric_grasp,
                    grasp_config=grasp_config,
                    planning_error=exc,
                    run_dir=run_dir,
                    point_count=point_cloud.filtered_point_count,
                    selected_option_name=str(selected_result["name"]),
                    diagnostics=diagnostics,
                    applied_bbox_tuple=applied_bbox_tuple,
                    raw_bbox_tuple=bbox_tuple,
                    rotate_clockwise_90=bool(selected_result["rotated_observation_clockwise_90"]),
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    current_state=current_state,
                    simple_ik_status=simple_ik_status,
                    dry_run=dry_run,
                )
            raise
        trajectory = [
            {
                "name": waypoint.name,
                "joint_targets": {str(key): float(value) for key, value in waypoint.joint_targets.items()},
                "settle_s": float(waypoint.settle_s),
            }
            for waypoint in motion_plan.waypoints
        ]

        pipeline_result = {
            "success": True,
            "scene_xml_path": "",
            "point_cloud_count": point_cloud.filtered_point_count,
            "selected_grasp_score": float(resolved_target.score),
            "planner_backend": motion_plan.backend,
            "trajectory": trajectory,
            "intermediate": {
                "geometric_grasp": geometric_grasp,
                "motion_plan_metadata": motion_plan.metadata,
                "simple_ik": simple_ik_status,
                "target_bbox_2d": list(applied_bbox_tuple),
                "raw_detection_bbox_2d": list(bbox_tuple),
                "selected_pointcloud_option": selected_result["name"],
                "pointcloud_options": diagnostics["options"],
                "rotated_observation_clockwise_90": bool(selected_result["rotated_observation_clockwise_90"]),
                "camera_intrinsics": intrinsics,
                "camera_extrinsics": extrinsics,
                "current_state_for_planning": current_state,
            },
            "error": None,
        }
        (run_dir / "real_pointcloud_plan.json").write_text(
            json.dumps(self._to_jsonable(pipeline_result), indent=2),
            encoding="utf-8",
        )

        plan_summary = GraspPlanResult(
            pipeline_mode="real_pointcloud",
            planner_backend=motion_plan.backend,
            target_bbox_xyxy=list(resolved_target.bbox_xyxy),
            point_cloud_count=point_cloud.filtered_point_count,
            selected_grasp_score=float(resolved_target.score),
            trajectory_waypoint_count=len(trajectory),
            pipeline_run_dir=str(run_dir),
            success=True,
            note="Planned from the latest Stretch head D435i depth frame and the resolved target bbox.",
        )
        dispatch_payload = {
            "pipeline_mode": "real_pointcloud",
            "planner_backend": motion_plan.backend,
            "target_bbox_2d": list(applied_bbox_tuple),
            "target_bbox_xyxy": list(resolved_target.bbox_xyxy),
            "trajectory": trajectory,
            "run_dir": str(run_dir),
            "resolved_target": resolved_target.model_dump(),
            "geometric_grasp": self._to_jsonable(geometric_grasp),
            "motion_plan_metadata": self._to_jsonable(motion_plan.metadata),
            "simple_ik": simple_ik_status,
            "selected_pointcloud_option": selected_result["name"],
            "pointcloud_diagnostics_path": str(run_dir / "real_pointcloud_diagnostics.json"),
        }
        return {
            "ok": True,
            "dry_run": dry_run,
            "pipeline_mode": "real_pointcloud",
            "plan_summary": plan_summary.model_dump(),
            "pipeline_result": self._to_jsonable(pipeline_result),
            "dispatch_payload": dispatch_payload,
        }

    def plan_for_target(
        self,
        resolved_target: ResolvedTarget,
        dry_run: bool = False,
        observation_metadata: Dict[str, Any] | None = None,
        rotate_clockwise_90: bool = True,
    ) -> Dict[str, Any]:
        if self.mode == "mock":
            return self._mock_plan(resolved_target, dry_run=dry_run)
        if self.mode == "local_sim":
            return self._local_sim_plan(resolved_target, dry_run=dry_run)
        if self.mode in {"real_pointcloud", "real_depth"}:
            return self._real_pointcloud_plan(
                resolved_target,
                observation_metadata=observation_metadata,
                dry_run=dry_run,
                rotate_clockwise_90=rotate_clockwise_90,
            )
        raise RuntimeError(f"Unsupported ASK2ACT_PIPELINE_MODE: {self.mode}")
