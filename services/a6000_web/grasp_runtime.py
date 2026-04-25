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
        depth_m = self._load_depth_image_m(observation_metadata)
        intrinsics = self._load_intrinsics_matrix(observation_metadata)
        extrinsics = self._load_camera_extrinsics(observation_metadata)
        if rotate_clockwise_90:
            original_shape = tuple(int(v) for v in depth_m.shape)
            depth_m = np.rot90(depth_m, k=-1)
            intrinsics = rotate_intrinsics_clockwise_90(intrinsics, original_depth_shape=original_shape)
            extrinsics = rotate_camera_extrinsics_clockwise_90(extrinsics)

        scene_config, _head_config = load_scene_config(self.scene_config_path or self._default_scene_config_path())
        grasp_config = load_grasp_config(self.grasp_config_path or self._default_grasp_config_path())
        if os.getenv("ASK2ACT_REAL_ALLOW_APPROXIMATE_TOPDOWN_FALLBACK", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            grasp_config = replace(grasp_config, allow_approximate_topdown_fallback=True)
        point_cloud = PointCloudGenerator().generate(
            depth_image=depth_m,
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
            table_top_z_m=scene_config.table_top_z_m,
            table_margin_m=scene_config.table_clearance_margin_m,
            z_min_m=grasp_config.z_min_m,
            z_max_m=min(grasp_config.z_max_m, scene_config.table_top_z_m + 0.20),
            target_bbox_2d=bbox_tuple,
        )
        min_points = int(os.getenv("ASK2ACT_REAL_MIN_POINT_CLOUD_COUNT", "30"))
        if point_cloud.filtered_point_count < min_points:
            raise RuntimeError(
                f"Target point cloud has too few points: {point_cloud.filtered_point_count} < {min_points}. "
                "Check bbox/depth alignment, head pose, and camera extrinsics."
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
        current_state = self._default_robot_state()
        motion_plan = MotionPlanner(scene_config, grasp_config).plan_to_grasp(candidate, current_state)
        trajectory = [
            {
                "name": waypoint.name,
                "joint_targets": {str(key): float(value) for key, value in waypoint.joint_targets.items()},
                "settle_s": float(waypoint.settle_s),
            }
            for waypoint in motion_plan.waypoints
        ]

        run_dir = self._default_run_dir()
        np.save(run_dir / "target_depth_m.npy", depth_m)
        save_point_cloud(point_cloud.world_points_xyz, run_dir / "target_cloud_world.ply")
        save_point_cloud(point_cloud.camera_points_xyz, run_dir / "target_cloud_camera.ply")

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
                "target_bbox_2d": list(bbox_tuple),
                "rotated_observation_clockwise_90": bool(rotate_clockwise_90),
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
            "target_bbox_2d": list(bbox_tuple),
            "target_bbox_xyxy": list(resolved_target.bbox_xyxy),
            "trajectory": trajectory,
            "run_dir": str(run_dir),
            "resolved_target": resolved_target.model_dump(),
            "geometric_grasp": self._to_jsonable(geometric_grasp),
            "motion_plan_metadata": self._to_jsonable(motion_plan.metadata),
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
