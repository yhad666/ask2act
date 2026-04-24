from __future__ import annotations

import json
import os
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

    def plan_for_target(self, resolved_target: ResolvedTarget, dry_run: bool = False) -> Dict[str, Any]:
        if self.mode == "mock":
            return self._mock_plan(resolved_target, dry_run=dry_run)
        if self.mode == "local_sim":
            return self._local_sim_plan(resolved_target, dry_run=dry_run)
        raise RuntimeError(f"Unsupported ASK2ACT_PIPELINE_MODE: {self.mode}")
