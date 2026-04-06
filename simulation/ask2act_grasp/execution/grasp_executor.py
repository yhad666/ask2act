from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from stretch_mujoco.enums.actuators import Actuators

from ask2act_grasp.grasp.grasp_generator import ContactGraspNetWrapper
from ask2act_grasp.grasp.grasp_selector import GraspSelector
from ask2act_grasp.perception.head_alignment import HeadAligner
from ask2act_grasp.perception.point_cloud_gen import PointCloudGenerator
from ask2act_grasp.planning.motion_planner import MotionPlanner
from ask2act_grasp.types import PipelineContext, PipelineResult
from ask2act_grasp.utils.visualization import save_point_cloud


class GraspExecutor:
    def __init__(self, context: PipelineContext) -> None:
        self.context = context
        self.head_aligner = HeadAligner(context.head_config)
        self.point_cloud_gen = PointCloudGenerator()
        self.grasp_generator = ContactGraspNetWrapper(context.grasp_config)
        self.grasp_selector = GraspSelector(context.scene_config, context.grasp_config)
        self.motion_planner = MotionPlanner(context.scene_config, context.grasp_config)

    def run(self, sim, target_bbox_2d: tuple[int, int, int, int] | None = None) -> PipelineResult:
        started_at = time.time()
        artifacts_dir = self.context.run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        try:
            head_observation = self.head_aligner.align_and_capture(sim)
            point_cloud = self.point_cloud_gen.generate(
                depth_image=head_observation.depth_image,
                camera_intrinsics=head_observation.camera_intrinsics,
                camera_extrinsics=head_observation.camera_extrinsics,
                table_top_z_m=self.context.scene_config.table_top_z_m,
                table_margin_m=self.context.scene_config.table_clearance_margin_m,
                z_min_m=self.context.grasp_config.z_min_m,
                z_max_m=self.context.grasp_config.z_max_m,
                target_bbox_2d=target_bbox_2d,
            )
            save_point_cloud(point_cloud.world_points_xyz, artifacts_dir / "scene_points.pcd")
            grasp_candidates = self.grasp_generator.generate(point_cloud.world_points_xyz)
            robot_state = self._read_robot_state(sim)
            selected = self.grasp_selector.select_best(grasp_candidates, robot_state=robot_state)
            if selected is None:
                result = PipelineResult(
                    success=False,
                    scene_xml_path=str(self.context.scene_xml_path),
                    point_cloud_count=point_cloud.filtered_point_count,
                    selected_grasp_score=None,
                    planner_backend=self.context.grasp_config.planner_backend,
                    trajectory=[],
                    intermediate={
                        "head_camera_source": head_observation.camera_source,
                        "head_pan_command_rad": self.context.head_config.head_pan_rad,
                        "head_tilt_command_rad": self.context.head_config.head_tilt_rad,
                        "head_pan_actual_rad": robot_state["head_pan"],
                        "head_tilt_actual_rad": robot_state["head_tilt"],
                        "point_cloud_count": point_cloud.filtered_point_count,
                        "grasp_candidate_count": len(grasp_candidates),
                    },
                    error="No feasible grasp candidate found.",
                )
                self._write_result(result)
                return result

            plan = self.motion_planner.plan_to_grasp(selected, robot_state)
            trajectory_trace = self._execute_plan(sim, plan)
            result = PipelineResult(
                success=True,
                scene_xml_path=str(self.context.scene_xml_path),
                point_cloud_count=point_cloud.filtered_point_count,
                selected_grasp_score=float(selected.score),
                planner_backend=plan.backend,
                trajectory=trajectory_trace,
                intermediate={
                    "head_camera_source": head_observation.camera_source,
                    "grasp_candidate_count": len(grasp_candidates),
                    "selected_grasp_source": selected.source,
                    "selected_grasp_position_m": selected.position_m.tolist(),
                    "elapsed_s": round(time.time() - started_at, 3),
                },
            )
            self._write_result(result)
            return result
        except Exception as exc:
            result = PipelineResult(
                success=False,
                scene_xml_path=str(self.context.scene_xml_path),
                point_cloud_count=0,
                selected_grasp_score=None,
                planner_backend=self.context.grasp_config.planner_backend,
                trajectory=[],
                error=str(exc),
            )
            self._write_result(result)
            return result

    def _execute_plan(self, sim, plan) -> list[dict]:
        trace: list[dict] = []
        actuator_map = {
            "base_rotate": Actuators.base_rotate,
            "lift": Actuators.lift,
            "arm": Actuators.arm,
            "wrist_yaw": Actuators.wrist_yaw,
            "wrist_pitch": Actuators.wrist_pitch,
            "wrist_roll": Actuators.wrist_roll,
            "stretch_gripper": Actuators.gripper,
        }
        for waypoint in plan.waypoints:
            for joint_name, target in waypoint.joint_targets.items():
                actuator = actuator_map[joint_name]
                if actuator == Actuators.base_rotate:
                    current_theta = float(sim.pull_status().base.theta)
                    sim.move_by(actuator, float(target - current_theta))
                    sim.wait_while_is_moving(actuator, timeout=8.0)
                else:
                    sim.move_to(actuator, float(target))
                    sim.wait_until_at_setpoint(actuator, timeout=8.0)
            trace.append({"name": waypoint.name, "joint_targets": waypoint.joint_targets})
        return trace

    def _read_robot_state(self, sim) -> dict[str, float]:
        status = sim.pull_status()
        return {
            "lift": float(status.lift.pos),
            "arm": float(status.arm.pos),
            "wrist_yaw": float(status.wrist_yaw.pos),
            "wrist_pitch": float(status.wrist_pitch.pos),
            "wrist_roll": float(status.wrist_roll.pos),
            "head_pan": float(status.head_pan.pos),
            "head_tilt": float(status.head_tilt.pos),
            "stretch_gripper": float(status.gripper.pos),
            "base_rotate": float(status.base.theta),
        }

    def _write_result(self, result: PipelineResult) -> None:
        (self.context.run_dir / "pipeline_result.json").write_text(
            json.dumps(result.__dict__, indent=2, default=_json_default)
        )


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
