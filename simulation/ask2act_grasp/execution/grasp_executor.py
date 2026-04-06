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
from ask2act_grasp.types import GraspCandidate, PipelineContext, PipelineResult
from ask2act_grasp.utils.tf_utils import pose_from_axes
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
            self._prepare_start_pose(sim)
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
            grasp_candidates = self._inject_single_cup_oracle_candidate(grasp_candidates)
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
                    "planner_metadata": plan.metadata,
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
            waypoint_ok = True
            for joint_name, target in waypoint.joint_targets.items():
                actuator = actuator_map[joint_name]
                tolerance = 0.05
                if waypoint.name == "extend_toward_cup" and joint_name == "arm":
                    tolerance = 0.10
                elif waypoint.name == "grasp" and joint_name == "arm":
                    tolerance = 0.10
                if actuator == Actuators.base_rotate:
                    current_theta = float(sim.pull_status().base.theta)
                    sim.move_by(actuator, float(target - current_theta))
                    settled = sim.wait_while_is_moving(actuator, timeout=60.0)
                    waypoint_ok = waypoint_ok and bool(settled)
                elif waypoint.name == "close_gripper" and joint_name == "stretch_gripper":
                    before_status = sim.pull_status()
                    sim.move_to(actuator, float(target))
                    sim.wait_while_is_moving(actuator, timeout=10.0)
                    after_status = sim.pull_status()
                    before_pos = float(before_status.gripper.pos)
                    after_pos = float(after_status.gripper.pos)
                    # Contact-limited grasp closes should count as success once the
                    # gripper closes to about the object's diameter or clearly moves inward.
                    reached = after_pos <= 0.1 or after_pos < before_pos - 0.02
                    waypoint_ok = waypoint_ok and bool(reached)
                else:
                    before_status = sim.pull_status()
                    sim.move_to(actuator, float(target))
                    reached = sim.wait_until_at_setpoint(actuator, timeout=60.0, position_tolerance=tolerance)
                    waypoint_ok = waypoint_ok and bool(reached)
            trace.append({"name": waypoint.name, "joint_targets": waypoint.joint_targets, "ok": waypoint_ok})
            if not waypoint_ok:
                raise RuntimeError(f"Failed to reach waypoint {waypoint.name}")
        return trace

    def _prepare_start_pose(self, sim) -> None:
        # Tuck first, then retract the arm. Do not raise first: the wrist/arm must
        # get out of the table edge region before any vertical motion.
        startup_targets = [
            (Actuators.gripper, 0.045, 20.0, 0.05),
            (Actuators.wrist_roll, 0.0, 20.0, 0.05),
            (Actuators.wrist_yaw, self.context.grasp_config.oracle_tucked_wrist_yaw_rad, 20.0, 0.10),
            (Actuators.wrist_pitch, 0.00, 20.0, 0.06),
            (Actuators.arm, 0.0, 30.0, 0.06),
        ]
        for actuator, target, timeout_s, tolerance in startup_targets:
            sim.move_to(actuator, float(target))
            if not sim.wait_until_at_setpoint(actuator, timeout=timeout_s, position_tolerance=tolerance):
                raise RuntimeError(f"startup pose failed at actuator {actuator.name}")

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

    def _inject_single_cup_oracle_candidate(self, candidates: list[GraspCandidate]) -> list[GraspCandidate]:
        scene = self.context.scene_config
        cup_x, cup_y, _ = scene.cup_position_m
        grasp_z = scene.table_top_z_m + scene.cup_height_m * self.context.grasp_config.oracle_grasp_height_ratio
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        y_axis = np.array([0.0, -1.0, 0.0], dtype=float)
        z_axis = np.array([0.0, 0.0, -1.0], dtype=float)
        oracle_pose = pose_from_axes(
            np.array([cup_x, cup_y, grasp_z], dtype=float),
            x_axis,
            y_axis,
            z_axis,
        )
        oracle_candidate = GraspCandidate(
            pose_4x4=oracle_pose,
            score=0.92,
            width_m=min(scene.cup_radius_m * 2.0 * 0.9, self.context.grasp_config.max_gripper_width_m),
            source="scene_oracle_single_cup",
            metadata={
                "grasp_style": "side_midline",
                "preferred_base_rotate_rad": 0.0,
                "preferred_wrist_yaw_rad": 0.0,
                "preferred_wrist_pitch_rad": self.context.grasp_config.oracle_side_grasp_wrist_pitch_rad,
            },
        )
        merged = [oracle_candidate, *candidates]
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged

    def _write_result(self, result: PipelineResult) -> None:
        (self.context.run_dir / "pipeline_result.json").write_text(
            json.dumps(result.__dict__, indent=2, default=_json_default)
        )


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
