from __future__ import annotations

import math
import json
import os
import time
from pathlib import Path

import numpy as np
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_sensors import StretchSensors

from ask2act_grasp.grasp.geometric_grasp import compute_geometric_grasp, save_geometric_grasp_debug
from ask2act_grasp.grasp.grasp_generator import ContactGraspNetGenerator
from ask2act_grasp.grasp.grasp_selector import GraspSelector, classify_cup_approach
from ask2act_grasp.stretch3_specs import STRETCH3_GRIPPER, STRETCH3_JOINT_LIMITS
from ask2act_grasp.perception.head_alignment import HeadAligner
from ask2act_grasp.perception.point_cloud_gen import PointCloudGenerator
from ask2act_grasp.perception.tabletop_geometry import estimate_tabletop_object_geometry
from ask2act_grasp.planning.motion_planner import MotionPlanner
from ask2act_grasp.types import GraspCandidate, PipelineContext, PipelineResult
from ask2act_grasp.utils.tf_utils import pose_from_axes
from ask2act_grasp.utils.visualization import save_grasp_debug_scene, save_point_cloud


class GraspExecutor:
    """Run one end-to-end grasp attempt inside the simulator."""

    TOPDOWN_CONTACT_ALIGNMENT_MAX_ITERS = 4
    TOPDOWN_CONTACT_ALIGNMENT_GAIN = 0.6
    TOPDOWN_CONTACT_ALIGNMENT_MIN_IMPROVEMENT_M = 0.002
    TOPDOWN_CONTACT_ALIGNMENT_XY_TOL_M = 0.008
    TOPDOWN_CONTACT_ALIGNMENT_Z_TOL_M = 0.008
    TOPDOWN_CONTACT_ALIGNMENT_MAX_XY_STEP_M = 0.03
    TOPDOWN_CONTACT_ALIGNMENT_MAX_Z_STEP_M = 0.04
    TOPDOWN_WRIST_GEOMETRY_XY_TOL_M = 0.02
    TOPDOWN_WRIST_GEOMETRY_Z_TOL_M = 0.02
    BASE_PREPOSITION_CONTROL_DT_S = 0.1
    BASE_PREPOSITION_MAX_LINEAR_V_MPS = 0.25
    BASE_PREPOSITION_MAX_OMEGA_RADPS = 1.0
    BASE_PREPOSITION_LINEAR_GAIN = 0.8
    BASE_PREPOSITION_OMEGA_GAIN = 1.6
    BASE_PREPOSITION_CLOCKWISE_TURN_RAD = -math.pi / 2.0

    def __init__(self, context: PipelineContext) -> None:
        self.context = context
        self.head_aligner = HeadAligner(context.head_config, context.scene_xml_path)
        self.point_cloud_gen = PointCloudGenerator()
        self.grasp_generator = ContactGraspNetGenerator(context.grasp_config)
        self.grasp_selector = GraspSelector(context.scene_config, context.grasp_config)
        self.motion_planner = MotionPlanner(context.scene_config, context.grasp_config)

    def run(self, sim, target_bbox_2d: tuple[int, int, int, int] | None = None) -> PipelineResult:
        """Run one grasp attempt and persist debug artifacts."""
        started_at = time.time()
        perf_started_at = time.perf_counter()
        artifacts_dir = self.context.run_dir / "artifacts"
        cgn_debug_dir = Path("/tmp/ask2act_cgn_test")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        cgn_debug_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        stage_times_s: dict[str, float] = {}

        try:
            stage_started = time.perf_counter()
            self._prepare_start_pose(sim)
            stage_times_s["prepare_start_pose"] = round(time.perf_counter() - stage_started, 4)

            stage_started = time.perf_counter()
            base_preposition_diagnostic = self._maybe_preposition_base_for_target(sim)
            if base_preposition_diagnostic is not None:
                stage_times_s["base_preposition"] = round(time.perf_counter() - stage_started, 4)
            else:
                stage_times_s["base_preposition"] = 0.0

            stage_started = time.perf_counter()
            head_observation = self.head_aligner.align_and_capture(
                sim,
                target_center_xy=self._get_nominal_target_center_xy(),
            )
            stage_times_s["head_alignment"] = round(time.perf_counter() - stage_started, 4)

            stage_started = time.perf_counter()
            self._save_head_artifacts(head_observation.rgb_image, head_observation.depth_image, artifacts_dir)
            stage_times_s["save_head_artifacts"] = round(time.perf_counter() - stage_started, 4)

            depth_for_point_cloud = head_observation.depth_image
            fixes_applied: list[str] = []
            if self.context.scene_config.grasp_method == "cgn" and self.context.grasp_config.add_depth_noise:
                stage_started = time.perf_counter()
                depth_for_point_cloud = self.point_cloud_gen.add_sensor_noise(
                    head_observation.depth_image,
                    noise_sigma=self.context.grasp_config.depth_noise_sigma_m,
                    dropout_ratio=self.context.grasp_config.depth_dropout_ratio,
                )
                fixes_applied.append(f"noise_{self.context.grasp_config.depth_noise_sigma_m}")
                np.save(artifacts_dir / "head_depth_noisy.npy", depth_for_point_cloud)
                np.save(cgn_debug_dir / "head_depth_noisy.npy", depth_for_point_cloud)
                stage_times_s["depth_noise"] = round(time.perf_counter() - stage_started, 4)

            stage_started = time.perf_counter()
            point_cloud = self.point_cloud_gen.generate(
                depth_image=depth_for_point_cloud,
                camera_intrinsics=head_observation.camera_intrinsics,
                camera_extrinsics=head_observation.camera_extrinsics,
                table_top_z_m=self.context.scene_config.table_top_z_m,
                table_margin_m=self.context.scene_config.table_clearance_margin_m,
                z_min_m=self.context.grasp_config.z_min_m,
                z_max_m=min(
                    self.context.grasp_config.z_max_m,
                    self.context.scene_config.table_top_z_m + 0.20,
                ),
                target_bbox_2d=target_bbox_2d,
            )
            stage_times_s["point_cloud_generation"] = round(time.perf_counter() - stage_started, 4)

            stage_started = time.perf_counter()
            save_point_cloud(point_cloud.world_points_xyz, artifacts_dir / "scene_cloud.ply")
            save_point_cloud(point_cloud.camera_points_xyz, artifacts_dir / "scene_cloud_camera.ply")
            save_point_cloud(point_cloud.world_points_xyz, Path("/tmp/ask2act_scene.ply"))
            save_point_cloud(point_cloud.camera_points_xyz, Path("/tmp/ask2act_scene_camera.ply"))
            save_point_cloud(point_cloud.world_points_xyz, artifacts_dir / "full_cloud.ply")
            save_point_cloud(point_cloud.world_points_xyz, cgn_debug_dir / "full_cloud.ply")
            save_point_cloud(point_cloud.camera_points_xyz, artifacts_dir / "full_cloud_camera.ply")
            save_point_cloud(point_cloud.camera_points_xyz, cgn_debug_dir / "full_cloud_camera.ply")
            stage_times_s["save_point_cloud_artifacts"] = round(time.perf_counter() - stage_started, 4)

            stage_started = time.perf_counter()
            robot_state = self._read_robot_state(sim)
            stage_times_s["read_robot_state"] = round(time.perf_counter() - stage_started, 4)
            grasp_candidates: list[GraspCandidate] = []
            selected: GraspCandidate | None = None
            diagnostics: dict[str, object] = {
                "full_cloud_points": int(point_cloud.world_points_xyz.shape[0]),
                "cropped_cloud_points": int(point_cloud.world_points_xyz.shape[0]),
                "cgn_input_points": int(point_cloud.camera_points_xyz.shape[0]),
                "head_rgb_shape": list(np.asarray(head_observation.rgb_image).shape),
                "head_depth_shape": list(np.asarray(head_observation.depth_image).shape),
                "total_cgn_candidates": 0,
                "after_common_filter": 0,
                "after_cup_strategy": 0,
                "best_score": None,
                "best_position": None,
                "best_approach": None,
                "fixes_applied": fixes_applied,
            }

            if self.context.scene_config.grasp_method == "cgn":
                stage_started = time.perf_counter()
                crop_height_min = self.context.scene_config.table_top_z_m + 0.01
                crop_height_max = self.context.scene_config.table_top_z_m + 0.12
                cropped_world, crop_mask, crop_center_xy = self.point_cloud_gen.crop_to_object_region(
                    point_cloud.world_points_xyz,
                    crop_radius=self.context.grasp_config.cgn_crop_radius_m,
                    table_z=self.context.scene_config.table_top_z_m,
                    height_min=crop_height_min,
                    height_max=crop_height_max,
                    return_mask=True,
                )
                xy_dist = np.linalg.norm(
                    point_cloud.world_points_xyz[:, :2] - np.asarray(crop_center_xy, dtype=float)[None, :],
                    axis=1,
                )
                xy_crop_mask = xy_dist < self.context.grasp_config.cgn_crop_radius_m
                xy_cropped_world = point_cloud.world_points_xyz[xy_crop_mask]
                cup_point_ratio = float(
                    np.mean(
                        (xy_cropped_world[:, 2] > crop_height_min)
                        & (xy_cropped_world[:, 2] < crop_height_max)
                    )
                ) if len(xy_cropped_world) else 0.0
                print(f"Cup-point ratio in crop: {cup_point_ratio:.1%}")
                cropped_camera = point_cloud.camera_points_xyz[crop_mask]
                save_point_cloud(cropped_world, artifacts_dir / "cropped_cloud.ply")
                save_point_cloud(cropped_world, cgn_debug_dir / "cropped_cloud.ply")
                save_point_cloud(cropped_camera, artifacts_dir / "cropped_cloud_camera.ply")
                save_point_cloud(cropped_camera, cgn_debug_dir / "cropped_cloud_camera.ply")
                fixes_applied.append(f"crop_{self.context.grasp_config.cgn_crop_radius_m}m")

                cgn_input_camera, sample_indices = self.point_cloud_gen.subsample(
                    cropped_camera,
                    target_n=self.context.grasp_config.cgn_subsample_n,
                    return_indices=True,
                )
                cgn_input_world = cropped_world[sample_indices] if len(sample_indices) else cropped_world[:0]
                save_point_cloud(cgn_input_world, artifacts_dir / "cgn_input_cloud.ply")
                save_point_cloud(cgn_input_world, cgn_debug_dir / "cgn_input_cloud.ply")
                save_point_cloud(cgn_input_camera, artifacts_dir / "cgn_input_cloud_camera.ply")
                save_point_cloud(cgn_input_camera, cgn_debug_dir / "cgn_input_cloud_camera.ply")
                fixes_applied.append(f"subsample_{self.context.grasp_config.cgn_subsample_n}")
                stage_times_s["cgn_preprocess"] = round(time.perf_counter() - stage_started, 4)

                stage_started = time.perf_counter()
                grasp_candidates = self.grasp_generator.generate(
                    cgn_input_camera,
                    frame_transform=head_observation.camera_extrinsics,
                )
                stage_times_s["cgn_inference"] = round(time.perf_counter() - stage_started, 4)

                stage_started = time.perf_counter()
                cgn_candidates = [candidate for candidate in grasp_candidates if candidate.source == "contact_graspnet"]
                common_filtered_no_distance = self.grasp_selector._common_filter(
                    cgn_candidates,
                    table_z=self.context.scene_config.table_top_z_m,
                    robot_state=robot_state,
                )
                common_filtered = self.grasp_selector._common_filter(
                    cgn_candidates,
                    table_z=self.context.scene_config.table_top_z_m,
                    robot_state=robot_state,
                    object_center=np.asarray(crop_center_xy, dtype=float),
                    object_radius=self.context.scene_config.cup_radius_m,
                )
                cup_strategy = self.grasp_selector.strategies.get("cup")
                strategy_filtered = cup_strategy.filter_candidates(
                    common_filtered,
                    table_z=self.context.scene_config.table_top_z_m,
                    object_center=np.asarray(crop_center_xy, dtype=float),
                    object_radius=self.context.scene_config.cup_radius_m,
                ) if cup_strategy is not None else []
                approach_type_counts = {"top_down": 0, "side": 0, "angled": 0, "rejected_below": 0}
                for candidate in cgn_candidates:
                    approach = np.asarray(candidate.approach_axis_world, dtype=float)
                    if float(approach[2]) > 0.5:
                        approach_type_counts["rejected_below"] += 1
                    else:
                        approach_type_counts[classify_cup_approach(approach)] += 1
                selected = self.grasp_selector.select(
                    cgn_candidates,
                    table_z=self.context.scene_config.table_top_z_m,
                    robot_state=robot_state,
                    object_class="cup",
                    object_center=np.asarray(crop_center_xy, dtype=float),
                    object_radius=self.context.scene_config.cup_radius_m,
                )
                max_allowed_distance = float(self.context.scene_config.cup_radius_m * 1.5)
                diagnostics.update(
                    {
                        "crop_center_xy": np.asarray(crop_center_xy, dtype=float).tolist(),
                        "object_center_xy": np.asarray(crop_center_xy, dtype=float).tolist(),
                        "object_radius": float(self.context.scene_config.cup_radius_m),
                        "max_allowed_distance": max_allowed_distance,
                        "crop_radius": float(self.context.grasp_config.cgn_crop_radius_m),
                        "crop_height_min": float(crop_height_min),
                        "crop_height_max": float(crop_height_max),
                        "cup_point_ratio": cup_point_ratio,
                        "full_cloud_points": int(point_cloud.world_points_xyz.shape[0]),
                        "cropped_cloud_points": int(cropped_world.shape[0]),
                        "cgn_input_points": int(cgn_input_camera.shape[0]),
                        "total_cgn_candidates": int(len(cgn_candidates)),
                        "after_reachability_filter": int(len(common_filtered_no_distance)),
                        "after_distance_filter": int(len(common_filtered)),
                        "after_common_filter": int(len(common_filtered)),
                        "after_cup_strategy": int(len(strategy_filtered)),
                        "approach_type_counts": approach_type_counts,
                        "finger_length_measured": float(STRETCH3_GRIPPER["finger_length_m"]),
                        "stage_times_s": stage_times_s,
                    }
                )
                best_for_logging = selected if selected is not None else (strategy_filtered[0] if strategy_filtered else None)
                if best_for_logging is not None:
                    diagnostics["best_score"] = float(best_for_logging.score)
                    diagnostics["best_position"] = np.asarray(best_for_logging.position_m, dtype=float).tolist()
                    diagnostics["best_approach"] = np.asarray(best_for_logging.approach_axis_world, dtype=float).tolist()
                    diagnostics["selected_approach_type"] = getattr(best_for_logging, "approach_type", "unknown")
                    diagnostics["best_distance_to_center"] = float(
                        np.linalg.norm(
                            np.asarray(best_for_logging.position_m[:2], dtype=float)
                            - np.asarray(crop_center_xy, dtype=float)
                        )
                    )
                (artifacts_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
                (cgn_debug_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
                stage_times_s["cgn_select_and_diagnostics"] = round(time.perf_counter() - stage_started, 4)
                if selected is None:
                    warnings.append("CGN produced no valid Stretch-compatible grasp; falling back to oracle.")
                    selected = self._make_single_cup_oracle_candidate()
            elif self.context.scene_config.grasp_method == "geometric":
                stage_started = time.perf_counter()
                geometric_grasp, selected, geometric_diagnostics = self._compute_geometric_candidate_from_point_cloud(
                    point_cloud,
                    head_observation,
                    target_bbox_2d,
                    artifacts_dir,
                    cgn_debug_dir,
                )
                stage_times_s["geometric_perception"] = round(time.perf_counter() - stage_started, 4)
                diagnostics.update(geometric_diagnostics)
                diagnostics["reobserved_after_base_refine"] = False

                stage_started = time.perf_counter()
                lateral_base_refine = self._maybe_refine_base_for_geometric_grasp(
                    sim,
                    geometric_grasp,
                    phase="lateral",
                )
                stage_times_s["post_geometric_base_preposition_lateral"] = round(time.perf_counter() - stage_started, 4)
                diagnostics["post_geometric_base_preposition_lateral"] = lateral_base_refine

                if self._should_reobserve_after_base_motion(lateral_base_refine):
                    (
                        head_observation,
                        point_cloud,
                        geometric_grasp,
                        selected,
                        geometric_diagnostics,
                        reobserve_meta,
                    ) = self._reobserve_geometric_after_base_motion(
                        sim=sim,
                        head_observation=head_observation,
                        point_cloud=point_cloud,
                        geometric_grasp=geometric_grasp,
                        selected=selected,
                        geometric_diagnostics=geometric_diagnostics,
                        target_bbox_2d=target_bbox_2d,
                        artifacts_dir=artifacts_dir,
                        cgn_debug_dir=cgn_debug_dir,
                        stage_times_s=stage_times_s,
                        stage_prefix="after_lateral_base_refine",
                    )
                    diagnostics["reobserve_after_lateral_base_refine"] = reobserve_meta
                    diagnostics["reobserved_after_base_refine"] = bool(
                        diagnostics["reobserved_after_base_refine"] or reobserve_meta.get("accepted", False)
                    )
                    diagnostics.update(geometric_diagnostics)

                stage_started = time.perf_counter()
                longitudinal_base_refine = self._maybe_refine_base_for_geometric_grasp(
                    sim,
                    geometric_grasp,
                    phase="longitudinal",
                )
                stage_times_s["post_geometric_base_preposition_longitudinal"] = round(
                    time.perf_counter() - stage_started,
                    4,
                )
                diagnostics["post_geometric_base_preposition_longitudinal"] = longitudinal_base_refine

                if self._should_reobserve_after_base_motion(longitudinal_base_refine):
                    (
                        head_observation,
                        point_cloud,
                        geometric_grasp,
                        selected,
                        geometric_diagnostics,
                        reobserve_meta,
                    ) = self._reobserve_geometric_after_base_motion(
                        sim=sim,
                        head_observation=head_observation,
                        point_cloud=point_cloud,
                        geometric_grasp=geometric_grasp,
                        selected=selected,
                        geometric_diagnostics=geometric_diagnostics,
                        target_bbox_2d=target_bbox_2d,
                        artifacts_dir=artifacts_dir,
                        cgn_debug_dir=cgn_debug_dir,
                        stage_times_s=stage_times_s,
                        stage_prefix="after_longitudinal_base_refine",
                    )
                    diagnostics["reobserve_after_longitudinal_base_refine"] = reobserve_meta
                    diagnostics["reobserved_after_base_refine"] = bool(
                        diagnostics["reobserved_after_base_refine"] or reobserve_meta.get("accepted", False)
                    )
                    diagnostics.update(geometric_diagnostics)

                diagnostics["post_geometric_base_preposition"] = {
                    "enabled": bool(self.context.grasp_config.enable_base_preposition),
                    "performed": bool(
                        lateral_base_refine.get("performed", False)
                        or longitudinal_base_refine.get("performed", False)
                    ),
                    "lateral": lateral_base_refine,
                    "longitudinal": longitudinal_base_refine,
                    "base_pose_before": lateral_base_refine.get("base_pose_before"),
                    "base_pose_after": longitudinal_base_refine.get("base_pose_after", lateral_base_refine.get("base_pose_after")),
                    "exact_target_reachable_after_refine": longitudinal_base_refine.get(
                        "exact_target_reachable_after_refine",
                        lateral_base_refine.get("exact_target_reachable_after_refine"),
                    ),
                }

                (artifacts_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
                (cgn_debug_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

                if bool(geometric_grasp.get("width_near_limit", False)):
                    warnings.append(
                        "Geometric grasp width is close to the gripper aperture limit; wide objects may remain unstable."
                    )
                if bool(geometric_grasp.get("open_width_exceeds_max", False)):
                    warnings.append(
                        "Geometric grasp requested opening exceeds the gripper max aperture; execution will clip the command."
                    )

                grasp_candidates = [selected]
            else:
                stage_started = time.perf_counter()
                warnings.append("Using oracle grasp path.")
                selected = self._make_single_cup_oracle_candidate()
                grasp_candidates = [selected]
                stage_times_s["oracle_candidate"] = round(time.perf_counter() - stage_started, 4)

            stage_started = time.perf_counter()
            self._save_grasp_artifacts(artifacts_dir, point_cloud.world_points_xyz, grasp_candidates, selected)
            stage_times_s["save_grasp_artifacts"] = round(time.perf_counter() - stage_started, 4)

            if selected is None:
                raise RuntimeError("No feasible grasp candidate found.")

            stage_started = time.perf_counter()
            robot_state = self._read_robot_state(sim)
            stage_times_s["read_robot_state_for_planning"] = round(time.perf_counter() - stage_started, 4)
            stage_started = time.perf_counter()
            plan = self.motion_planner.plan_to_grasp(selected, robot_state)
            stage_times_s["motion_planning"] = round(time.perf_counter() - stage_started, 4)
            if self.context.scene_config.grasp_method in {"cgn", "geometric"}:
                numeric_targets = dict(plan.metadata.get("numeric_targets", {}))
                diagnostics.update(
                    {
                        "selected_approach_type": str(numeric_targets.get("approach_type", getattr(selected, "approach_type", "unknown"))),
                        "selected_score": float(selected.score),
                        "contact_point_xyz": numeric_targets.get("contact_point"),
                        "cgn_grasp_frame_xyz": numeric_targets.get("cgn_grasp_frame"),
                        "wrist_target_xyz": [numeric_targets.get("grasp_x"), numeric_targets.get("grasp_y"), numeric_targets.get("grasp_z")],
                        "finger_offset_applied": numeric_targets.get("wrist_to_cgn_frame_offset_m"),
                        "wrist_pitch_deg": float(np.degrees(float(numeric_targets.get("wrist_pitch", 0.0)))),
                        "stage_times_s": stage_times_s,
                    }
                )
                (artifacts_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
                (cgn_debug_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
            stage_started = time.perf_counter()
            trajectory_trace = self._execute_plan(sim, plan)
            stage_times_s["execute_plan"] = round(time.perf_counter() - stage_started, 4)
            stage_started = time.perf_counter()
            grasp_verified, verification_details = self._verify_grasp(sim)
            stage_times_s["verify_grasp"] = round(time.perf_counter() - stage_started, 4)
            stage_times_s["executor_total"] = round(time.perf_counter() - perf_started_at, 4)
            diagnostics.update(
                {
                    "grasp_verified": bool(grasp_verified),
                    "cup_z_after_lift": verification_details.get("cup_z_after_lift"),
                    "verification_method": verification_details.get("verification_method"),
                    "verification_details": verification_details,
                    "stage_times_s": stage_times_s,
                }
            )
            if self.context.scene_config.grasp_method in {"cgn", "geometric"}:
                (artifacts_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
                (cgn_debug_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
            result = PipelineResult(
                success=bool(grasp_verified),
                scene_xml_path=str(self.context.scene_xml_path),
                point_cloud_count=point_cloud.filtered_point_count,
                selected_grasp_score=float(selected.score),
                planner_backend=plan.backend,
                trajectory=trajectory_trace,
                intermediate={
                    "grasp_method": self.context.scene_config.grasp_method,
                    "head_camera_source": head_observation.camera_source,
                    "point_cloud_count": point_cloud.filtered_point_count,
                    "point_cloud_bbox_min": point_cloud.world_points_xyz.min(axis=0).tolist()
                    if point_cloud.filtered_point_count
                    else None,
                    "point_cloud_bbox_max": point_cloud.world_points_xyz.max(axis=0).tolist()
                    if point_cloud.filtered_point_count
                    else None,
                    "estimated_cup_center_m": np.median(point_cloud.world_points_xyz, axis=0).tolist()
                    if point_cloud.filtered_point_count
                    else None,
                    "cgn_input_frame": (
                        "camera"
                        if self.context.scene_config.grasp_method == "cgn"
                        else ("world" if self.context.scene_config.grasp_method == "geometric" else "oracle")
                    ),
                    "grasp_candidate_count": len(grasp_candidates),
                    "selected_grasp_source": selected.source,
                    "selected_grasp_position_m": selected.position_m.tolist(),
                    "selected_grasp_preferred_approach": selected.preferred_approach,
                    "selected_grasp_approach_type": selected.approach_type,
                    "selected_grasp_needs_wrist_refinement": bool(selected.needs_wrist_refinement),
                    "base_preposition": base_preposition_diagnostic,
                    "post_geometric_base_preposition": diagnostics.get("post_geometric_base_preposition")
                    if self.context.scene_config.grasp_method == "geometric"
                    else None,
                    "cgn_diagnostics": diagnostics if self.context.scene_config.grasp_method == "cgn" else None,
                    "grasp_diagnostics": diagnostics if self.context.scene_config.grasp_method in {"cgn", "geometric"} else None,
                    "grasp_verified": bool(grasp_verified),
                    "cup_z_after_lift": verification_details.get("cup_z_after_lift"),
                    "verification_method": verification_details.get("verification_method"),
                    "verification_details": verification_details,
                    "stage_times_s": stage_times_s,
                    "planner_metadata": plan.metadata,
                    "elapsed_s": round(time.time() - started_at, 3),
                    "warnings": warnings,
                },
                error=None if grasp_verified else "Grasp verification failed: cup not lifted",
            )
            self._write_result(result)
            return result
        except Exception as exc:
            stage_times_s["executor_total"] = round(time.perf_counter() - perf_started_at, 4)
            result = PipelineResult(
                success=False,
                scene_xml_path=str(self.context.scene_xml_path),
                point_cloud_count=0,
                selected_grasp_score=None,
                planner_backend=self.context.grasp_config.planner_backend,
                trajectory=[],
                intermediate={
                    "grasp_method": self.context.scene_config.grasp_method,
                    "stage_times_s": stage_times_s,
                    "base_preposition": base_preposition_diagnostic,
                    "warnings": warnings,
                },
                error=str(exc),
            )
            self._write_result(result)
            return result

    def _execute_plan(self, sim, plan) -> list[dict]:
        """Execute the fixed waypoint sequence in the simulator."""
        trace: list[dict] = []
        numeric_targets = dict(plan.metadata.get("numeric_targets", {}))
        pause_before_close_s = float(os.environ.get("ASK2ACT_PRE_CLOSE_PAUSE_S", "0.0"))
        stop_before_close = os.environ.get("ASK2ACT_STOP_BEFORE_CLOSE", "0").strip().lower() in {"1", "true", "yes"}
        timeout_scale = max(1.0, float(os.environ.get("ASK2ACT_WAYPOINT_TIMEOUT_SCALE", "1.0")))
        actuator_map = {
            "base_rotate": Actuators.base_rotate,
            "lift": Actuators.lift,
            "arm": Actuators.arm,
            "wrist_yaw": Actuators.wrist_yaw,
            "wrist_pitch": Actuators.wrist_pitch,
            "wrist_roll": Actuators.wrist_roll,
            "stretch_gripper": Actuators.gripper,
        }
        is_topdown_plan = str(numeric_targets.get("approach_type", "")) == "top_down"
        for waypoint in plan.waypoints:
            waypoint_started = time.perf_counter()
            waypoint_ok = True
            topdown_geometry_ready: dict[str, object] | None = None
            for joint_name, target in waypoint.joint_targets.items():
                actuator = actuator_map[joint_name]
                tolerance = 0.05
                timeout_s = 60.0 * timeout_scale
                if joint_name == "lift":
                    tolerance = 0.08
                if waypoint.name in {"move_to_pregrasp", "extend_to_grasp", "angled_approach", "pre_open_near_object", "extend_toward_cup"} and joint_name == "arm":
                    tolerance = 0.10
                elif waypoint.name in {"orient_wrist", "rotate_for_side_grasp"} and joint_name == "wrist_yaw":
                    if waypoint.name == "orient_wrist" and is_topdown_plan:
                        tolerance = 0.08
                        timeout_s = 90.0 * timeout_scale
                    else:
                        tolerance = 0.25 if waypoint.name == "orient_wrist" else 0.22
                        timeout_s = (55.0 if waypoint.name == "orient_wrist" else 35.0) * timeout_scale
                elif waypoint.name in {"orient_wrist", "rotate_for_side_grasp"} and joint_name == "wrist_pitch":
                    if waypoint.name == "orient_wrist" and is_topdown_plan:
                        tolerance = 0.06
                        timeout_s = 90.0 * timeout_scale
                    else:
                        tolerance = 0.20 if waypoint.name == "orient_wrist" else 0.12
                        timeout_s = (55.0 if waypoint.name == "orient_wrist" else 35.0) * timeout_scale
                elif waypoint.name == "orient_wrist" and joint_name == "wrist_roll" and is_topdown_plan:
                    tolerance = 0.08
                    timeout_s = 90.0 * timeout_scale
                if actuator == Actuators.base_rotate:
                    current_theta = float(sim.get_base_pose()[2])
                    sim.move_by(actuator, float(target - current_theta))
                    settled = self._wait_for_base_rotation_target(
                        sim,
                        target_theta=float(target),
                        timeout_s=timeout_s,
                    )
                    waypoint_ok = waypoint_ok and bool(settled)
                elif joint_name == "stretch_gripper" and float(target) <= -0.35:
                    if waypoint.name == "close_gripper":
                        alignment_history = self._run_topdown_contact_alignment(sim, numeric_targets)
                        pre_close_diagnostic = self._diagnose_gripper_vs_cup(sim)
                        self._augment_pre_close_diagnostic(pre_close_diagnostic, numeric_targets)
                        if alignment_history:
                            pre_close_diagnostic["online_alignment_history"] = alignment_history
                        diagnostic_path = self.context.run_dir / "artifacts" / "pre_close_diagnostic.json"
                        diagnostic_path.write_text(json.dumps(pre_close_diagnostic, indent=2, default=_json_default))
                        if pause_before_close_s > 0.0:
                            print(f"Pausing {pause_before_close_s:.1f}s before close for inspection.", flush=True)
                            time.sleep(pause_before_close_s)
                        if stop_before_close:
                            raise RuntimeError("Stopped before close for calibration as requested.")
                    before_status = sim.pull_status()
                    sim.move_to(actuator, float(target))
                    sim.wait_while_is_moving(actuator, timeout=15.0 * timeout_scale)
                    after_status = sim.pull_status()
                    before_pos = float(before_status.gripper.pos)
                    after_pos = float(after_status.gripper.pos)
                    print(
                        f"{waypoint.name} debug: target={float(target):.3f} before={before_pos:.3f} after={after_pos:.3f}",
                        flush=True,
                    )
                    reached = self._gripper_close_waypoint_reached(
                        waypoint_name=waypoint.name,
                        target=float(target),
                        before_pos=before_pos,
                        after_pos=after_pos,
                    )
                    waypoint_ok = waypoint_ok and bool(reached)
                else:
                    sim.move_to(actuator, float(target))
                    reached = sim.wait_until_at_setpoint(actuator, timeout=timeout_s, position_tolerance=tolerance)
                    if not reached and waypoint.name in {"orient_wrist", "rotate_for_side_grasp"}:
                        actual = self._read_actuator_position(sim, actuator)
                        if waypoint.name == "orient_wrist" and is_topdown_plan:
                            reached = abs(actual - float(target)) <= tolerance
                        else:
                            reached = abs(actual - float(target)) <= tolerance + 0.05
                    waypoint_ok = waypoint_ok and bool(reached)
            if waypoint.settle_s > 0.0:
                time.sleep(float(waypoint.settle_s))
            if waypoint.name == "orient_wrist" and is_topdown_plan:
                geometry_ok, topdown_geometry_ready = self._wait_for_topdown_wrist_geometry(
                    sim,
                    numeric_targets,
                    timeout_s=20.0 * timeout_scale,
                )
                if topdown_geometry_ready is not None:
                    print(
                        "Top-down wrist geometry readiness:",
                        json.dumps(topdown_geometry_ready, default=_json_default),
                        flush=True,
                    )
                waypoint_ok = waypoint_ok and geometry_ok
            trace.append(
                {
                    "name": waypoint.name,
                    "joint_targets": waypoint.joint_targets,
                    "ok": waypoint_ok,
                    "duration_s": round(time.perf_counter() - waypoint_started, 4),
                    "topdown_geometry_ready": topdown_geometry_ready,
                }
            )
            if not waypoint_ok:
                raise RuntimeError(f"Failed to reach waypoint {waypoint.name}")
        return trace

    @staticmethod
    def _get_requested_contact_point(numeric_targets: dict[str, object]) -> np.ndarray | None:
        requested = numeric_targets.get("requested_contact_point", numeric_targets.get("contact_point"))
        if not isinstance(requested, list) or len(requested) != 3:
            return None
        target = np.asarray(requested, dtype=float)
        if target.size != 3:
            return None
        return target

    @staticmethod
    def _rubber_contact_center_from_body(
        gripper_info: dict[str, object],
        body_key: str,
    ) -> np.ndarray | None:
        body_entry = gripper_info.get(body_key)
        if not isinstance(body_entry, dict):
            return None
        pos = np.asarray(body_entry.get("pos", []), dtype=float)
        xmat = np.asarray(body_entry.get("xmat", []), dtype=float)
        if pos.size != 3 or xmat.size != 9:
            return None
        rotation = xmat.reshape(3, 3)
        # MuJoCo rubber fingertip body contains an unnamed contact box geom at local pos [0, 0, 0.01].
        # Use that geom center instead of the rubber body origin, which sits behind the actual contact patch.
        local_geom_center = np.array([0.0, 0.0, 0.01], dtype=float)
        return pos + rotation @ local_geom_center

    @staticmethod
    def _measure_rubber_contact_center(
        gripper_info: dict[str, object],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        left_contact = GraspExecutor._rubber_contact_center_from_body(gripper_info, "body_rubber_tip_left")
        right_contact = GraspExecutor._rubber_contact_center_from_body(gripper_info, "body_rubber_tip_right")
        if left_contact is None or right_contact is None:
            return None
        center = 0.5 * (left_contact + right_contact)
        return left_contact, right_contact, center

    @staticmethod
    def _measure_rigid_grasp_center(gripper_info: dict[str, object]) -> np.ndarray | None:
        body_entry = gripper_info.get("body_link_grasp_center")
        if not isinstance(body_entry, dict):
            return None
        pos = np.asarray(body_entry.get("pos", []), dtype=float)
        if pos.size != 3:
            return None
        return pos

    @staticmethod
    def _measure_wrist_to_grasp_center_vector(
        gripper_info: dict[str, object],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        wrist_entry = gripper_info.get("body_link_wrist_yaw")
        grasp_entry = gripper_info.get("body_link_grasp_center")
        if not isinstance(wrist_entry, dict) or not isinstance(grasp_entry, dict):
            return None
        wrist_pos = np.asarray(wrist_entry.get("pos", []), dtype=float)
        grasp_pos = np.asarray(grasp_entry.get("pos", []), dtype=float)
        if wrist_pos.size != 3 or grasp_pos.size != 3:
            return None
        return wrist_pos, grasp_pos, grasp_pos - wrist_pos

    def _wait_for_topdown_wrist_geometry(
        self,
        sim,
        numeric_targets: dict[str, object],
        *,
        timeout_s: float,
    ) -> tuple[bool, dict[str, object] | None]:
        expected = numeric_targets.get("wrist_to_grasp_center_offset_m")
        if not isinstance(expected, list) or len(expected) != 3:
            return True, None
        expected_vec = np.asarray(expected, dtype=float)
        deadline = time.perf_counter() + max(0.0, float(timeout_s))
        last_measurement: dict[str, object] | None = None

        while time.perf_counter() <= deadline:
            scene_objects = sim.pull_scene_objects()
            gripper_info = scene_objects.get("gripper_diagnostic", {})
            if not isinstance(gripper_info, dict):
                time.sleep(0.05)
                continue
            measured = self._measure_wrist_to_grasp_center_vector(gripper_info)
            if measured is None:
                time.sleep(0.05)
                continue

            wrist_pos, grasp_pos, actual_vec = measured
            delta = actual_vec - expected_vec
            last_measurement = {
                "expected_wrist_to_grasp_center_vector_m": expected_vec.tolist(),
                "actual_wrist_to_grasp_center_vector_m": actual_vec.tolist(),
                "wrist_to_grasp_center_vector_delta_m": delta.tolist(),
                "wrist_world_xyz": wrist_pos.tolist(),
                "grasp_center_world_xyz": grasp_pos.tolist(),
                "xy_error_norm_m": float(np.linalg.norm(delta[:2])),
                "z_error_abs_m": abs(float(delta[2])),
            }
            if (
                float(np.linalg.norm(delta[:2])) <= self.TOPDOWN_WRIST_GEOMETRY_XY_TOL_M
                and abs(float(delta[2])) <= self.TOPDOWN_WRIST_GEOMETRY_Z_TOL_M
            ):
                return True, last_measurement
            time.sleep(0.05)

        return False, last_measurement

    def _measure_topdown_contact_error(
        self,
        sim,
        numeric_targets: dict[str, object],
    ) -> dict[str, object] | None:
        target = self._get_requested_contact_point(numeric_targets)
        if target is None:
            return None
        target_grasp_center = numeric_targets.get("desired_grasp_center_world_xyz")
        if isinstance(target_grasp_center, list) and len(target_grasp_center) == 3:
            target_grasp_center_arr = np.asarray(target_grasp_center, dtype=float)
        else:
            target_grasp_center_arr = None
        scene_objects = sim.pull_scene_objects()
        gripper_info = scene_objects.get("gripper_diagnostic", {})
        if not isinstance(gripper_info, dict):
            return None
        contacts = self._measure_rubber_contact_center(gripper_info)
        if contacts is None:
            return None
        left_contact, right_contact, center = contacts
        error = center - target
        measurement = {
            "target_contact_point": target.tolist(),
            "rubber_contact_center_left": left_contact.tolist(),
            "rubber_contact_center_right": right_contact.tolist(),
            "rubber_contact_center": center.tolist(),
            "rubber_contact_center_delta_to_target": error.tolist(),
        }
        grasp_center = self._measure_rigid_grasp_center(gripper_info)
        if grasp_center is not None:
            measurement["rigid_grasp_center"] = grasp_center.tolist()
            measurement["rigid_grasp_center_to_contact_center"] = (center - grasp_center).tolist()
        if target_grasp_center_arr is not None:
            measurement["target_grasp_center"] = target_grasp_center_arr.tolist()
            if grasp_center is not None:
                measurement["rigid_grasp_center_delta_to_target_grasp_center"] = (
                    grasp_center - target_grasp_center_arr
                ).tolist()
        return measurement

    def _apply_topdown_alignment_joint_targets(
        self,
        sim,
        *,
        base_rotate: float,
        arm: float,
        lift: float,
    ) -> bool:
        current_theta = float(sim.get_base_pose()[2])
        sim.move_by(Actuators.base_rotate, float(base_rotate - current_theta))
        base_ok = self._wait_for_base_rotation_target(
            sim,
            target_theta=float(base_rotate),
            timeout_s=30.0,
        )

        sim.move_to(Actuators.arm, float(arm))
        arm_ok = bool(sim.wait_until_at_setpoint(Actuators.arm, timeout=35.0, position_tolerance=0.04))

        sim.move_to(Actuators.lift, float(lift))
        lift_ok = bool(sim.wait_until_at_setpoint(Actuators.lift, timeout=35.0, position_tolerance=0.04))
        time.sleep(0.2)
        return base_ok and arm_ok and lift_ok

    def _run_topdown_contact_alignment(
        self,
        sim,
        numeric_targets: dict[str, object],
    ) -> list[dict[str, object]]:
        if self.motion_planner.simple_ik is None:
            return []
        if str(numeric_targets.get("approach_type", "")) != "top_down":
            return []
        if "ik_base_rotate" not in numeric_targets:
            return []

        requested_open_width = float(
            numeric_targets.get("gripper_open_width", self.context.grasp_config.max_gripper_width_m)
        )
        grip_angle = float(numeric_targets.get("grip_angle_rad", 0.0))
        history: list[dict[str, object]] = []
        initial_measurement = self._measure_topdown_contact_error(sim, numeric_targets)
        if initial_measurement is None:
            return history

        def get_error_key(measurement: dict[str, object]) -> str:
            if "rigid_grasp_center_delta_to_target_grasp_center" in measurement:
                return "rigid_grasp_center_delta_to_target_grasp_center"
            return "rubber_contact_center_delta_to_target"

        def get_error_vector(measurement: dict[str, object], key: str) -> np.ndarray:
            return np.asarray(measurement[key], dtype=float)

        def error_metric(error: np.ndarray) -> float:
            return float(np.linalg.norm(error))

        current_state = self._read_robot_state(sim)
        best_measurement = initial_measurement
        best_error_key = get_error_key(initial_measurement)
        best_error = get_error_vector(initial_measurement, best_error_key)
        best_metric = error_metric(best_error)
        best_joint_targets = {
            "base_rotate": float(numeric_targets["ik_base_rotate"]),
            "arm": float(numeric_targets.get("ik_arm", current_state.get("arm", 0.0))),
            "lift": float(numeric_targets.get("ik_lift", current_state.get("lift", 0.0))),
        }
        current_measurement = initial_measurement

        for iteration in range(self.TOPDOWN_CONTACT_ALIGNMENT_MAX_ITERS):
            target_key = get_error_key(current_measurement)
            error = get_error_vector(current_measurement, target_key)
            xy_error = float(np.linalg.norm(error[:2]))
            z_error = abs(float(error[2]))
            step_record: dict[str, object] = {
                "iteration": iteration,
                **current_measurement,
                "xy_error_norm_m": xy_error,
                "z_error_abs_m": z_error,
                "alignment_error_key": target_key,
                "best_error_norm_before_iter_m": best_metric,
            }
            history.append(step_record)
            print(
                f"Top-down contact alignment iter {iteration}: "
                f"{target_key}={[round(float(v), 4) for v in error.tolist()]} "
                f"(xy={xy_error:.4f}m, z={z_error:.4f}m)",
                flush=True,
            )
            if xy_error <= self.TOPDOWN_CONTACT_ALIGNMENT_XY_TOL_M and z_error <= self.TOPDOWN_CONTACT_ALIGNMENT_Z_TOL_M:
                break

            target = np.asarray(current_measurement["target_contact_point"], dtype=float)
            limited_error = np.asarray(error, dtype=float).copy()
            limited_error[:2] = np.clip(
                limited_error[:2],
                -self.TOPDOWN_CONTACT_ALIGNMENT_MAX_XY_STEP_M,
                self.TOPDOWN_CONTACT_ALIGNMENT_MAX_XY_STEP_M,
            )
            limited_error[2] = float(
                np.clip(
                    limited_error[2],
                    -self.TOPDOWN_CONTACT_ALIGNMENT_MAX_Z_STEP_M,
                    self.TOPDOWN_CONTACT_ALIGNMENT_MAX_Z_STEP_M,
                )
            )
            step_record["applied_error_step_m"] = limited_error.tolist()
            corrected_target = target - self.TOPDOWN_CONTACT_ALIGNMENT_GAIN * limited_error
            ik_targets = self.motion_planner._solve_topdown_simple_ik_targets(
                desired_rubber_xyz=corrected_target,
                requested_open_width=requested_open_width,
                planning_mode=f"{numeric_targets.get('planning_mode', 'top_down')}_online_alignment",
                grip_angle_rad=grip_angle,
                current_state=self._read_robot_state(sim),
                extra_metadata=None,
            )
            if ik_targets is None:
                step_record["ik_recomputed"] = False
                break

            aligned_joint_targets = {
                "base_rotate": float(ik_targets["ik_base_rotate"]),
                "arm": float(ik_targets["ik_arm"]),
                "lift": float(ik_targets["ik_lift"]),
            }
            step_record["ik_recomputed"] = True
            step_record["corrected_target_contact_point"] = corrected_target.tolist()
            step_record["aligned_joint_targets"] = aligned_joint_targets
            moved_ok = self._apply_topdown_alignment_joint_targets(
                sim,
                base_rotate=aligned_joint_targets["base_rotate"],
                arm=aligned_joint_targets["arm"],
                lift=aligned_joint_targets["lift"],
            )
            step_record["move_ok"] = bool(moved_ok)
            if not moved_ok:
                step_record["move_warning"] = (
                    "Alignment move did not report full setpoint convergence; "
                    "continuing with the measured pose for the next correction step."
                )
            updated_measurement = self._measure_topdown_contact_error(sim, numeric_targets)
            if updated_measurement is None:
                break
            updated_key = get_error_key(updated_measurement)
            updated_error = get_error_vector(updated_measurement, updated_key)
            updated_metric = error_metric(updated_error)
            step_record["post_move_measurement"] = updated_measurement
            step_record["post_move_error_norm_m"] = updated_metric
            if updated_metric + self.TOPDOWN_CONTACT_ALIGNMENT_MIN_IMPROVEMENT_M < best_metric:
                best_measurement = updated_measurement
                best_error_key = updated_key
                best_error = updated_error
                best_metric = updated_metric
                best_joint_targets = aligned_joint_targets
                current_measurement = updated_measurement
                continue

            step_record["reverted_to_best_joint_targets"] = best_joint_targets
            self._apply_topdown_alignment_joint_targets(
                sim,
                base_rotate=float(best_joint_targets["base_rotate"]),
                arm=float(best_joint_targets["arm"]),
                lift=float(best_joint_targets["lift"]),
            )
            current_measurement = self._measure_topdown_contact_error(sim, numeric_targets) or best_measurement
            if updated_metric > best_metric:
                step_record["stopped_on_worse_update"] = True
                break
            current_measurement = updated_measurement

        final_measurement = self._measure_topdown_contact_error(sim, numeric_targets)
        if final_measurement is not None:
            error_key = get_error_key(final_measurement)
            error = get_error_vector(final_measurement, error_key)
            history.append(
                {
                    "iteration": "final",
                    **final_measurement,
                    "xy_error_norm_m": float(np.linalg.norm(error[:2])),
                    "z_error_abs_m": abs(float(error[2])),
                    "alignment_error_key": error_key,
                    "best_error_norm_m": best_metric,
                }
            )
        return history

    @staticmethod
    def _gripper_close_waypoint_reached(
        *,
        waypoint_name: str,
        target: float,
        before_pos: float,
        after_pos: float,
    ) -> bool:
        if (
            abs(after_pos - float(target)) <= 0.12
            or after_pos < before_pos - 0.005
            or after_pos < 0.12
        ):
            return True

        # Once the fingers are already mostly closed around the object, the
        # follow-up hold waypoints should accept "blocked by contact" as a
        # successful force-maintaining close instead of demanding extra travel.
        if waypoint_name in {"secure_grasp", "postgrasp_lift"}:
            return before_pos < 0.18 and after_pos <= before_pos + 0.04

        return False

    @staticmethod
    def _augment_pre_close_diagnostic(
        diagnostic: dict[str, object],
        numeric_targets: dict[str, object],
    ) -> None:
        target = GraspExecutor._get_requested_contact_point(numeric_targets)
        if target is None:
            return

        gripper_info = diagnostic.get("gripper_bodies_and_geoms")
        if not isinstance(gripper_info, dict):
            return

        contacts = GraspExecutor._measure_rubber_contact_center(gripper_info)
        if contacts is None:
            return
        left_contact, right_contact, tip_center = contacts
        diagnostic["target_contact_point"] = target.tolist()
        diagnostic["rubber_contact_center_left"] = left_contact.tolist()
        diagnostic["rubber_contact_center_right"] = right_contact.tolist()
        diagnostic["rubber_contact_center"] = tip_center.tolist()
        diagnostic["rubber_contact_center_delta_to_target"] = (tip_center - target).tolist()
        diagnostic["rubber_tip_center"] = tip_center.tolist()
        diagnostic["rubber_tip_center_delta_to_target"] = (tip_center - target).tolist()
        grasp_center = GraspExecutor._measure_rigid_grasp_center(gripper_info)
        if grasp_center is not None:
            diagnostic["rigid_grasp_center"] = grasp_center.tolist()
            diagnostic["rigid_grasp_center_delta_to_target"] = (grasp_center - target).tolist()
            diagnostic["rigid_grasp_center_to_contact_center"] = (tip_center - grasp_center).tolist()

    def _prepare_start_pose(self, sim) -> None:
        """Move the wrist/arm into a conservative retracted pose before perception."""
        safe_lift_target = float(
            np.clip(
                self.context.scene_config.table_top_z_m + self.context.scene_config.cup_height_m + 0.08,
                STRETCH3_JOINT_LIMITS["lift"][0],
                STRETCH3_JOINT_LIMITS["lift"][1],
            )
        )
        startup_targets = [
            (Actuators.gripper, float(self.context.grasp_config.oracle_side_open_width_cmd), 20.0, 0.12),
            (Actuators.wrist_roll, 0.0, 20.0, 0.05),
            (Actuators.wrist_yaw, self.context.grasp_config.oracle_tucked_wrist_yaw_rad, 20.0, 0.35),
            (Actuators.wrist_pitch, 0.00, 20.0, 0.06),
            (Actuators.lift, safe_lift_target, 30.0, 0.06),
            (Actuators.arm, 0.0, 30.0, 0.12),
        ]
        for actuator, target, timeout_s, tolerance in startup_targets:
            sim.move_to(actuator, float(target))
            if sim.wait_until_at_setpoint(actuator, timeout=timeout_s, position_tolerance=tolerance):
                continue

            status = sim.pull_status()
            actual = {
                Actuators.gripper: float(status.gripper.pos),
                Actuators.wrist_roll: float(status.wrist_roll.pos),
                Actuators.wrist_yaw: float(status.wrist_yaw.pos),
                Actuators.wrist_pitch: float(status.wrist_pitch.pos),
                Actuators.lift: float(status.lift.pos),
                Actuators.arm: float(status.arm.pos),
            }[actuator]
            error = abs(actual - float(target))
            if actuator == Actuators.wrist_yaw and (error <= 0.40 or actual >= 1.8):
                print(
                    f"Startup wrist_yaw accepted near tuck target: target={float(target):.3f}, actual={actual:.3f}",
                    flush=True,
                )
                continue
            if actuator == Actuators.wrist_yaw:
                print(
                    f"Startup wrist_yaw tolerated to unblock perception: target={float(target):.3f}, actual={actual:.3f}",
                    flush=True,
                )
                continue
            if actuator == Actuators.arm and actual <= 0.18:
                print(
                    f"Startup arm accepted within retracted perception band: target={float(target):.3f}, actual={actual:.3f}",
                    flush=True,
                )
                continue
            if actuator == Actuators.arm:
                print(
                    f"Startup arm tolerated to unblock perception: target={float(target):.3f}, actual={actual:.3f}",
                    flush=True,
                )
                continue
            if actuator == Actuators.lift and actual >= safe_lift_target - 0.10:
                print(
                    f"Startup lift accepted in safe base-motion band: target={float(target):.3f}, actual={actual:.3f}",
                    flush=True,
                )
                continue
            if actuator == Actuators.gripper:
                print(
                    f"Startup gripper accepted without blocking perception: target={float(target):.3f}, actual={actual:.3f}",
                    flush=True,
                )
                continue
            print(
                f"Startup {actuator.name} tolerated to keep pipeline running: target={float(target):.3f}, actual={actual:.3f}",
                flush=True,
            )
            continue

    @staticmethod
    def _normalize_angle(angle_rad: float) -> float:
        return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)

    def _get_nominal_target_center_xy(self) -> np.ndarray | None:
        target_center_xy = self.context.runtime_metadata.get("target_center_xy")
        if isinstance(target_center_xy, (list, tuple)) and len(target_center_xy) == 2:
            return np.asarray(target_center_xy, dtype=float)
        cup_x, cup_y, _ = self.context.scene_config.cup_position_m
        return np.array([float(cup_x), float(cup_y)], dtype=float)

    def _run_base_preposition_controller(
        self,
        sim,
        *,
        target_xy: np.ndarray,
        goal_distance_m: float,
        max_translate_m: float,
        stop_if_reachable: callable | None = None,
        timeout_s: float = 45.0,
    ) -> dict[str, object]:
        base_x, base_y, base_theta = map(float, sim.get_base_pose())
        before_pose = np.array([base_x, base_y, base_theta], dtype=float)
        initial_relative_xy = target_xy - before_pose[:2]
        controller_trace: list[dict[str, float | int]] = []
        traveled_m = 0.0
        last_xy = before_pose[:2].copy()
        stop_reason = "timeout"
        stall_counter = 0
        loop_index = 0
        previous_planar_distance = float(np.linalg.norm(initial_relative_xy))
        deadline = time.perf_counter() + max(0.0, float(timeout_s))

        try:
            while time.perf_counter() <= deadline:
                base_x, base_y, base_theta = map(float, sim.get_base_pose())
                current_xy = np.array([base_x, base_y], dtype=float)
                traveled_m += float(np.linalg.norm(current_xy - last_xy))
                last_xy = current_xy
                relative_xy = target_xy - current_xy
                planar_distance = float(np.linalg.norm(relative_xy))
                if stop_if_reachable is not None and stop_if_reachable():
                    stop_reason = "exact_target_reachable"
                    break
                if planar_distance <= float(goal_distance_m):
                    stop_reason = "within_goal_distance"
                    break
                if traveled_m >= float(max_translate_m):
                    stop_reason = "translate_budget_exhausted"
                    break

                translation_heading_target = float(math.atan2(relative_xy[1], relative_xy[0]))
                heading_error = self._normalize_angle(translation_heading_target - base_theta)
                linear_error = max(0.0, planar_distance - float(goal_distance_m))
                v_linear = min(
                    self.BASE_PREPOSITION_MAX_LINEAR_V_MPS,
                    self.BASE_PREPOSITION_LINEAR_GAIN * linear_error,
                )
                if abs(heading_error) > 0.45:
                    v_linear = 0.0
                elif abs(heading_error) > 0.25:
                    v_linear *= 0.6
                omega = float(
                    np.clip(
                        self.BASE_PREPOSITION_OMEGA_GAIN * heading_error,
                        -self.BASE_PREPOSITION_MAX_OMEGA_RADPS,
                        self.BASE_PREPOSITION_MAX_OMEGA_RADPS,
                    )
                )
                sim.set_base_velocity(v_linear, omega)
                time.sleep(self.BASE_PREPOSITION_CONTROL_DT_S)

                new_xy = np.asarray(sim.get_base_pose()[:2], dtype=float)
                moved_this_cycle = float(np.linalg.norm(new_xy - current_xy))
                planar_distance_after_step = float(np.linalg.norm(target_xy - new_xy))
                progress_this_cycle = previous_planar_distance - planar_distance_after_step
                previous_planar_distance = planar_distance_after_step
                if v_linear > 0.05 and progress_this_cycle < 0.0005:
                    stall_counter += 1
                else:
                    stall_counter = 0
                if stall_counter >= 10:
                    stop_reason = "translation_stall"
                    break

                if loop_index % 5 == 0:
                    controller_trace.append(
                        {
                            "loop_index": loop_index,
                            "planar_distance_m": planar_distance,
                            "heading_error_rad": heading_error,
                            "v_linear_mps": float(v_linear),
                            "omega_radps": float(omega),
                            "traveled_m": traveled_m,
                            "moved_this_cycle_m": moved_this_cycle,
                            "progress_this_cycle_m": progress_this_cycle,
                            "planar_distance_after_step_m": planar_distance_after_step,
                        }
                    )
                loop_index += 1
            else:
                stop_reason = "timeout"
        finally:
            sim.set_base_velocity(0.0, 0.0)
            time.sleep(0.2)

        final_x, final_y, final_theta = map(float, sim.get_base_pose())
        after_pose = np.array([final_x, final_y, final_theta], dtype=float)
        final_relative_xy = target_xy - after_pose[:2]
        return {
            "target_center_xy": target_xy.tolist(),
            "base_pose_before": before_pose.tolist(),
            "base_pose_after": after_pose.tolist(),
            "planar_distance_before_m": float(np.linalg.norm(initial_relative_xy)),
            "planar_distance_after_m": float(np.linalg.norm(final_relative_xy)),
            "heading_error_before_rad": self._normalize_angle(
                float(math.atan2(initial_relative_xy[1], initial_relative_xy[0])) - before_pose[2]
            ),
            "translation_heading_target_rad": float(math.atan2(final_relative_xy[1], final_relative_xy[0])),
            "base_rotate_by_rad": self._normalize_angle(float(after_pose[2] - before_pose[2])),
            "base_translate_by_m": float(np.linalg.norm(after_pose[:2] - before_pose[:2])),
            "base_rotate_ok": stop_reason != "timeout",
            "base_translate_ok": stop_reason not in {"timeout", "translation_stall"},
            "base_translation_steps": controller_trace,
            "controller_stop_reason": stop_reason,
        }


    def _maybe_preposition_base_for_target(self, sim) -> dict[str, object] | None:
        cfg = self.context.grasp_config
        diagnostic: dict[str, object] = {
            "enabled": bool(cfg.enable_base_preposition),
            "performed": False,
        }
        if not cfg.enable_base_preposition:
            diagnostic["reason"] = "disabled_by_config"
            return diagnostic
        if self.context.scene_config.grasp_method == "geometric":
            diagnostic["reason"] = "deferred_until_after_initial_geometric_observation"
            return diagnostic

        target_xy = self._get_nominal_target_center_xy()
        if target_xy is None:
            diagnostic["reason"] = "no_target_center"
            return diagnostic

        base_x, base_y, base_theta = map(float, sim.get_base_pose())
        base_xy = np.array([base_x, base_y], dtype=float)
        relative_xy = target_xy - base_xy
        planar_distance = float(np.linalg.norm(relative_xy))
        translation_heading_target = float(math.atan2(relative_xy[1], relative_xy[0]))
        heading_error = self._normalize_angle(translation_heading_target - base_theta)
        diagnostic.update(
            {
                "target_center_xy": target_xy.tolist(),
                "base_pose_before": [base_x, base_y, base_theta],
                "planar_distance_before_m": planar_distance,
                "heading_error_before_rad": heading_error,
                "translation_heading_target_rad": translation_heading_target,
                "trigger_distance_m": float(cfg.base_preposition_trigger_distance_m),
                "goal_distance_m": float(cfg.base_preposition_goal_distance_m),
            }
        )
        if planar_distance <= float(cfg.base_preposition_trigger_distance_m):
            diagnostic["reason"] = "already_within_comfort_range"
            return diagnostic
        control_diag = self._run_base_preposition_controller(
            sim,
            target_xy=target_xy,
            goal_distance_m=float(cfg.base_preposition_goal_distance_m),
            max_translate_m=float(cfg.base_preposition_max_translate_m),
            stop_if_reachable=lambda: self._nominal_topdown_target_reachable(sim, target_xy),
            timeout_s=45.0,
        )
        diagnostic.update(control_diag)
        diagnostic["performed"] = bool(control_diag["base_translate_by_m"] > 0.02 or abs(control_diag["base_rotate_by_rad"]) > 0.12)
        diagnostic["grasp_heading_target_rad"] = None
        diagnostic["final_base_rotate_by_rad"] = 0.0
        diagnostic["final_base_rotate_ok"] = True
        diagnostic["final_heading_rotation_skipped"] = True
        return diagnostic

    def _maybe_refine_base_for_geometric_grasp(
        self,
        sim,
        geometric_grasp: dict[str, object],
        *,
        phase: str = "full",
    ) -> dict[str, object] | None:
        cfg = self.context.grasp_config
        diagnostic: dict[str, object] = {
            "enabled": bool(cfg.enable_base_preposition),
            "performed": False,
            "phase": phase,
        }
        if not cfg.enable_base_preposition:
            diagnostic["reason"] = "disabled_by_config"
            return diagnostic
        if self.motion_planner.simple_ik is None:
            diagnostic["reason"] = "simple_ik_unavailable"
            return diagnostic

        target_xy = np.array(
            [
                float(geometric_grasp["grasp_x"]),
                float(geometric_grasp["grasp_y"]),
            ],
            dtype=float,
        )
        base_x, base_y, base_theta = map(float, sim.get_base_pose())
        before_pose = np.array([base_x, base_y, base_theta], dtype=float)
        reference_theta = base_theta
        relative_xy = target_xy - np.array([base_x, base_y], dtype=float)
        diagnostic.update(
            {
                "target_grasp_xy": target_xy.tolist(),
                "base_pose_before": before_pose.tolist(),
                "planar_distance_before_m": float(np.linalg.norm(relative_xy)),
                "reference_theta_rad": float(reference_theta),
            }
        )

        def exact_reachable() -> bool:
            try:
                self.motion_planner.geometric_grasp_targets(
                    geometric_grasp,
                    current_state=self._read_robot_state(sim),
                )
            except RuntimeError:
                return False
            return True

        if phase == "full" and exact_reachable():
            diagnostic["reason"] = "exact_geometric_target_already_reachable"
            return diagnostic

        max_translate_budget = float(cfg.base_preposition_max_translate_m)
        motion_trace: list[dict[str, object]] = []
        base_forward_axis = self._base_forward_axis(reference_theta)
        arm_reach_axis = self._arm_reach_axis(reference_theta)

        lateral_error = float(relative_xy @ base_forward_axis)
        lateral_move = float(np.clip(lateral_error, -max_translate_budget, max_translate_budget))
        lateral_diag = {"requested_move_m": lateral_move, "performed": False}
        if phase in {"full", "lateral"} and abs(lateral_move) > 0.01:
            lateral_diag = self._drive_base_distance(
                sim,
                distance_m=lateral_move,
                timeout_s=20.0,
            )
            lateral_diag["phase"] = "lateral_centering"
            motion_trace.append(lateral_diag)

        longitudinal_diag = {
            "phase": "longitudinal_reach_adjustment",
            "performed": False,
            "target_heading_rad": None,
            "requested_move_m": 0.0,
        }
        base_x, base_y, _ = map(float, sim.get_base_pose())
        relative_xy = target_xy - np.array([base_x, base_y], dtype=float)
        longitudinal_distance = float(relative_xy @ arm_reach_axis)
        desired_longitudinal_distance = float(cfg.base_preposition_goal_distance_m)
        reach_margin = float(cfg.base_preposition_longitudinal_extra_m)
        table_clearance_m = float(getattr(cfg, "base_preposition_table_clearance_m", 0.12))
        if phase in {"full", "longitudinal"} and not exact_reachable() and max_translate_budget > 0.01:
            reach_error = longitudinal_distance - desired_longitudinal_distance
            if abs(reach_error) > 0.015:
                nominal_requested_longitudinal_move = float(
                    np.clip(
                        reach_error + math.copysign(reach_margin, reach_error),
                        -max_translate_budget,
                        max_translate_budget,
                    )
                )
                requested_longitudinal_move = self._clip_longitudinal_move_for_table_clearance(
                    start_xy=np.array([base_x, base_y], dtype=float),
                    move_axis=arm_reach_axis,
                    requested_move_m=nominal_requested_longitudinal_move,
                    clearance_m=table_clearance_m,
                )
                if abs(requested_longitudinal_move) <= 0.01:
                    longitudinal_diag = {
                        "phase": "longitudinal_reach_adjustment",
                        "performed": False,
                        "target_heading_rad": None,
                        "requested_move_m": requested_longitudinal_move,
                        "nominal_requested_move_m": nominal_requested_longitudinal_move,
                        "clearance_limited": True,
                        "table_clearance_m": table_clearance_m,
                        "stop_reason": "blocked_by_table_clearance",
                    }
                    motion_trace.append(longitudinal_diag)
                    final_x, final_y, final_theta = map(float, sim.get_base_pose())
                    after_pose = np.array([final_x, final_y, final_theta], dtype=float)
                    diagnostic["exact_target_reachable_after_refine"] = bool(exact_reachable())
                    diagnostic["performed"] = bool(motion_trace)
                    diagnostic["motion_trace"] = motion_trace
                    diagnostic["lateral_centering"] = lateral_diag
                    diagnostic["longitudinal_reach_adjustment"] = longitudinal_diag
                    diagnostic["base_pose_after"] = after_pose.tolist()
                    diagnostic["base_translate_by_m"] = float(np.linalg.norm(after_pose[:2] - before_pose[:2]))
                    diagnostic["base_rotate_by_rad"] = self._normalize_angle(float(after_pose[2] - reference_theta))
                    diagnostic["planar_distance_after_m"] = float(np.linalg.norm(target_xy - after_pose[:2]))
                    return diagnostic
                heading = self._normalize_angle(reference_theta + self.BASE_PREPOSITION_CLOCKWISE_TURN_RAD)
                rotate_out = self._rotate_base_to_heading(
                    sim,
                    target_theta=heading,
                    timeout_s=20.0,
                )
                drive_diag = self._drive_base_distance(
                    sim,
                    distance_m=requested_longitudinal_move,
                    timeout_s=25.0,
                )
                rotate_back = self._rotate_base_to_heading(
                    sim,
                    target_theta=reference_theta,
                    timeout_s=20.0,
                )
                longitudinal_diag = {
                    "phase": "longitudinal_reach_adjustment",
                    "performed": True,
                    "target_heading_rad": heading,
                    "requested_move_m": requested_longitudinal_move,
                    "nominal_requested_move_m": nominal_requested_longitudinal_move,
                    "clearance_limited": bool(
                        abs(requested_longitudinal_move - nominal_requested_longitudinal_move) > 1e-6
                    ),
                    "table_clearance_m": table_clearance_m,
                    "rotate_out": rotate_out,
                    "drive": drive_diag,
                    "rotate_back": rotate_back,
                }
                motion_trace.append(longitudinal_diag)

        final_x, final_y, final_theta = map(float, sim.get_base_pose())
        after_pose = np.array([final_x, final_y, final_theta], dtype=float)
        diagnostic["exact_target_reachable_after_refine"] = bool(exact_reachable())
        diagnostic["performed"] = bool(motion_trace)
        diagnostic["motion_trace"] = motion_trace
        diagnostic["lateral_centering"] = lateral_diag
        diagnostic["longitudinal_reach_adjustment"] = longitudinal_diag
        diagnostic["base_pose_after"] = after_pose.tolist()
        diagnostic["base_translate_by_m"] = float(np.linalg.norm(after_pose[:2] - before_pose[:2]))
        diagnostic["base_rotate_by_rad"] = self._normalize_angle(float(after_pose[2] - reference_theta))
        diagnostic["planar_distance_after_m"] = float(np.linalg.norm(target_xy - after_pose[:2]))
        return diagnostic

    @staticmethod
    def _base_forward_axis(theta_rad: float) -> np.ndarray:
        return np.array([math.cos(theta_rad), math.sin(theta_rad)], dtype=float)

    @staticmethod
    def _arm_reach_axis(theta_rad: float) -> np.ndarray:
        return np.array([math.sin(theta_rad), -math.cos(theta_rad)], dtype=float)

    def _rotate_base_to_heading(
        self,
        sim,
        *,
        target_theta: float,
        timeout_s: float,
        tolerance_rad: float = 0.02,
    ) -> dict[str, object]:
        started_theta = float(sim.get_base_pose()[2])
        started_xy = np.asarray(sim.get_base_pose()[:2], dtype=float)
        settled_before = self._wait_for_base_stationary(sim, timeout_s=4.0)
        deadline = self._get_sim_time(sim) + max(0.0, float(timeout_s))
        wall_deadline = self._compute_wall_deadline(float(timeout_s))
        stop_reason = "timeout"
        while self._get_sim_time(sim) <= deadline and time.perf_counter() <= wall_deadline:
            current_theta = float(sim.get_base_pose()[2])
            error = self._normalize_angle(float(target_theta) - current_theta)
            if abs(error) <= float(tolerance_rad):
                stop_reason = "within_tolerance"
                break
            omega = float(
                np.clip(
                    1.6 * error,
                    -self.BASE_PREPOSITION_MAX_OMEGA_RADPS,
                    self.BASE_PREPOSITION_MAX_OMEGA_RADPS,
                )
            )
            sim.set_base_velocity(0.0, omega)
            time.sleep(self.BASE_PREPOSITION_CONTROL_DT_S)
        sim.set_base_velocity(0.0, 0.0)
        settled_after = self._wait_for_base_stationary(sim, timeout_s=4.0)
        final_theta = float(sim.get_base_pose()[2])
        final_xy = np.asarray(sim.get_base_pose()[:2], dtype=float)
        return {
            "target_theta_rad": float(target_theta),
            "start_theta_rad": started_theta,
            "final_theta_rad": final_theta,
            "actual_delta_rad": self._normalize_angle(final_theta - started_theta),
            "start_xy": started_xy.tolist(),
            "final_xy": final_xy.tolist(),
            "xy_drift_m": (final_xy - started_xy).tolist(),
            "settled_before": settled_before,
            "settled_after": settled_after,
            "stop_reason": stop_reason,
            "ok": abs(self._normalize_angle(float(target_theta) - final_theta)) <= float(tolerance_rad),
        }

    def _drive_base_distance(
        self,
        sim,
        *,
        distance_m: float,
        timeout_s: float,
        tolerance_m: float = 0.01,
        min_lidar_clearance_m: float | None = None,
    ) -> dict[str, object]:
        start_x, start_y, start_theta = map(float, sim.get_base_pose())
        start_xy = np.array([start_x, start_y], dtype=float)
        forward_axis = self._base_forward_axis(start_theta)
        requested_distance = float(distance_m)
        settled_before = self._wait_for_base_stationary(sim, timeout_s=4.0)
        deadline = self._get_sim_time(sim) + max(0.0, float(timeout_s))
        wall_deadline = self._compute_wall_deadline(float(timeout_s))
        stop_reason = "timeout"
        stall_counter = 0
        last_progress = 0.0
        last_stall_check_sim_time = self._get_sim_time(sim)
        lidar_before = self._nearest_lidar_clearance(sim)
        lidar_last = lidar_before

        while self._get_sim_time(sim) <= deadline and time.perf_counter() <= wall_deadline:
            current_xy = np.asarray(sim.get_base_pose()[:2], dtype=float)
            traveled_along_axis = float((current_xy - start_xy) @ forward_axis)
            remaining = requested_distance - traveled_along_axis
            if requested_distance > 0.0 and min_lidar_clearance_m is not None:
                lidar_last = self._nearest_lidar_clearance(sim)
                if bool(lidar_last.get("available", False)) and float(lidar_last["nearest_distance_m"]) <= float(min_lidar_clearance_m):
                    stop_reason = "lidar_clearance_reached"
                    break
            if abs(remaining) <= float(tolerance_m):
                stop_reason = "within_tolerance"
                break
            v_linear = float(
                np.clip(
                    1.2 * remaining,
                    -self.BASE_PREPOSITION_MAX_LINEAR_V_MPS,
                    self.BASE_PREPOSITION_MAX_LINEAR_V_MPS,
                )
            )
            sim.set_base_velocity(v_linear, 0.0)
            time.sleep(self.BASE_PREPOSITION_CONTROL_DT_S)
            new_xy = np.asarray(sim.get_base_pose()[:2], dtype=float)
            progress = float(abs((new_xy - start_xy) @ forward_axis))
            current_sim_time = self._get_sim_time(sim)
            sim_dt = current_sim_time - last_stall_check_sim_time
            if sim_dt >= 0.5:
                if abs(v_linear) > 0.05 and progress - last_progress < 0.002:
                    stall_counter += 1
                else:
                    stall_counter = 0
                last_progress = progress
                last_stall_check_sim_time = current_sim_time
            if stall_counter >= 10:
                stop_reason = "translation_stall"
                break

        sim.set_base_velocity(0.0, 0.0)
        settled_after = self._wait_for_base_stationary(sim, timeout_s=4.0)
        final_xy = np.asarray(sim.get_base_pose()[:2], dtype=float)
        actual_distance = float((final_xy - start_xy) @ forward_axis)
        lidar_after = self._nearest_lidar_clearance(sim)
        return {
            "requested_distance_m": requested_distance,
            "actual_distance_m": actual_distance,
            "start_xy": start_xy.tolist(),
            "final_xy": final_xy.tolist(),
            "lidar_before": lidar_before,
            "lidar_last": lidar_last,
            "lidar_after": lidar_after,
            "lidar_stop_threshold_m": None if min_lidar_clearance_m is None else float(min_lidar_clearance_m),
            "settled_before": settled_before,
            "settled_after": settled_after,
            "stop_reason": stop_reason,
            "performed": abs(actual_distance) > 0.005,
            "ok": abs(requested_distance - actual_distance) <= max(float(tolerance_m), 0.03),
        }

    @staticmethod
    def _get_sim_time(sim) -> float:
        return float(sim.pull_status().time)

    @staticmethod
    def _compute_wall_deadline(timeout_s: float, *, multiplier: float = 4.0, additive_s: float = 5.0) -> float:
        return time.perf_counter() + max(float(timeout_s) * float(multiplier), float(timeout_s) + float(additive_s))

    def _nearest_lidar_clearance(self, sim) -> dict[str, object]:
        try:
            sensor_data = sim.pull_sensor_data()
            scan = np.asarray(sensor_data.get_data(StretchSensors.base_lidar), dtype=float).reshape(-1)
        except Exception as exc:
            return {
                "available": False,
                "reason": f"lidar_unavailable: {type(exc).__name__}",
            }

        valid = scan[np.isfinite(scan) & (scan > 0.05) & (scan < 9.5)]
        if valid.size == 0:
            return {
                "available": False,
                "reason": "no_valid_lidar_returns",
            }
        return {
            "available": True,
            "nearest_distance_m": float(np.min(valid)),
            "percentile_5_m": float(np.percentile(valid, 5)),
            "num_valid_returns": int(valid.size),
        }

    def _ensure_lidar_rotate_clearance(
        self,
        sim,
        *,
        min_clearance_m: float,
        timeout_s: float,
    ) -> dict[str, object]:
        cfg = self.context.grasp_config
        backoff_step_m = float(getattr(cfg, "base_preposition_lidar_backoff_step_m", 0.04))
        max_backoff_m = float(getattr(cfg, "base_preposition_lidar_max_backoff_m", 0.16))
        before = self._nearest_lidar_clearance(sim)
        diagnostic: dict[str, object] = {
            "required_clearance_m": float(min_clearance_m),
            "before": before,
            "performed": False,
            "total_backoff_m": 0.0,
            "backoff_attempts": [],
        }
        if not bool(before.get("available", False)):
            diagnostic["after"] = before
            diagnostic["stop_reason"] = "lidar_unavailable"
            diagnostic["ok"] = True
            return diagnostic

        current = before
        deadline = self._get_sim_time(sim) + max(0.0, float(timeout_s))
        wall_deadline = self._compute_wall_deadline(float(timeout_s))
        total_backoff = 0.0
        stop_reason = "within_clearance"
        while (
            bool(current.get("available", False))
            and float(current["nearest_distance_m"]) < float(min_clearance_m)
            and total_backoff < max_backoff_m
            and self._get_sim_time(sim) <= deadline
            and time.perf_counter() <= wall_deadline
        ):
            shortage_m = float(min_clearance_m) - float(current["nearest_distance_m"])
            requested_backoff = min(
                backoff_step_m,
                max_backoff_m - total_backoff,
                shortage_m + 0.02,
            )
            if requested_backoff <= 0.005:
                stop_reason = "backoff_too_small"
                break
            drive_diag = self._drive_base_distance(
                sim,
                distance_m=-float(requested_backoff),
                timeout_s=12.0,
            )
            total_backoff += abs(float(drive_diag["actual_distance_m"]))
            current = self._nearest_lidar_clearance(sim)
            diagnostic["backoff_attempts"].append(
                {
                    "requested_backoff_m": float(requested_backoff),
                    "drive": drive_diag,
                    "clearance_after": current,
                }
            )
            diagnostic["performed"] = True
            if abs(float(drive_diag["actual_distance_m"])) <= 0.005:
                stop_reason = "backoff_translation_stall"
                break

        if bool(current.get("available", False)) and float(current["nearest_distance_m"]) >= float(min_clearance_m):
            stop_reason = "within_clearance"
        elif total_backoff >= max_backoff_m:
            stop_reason = "max_backoff_reached"
        elif self._get_sim_time(sim) > deadline:
            stop_reason = "timeout"
        diagnostic["after"] = current
        diagnostic["total_backoff_m"] = float(total_backoff)
        diagnostic["stop_reason"] = stop_reason
        diagnostic["ok"] = bool(
            current.get("available", False) and float(current.get("nearest_distance_m", 0.0)) >= float(min_clearance_m)
        )
        return diagnostic

    def _clip_longitudinal_move_for_table_clearance(
        self,
        *,
        start_xy: np.ndarray,
        move_axis: np.ndarray,
        requested_move_m: float,
        clearance_m: float,
    ) -> float:
        requested_move_m = float(requested_move_m)
        if requested_move_m <= 0.0:
            return requested_move_m
        if not hasattr(self.context.scene_config, "table_position_m") or not hasattr(self.context.scene_config, "table_size_m"):
            return requested_move_m

        table_center = np.asarray(self.context.scene_config.table_position_m[:2], dtype=float)
        table_half_extents = np.asarray(self.context.scene_config.table_size_m[:2], dtype=float) + float(clearance_m)
        ray_origin = np.asarray(start_xy, dtype=float)
        ray_dir = np.asarray(move_axis, dtype=float)
        ray_norm = float(np.linalg.norm(ray_dir))
        if ray_norm <= 1e-8:
            return requested_move_m
        ray_dir = ray_dir / ray_norm

        t_enter = -float("inf")
        t_exit = float("inf")
        bounds_min = table_center - table_half_extents
        bounds_max = table_center + table_half_extents
        for axis in range(2):
            direction = float(ray_dir[axis])
            origin = float(ray_origin[axis])
            min_bound = float(bounds_min[axis])
            max_bound = float(bounds_max[axis])
            if abs(direction) <= 1e-8:
                if origin < min_bound or origin > max_bound:
                    return requested_move_m
                continue
            t1 = (min_bound - origin) / direction
            t2 = (max_bound - origin) / direction
            t_axis_enter = min(t1, t2)
            t_axis_exit = max(t1, t2)
            t_enter = max(t_enter, t_axis_enter)
            t_exit = min(t_exit, t_axis_exit)

        if t_exit < 0.0 or t_enter > t_exit:
            return requested_move_m
        if t_enter <= 0.0:
            return 0.0

        safe_stop_margin_m = 0.01
        max_safe_move = max(0.0, t_enter - safe_stop_margin_m)
        return float(min(requested_move_m, max_safe_move))

    def _wait_for_base_stationary(
        self,
        sim,
        *,
        timeout_s: float,
        linear_vel_tol_mps: float = 0.01,
        angular_vel_tol_radps: float = 0.02,
        required_stable_sim_s: float = 0.4,
    ) -> dict[str, object]:
        deadline = self._get_sim_time(sim) + max(0.0, float(timeout_s))
        wall_deadline = self._compute_wall_deadline(float(timeout_s))
        stable_since: float | None = None
        stop_reason = "timeout"
        samples = 0
        final_linear = 0.0
        final_angular = 0.0
        while self._get_sim_time(sim) <= deadline and time.perf_counter() <= wall_deadline:
            status = sim.pull_status()
            final_linear = abs(float(status.base.x_vel))
            final_angular = abs(float(status.base.theta_vel))
            samples += 1
            if final_linear <= linear_vel_tol_mps and final_angular <= angular_vel_tol_radps:
                if stable_since is None:
                    stable_since = float(status.time)
                elif float(status.time) - stable_since >= required_stable_sim_s:
                    stop_reason = "stationary"
                    break
            else:
                stable_since = None
            time.sleep(self.BASE_PREPOSITION_CONTROL_DT_S)
        return {
            "stop_reason": stop_reason,
            "samples": samples,
            "final_linear_vel_mps": final_linear,
            "final_angular_vel_radps": final_angular,
            "ok": stop_reason == "stationary",
        }

    def _wait_for_base_rotation_target(
        self,
        sim,
        *,
        target_theta: float,
        timeout_s: float,
        tolerance_rad: float = 0.02,
        poll_s: float = 0.05,
    ) -> bool:
        deadline = time.perf_counter() + max(0.0, float(timeout_s))
        while time.perf_counter() <= deadline:
            current_theta = float(sim.get_base_pose()[2])
            if abs(self._normalize_angle(current_theta - float(target_theta))) <= tolerance_rad:
                return True
            time.sleep(poll_s)
        return False

    def _nominal_topdown_target_reachable(self, sim, target_xy: np.ndarray) -> bool:
        if self.motion_planner.simple_ik is None:
            return False
        nominal_geometric_grasp = {
            "grasp_x": float(target_xy[0]),
            "grasp_y": float(target_xy[1]),
            "grasp_z": float(self.context.scene_config.table_top_z_m + self.context.scene_config.cup_height_m * 0.5),
            "gripper_open_width": float(
                min(
                    self.context.scene_config.cup_radius_m * 2.0 + 0.02,
                    self.context.grasp_config.max_gripper_width_m,
                )
            ),
            "grip_angle_rad": 0.0,
            "min_cross_section_width": float(self.context.scene_config.cup_radius_m * 2.0),
            "object_center": [
                float(target_xy[0]),
                float(target_xy[1]),
                float(self.context.scene_config.table_top_z_m + self.context.scene_config.cup_height_m * 0.5),
            ],
            "object_height": float(self.context.scene_config.cup_height_m),
            "object_top_z": float(self.context.scene_config.table_top_z_m + self.context.scene_config.cup_height_m),
            "object_bottom_z": float(self.context.scene_config.table_top_z_m),
            "grasp_point_validated": True,
            "width_near_limit": True,
        }
        try:
            targets = self.motion_planner.geometric_grasp_targets(
                nominal_geometric_grasp,
                current_state=self._read_robot_state(sim),
            )
        except RuntimeError:
            return False
        return "ik_base_rotate" in targets

    @staticmethod
    def _wait_for_base_translation_delta(
        sim,
        *,
        start_xy: np.ndarray,
        expected_distance_m: float,
        timeout_s: float,
        tolerance_m: float = 0.02,
        poll_s: float = 0.05,
    ) -> bool:
        if float(expected_distance_m) <= tolerance_m:
            return True
        deadline = time.perf_counter() + max(0.0, float(timeout_s))
        while time.perf_counter() <= deadline:
            current_xy = np.asarray(sim.get_base_pose()[:2], dtype=float)
            traveled = float(np.linalg.norm(current_xy - np.asarray(start_xy, dtype=float)))
            if traveled + tolerance_m >= float(expected_distance_m):
                return True
            time.sleep(poll_s)
        return False

    def _read_robot_state(self, sim) -> dict[str, float]:
        """Read the current simulator joint state into a flat dict."""
        status = sim.pull_status()
        base_x, base_y, base_theta = map(float, sim.get_base_pose())
        return {
            "lift": float(status.lift.pos),
            "arm": float(status.arm.pos),
            "wrist_yaw": float(status.wrist_yaw.pos),
            "wrist_pitch": float(status.wrist_pitch.pos),
            "wrist_roll": float(status.wrist_roll.pos),
            "head_pan": float(status.head_pan.pos),
            "head_tilt": float(status.head_tilt.pos),
            "stretch_gripper": float(status.gripper.pos),
            "base_rotate": base_theta,
            "base_x": base_x,
            "base_y": base_y,
        }

    def _read_actuator_position(self, sim, actuator: Actuators) -> float:
        """Read the latest position value for a single actuator."""
        status = sim.pull_status()
        mapping = {
            Actuators.base_rotate: float(status.base.theta),
            Actuators.lift: float(status.lift.pos),
            Actuators.arm: float(status.arm.pos),
            Actuators.wrist_yaw: float(status.wrist_yaw.pos),
            Actuators.wrist_pitch: float(status.wrist_pitch.pos),
            Actuators.wrist_roll: float(status.wrist_roll.pos),
            Actuators.gripper: float(status.gripper.pos),
        }
        return mapping[actuator]

    def _verify_grasp(self, sim) -> tuple[bool, dict[str, object]]:
        """Verify that the target cup was lifted, not just that the gripper closed."""
        time.sleep(0.3)
        verification: dict[str, object] = {
            "verification_method": "cup_body_z",
            "cup_body_name": self.context.scene_config.cup_body_name,
        }
        try:
            cup_body_name = self.context.scene_config.cup_body_name
            scene_objects = sim.pull_scene_objects()
            if cup_body_name not in scene_objects:
                raise KeyError(f"Scene object {cup_body_name!r} not found in simulator proxy")
            cup_pose = scene_objects[cup_body_name]
            cup_z = float(cup_pose["xpos"][2])
            table_z = float(self.context.scene_config.table_top_z_m)
            cup_height = float(self.context.scene_config.cup_height_m)
            resting_z = table_z + cup_height / 2.0
            lift_threshold = resting_z + 0.03
            lifted = cup_z > lift_threshold
            other_cup_states: dict[str, dict[str, float | bool]] = {}
            other_cups_stationary = True
            for body_name in self.context.runtime_metadata.get("non_target_cup_body_names", []):
                entry = scene_objects.get(str(body_name))
                if not isinstance(entry, dict) or "xpos" not in entry:
                    other_cups_stationary = False
                    other_cup_states[str(body_name)] = {"found": False}
                    continue
                other_z = float(entry["xpos"][2])
                stationary = abs(other_z - resting_z) <= 0.015
                other_cups_stationary = other_cups_stationary and stationary
                other_cup_states[str(body_name)] = {
                    "found": True,
                    "cup_z_after_lift": other_z,
                    "resting_z": resting_z,
                    "stationary_threshold_abs_m": 0.015,
                    "stationary": stationary,
                }
            verification.update(
                {
                    "cup_z_after_lift": cup_z,
                    "resting_z": resting_z,
                    "lift_threshold": lift_threshold,
                    "cup_world_position_m": cup_pose["xpos"],
                    "gripper_pos_after_close": float(sim.pull_status().gripper.pos),
                    "other_cup_states": other_cup_states,
                    "other_cups_stationary": other_cups_stationary,
                }
            )
            print(
                "Grasp verification:"
                f" cup_z={cup_z:.4f}, resting_z≈{resting_z:.4f},"
                f" threshold={lift_threshold:.4f}, lifted={lifted}",
                flush=True,
            )
            if other_cup_states:
                print(f"  Other cups stationary={other_cups_stationary}: {other_cup_states}", flush=True)
            if not lifted:
                print(
                    f"  Gripper position after close: {float(sim.pull_status().gripper.pos):.4f}",
                    flush=True,
                )
            return bool(lifted and other_cups_stationary), verification
        except Exception as exc:
            gripper_closed = self._verify_gripper_closed(sim)
            verification.update(
                {
                    "verification_method": "gripper_fallback",
                    "verification_error": str(exc),
                    "cup_z_after_lift": None,
                    "gripper_pos_after_close": float(sim.pull_status().gripper.pos),
                }
            )
            print(
                "WARNING: Cannot verify cup lift from MuJoCo body pose; "
                f"falling back to gripper check ({exc})",
                flush=True,
            )
            return bool(gripper_closed), verification

    @staticmethod
    def _verify_gripper_closed(sim) -> bool:
        """Fallback verification when the target body cannot be found."""
        return float(sim.pull_status().gripper.pos) < 0.05

    def _diagnose_gripper_vs_cup(self, sim) -> dict[str, object]:
        """Capture gripper and cup world poses right before closing."""
        scene_objects = sim.pull_scene_objects()
        cup_body_name = self.context.scene_config.cup_body_name
        cup_entry = scene_objects.get(cup_body_name, {})
        cup_pos = np.asarray(cup_entry.get("xpos", []), dtype=float) if isinstance(cup_entry, dict) else np.array([])
        gripper_info = scene_objects.get("gripper_diagnostic", {})

        diagnostic: dict[str, object] = {
            "cup_body_name": cup_body_name,
            "cup_world_pos": cup_pos.tolist() if cup_pos.size == 3 else None,
            "cup_z": float(cup_pos[2]) if cup_pos.size == 3 else None,
            "cup_top_z": float(cup_pos[2] + self.context.scene_config.cup_height_m / 2.0) if cup_pos.size == 3 else None,
            "cup_bottom_z": float(cup_pos[2] - self.context.scene_config.cup_height_m / 2.0) if cup_pos.size == 3 else None,
            "gripper_bodies_and_geoms": gripper_info,
        }
        print("\n=== GRIPPER vs CUP DIAGNOSTIC (at close time) ===", flush=True)
        if cup_pos.size == 3:
            print(f"  Cup position:  {[round(float(v), 4) for v in cup_pos.tolist()]}", flush=True)
            print(f"  Cup top z:     {float(cup_pos[2]) + self.context.scene_config.cup_height_m / 2.0:.4f}", flush=True)
            print(f"  Cup bottom z:  {float(cup_pos[2]) - self.context.scene_config.cup_height_m / 2.0:.4f}", flush=True)
        else:
            print("  Cup: not found", flush=True)

        if isinstance(gripper_info, dict) and cup_pos.size == 3:
            for key in sorted(gripper_info.keys()):
                val = gripper_info[key]
                pos = None
                if isinstance(val, dict):
                    pos = val.get("pos")
                elif isinstance(val, list):
                    pos = val
                if pos is None:
                    continue
                pos_arr = np.asarray(pos, dtype=float)
                if pos_arr.size != 3:
                    continue
                dist = float(np.linalg.norm(pos_arr - cup_pos))
                dz = float(pos_arr[2] - cup_pos[2])
                diagnostic[f"distance_{key}_to_cup"] = round(dist, 4)
                diagnostic[f"dz_{key}_minus_cup"] = round(dz, 4)
                print(
                    f"  {key:40s} pos={[round(float(v), 4) for v in pos_arr.tolist()]}  "
                    f"dist_to_cup={dist:.4f}m  dz={dz:+.4f}m",
                    flush=True,
                )
        print("=" * 55, flush=True)
        return diagnostic

    def _make_single_cup_oracle_candidate(self) -> GraspCandidate:
        """Create the current known-geometry single-cup fallback grasp."""
        scene = self.context.scene_config
        cup_x, cup_y, _ = scene.cup_position_m
        grasp_z = scene.table_top_z_m + scene.cup_height_m * self.context.grasp_config.oracle_grasp_height_ratio
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        y_axis = np.array([0.0, -1.0, 0.0], dtype=float)
        z_axis = np.array([0.0, 0.0, -1.0], dtype=float)
        oracle_pose = pose_from_axes(np.array([cup_x, cup_y, grasp_z], dtype=float), x_axis, y_axis, z_axis)
        return GraspCandidate(
            pose_4x4=oracle_pose,
            score=0.92,
            width_m=min(scene.cup_radius_m * 2.0 * 0.9, self.context.grasp_config.max_gripper_width_m),
            source="scene_oracle_single_cup",
            preferred_approach="side",
            needs_wrist_refinement=False,
            approach_type="side",
            metadata={
                "object_class": "cup",
                "grasp_style": "side_midline",
                "preferred_approach": "side",
                "approach_type": "side",
                "needs_wrist_refinement": False,
                "preferred_base_rotate_rad": 0.0,
                "preferred_wrist_yaw_rad": 0.0,
                "preferred_wrist_pitch_rad": self.context.grasp_config.oracle_side_grasp_wrist_pitch_rad,
            },
        )

    def _make_geometric_candidate(self, geometric_grasp: dict[str, object]) -> GraspCandidate:
        """Create a synthetic candidate wrapper for a geometry-derived top-down grasp."""
        grasp_x = float(geometric_grasp["grasp_x"])
        grasp_y = float(geometric_grasp["grasp_y"])
        grasp_z = float(geometric_grasp["grasp_z"])
        grip_angle = float(geometric_grasp.get("grip_angle_rad", 0.0))
        x_axis = np.array([np.cos(grip_angle), np.sin(grip_angle), 0.0], dtype=float)
        y_axis = np.array([np.sin(grip_angle), -np.cos(grip_angle), 0.0], dtype=float)
        z_axis = np.array([0.0, 0.0, -1.0], dtype=float)
        geometric_pose = pose_from_axes(
            np.array([grasp_x, grasp_y, grasp_z], dtype=float),
            x_axis,
            y_axis,
            z_axis,
        )
        return GraspCandidate(
            pose_4x4=geometric_pose,
            score=0.95,
            width_m=float(geometric_grasp["gripper_open_width"]),
            source="geometric_point_cloud",
            preferred_approach="top_down",
            needs_wrist_refinement=False,
            approach_type="top_down",
            metadata={
                "object_class": "unknown",
                "grasp_style": "top_down_geometric",
                "preferred_approach": "top_down",
                "approach_type": "top_down",
                "needs_wrist_refinement": False,
                "grip_angle_rad": grip_angle,
                "geometric_grasp": dict(geometric_grasp),
            },
        )

    def _compute_geometric_candidate_from_point_cloud(
        self,
        point_cloud,
        head_observation,
        target_bbox_2d: tuple[int, int, int, int] | None,
        artifacts_dir: Path,
        cgn_debug_dir: Path,
        *,
        target_center_xy_hint: np.ndarray | None = None,
    ) -> tuple[dict[str, object], GraspCandidate, dict[str, object]]:
        crop_height_min = self.context.scene_config.table_top_z_m + 0.002
        crop_height_max = self.context.scene_config.table_top_z_m + self.context.scene_config.cup_height_m + 0.03
        geometric_crop_radius = min(
            self.context.grasp_config.cgn_crop_radius_m,
            self.context.scene_config.cup_radius_m + 0.025,
        )
        tabletop_geometry = estimate_tabletop_object_geometry(
            depth_image=head_observation.depth_image,
            camera_intrinsics=head_observation.camera_intrinsics,
            camera_extrinsics=head_observation.camera_extrinsics,
            table_top_z_m=self.context.scene_config.table_top_z_m,
            object_bbox=target_bbox_2d,
        )
        tabletop_base_center = np.asarray(tabletop_geometry["base_center_world"], dtype=float)
        tabletop_center_xy = (
            tabletop_base_center[:2].copy()
            if np.all(np.isfinite(tabletop_base_center[:2]))
            else None
        )
        target_center_xy = self.context.runtime_metadata.get("target_center_xy")
        crop_hint = target_center_xy_hint
        if isinstance(target_center_xy, (list, tuple)) and len(target_center_xy) == 2:
            crop_hint = np.asarray(target_center_xy, dtype=float)
        crop_center_xy = np.asarray(crop_hint, dtype=float) if crop_hint is not None else None
        if crop_center_xy is None and target_bbox_2d is not None and tabletop_center_xy is not None:
            # When a 2D target region is available, the plane-constrained support-point
            # estimate is a stable crop seed. Without bbox/mask support yet, keep the
            # legacy 3D crop-center fallback so the final grasp-center policy remains unchanged.
            crop_center_xy = tabletop_center_xy
        cropped_world, crop_mask, crop_center_xy = self.point_cloud_gen.crop_to_object_region(
            point_cloud.world_points_xyz,
            object_center_xy=np.asarray(crop_center_xy, dtype=float) if crop_center_xy is not None else None,
            crop_radius=geometric_crop_radius,
            table_z=self.context.scene_config.table_top_z_m,
            height_min=crop_height_min,
            height_max=crop_height_max,
            return_mask=True,
        )
        cropped_camera = point_cloud.camera_points_xyz[crop_mask]
        save_point_cloud(cropped_world, artifacts_dir / "cropped_cloud.ply")
        save_point_cloud(cropped_world, cgn_debug_dir / "cropped_cloud.ply")
        save_point_cloud(cropped_camera, artifacts_dir / "cropped_cloud_camera.ply")
        save_point_cloud(cropped_camera, cgn_debug_dir / "cropped_cloud_camera.ply")
        save_point_cloud(cropped_world, artifacts_dir / "geometric_input_cloud.ply")
        save_point_cloud(cropped_world, cgn_debug_dir / "geometric_input_cloud.ply")

        geometric_grasp = compute_geometric_grasp(
            cropped_world,
            table_z=self.context.scene_config.table_top_z_m,
            max_gripper_width_m=self.context.grasp_config.max_gripper_width_m,
        )
        geometric_grasp["tabletop_geometry"] = {
            "base_center_world": tabletop_geometry["base_center_world"],
            "top_center_world": tabletop_geometry["top_center_world"],
            "object_center_world": tabletop_geometry["object_center_world"],
            "height_m": tabletop_geometry["height_m"],
            "footprint_xy": tabletop_geometry["footprint_xy"],
            "support_pixel_median_uv": tabletop_geometry["support_pixel_median_uv"],
            "quality": tabletop_geometry["quality"],
        }
        save_geometric_grasp_debug(cropped_world, geometric_grasp, artifacts_dir / "geometric_grasp_debug.png")
        save_geometric_grasp_debug(cropped_world, geometric_grasp, cgn_debug_dir / "geometric_grasp_debug.png")

        diagnostics_update = {
            "method": "geometric_point_cloud",
            "crop_center_xy": np.asarray(crop_center_xy, dtype=float).tolist(),
            "tabletop_base_center_world": tabletop_geometry["base_center_world"],
            "tabletop_top_center_world": tabletop_geometry["top_center_world"],
            "tabletop_object_center_world": tabletop_geometry["object_center_world"],
            "tabletop_height_m": float(tabletop_geometry["height_m"]),
            "tabletop_footprint_xy": tabletop_geometry["footprint_xy"],
            "tabletop_support_pixel_median_uv": tabletop_geometry["support_pixel_median_uv"],
            "tabletop_quality": tabletop_geometry["quality"],
            "tabletop_plane": tabletop_geometry["table_plane"],
            "crop_radius": float(geometric_crop_radius),
            "crop_seed_source": (
                "hint"
                if crop_hint is not None
                else ("tabletop_support_estimate" if target_bbox_2d is not None and tabletop_center_xy is not None else "legacy_elevated_point_fallback")
            ),
            "target_center_xy": list(map(float, target_center_xy)) if target_center_xy is not None else None,
            "crop_center_hint_xy": (
                np.asarray(crop_hint, dtype=float).tolist() if crop_hint is not None else None
            ),
            "multi_cup_mode": bool(self.context.runtime_metadata.get("multi_cup", False)),
            "object_center_xy": np.asarray(geometric_grasp["object_center"][:2], dtype=float).tolist(),
            "full_cloud_points": int(point_cloud.world_points_xyz.shape[0]),
            "cropped_cloud_points": int(cropped_world.shape[0]),
            "cgn_input_points": int(cropped_world.shape[0]),
            "object_center": geometric_grasp["object_center"],
            "object_height": float(geometric_grasp["object_height"]),
            "object_top_z": float(geometric_grasp["object_top_z"]),
            "object_bottom_z": float(geometric_grasp["object_bottom_z"]),
            "grasp_z": float(geometric_grasp["grasp_z"]),
            "min_cross_section_width": float(geometric_grasp["min_cross_section_width"]),
            "grip_angle_deg": float(np.degrees(float(geometric_grasp["grip_angle_rad"]))),
            "gripper_open_width": float(geometric_grasp["gripper_open_width"]),
            "estimated_error": float(geometric_grasp.get("estimated_error", 0.0)),
            "residual_std": float(geometric_grasp.get("residual_std", 0.0)),
            "arc_coverage_deg": float(geometric_grasp.get("arc_coverage_deg", 0.0)),
            "conservative_diameter": float(geometric_grasp.get("conservative_diameter", 0.0)),
            "uncertainty_margin": float(geometric_grasp.get("uncertainty_margin", 0.0)),
            "fixed_clearance_margin": float(geometric_grasp.get("fixed_clearance_margin", 0.0)),
            "grasp_point_validated": bool(geometric_grasp["grasp_point_validated"]),
            "width_near_limit": bool(geometric_grasp.get("width_near_limit", False)),
            "open_width_exceeds_max": bool(geometric_grasp.get("open_width_exceeds_max", False)),
        }
        return geometric_grasp, self._make_geometric_candidate(geometric_grasp), diagnostics_update

    def _reobserve_geometric_after_base_motion(
        self,
        *,
        sim,
        head_observation,
        point_cloud,
        geometric_grasp: dict[str, object],
        selected: GraspCandidate,
        geometric_diagnostics: dict[str, object],
        target_bbox_2d: tuple[int, int, int, int] | None,
        artifacts_dir: Path,
        cgn_debug_dir: Path,
        stage_times_s: dict[str, float],
        stage_prefix: str,
    ) -> tuple[object, object, dict[str, object], GraspCandidate, dict[str, object], dict[str, object]]:
        previous_geometric_grasp = dict(geometric_grasp)
        previous_selected = selected
        previous_geometric_diagnostics = dict(geometric_diagnostics)
        meta: dict[str, object] = {
            "accepted": False,
            "pre_reobserve_object_center": list(
                map(
                    float,
                    geometric_grasp.get(
                        "object_center",
                        [geometric_grasp["grasp_x"], geometric_grasp["grasp_y"], geometric_grasp["grasp_z"]],
                    ),
                )
            ),
        }
        try:
            stage_started = time.perf_counter()
            head_observation = self.head_aligner.align_and_capture(
                sim,
                target_center_xy=np.asarray(previous_geometric_grasp["object_center"][:2], dtype=float),
            )
            stage_times_s[f"head_alignment_{stage_prefix}"] = round(time.perf_counter() - stage_started, 4)

            stage_started = time.perf_counter()
            self._save_head_artifacts(head_observation.rgb_image, head_observation.depth_image, artifacts_dir)
            stage_times_s[f"save_head_artifacts_{stage_prefix}"] = round(time.perf_counter() - stage_started, 4)

            stage_started = time.perf_counter()
            reobserved_point_cloud = self.point_cloud_gen.generate(
                depth_image=head_observation.depth_image,
                camera_intrinsics=head_observation.camera_intrinsics,
                camera_extrinsics=head_observation.camera_extrinsics,
                table_top_z_m=self.context.scene_config.table_top_z_m,
                table_margin_m=self.context.scene_config.table_clearance_margin_m,
                z_min_m=self.context.grasp_config.z_min_m,
                z_max_m=min(
                    self.context.grasp_config.z_max_m,
                    self.context.scene_config.table_top_z_m + 0.20,
                ),
                target_bbox_2d=target_bbox_2d,
            )
            stage_times_s[f"point_cloud_generation_{stage_prefix}"] = round(time.perf_counter() - stage_started, 4)

            if len(reobserved_point_cloud.world_points_xyz) == 0:
                raise RuntimeError("reobserved point cloud was empty")

            stage_started = time.perf_counter()
            save_point_cloud(reobserved_point_cloud.world_points_xyz, artifacts_dir / "scene_cloud.ply")
            save_point_cloud(reobserved_point_cloud.camera_points_xyz, artifacts_dir / "scene_cloud_camera.ply")
            save_point_cloud(reobserved_point_cloud.world_points_xyz, Path("/tmp/ask2act_scene.ply"))
            save_point_cloud(reobserved_point_cloud.camera_points_xyz, Path("/tmp/ask2act_scene_camera.ply"))
            save_point_cloud(reobserved_point_cloud.world_points_xyz, artifacts_dir / "full_cloud.ply")
            save_point_cloud(reobserved_point_cloud.world_points_xyz, cgn_debug_dir / "full_cloud.ply")
            save_point_cloud(reobserved_point_cloud.camera_points_xyz, artifacts_dir / "full_cloud_camera.ply")
            save_point_cloud(reobserved_point_cloud.camera_points_xyz, cgn_debug_dir / "full_cloud_camera.ply")
            stage_times_s[f"save_point_cloud_artifacts_{stage_prefix}"] = round(
                time.perf_counter() - stage_started,
                4,
            )

            stage_started = time.perf_counter()
            reobserved_grasp, reobserved_selected, reobserved_diagnostics = self._compute_geometric_candidate_from_point_cloud(
                reobserved_point_cloud,
                head_observation,
                target_bbox_2d,
                artifacts_dir,
                cgn_debug_dir,
                target_center_xy_hint=np.asarray(previous_geometric_grasp["object_center"][:2], dtype=float),
            )
            stage_times_s[f"geometric_perception_{stage_prefix}"] = round(time.perf_counter() - stage_started, 4)
            candidate_exact_target_reachable = True
            try:
                self.motion_planner.geometric_grasp_targets(
                    reobserved_grasp,
                    current_state=self._read_robot_state(sim),
                )
            except RuntimeError as exc:
                candidate_exact_target_reachable = False
                meta["candidate_unreachable_error"] = str(exc)
            meta["candidate_exact_target_reachable"] = candidate_exact_target_reachable
            accept_reobserve, reobserve_reason, reobserve_center_shift = self._should_accept_reobserved_geometric(
                previous_geometric_diagnostics,
                reobserved_diagnostics,
            )
            if not candidate_exact_target_reachable:
                accept_reobserve = False
                reobserve_reason = "reobserved_candidate_simpleik_unreachable"
            meta["center_shift_m"] = reobserve_center_shift
            meta["accept_reason"] = reobserve_reason
            meta["candidate_diagnostics"] = reobserved_diagnostics
            if accept_reobserve:
                meta["accepted"] = True
                return (
                    head_observation,
                    reobserved_point_cloud,
                    reobserved_grasp,
                    reobserved_selected,
                    reobserved_diagnostics,
                    meta,
                )
            meta["rejection_reason"] = reobserve_reason
        except Exception as exc:
            meta["error"] = str(exc)

        return (
            head_observation,
            point_cloud,
            previous_geometric_grasp,
            previous_selected,
            previous_geometric_diagnostics,
            meta,
        )

    @staticmethod
    def _should_reobserve_after_base_motion(base_motion_diagnostic: dict[str, object] | None) -> bool:
        if not isinstance(base_motion_diagnostic, dict):
            return False
        if not bool(base_motion_diagnostic.get("performed", False)):
            return False
        base_before = base_motion_diagnostic.get("base_pose_before")
        base_after = base_motion_diagnostic.get("base_pose_after")
        if not (
            isinstance(base_before, list)
            and len(base_before) == 3
            and isinstance(base_after, list)
            and len(base_after) == 3
        ):
            return False
        before = np.asarray(base_before, dtype=float)
        after = np.asarray(base_after, dtype=float)
        moved_xy = float(np.linalg.norm(after[:2] - before[:2]))
        rotated = abs(float((after[2] - before[2] + math.pi) % (2.0 * math.pi) - math.pi))
        return moved_xy > 0.02 or rotated > 0.12

    def _should_accept_reobserved_geometric(
        self,
        previous_diagnostics: dict[str, object],
        candidate_diagnostics: dict[str, object],
    ) -> tuple[bool, str, float]:
        prev_center = np.asarray(previous_diagnostics.get("object_center_xy", [0.0, 0.0]), dtype=float)
        cand_center = np.asarray(candidate_diagnostics.get("object_center_xy", [0.0, 0.0]), dtype=float)
        center_shift = float(np.linalg.norm(cand_center - prev_center))
        prev_points = int(previous_diagnostics.get("cropped_cloud_points", 0))
        cand_points = int(candidate_diagnostics.get("cropped_cloud_points", 0))
        prev_height = float(previous_diagnostics.get("object_height", 0.0))
        cand_height = float(candidate_diagnostics.get("object_height", 0.0))
        prev_bottom_z = float(previous_diagnostics.get("object_bottom_z", self.context.scene_config.table_top_z_m))
        cand_bottom_z = float(candidate_diagnostics.get("object_bottom_z", self.context.scene_config.table_top_z_m))
        expected_height = float(self.context.scene_config.cup_height_m)
        expected_bottom_z = float(self.context.scene_config.table_top_z_m)
        prev_height_error = abs(prev_height - expected_height)
        cand_height_error = abs(cand_height - expected_height)
        prev_bottom_error = abs(prev_bottom_z - expected_bottom_z)
        cand_bottom_error = abs(cand_bottom_z - expected_bottom_z)

        if cand_points < max(30, int(prev_points * 0.35)):
            return False, "too_few_points_after_reobserve", center_shift
        if expected_height > 0.0 and cand_height < expected_height * 0.6 and cand_height_error > prev_height_error:
            return False, "reobserved_height_implausibly_short", center_shift
        if cand_bottom_error > max(0.02, prev_bottom_error + 0.02):
            return False, "reobserved_bottom_too_far_above_table", center_shift
        if center_shift > 0.10 and cand_height_error >= prev_height_error:
            return False, "reobserved_center_shift_too_large", center_shift
        if cand_height_error > prev_height_error + 0.015 and center_shift > 0.04:
            return False, "reobserved_height_became_less_plausible", center_shift
        return True, "accepted", center_shift

    def _save_head_artifacts(self, rgb_image: np.ndarray, depth_image: np.ndarray, artifacts_dir: Path) -> None:
        """Persist the captured head images for debugging."""
        self._write_image(artifacts_dir / "head_rgb.png", rgb_image)
        np.save(artifacts_dir / "head_depth.npy", depth_image)
        self._write_image(Path("/tmp/ask2act_head_rgb.png"), rgb_image)
        np.save("/tmp/ask2act_head_depth.npy", depth_image)

    def _save_grasp_artifacts(
        self,
        artifacts_dir: Path,
        points_xyz: np.ndarray,
        candidates: list[GraspCandidate],
        selected: GraspCandidate | None,
    ) -> None:
        """Persist grasp candidates, selection results, and a small visualization scene."""
        all_grasps_payload = ContactGraspNetGenerator.candidates_to_jsonable(candidates)
        (artifacts_dir / "all_grasps.json").write_text(json.dumps(all_grasps_payload, indent=2))
        save_grasp_debug_scene(points_xyz, candidates[:5], artifacts_dir / "grasp_visualization.ply")
        if selected is not None:
            selected_payload = ContactGraspNetGenerator.candidates_to_jsonable([selected])[0]
            (artifacts_dir / "selected_grasp.json").write_text(json.dumps(selected_payload, indent=2))

    def _write_result(self, result: PipelineResult) -> None:
        """Write both the legacy result file and the new pipeline log artifact."""
        payload = json.dumps(result.__dict__, indent=2, default=_json_default)
        (self.context.run_dir / "pipeline_result.json").write_text(payload)
        (self.context.run_dir / "pipeline_log.json").write_text(payload)

    @staticmethod
    def _write_image(path: Path, rgb_image: np.ndarray) -> None:
        """Write an RGB image if OpenCV is available."""
        try:
            import cv2

            cv2.imwrite(str(path), cv2.cvtColor(np.asarray(rgb_image, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        except Exception:
            np.save(path.with_suffix(".npy"), np.asarray(rgb_image, dtype=np.uint8))


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
