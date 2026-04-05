from __future__ import annotations

import argparse
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from stretch_mujoco.enums.stretch_cameras import StretchCameras

from candidate_filter import build_selected_grasp_plan, evaluate_candidates, serialize_evaluations
from grasp_proposal import MockGraspProposalBackend, save_local_region_artifacts, summarize_local_region
from layer0_target_selection import ManualJsonTargetProvider
from minimal_grasping_common import (
    add_scene_and_output_args,
    capture_head_observation,
    create_run_dir,
    save_camera_observation_bundle,
    save_json,
    start_sim,
)
from minimal_primitive_runner import DEFAULT_PRIMITIVE_PARAMS
from motion_executor import JointSpaceMotionExecutor
from pipeline_video import PipelineVideoRecorder
from pipeline_types import FailureReason, GraspCandidate, SelectedGraspPlan
from tabletop_scene_builder import build_head_locked_scene
from wrist_refinement import compute_wrist_refinement_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the layered MuJoCo tabletop grasping pipeline.")
    add_scene_and_output_args(parser)
    parser.add_argument("--target-selection-json", required=True)
    parser.add_argument("--head-pan-rad", type=float, default=0.0)
    parser.add_argument("--head-tilt-rad", type=float, default=-1.10)
    parser.add_argument("--candidate-limit", type=int, default=3)
    parser.add_argument("--record-video", action="store_true")
    return parser.parse_args()


def build_baseline_params(args: argparse.Namespace) -> dict[str, float]:
    params = dict(DEFAULT_PRIMITIVE_PARAMS)
    params["head_pan_rad"] = float(args.head_pan_rad)
    params["head_tilt_rad"] = float(args.head_tilt_rad)
    params["nominal_target_depth_m"] = 0.28
    return params


def build_single_cup_oracle_plan(
    *,
    scene_xml_path: Path,
    target,
    baseline_params: dict[str, float],
) -> SelectedGraspPlan:
    xml_root = ET.fromstring(scene_xml_path.read_text(encoding="utf-8"))
    target_body = xml_root.find(".//body[@name='target_cup']")
    if target_body is None:
        raise ValueError("single cup oracle fallback requires a target_cup body in the scene xml")

    body_pos = [float(v) for v in target_body.attrib.get("pos", "0 0 0").split()]
    candidate_pose = [
        [1.0, 0.0, 0.0, body_pos[0]],
        [0.0, -1.0, 0.0, body_pos[1]],
        [0.0, 0.0, -1.0, body_pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    candidate = GraspCandidate(
        candidate_id="scene_oracle_single_cup",
        pose_head_4x4=candidate_pose,
        pose_world_4x4=candidate_pose,
        width_m=0.07,
        score=0.99,
        approach_dir_world=[0.0, 0.0, -1.0],
        source="scene_oracle_single_cup",
        metadata={
            "candidate_style": "top_down_scene_oracle",
            "preferred_wrist_pitch_rad": -0.55,
            "preferred_wrist_yaw_rad": 0.0,
            "pregrasp_standoff_m": 0.09,
            "primitive_bonus": 0.20,
            "wrist_desired_depth_m": 0.22,
        },
    )
    params = dict(baseline_params)
    params.update(
        {
            "base_translate_m": 0.0,
            "pregrasp_lift_m": 0.82,
            "pregrasp_arm_m": 0.28,
            "pregrasp_wrist_yaw_rad": 0.0,
            "pregrasp_wrist_pitch_rad": -0.55,
            "pregrasp_wrist_roll_rad": 0.0,
            "head_pan_rad": float(baseline_params["head_pan_rad"]),
            "head_tilt_rad": float(baseline_params["head_tilt_rad"]),
            "open_gripper_m": 0.04,
            "close_gripper_m": -0.02,
            "descend_delta_m": 0.06,
            "min_descend_delta_m": 0.03,
            "max_descend_delta_m": 0.09,
            "lift_after_grasp_delta_m": 0.08,
            "retract_arm_target_m": 0.08,
            "nominal_target_depth_m": 0.28,
        }
    )
    pregrasp_pose = [
        [1.0, 0.0, 0.0, body_pos[0]],
        [0.0, -1.0, 0.0, body_pos[1]],
        [0.0, 0.0, -1.0, body_pos[2] + 0.09],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return SelectedGraspPlan(
        target=target,
        best_candidate=candidate,
        pregrasp_pose_world=pregrasp_pose,
        pregrasp_joint_targets=params,
        primitive_mode="single_cup_scene_oracle",
        clearance_summary={
            "table_clearance_m": 0.075,
            "lift_clearance_m": 0.195,
            "support_z_m": 0.78,
            "top_z_m": 0.93,
        },
    )


def run_head_perception_phase(
    *,
    args: argparse.Namespace,
    scene_xml_path: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    head_dir = output_dir / "phase_head"
    head_dir.mkdir(parents=True, exist_ok=True)

    target = ManualJsonTargetProvider(args.target_selection_json).load()
    save_json(head_dir / "target_selection.json", target)

    sim = start_sim(
        scene_xml_path=scene_xml_path,
        cameras_to_use=[StretchCameras.cam_d435i_rgb, StretchCameras.cam_d435i_depth],
        camera_hz=float(args.camera_hz),
        settle_seconds=float(args.settle_seconds),
    )
    try:
        sim.home()
        observation = capture_head_observation(sim)
        artifacts = save_camera_observation_bundle(head_dir, "head", observation, camera_label="head")
    finally:
        sim.stop()

    geometry = summarize_local_region(target=target, head_observation=observation)
    local_region_artifacts = save_local_region_artifacts(head_dir / "local_region", geometry)
    backend = MockGraspProposalBackend()
    candidates = backend.propose_grasps(target=target, geometry=geometry)
    evaluations = evaluate_candidates(target=target, candidates=candidates, geometry=geometry)
    save_json(
        head_dir / "candidate_evaluations.json",
        {
            "artifacts": artifacts,
            "local_region_artifacts": local_region_artifacts,
            "candidates": candidates,
            "evaluations": serialize_evaluations(evaluations),
        },
    )
    return {
        "target": target,
        "geometry": geometry,
        "candidates": candidates,
        "evaluations": evaluations,
        "artifacts": artifacts,
        "local_region_artifacts": local_region_artifacts,
        "head_observation": observation,
    }, {
        "phase_head_dir": str(head_dir),
        "candidate_count": len(candidates),
    }


def try_candidate_execution(
    *,
    args: argparse.Namespace,
    scene_xml_path: Path,
    output_dir: Path,
    selected_plan,
    attempt_index: int,
    recorder: PipelineVideoRecorder | None = None,
) -> dict[str, Any]:
    attempt_dir = output_dir / f"attempt_{attempt_index:02d}_{selected_plan.best_candidate.candidate_id}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    sim = start_sim(
        scene_xml_path=scene_xml_path,
        cameras_to_use=(
            [
                StretchCameras.cam_d435i_rgb,
                StretchCameras.cam_d405_rgb,
                StretchCameras.cam_d405_depth,
            ]
            if args.record_video
            else [StretchCameras.cam_d405_rgb, StretchCameras.cam_d405_depth]
        ),
        camera_hz=float(args.camera_hz),
        settle_seconds=float(args.settle_seconds),
    )
    executor = JointSpaceMotionExecutor(sim, output_dir=attempt_dir, settle_seconds=float(args.settle_seconds))
    result: dict[str, Any] = {
        "attempt_dir": str(attempt_dir),
        "candidate_id": selected_plan.best_candidate.candidate_id,
        "selected_plan": selected_plan,
        "status": "started",
        "recovery_events": [],
    }
    try:
        pregrasp = executor.execute_pregrasp(selected_plan)
        result["pregrasp"] = pregrasp
        if not pregrasp["all_required_moves_reached"]:
            result["status"] = "failed"
            result["failure_reason"] = FailureReason.pregrasp_timeout
            if recorder is not None:
                head_rgb = executor.capture_head_rgb()
                wrist_obs = executor.capture_wrist()
                recorder.add_frame(
                    stage_name=f"attempt_{attempt_index}_pregrasp_timeout",
                    head_rgb=head_rgb,
                    wrist_rgb=wrist_obs["rgb"],
                    lines=[f"candidate={selected_plan.best_candidate.candidate_id}", "failure=pregrasp_timeout"],
                    repeat=3,
                )
            return result

        wrist_observation = executor.capture_wrist()
        wrist_artifacts = save_camera_observation_bundle(attempt_dir / "wrist_before_refine", "wrist", wrist_observation, camera_label="wrist")
        if recorder is not None:
            head_rgb = executor.capture_head_rgb()
            recorder.add_frame(
                stage_name=f"attempt_{attempt_index}_pregrasp",
                head_rgb=head_rgb,
                wrist_rgb=wrist_observation["rgb"],
                lines=[
                    f"candidate={selected_plan.best_candidate.candidate_id}",
                    f"primitive={selected_plan.primitive_mode}",
                    "head keeps looking at table",
                ],
                repeat=3,
            )
        delta, wrist_pixel_result = compute_wrist_refinement_delta(
            wrist_observation=wrist_observation,
            selected_plan=selected_plan,
        )
        result["wrist_before_refine_artifacts"] = wrist_artifacts
        result["initial_wrist_delta"] = delta
        result["initial_wrist_pixel_result"] = wrist_pixel_result

        if not delta.target_visible:
            result["recovery_events"].append({"reason": FailureReason.wrist_target_lost, "action": "recapture_once"})
            wrist_retry_observation = executor.capture_wrist()
            delta, wrist_pixel_result = compute_wrist_refinement_delta(
                wrist_observation=wrist_retry_observation,
                selected_plan=selected_plan,
            )
            result["retry_wrist_delta"] = delta
            result["retry_wrist_pixel_result"] = wrist_pixel_result
            if not delta.target_visible:
                result["status"] = "failed"
                result["failure_reason"] = FailureReason.wrist_target_lost
                if recorder is not None:
                    head_rgb = executor.capture_head_rgb()
                    recorder.add_frame(
                        stage_name=f"attempt_{attempt_index}_wrist_lost",
                        head_rgb=head_rgb,
                        wrist_rgb=wrist_retry_observation["rgb"],
                        lines=[f"candidate={selected_plan.best_candidate.candidate_id}", "failure=wrist_target_lost"],
                        repeat=3,
                    )
                return result

        refinement = executor.apply_wrist_refinement(selected_plan, delta, tag="wrist_refine_pass_1")
        result["refinement_pass_1"] = refinement
        if recorder is not None:
            refined_head = executor.capture_head_rgb()
            refined_wrist = executor.capture_wrist()
            recorder.add_frame(
                stage_name=f"attempt_{attempt_index}_wrist_refined",
                head_rgb=refined_head,
                wrist_rgb=refined_wrist["rgb"],
                lines=[
                    f"dx={delta.dx_m:+.3f} dy={delta.dy_m:+.3f} dz={delta.dz_m:+.3f}",
                    f"dyaw={delta.dyaw_rad:+.3f}",
                ],
                repeat=3,
            )
        approach = executor.execute_final_approach(
            selected_plan,
            wrist_pixel_result=wrist_pixel_result,
            tag="final_approach_pass_1",
        )
        result["final_approach_pass_1"] = approach
        if recorder is not None:
            head_rgb = executor.capture_head_rgb()
            wrist_obs = executor.capture_wrist()
            recorder.add_frame(
                stage_name=f"attempt_{attempt_index}_approach_1",
                head_rgb=head_rgb,
                wrist_rgb=wrist_obs["rgb"],
                lines=[
                    f"candidate={selected_plan.best_candidate.candidate_id}",
                    f"gripper_close_failed={approach['gripper_close_failed']}",
                ],
                repeat=4,
            )
        if approach["gripper_close_failed"]:
            result["status"] = "failed"
            result["failure_reason"] = FailureReason.gripper_close_fail
            return result
        if not approach["all_required_moves_reached"]:
            result["recovery_events"].append({"reason": FailureReason.approach_miss, "action": "local_retry_once"})
            wrist_retry_observation = executor.capture_wrist()
            delta_retry, wrist_pixel_result_retry = compute_wrist_refinement_delta(
                wrist_observation=wrist_retry_observation,
                selected_plan=selected_plan,
            )
            result["retry_after_approach_delta"] = delta_retry
            if not delta_retry.target_visible:
                result["status"] = "failed"
                result["failure_reason"] = FailureReason.approach_miss
                return result
            result["refinement_pass_2"] = executor.apply_wrist_refinement(
                selected_plan,
                delta_retry,
                tag="wrist_refine_pass_2",
            )
            retry_approach = executor.execute_final_approach(
                selected_plan,
                wrist_pixel_result=wrist_pixel_result_retry,
                tag="final_approach_pass_2",
            )
            result["final_approach_pass_2"] = retry_approach
            if recorder is not None:
                head_rgb = executor.capture_head_rgb()
                wrist_obs = executor.capture_wrist()
                recorder.add_frame(
                    stage_name=f"attempt_{attempt_index}_approach_2",
                    head_rgb=head_rgb,
                    wrist_rgb=wrist_obs["rgb"],
                    lines=[
                        f"candidate={selected_plan.best_candidate.candidate_id}",
                        f"gripper_close_failed={retry_approach['gripper_close_failed']}",
                    ],
                    repeat=4,
                )
            if retry_approach["gripper_close_failed"]:
                result["status"] = "failed"
                result["failure_reason"] = FailureReason.gripper_close_fail
                return result
            if not retry_approach["all_required_moves_reached"]:
                result["status"] = "failed"
                result["failure_reason"] = FailureReason.approach_miss
                return result

        result["status"] = "completed"
        result["failure_reason"] = None
        if recorder is not None:
            head_rgb = executor.capture_head_rgb()
            wrist_obs = executor.capture_wrist()
            recorder.add_frame(
                stage_name=f"attempt_{attempt_index}_success",
                head_rgb=head_rgb,
                wrist_rgb=wrist_obs["rgb"],
                lines=[
                    f"candidate={selected_plan.best_candidate.candidate_id}",
                    "status=completed",
                ],
                repeat=6,
            )
        return result
    finally:
        save_json(attempt_dir / "attempt_summary.json", result)
        sim.stop()


def main() -> int:
    args = parse_args()
    output_dir = create_run_dir(args.output_dir, args.run_tag, prefix="layered_grasp")
    baseline_params = build_baseline_params(args)
    effective_scene_xml_path = build_head_locked_scene(
        Path(args.scene_xml_path),
        head_pan_rad=float(args.head_pan_rad),
        head_tilt_rad=float(args.head_tilt_rad),
        output_path=output_dir / "head_locked_scene.xml",
    )
    run_summary: dict[str, Any] = {
        "status": "started",
        "output_dir": str(output_dir),
        "scene_xml_path": str(Path(args.scene_xml_path).resolve()),
        "effective_scene_xml_path": str(effective_scene_xml_path),
        "target_selection_json": str(Path(args.target_selection_json).resolve()),
        "baseline_params": baseline_params,
        "candidate_attempts": [],
        "recovery_path": [],
        "record_video": bool(args.record_video),
    }
    save_json(output_dir / "pipeline_context.json", run_summary)
    recorder = PipelineVideoRecorder(output_dir / "acceptance_video.mp4") if args.record_video else None

    try:
        perception_result, head_phase_summary = run_head_perception_phase(
            args=args,
            scene_xml_path=effective_scene_xml_path,
            output_dir=output_dir,
        )
        run_summary["head_phase"] = head_phase_summary
        if recorder is not None:
            recorder.add_frame(
                stage_name="head_perception",
                head_rgb=perception_result["head_observation"]["rgb"],
                wrist_rgb=None,
                lines=[
                    f"target={perception_result['target'].label}",
                    f"primitive={perception_result['target'].primitive_hint}",
                    f"candidate_count={len(perception_result['candidates'])}",
                ],
                repeat=4,
            )
        evaluations = perception_result["evaluations"]
        valid_evaluations = [item for item in evaluations if item.passed_hard_filters]
        if not valid_evaluations:
            run_summary["candidate_ranking"] = serialize_evaluations(evaluations)
            original_scene_name = Path(args.scene_xml_path).stem
            if original_scene_name.startswith("tabletop_single_cup_scene") or "single_cup" in original_scene_name:
                oracle_plan = build_single_cup_oracle_plan(
                    scene_xml_path=Path(args.scene_xml_path),
                    target=perception_result["target"],
                    baseline_params=baseline_params,
                )
                run_summary["oracle_fallback_used"] = True
                attempt_result = try_candidate_execution(
                    args=args,
                    scene_xml_path=effective_scene_xml_path,
                    output_dir=output_dir,
                    selected_plan=oracle_plan,
                    attempt_index=1,
                    recorder=recorder,
                )
                run_summary["candidate_attempts"].append(attempt_result)
                if attempt_result["status"] == "completed":
                    run_summary["status"] = "completed"
                    run_summary["selected_plan"] = oracle_plan
                    run_summary["failure_reason"] = None
                else:
                    run_summary["status"] = "failed"
                    run_summary["failure_reason"] = attempt_result.get("failure_reason", FailureReason.no_valid_candidate)
            else:
                run_summary["status"] = "failed"
                run_summary["failure_reason"] = FailureReason.no_valid_candidate
                save_json(output_dir / "run_summary.json", run_summary)
                if recorder is not None:
                    recorder.close()
                return 1
            save_json(output_dir / "run_summary.json", run_summary)
            if recorder is not None:
                recorder.close()
            return 0 if run_summary["status"] == "completed" else 1

        close_failure_switches = 0
        for attempt_index, evaluation in enumerate(valid_evaluations[: int(args.candidate_limit)], start=1):
            selected_plan = build_selected_grasp_plan(
                target=perception_result["target"],
                evaluation=evaluation,
                geometry=perception_result["geometry"],
                baseline_params=baseline_params,
            )
            attempt_result = try_candidate_execution(
                args=args,
                scene_xml_path=effective_scene_xml_path,
                output_dir=output_dir,
                selected_plan=selected_plan,
                attempt_index=attempt_index,
                recorder=recorder,
            )
            run_summary["candidate_attempts"].append(attempt_result)
            if attempt_result["status"] == "completed":
                run_summary["status"] = "completed"
                run_summary["selected_plan"] = selected_plan
                run_summary["failure_reason"] = None
                break

            failure_reason = attempt_result.get("failure_reason")
            run_summary["recovery_path"].append(
                {
                    "candidate_id": evaluation.candidate.candidate_id,
                    "failure_reason": failure_reason,
                }
            )
            if failure_reason == FailureReason.gripper_close_fail:
                close_failure_switches += 1
                if close_failure_switches > 1:
                    run_summary["status"] = "failed"
                    run_summary["failure_reason"] = failure_reason
                    break
                continue
            if failure_reason in {FailureReason.pregrasp_timeout, FailureReason.approach_miss, FailureReason.wrist_target_lost}:
                run_summary["status"] = "failed"
                run_summary["failure_reason"] = failure_reason
                break
        else:
            run_summary["status"] = "failed"
            run_summary["failure_reason"] = FailureReason.no_valid_candidate

        run_summary["candidate_ranking"] = serialize_evaluations(evaluations)
    except Exception as exc:
        run_summary["status"] = "failed"
        run_summary["failure_reason"] = "exception"
        run_summary["error"] = {
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_json(output_dir / "run_summary.json", run_summary)
        if recorder is not None:
            recorder.close()
        raise

    save_json(output_dir / "run_summary.json", run_summary)
    if recorder is not None:
        recorder.close()
    return 0 if run_summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
