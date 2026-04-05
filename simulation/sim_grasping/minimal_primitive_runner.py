from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators

from minimal_grasping_common import (
    add_scene_and_output_args,
    create_run_dir,
    heuristic_descend_delta_m,
    move_to_and_record,
    save_json,
    sim_pose_snapshot,
    start_wrist_sim,
)


DEFAULT_PRIMITIVE_PARAMS = {
    "pregrasp_lift_m": 0.80,
    "pregrasp_arm_m": 0.24,
    "pregrasp_wrist_yaw_rad": 0.0,
    "pregrasp_wrist_pitch_rad": -0.55,
    "pregrasp_wrist_roll_rad": 0.0,
    "head_pan_rad": 0.0,
    "head_tilt_rad": -0.55,
    "open_gripper_m": 0.03,
    "close_gripper_m": -0.02,
    "base_descend_delta_m": 0.06,
    "min_descend_delta_m": 0.03,
    "max_descend_delta_m": 0.09,
    "lift_after_grasp_delta_m": 0.08,
    "retract_arm_target_m": 0.08,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a conservative minimal grasping primitive sequence.")
    add_scene_and_output_args(parser)
    parser.add_argument("--pregrasp-lift-m", type=float, default=DEFAULT_PRIMITIVE_PARAMS["pregrasp_lift_m"])
    parser.add_argument("--pregrasp-arm-m", type=float, default=DEFAULT_PRIMITIVE_PARAMS["pregrasp_arm_m"])
    parser.add_argument(
        "--pregrasp-wrist-yaw-rad",
        type=float,
        default=DEFAULT_PRIMITIVE_PARAMS["pregrasp_wrist_yaw_rad"],
    )
    parser.add_argument(
        "--pregrasp-wrist-pitch-rad",
        type=float,
        default=DEFAULT_PRIMITIVE_PARAMS["pregrasp_wrist_pitch_rad"],
    )
    parser.add_argument(
        "--pregrasp-wrist-roll-rad",
        type=float,
        default=DEFAULT_PRIMITIVE_PARAMS["pregrasp_wrist_roll_rad"],
    )
    parser.add_argument("--head-pan-rad", type=float, default=DEFAULT_PRIMITIVE_PARAMS["head_pan_rad"])
    parser.add_argument("--head-tilt-rad", type=float, default=DEFAULT_PRIMITIVE_PARAMS["head_tilt_rad"])
    parser.add_argument("--open-gripper-m", type=float, default=DEFAULT_PRIMITIVE_PARAMS["open_gripper_m"])
    parser.add_argument("--close-gripper-m", type=float, default=DEFAULT_PRIMITIVE_PARAMS["close_gripper_m"])
    parser.add_argument(
        "--descend-delta-m",
        type=float,
        default=DEFAULT_PRIMITIVE_PARAMS["base_descend_delta_m"],
        help="Fallback descend delta when no target pixel result is provided.",
    )
    parser.add_argument(
        "--min-descend-delta-m",
        type=float,
        default=DEFAULT_PRIMITIVE_PARAMS["min_descend_delta_m"],
    )
    parser.add_argument(
        "--max-descend-delta-m",
        type=float,
        default=DEFAULT_PRIMITIVE_PARAMS["max_descend_delta_m"],
    )
    parser.add_argument(
        "--lift-after-grasp-delta-m",
        type=float,
        default=DEFAULT_PRIMITIVE_PARAMS["lift_after_grasp_delta_m"],
    )
    parser.add_argument(
        "--retract-arm-target-m",
        type=float,
        default=DEFAULT_PRIMITIVE_PARAMS["retract_arm_target_m"],
    )
    parser.add_argument("--target-depth-m", type=float, default=None)
    parser.add_argument("--nominal-target-depth-m", type=float, default=0.28)
    parser.add_argument("--skip-home", action="store_true")
    return parser.parse_args()


def compute_descend_delta_m(params: dict[str, float], target_pixel_result: dict[str, Any] | None) -> tuple[float, dict[str, Any]]:
    if target_pixel_result is None:
        descend_delta_m = float(params["descend_delta_m"])
        return descend_delta_m, {"mode": "fixed", "source": "cli_default"}

    depth_m = float(target_pixel_result["depth_m"])
    descend_delta_m = heuristic_descend_delta_m(
        depth_m,
        nominal_target_depth_m=float(params["nominal_target_depth_m"]),
        base_delta_m=float(params["descend_delta_m"]),
        min_delta_m=float(params["min_descend_delta_m"]),
        max_delta_m=float(params["max_descend_delta_m"]),
    )
    return descend_delta_m, {
        "mode": "depth_heuristic",
        "depth_m": depth_m,
        "nominal_target_depth_m": float(params["nominal_target_depth_m"]),
    }


def execute_named_move(
    sim: StretchMujocoSimulator,
    *,
    output_dir: Path,
    name: str,
    actuator: Actuators,
    target: float,
) -> dict[str, Any]:
    before_snapshot = sim_pose_snapshot(sim)
    result = move_to_and_record(sim, actuator, target)
    after_snapshot = sim_pose_snapshot(sim)
    payload = {
        "name": name,
        "type": "move_to",
        "actuator": actuator.name,
        "target": float(target),
        "result": result,
        "before_snapshot": before_snapshot,
        "after_snapshot": after_snapshot,
    }
    save_json(output_dir / f"{name}.json", payload)
    return payload


def execute_home(sim: StretchMujocoSimulator, *, output_dir: Path, settle_seconds: float) -> dict[str, Any]:
    before_snapshot = sim_pose_snapshot(sim)
    sim.home()
    time.sleep(settle_seconds)
    after_snapshot = sim_pose_snapshot(sim)
    payload = {
        "name": "go_home",
        "type": "home",
        "settle_seconds": float(settle_seconds),
        "before_snapshot": before_snapshot,
        "after_snapshot": after_snapshot,
    }
    save_json(output_dir / "go_home.json", payload)
    return payload


def execute_pregrasp_sequence(
    sim: StretchMujocoSimulator,
    *,
    output_dir: Path,
    params: dict[str, float],
    settle_seconds: float,
    skip_home: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace: list[dict[str, Any]] = []
    if not skip_home:
        trace.append(execute_home(sim, output_dir=output_dir, settle_seconds=settle_seconds))

    trace.append(
        execute_named_move(
            sim,
            output_dir=output_dir,
            name="go_pregrasp_lift",
            actuator=Actuators.lift,
            target=float(params["pregrasp_lift_m"]),
        )
    )
    trace.append(
        execute_named_move(
            sim,
            output_dir=output_dir,
            name="go_pregrasp_arm",
            actuator=Actuators.arm,
            target=float(params["pregrasp_arm_m"]),
        )
    )
    trace.append(
        execute_named_move(
            sim,
            output_dir=output_dir,
            name="go_pregrasp_wrist_yaw",
            actuator=Actuators.wrist_yaw,
            target=float(params["pregrasp_wrist_yaw_rad"]),
        )
    )
    trace.append(
        execute_named_move(
            sim,
            output_dir=output_dir,
            name="go_pregrasp_wrist_pitch",
            actuator=Actuators.wrist_pitch,
            target=float(params["pregrasp_wrist_pitch_rad"]),
        )
    )
    trace.append(
        execute_named_move(
            sim,
            output_dir=output_dir,
            name="go_pregrasp_wrist_roll",
            actuator=Actuators.wrist_roll,
            target=float(params["pregrasp_wrist_roll_rad"]),
        )
    )
    trace.append(
        execute_named_move(
            sim,
            output_dir=output_dir,
            name="go_pregrasp_head_pan",
            actuator=Actuators.head_pan,
            target=float(params["head_pan_rad"]),
        )
    )
    trace.append(
        execute_named_move(
            sim,
            output_dir=output_dir,
            name="go_pregrasp_head_tilt",
            actuator=Actuators.head_tilt,
            target=float(params["head_tilt_rad"]),
        )
    )
    trace.append(
        execute_named_move(
            sim,
            output_dir=output_dir,
            name="open_gripper",
            actuator=Actuators.gripper,
            target=float(params["open_gripper_m"]),
        )
    )
    summary = {
        "stage": "pregrasp",
        "trace": trace,
        "all_required_moves_reached": all(item["result"]["reached"] for item in trace if item["type"] == "move_to"),
        "final_snapshot": sim_pose_snapshot(sim),
    }
    save_json(output_dir / "pregrasp_stage_summary.json", summary)
    return summary


def execute_final_approach_sequence(
    sim: StretchMujocoSimulator,
    *,
    output_dir: Path,
    params: dict[str, float],
    target_pixel_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    descend_delta_m, descend_meta = compute_descend_delta_m(params, target_pixel_result)
    trace: list[dict[str, Any]] = []
    pre_descend_lift = float(sim.pull_status().lift.pos)
    segment_1 = float(descend_delta_m * 0.4)
    segment_2 = float(descend_delta_m - segment_1)

    current_lift = pre_descend_lift
    for segment_index, delta in enumerate([segment_1, segment_2], start=1):
        current_lift = max(0.0, current_lift - delta)
        trace.append(
            {
                **execute_named_move(
                    sim,
                    output_dir=output_dir,
                    name=f"descend_segment_{segment_index}",
                    actuator=Actuators.lift,
                    target=current_lift,
                ),
                "descend_delta_segment_m": float(delta),
            }
        )

    close_move = execute_named_move(
        sim,
        output_dir=output_dir,
        name="close_gripper",
        actuator=Actuators.gripper,
        target=float(params["close_gripper_m"]),
    )
    trace.append(close_move)
    closed_after = float(close_move["result"]["after"])
    close_target = float(params["close_gripper_m"])
    gripper_close_failed = abs(closed_after - close_target) <= 0.004

    post_close_lift = float(sim.pull_status().lift.pos)
    lift_target = min(1.05, post_close_lift + float(params["lift_after_grasp_delta_m"]))
    trace.append(
        execute_named_move(
            sim,
            output_dir=output_dir,
            name="lift_after_grasp",
            actuator=Actuators.lift,
            target=lift_target,
        )
    )
    trace.append(
        execute_named_move(
            sim,
            output_dir=output_dir,
            name="retract_arm",
            actuator=Actuators.arm,
            target=float(params["retract_arm_target_m"]),
        )
    )
    summary = {
        "stage": "final_approach",
        "trace": trace,
        "descend_delta_m": float(descend_delta_m),
        "descend_meta": descend_meta,
        "gripper_close_failed": bool(gripper_close_failed),
        "all_required_moves_reached": all(item["result"]["reached"] for item in trace),
        "final_snapshot": sim_pose_snapshot(sim),
    }
    save_json(output_dir / "final_approach_stage_summary.json", summary)
    return summary


def run_primitive_sequence(
    sim: StretchMujocoSimulator,
    *,
    output_dir: Path,
    params: dict[str, float],
    settle_seconds: float,
    target_pixel_result: dict[str, Any] | None = None,
    skip_home: bool = False,
) -> dict[str, Any]:
    primitive_dir = output_dir / "primitive_trace"
    primitive_dir.mkdir(parents=True, exist_ok=True)

    pregrasp_summary = execute_pregrasp_sequence(
        sim,
        output_dir=primitive_dir,
        params=params,
        settle_seconds=settle_seconds,
        skip_home=skip_home,
    )
    final_approach_summary = execute_final_approach_sequence(
        sim,
        output_dir=primitive_dir,
        params=params,
        target_pixel_result=target_pixel_result,
    )
    trace = [*pregrasp_summary["trace"], *final_approach_summary["trace"]]
    descend_delta_m = final_approach_summary["descend_delta_m"]
    descend_meta = final_approach_summary["descend_meta"]

    summary = {
        "primitive_sequence": [
            "go_home",
            "go_pregrasp",
            "open_gripper",
            "descend_small_delta",
            "close_gripper",
            "lift_after_grasp",
            "retract_arm",
        ],
        "parameters": params,
        "target_pixel_result": target_pixel_result,
        "descend_delta_m": float(descend_delta_m),
        "descend_meta": descend_meta,
        "pregrasp_summary": pregrasp_summary,
        "final_approach_summary": final_approach_summary,
        "gripper_close_failed": final_approach_summary["gripper_close_failed"],
        "trace": trace,
        "final_snapshot": sim_pose_snapshot(sim),
    }
    save_json(output_dir / "primitive_runner_summary.json", summary)
    return summary


def args_to_params(args: argparse.Namespace) -> dict[str, float]:
    return {
        "pregrasp_lift_m": float(args.pregrasp_lift_m),
        "pregrasp_arm_m": float(args.pregrasp_arm_m),
        "pregrasp_wrist_yaw_rad": float(args.pregrasp_wrist_yaw_rad),
        "pregrasp_wrist_pitch_rad": float(args.pregrasp_wrist_pitch_rad),
        "pregrasp_wrist_roll_rad": float(args.pregrasp_wrist_roll_rad),
        "head_pan_rad": float(args.head_pan_rad),
        "head_tilt_rad": float(args.head_tilt_rad),
        "open_gripper_m": float(args.open_gripper_m),
        "close_gripper_m": float(args.close_gripper_m),
        "descend_delta_m": float(args.descend_delta_m),
        "min_descend_delta_m": float(args.min_descend_delta_m),
        "max_descend_delta_m": float(args.max_descend_delta_m),
        "lift_after_grasp_delta_m": float(args.lift_after_grasp_delta_m),
        "retract_arm_target_m": float(args.retract_arm_target_m),
        "nominal_target_depth_m": float(args.nominal_target_depth_m),
    }


def main() -> int:
    args = parse_args()
    output_dir = create_run_dir(args.output_dir, args.run_tag, prefix="primitive_runner")
    params = args_to_params(args)
    target_pixel_result = None
    if args.target_depth_m is not None:
        target_pixel_result = {"depth_m": float(args.target_depth_m)}

    save_json(
        output_dir / "primitive_runner_context.json",
        {
            "scene_xml_path": str(Path(args.scene_xml_path).resolve()),
            "output_dir": str(output_dir),
            "parameters": params,
            "target_pixel_result": target_pixel_result,
            "skip_home": bool(args.skip_home),
        },
    )

    sim = start_wrist_sim(
        scene_xml_path=Path(args.scene_xml_path),
        camera_hz=float(args.camera_hz),
        settle_seconds=float(args.settle_seconds),
    )
    try:
        run_primitive_sequence(
            sim,
            output_dir=output_dir,
            params=params,
            settle_seconds=float(args.settle_seconds),
            target_pixel_result=target_pixel_result,
            skip_home=bool(args.skip_home),
        )
    finally:
        sim.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
