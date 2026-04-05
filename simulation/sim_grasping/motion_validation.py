from __future__ import annotations

import argparse
import time
from pathlib import Path

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators

from validation_common import (
    DEFAULT_SCENE_PATH,
    MOTION_ROOT,
    ensure_dir,
    ensure_validation_dirs,
    matrix_to_rows,
    save_json,
    utc_timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small Stretch motion validation sequence.")
    parser.add_argument("--scene-xml-path", default=str(DEFAULT_SCENE_PATH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--settle-seconds", type=float, default=0.75)
    return parser.parse_args()


def pose_from_status(sim: StretchMujocoSimulator) -> dict[str, object]:
    status = sim.pull_status()
    return {
        "sim_time": float(status.time),
        "base_pose": [float(v) for v in sim.get_base_pose()],
        "ee_pose_4x4": matrix_to_rows(sim.get_ee_pose()),
        "status": status.to_dict(),
    }


def actuator_tolerance(actuator: Actuators) -> float:
    if actuator == Actuators.gripper:
        return 0.01
    if actuator in [Actuators.arm, Actuators.head_pan, Actuators.head_tilt]:
        return 0.02
    return 0.03


def move_to_and_record(sim: StretchMujocoSimulator, actuator: Actuators, target: float) -> dict[str, object]:
    before = actuator.get_position(sim.pull_status())
    sim.move_to(actuator, target)
    tolerance = actuator_tolerance(actuator)
    reached = sim.wait_until_at_setpoint(actuator, timeout=8.0, position_tolerance=tolerance)
    after = actuator.get_position(sim.pull_status())
    return {
        "actuator": actuator.name,
        "target": float(target),
        "before": float(before),
        "after": float(after),
        "reached": bool(reached),
        "tolerance": float(tolerance),
        "abs_error": float(abs(after - target)),
    }


def base_velocity_step(
    sim: StretchMujocoSimulator, *, v_linear: float, omega: float, duration_s: float
) -> dict[str, object]:
    before = sim.get_base_pose()
    sim.set_base_velocity(v_linear=v_linear, omega=omega)
    time.sleep(duration_s)
    sim.set_base_velocity(v_linear=0.0, omega=0.0)
    sim.wait_while_is_moving(Actuators.base_translate, timeout=6.0)
    after = sim.get_base_pose()
    return {
        "v_linear": float(v_linear),
        "omega": float(omega),
        "duration_s": float(duration_s),
        "before_pose": [float(v) for v in before],
        "after_pose": [float(v) for v in after],
        "delta_pose": [float(a - b) for a, b in zip(after, before)],
    }


def main() -> int:
    args = parse_args()
    ensure_validation_dirs()
    output_dir = Path(args.output_dir) if args.output_dir else MOTION_ROOT / "latest"
    output_dir = ensure_dir(output_dir)

    sim = StretchMujocoSimulator(
        scene_xml_path=str(Path(args.scene_xml_path).resolve()),
        cameras_to_use=[],
        camera_hz=10.0,
    )

    summary: dict[str, object] = {
        "wall_time_utc": utc_timestamp(),
        "scene_xml_path": str(Path(args.scene_xml_path).resolve()),
        "steps": [],
    }

    sim.start(headless=True)
    try:
        step_results: list[dict[str, object]] = []

        sim.home()
        time.sleep(args.settle_seconds)
        step_results.append({"name": "home", "result": pose_from_status(sim)})

        sim.stow()
        time.sleep(args.settle_seconds)
        step_results.append({"name": "stow", "result": pose_from_status(sim)})

        sim.home()
        time.sleep(args.settle_seconds)
        step_results.append({"name": "home_again", "result": pose_from_status(sim)})

        step_results.append(
            {"name": "base_forward", "result": base_velocity_step(sim, v_linear=0.15, omega=0.0, duration_s=1.5)}
        )
        step_results.append(
            {"name": "base_backward", "result": base_velocity_step(sim, v_linear=-0.15, omega=0.0, duration_s=1.5)}
        )
        step_results.append(
            {"name": "base_turn", "result": base_velocity_step(sim, v_linear=0.0, omega=0.45, duration_s=1.5)}
        )

        for actuator, target in [
            (Actuators.lift, 0.92),
            (Actuators.lift, 0.62),
            (Actuators.arm, 0.28),
            (Actuators.arm, 0.08),
            (Actuators.head_pan, -0.9),
            (Actuators.head_tilt, -0.65),
            (Actuators.head_pan, 0.0),
            (Actuators.head_tilt, -0.25),
            (Actuators.wrist_yaw, 1.0),
            (Actuators.wrist_pitch, -0.45),
            (Actuators.wrist_roll, 1.2),
            (Actuators.gripper, 0.03),
            (Actuators.gripper, 0.0),
        ]:
            step_results.append(
                {
                    "name": f"{actuator.name}_{target:+.2f}",
                    "result": move_to_and_record(sim, actuator, target),
                }
            )

        pre_grasp_targets = {
            Actuators.lift: 0.80,
            Actuators.arm: 0.24,
            Actuators.wrist_yaw: 0.0,
            Actuators.wrist_pitch: -0.55,
            Actuators.wrist_roll: 0.0,
            Actuators.gripper: 0.03,
            Actuators.head_pan: 0.0,
            Actuators.head_tilt: -0.55,
        }
        pre_grasp_log: list[dict[str, object]] = []
        for actuator, target in pre_grasp_targets.items():
            pre_grasp_log.append(move_to_and_record(sim, actuator, target))
        pre_grasp_log.append({"phase": "above_target", "pose": pose_from_status(sim)})
        pre_grasp_log.append({"phase": "lower", "move": move_to_and_record(sim, Actuators.lift, 0.68)})
        pre_grasp_log.append({"phase": "lift_back", "move": move_to_and_record(sim, Actuators.lift, 0.82)})
        pre_grasp_log.append({"phase": "retract", "move": move_to_and_record(sim, Actuators.arm, 0.08)})
        step_results.append({"name": "pre_grasp_template", "result": pre_grasp_log})

        summary["steps"] = step_results
        summary["final_pose"] = pose_from_status(sim)
        summary["joint_limits"] = {
            actuator.name: [float(v) for v in limits]
            for actuator, limits in sim.pull_joint_limits().items()
        }
    finally:
        sim.stop()

    save_json(output_dir / "motion_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
