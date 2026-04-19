from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ask2act_grasp.ik.simple_ik import SimpleIK
from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators


@dataclass
class CalibrationSample:
    lift: float
    arm: float
    gripper_cmd: float
    actual_lift: float
    actual_arm: float
    actual_gripper_cmd: float
    wrist_model_error_xyz: list[float]
    grasp_center_from_wrist_xyz: list[float]
    rubber_from_wrist_xyz: list[float]
    rubber_from_grasp_center_xyz: list[float]


def _extract_scene_positions(sim: StretchMujocoSimulator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gripper_info = sim.pull_scene_objects()["gripper_diagnostic"]
    wrist = np.array(gripper_info["body_link_wrist_yaw"]["pos"], dtype=float)
    grasp_center = np.array(gripper_info["body_link_grasp_center"]["pos"], dtype=float)
    left_body = gripper_info["body_rubber_tip_left"]
    right_body = gripper_info["body_rubber_tip_right"]
    left_pos = np.array(left_body["pos"], dtype=float)
    right_pos = np.array(right_body["pos"], dtype=float)
    left_rot = np.array(left_body["xmat"], dtype=float).reshape(3, 3)
    right_rot = np.array(right_body["xmat"], dtype=float).reshape(3, 3)
    local_geom_center = np.array([0.0, 0.0, 0.01], dtype=float)
    left = left_pos + left_rot @ local_geom_center
    right = right_pos + right_rot @ local_geom_center
    rubber_center = 0.5 * (left + right)
    return wrist, grasp_center, rubber_center


def _move_to_topdown_pose(
    sim: StretchMujocoSimulator,
    *,
    lift: float,
    arm: float,
    gripper_cmd: float,
) -> tuple[float, float, float]:
    commands = [
        (Actuators.wrist_pitch, -1.57, 35.0, 0.08),
        (Actuators.wrist_yaw, 0.0, 35.0, 0.08),
        (Actuators.wrist_roll, 0.0, 35.0, 0.08),
        (Actuators.lift, float(lift), 35.0, 0.02),
        (Actuators.arm, float(arm), 35.0, 0.02),
        (Actuators.gripper, float(gripper_cmd), 35.0, 0.02),
    ]
    for actuator, target, timeout_s, tolerance in commands:
        sim.move_to(actuator, float(target))
        sim.wait_until_at_setpoint(actuator, timeout=timeout_s, position_tolerance=tolerance)
    time.sleep(1.5)
    status = sim.pull_status()
    return float(status.lift.pos), float(status.arm.pos), float(status.gripper.pos)


def _fit_affine(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, float]:
    coeffs, *_ = np.linalg.lstsq(features, targets, rcond=None)
    pred = features @ coeffs
    rms = float(np.sqrt(np.mean(np.sum((pred - targets) ** 2, axis=1))))
    return coeffs, rms


def main() -> None:
    output_path = Path("/tmp/ask2act_topdown_offset_calibration.json")
    ik = SimpleIK()
    sim = StretchMujocoSimulator()
    sim.start(headless=True)
    samples: list[CalibrationSample] = []
    try:
        sim.home()
        time.sleep(2.0)
        lift_values = [0.72, 0.86, 0.98]
        arm_values = [0.22, 0.35, 0.48]
        gripper_values = [-0.30, 0.00, 0.28, 0.46]

        for lift in lift_values:
            for arm in arm_values:
                for gripper_cmd in gripper_values:
                    actual_lift, actual_arm, actual_gripper_cmd = _move_to_topdown_pose(
                        sim,
                        lift=lift,
                        arm=arm,
                        gripper_cmd=gripper_cmd,
                    )
                    wrist_world, grasp_center_world, rubber_world = _extract_scene_positions(sim)
                    fk_wrist = np.asarray(
                        ik.fk_rotary_base(
                            {
                                "joint_mobile_base_rotation": 0.0,
                                "joint_lift": actual_lift,
                                "joint_arm_l0": actual_arm,
                            }
                        ),
                        dtype=float,
                    )
                    samples.append(
                        CalibrationSample(
                            lift=lift,
                            arm=arm,
                            gripper_cmd=gripper_cmd,
                            actual_lift=actual_lift,
                            actual_arm=actual_arm,
                            actual_gripper_cmd=actual_gripper_cmd,
                            wrist_model_error_xyz=(wrist_world - fk_wrist).tolist(),
                            grasp_center_from_wrist_xyz=(grasp_center_world - wrist_world).tolist(),
                            rubber_from_wrist_xyz=(rubber_world - wrist_world).tolist(),
                            rubber_from_grasp_center_xyz=(rubber_world - grasp_center_world).tolist(),
                        )
                    )

        wrist_features = np.array(
            [[1.0, s.actual_lift, s.actual_arm] for s in samples],
            dtype=float,
        )
        wrist_targets = np.array([s.wrist_model_error_xyz for s in samples], dtype=float)
        wrist_coeffs, wrist_rms = _fit_affine(wrist_features, wrist_targets)

        rubber_features = np.array(
            [[1.0, s.actual_gripper_cmd] for s in samples],
            dtype=float,
        )
        rubber_targets = np.array([s.rubber_from_wrist_xyz for s in samples], dtype=float)
        rubber_coeffs, rubber_rms = _fit_affine(rubber_features, rubber_targets)

        grasp_center_from_wrist = np.array([s.grasp_center_from_wrist_xyz for s in samples], dtype=float)
        grasp_center_mean = grasp_center_from_wrist.mean(axis=0)
        grasp_center_rms = float(np.sqrt(np.mean(np.sum((grasp_center_from_wrist - grasp_center_mean) ** 2, axis=1))))

        rubber_from_grasp_center_targets = np.array([s.rubber_from_grasp_center_xyz for s in samples], dtype=float)
        rubber_from_grasp_center_coeffs, rubber_from_grasp_center_rms = _fit_affine(
            rubber_features,
            rubber_from_grasp_center_targets,
        )

        payload = {
            "wrist_model_error_affine_xyz_from_[1,lift,arm]": wrist_coeffs.tolist(),
            "wrist_model_error_fit_rms_m": wrist_rms,
            "grasp_center_from_wrist_constant_xyz": grasp_center_mean.tolist(),
            "grasp_center_from_wrist_fit_rms_m": grasp_center_rms,
            "rubber_from_wrist_affine_xyz_from_[1,gripper_cmd]": rubber_coeffs.tolist(),
            "rubber_from_wrist_fit_rms_m": rubber_rms,
            "rubber_from_grasp_center_affine_xyz_from_[1,gripper_cmd]": rubber_from_grasp_center_coeffs.tolist(),
            "rubber_from_grasp_center_fit_rms_m": rubber_from_grasp_center_rms,
            "samples": [asdict(sample) for sample in samples],
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        print(f"\nSaved calibration to {output_path}")
    finally:
        sim.stop()


if __name__ == "__main__":
    main()
