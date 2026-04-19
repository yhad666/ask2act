from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators


def _extract_positions(sim: StretchMujocoSimulator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scene_objects = sim.pull_scene_objects()
    gripper_info = scene_objects.get("gripper_diagnostic", {})
    if not isinstance(gripper_info, dict):
        raise RuntimeError("gripper_diagnostic not available from simulator")

    wrist = np.asarray(gripper_info["body_link_wrist_yaw"]["pos"], dtype=float)
    grasp_center = np.asarray(gripper_info["body_link_grasp_center"]["pos"], dtype=float)
    left_body = gripper_info["body_rubber_tip_left"]
    right_body = gripper_info["body_rubber_tip_right"]
    left_pos = np.asarray(left_body["pos"], dtype=float)
    right_pos = np.asarray(right_body["pos"], dtype=float)
    left_rot = np.asarray(left_body["xmat"], dtype=float).reshape(3, 3)
    right_rot = np.asarray(right_body["xmat"], dtype=float).reshape(3, 3)
    local_geom_center = np.array([0.0, 0.0, 0.01], dtype=float)
    left = left_pos + left_rot @ local_geom_center
    right = right_pos + right_rot @ local_geom_center
    return wrist, grasp_center, left, right


def _move_to_topdown_pose(sim: StretchMujocoSimulator, *, lift: float, arm: float) -> None:
    targets = [
        (Actuators.wrist_pitch, -1.57, 35.0, 0.10),
        (Actuators.wrist_yaw, 0.0, 35.0, 0.15),
        (Actuators.wrist_roll, 0.0, 20.0, 0.10),
        (Actuators.lift, float(lift), 30.0, 0.08),
        (Actuators.arm, float(arm), 30.0, 0.10),
    ]
    for actuator, target, timeout_s, tolerance in targets:
        sim.move_to(actuator, float(target))
        sim.wait_until_at_setpoint(actuator, timeout=timeout_s, position_tolerance=tolerance)
    time.sleep(2.0)


def main() -> None:
    sim = StretchMujocoSimulator()
    sim.start(headless=True)
    try:
        sim.home()
        time.sleep(2.0)

        _move_to_topdown_pose(sim, lift=0.8, arm=0.3)
        wrist_pos, grasp_center_pos, left_pos, right_pos = _extract_positions(sim)
        rubber_center = 0.5 * (left_pos + right_pos)
        offset = rubber_center - wrist_pos
        grasp_center_offset = grasp_center_pos - wrist_pos
        grasp_to_rubber = rubber_center - grasp_center_pos

        print("\n=== TOP-DOWN OFFSET MEASUREMENT ===")
        print(f"link_wrist_yaw position: {wrist_pos}")
        print(f"link_grasp_center:       {grasp_center_pos}")
        print(f"rubber_tip_left:         {left_pos}")
        print(f"rubber_tip_right:        {right_pos}")
        print(f"rubber_tip_center:       {rubber_center}")
        print("")
        print("OFFSET (grasp_center - wrist_yaw_link):")
        print(f"  dx = {grasp_center_offset[0]:.6f} m")
        print(f"  dy = {grasp_center_offset[1]:.6f} m")
        print(f"  dz = {grasp_center_offset[2]:.6f} m")
        print(f"  total = {np.linalg.norm(grasp_center_offset):.6f} m")
        print("")
        print("OFFSET (rubber_center - grasp_center):")
        print(f"  dx = {grasp_to_rubber[0]:.6f} m")
        print(f"  dy = {grasp_to_rubber[1]:.6f} m")
        print(f"  dz = {grasp_to_rubber[2]:.6f} m")
        print(f"  total = {np.linalg.norm(grasp_to_rubber):.6f} m")
        print("")
        print("OFFSET (rubber_center - wrist_yaw_link):")
        print(f"  dx = {offset[0]:.6f} m")
        print(f"  dy = {offset[1]:.6f} m")
        print(f"  dz = {offset[2]:.6f} m")
        print(f"  total = {np.linalg.norm(offset):.6f} m")
        print("")
        print("Put this in stretch3_specs.py:")
        print(
            "TOPDOWN_WRIST_TO_RUBBER_OFFSET = np.array("
            f"[{offset[0]:.6f}, {offset[1]:.6f}, {offset[2]:.6f}], dtype=float)"
        )

        _move_to_topdown_pose(sim, lift=0.5, arm=0.15)
        wrist_pos2, grasp_center_pos2, left_pos2, right_pos2 = _extract_positions(sim)
        rubber_center2 = 0.5 * (left_pos2 + right_pos2)
        offset2 = rubber_center2 - wrist_pos2
        grasp_center_offset2 = grasp_center_pos2 - wrist_pos2
        grasp_to_rubber2 = rubber_center2 - grasp_center_pos2

        print("\n=== VERIFICATION AT DIFFERENT POSE ===")
        print(f"Offset at pose 2: [{offset2[0]:.6f}, {offset2[1]:.6f}, {offset2[2]:.6f}]")
        print(f"Difference from pose 1: {np.linalg.norm(offset - offset2):.6f} m")
        print("(Should be < 0.005m if offset is truly constant)")

        output_path = Path("/tmp/ask2act_topdown_offset_measurement.json")
        output_path.write_text(
            json.dumps(
                {
                    "pose1_grasp_center_from_wrist": grasp_center_offset.tolist(),
                    "pose1_rubber_from_grasp_center": grasp_to_rubber.tolist(),
                    "pose1_offset": offset.tolist(),
                    "pose2_grasp_center_from_wrist": grasp_center_offset2.tolist(),
                    "pose2_rubber_from_grasp_center": grasp_to_rubber2.tolist(),
                    "pose2_offset": offset2.tolist(),
                    "difference_norm_m": float(np.linalg.norm(offset - offset2)),
                },
                indent=2,
            )
        )
        print(f"\nSaved measurement to {output_path}")
    finally:
        sim.stop()


if __name__ == "__main__":
    main()
