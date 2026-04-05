from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras

from minimal_grasping_common import (
    capture_rgb_frame,
    capture_head_observation,
    capture_wrist_observation,
    heuristic_descend_delta_m,
    save_json,
    sim_pose_snapshot,
)
from minimal_primitive_runner import (
    execute_final_approach_sequence,
    execute_named_move,
    execute_pregrasp_sequence,
)
from pipeline_types import SelectedGraspPlan, WristRefinementDelta


class MotionExecutor:
    def execute_pregrasp(self, *args, **kwargs):
        raise NotImplementedError

    def apply_wrist_refinement(self, *args, **kwargs):
        raise NotImplementedError

    def execute_final_approach(self, *args, **kwargs):
        raise NotImplementedError


class JointSpaceMotionExecutor(MotionExecutor):
    def __init__(self, sim: StretchMujocoSimulator, *, output_dir: Path, settle_seconds: float):
        self.sim = sim
        self.output_dir = output_dir
        self.settle_seconds = float(settle_seconds)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute_pregrasp(self, selected_plan: SelectedGraspPlan) -> dict[str, Any]:
        params = dict(selected_plan.pregrasp_joint_targets)
        base_translate_m = float(params.pop("base_translate_m", 0.0))
        pregrasp_dir = self.output_dir / "pregrasp"
        pregrasp_dir.mkdir(parents=True, exist_ok=True)

        base_move = None
        if abs(base_translate_m) > 1e-4:
            before_pose = [float(v) for v in self.sim.get_base_pose()]
            self.sim.move_by(Actuators.base_translate, base_translate_m)
            reached = self.sim.wait_while_is_moving(Actuators.base_translate, timeout=10.0)
            after_pose = [float(v) for v in self.sim.get_base_pose()]
            base_move = {
                "name": "base_translate_pregrasp",
                "delta_m": base_translate_m,
                "reached": bool(reached),
                "before_pose": before_pose,
                "after_pose": after_pose,
            }
            save_json(pregrasp_dir / "base_translate_pregrasp.json", base_move)

        sequence = execute_pregrasp_sequence(
            self.sim,
            output_dir=pregrasp_dir,
            params=params,
            settle_seconds=self.settle_seconds,
            skip_home=False,
        )
        sequence["base_move"] = base_move
        sequence["all_required_moves_reached"] = bool(
            sequence["all_required_moves_reached"] and (base_move is None or base_move["reached"])
        )
        save_json(pregrasp_dir / "pregrasp_summary.json", sequence)
        return sequence

    def apply_wrist_refinement(
        self,
        selected_plan: SelectedGraspPlan,
        delta: WristRefinementDelta,
        *,
        tag: str,
    ) -> dict[str, Any]:
        refine_dir = self.output_dir / tag
        refine_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "tag": tag,
            "delta": delta,
            "before_snapshot": sim_pose_snapshot(self.sim),
            "moves": [],
        }
        if not delta.target_visible:
            payload["status"] = "skipped"
            payload["reason"] = "target_not_visible_or_delta_out_of_limits"
            save_json(refine_dir / f"{tag}_summary.json", payload)
            return payload

        arm_status = self.sim.pull_status().arm.pos
        lift_status = self.sim.pull_status().lift.pos
        yaw_status = self.sim.pull_status().wrist_yaw.pos
        if abs(delta.dy_m) > 1e-4:
            self.sim.move_by(Actuators.base_translate, delta.dy_m)
            base_reached = self.sim.wait_while_is_moving(Actuators.base_translate, timeout=8.0)
            payload["moves"].append(
                {
                    "name": "base_translate_refine",
                    "delta": float(delta.dy_m),
                    "reached": bool(base_reached),
                    "after_pose": [float(v) for v in self.sim.get_base_pose()],
                }
            )
        if abs(delta.dx_m) > 1e-4:
            payload["moves"].append(
                execute_named_move(
                    self.sim,
                    output_dir=refine_dir,
                    name=f"{tag}_arm_refine",
                    actuator=Actuators.arm,
                    target=float(max(0.0, arm_status + delta.dx_m)),
                )
            )
        if abs(delta.dz_m) > 1e-4:
            payload["moves"].append(
                execute_named_move(
                    self.sim,
                    output_dir=refine_dir,
                    name=f"{tag}_lift_refine",
                    actuator=Actuators.lift,
                    target=float(max(0.0, min(1.05, lift_status - delta.dz_m))),
                )
            )
        if abs(delta.dyaw_rad) > 1e-4:
            payload["moves"].append(
                execute_named_move(
                    self.sim,
                    output_dir=refine_dir,
                    name=f"{tag}_yaw_refine",
                    actuator=Actuators.wrist_yaw,
                    target=float(yaw_status + delta.dyaw_rad),
                )
            )
        time.sleep(self.settle_seconds)
        payload["status"] = "completed"
        payload["after_snapshot"] = sim_pose_snapshot(self.sim)
        save_json(refine_dir / f"{tag}_summary.json", payload)
        return payload

    def execute_final_approach(
        self,
        selected_plan: SelectedGraspPlan,
        *,
        wrist_pixel_result: dict[str, Any] | None,
        tag: str,
    ) -> dict[str, Any]:
        params = dict(selected_plan.pregrasp_joint_targets)
        if wrist_pixel_result is not None:
            params["descend_delta_m"] = heuristic_descend_delta_m(
                float(wrist_pixel_result["depth_m"]),
                nominal_target_depth_m=float(params["nominal_target_depth_m"]),
                base_delta_m=float(params["descend_delta_m"]),
                min_delta_m=float(params["min_descend_delta_m"]),
                max_delta_m=float(params["max_descend_delta_m"]),
            )
        approach_dir = self.output_dir / tag
        approach_dir.mkdir(parents=True, exist_ok=True)
        result = execute_final_approach_sequence(
            self.sim,
            output_dir=approach_dir,
            params=params,
            target_pixel_result=wrist_pixel_result,
        )
        save_json(approach_dir / f"{tag}_summary.json", result)
        return result

    def capture_wrist(self) -> dict[str, Any]:
        return capture_wrist_observation(self.sim)

    def capture_head(self) -> dict[str, Any]:
        return capture_head_observation(self.sim)

    def capture_head_rgb(self):
        return capture_rgb_frame(self.sim, camera=StretchCameras.cam_d435i_rgb)
