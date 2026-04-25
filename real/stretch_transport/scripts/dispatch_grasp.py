from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


class GraspExecutionError(RuntimeError):
    def __init__(self, message: str, trace: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.trace = trace


def _artifact_root() -> Path:
    raw = os.getenv(
        "ASK2ACT_STRETCH_EXECUTE_ARTIFACT_ROOT",
        str(Path(__file__).resolve().parents[1] / "artifacts" / "executions"),
    )
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_artifact(payload: dict[str, Any]) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = _artifact_root() / f"execute_request_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_response_artifact(request_artifact_path: Path, response: dict[str, Any]) -> Path:
    response_path = request_artifact_path.with_name(
        request_artifact_path.name.replace("execute_request_", "execute_response_")
    )
    response["response_artifact_path"] = str(response_path)
    response_path.write_text(json.dumps(response, indent=2, default=str), encoding="utf-8")
    return response_path


def _trajectory_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    grasp_plan = payload.get("grasp_plan") or {}
    trajectory = grasp_plan.get("trajectory") or []
    if not isinstance(trajectory, list):
        raise RuntimeError("grasp_plan.trajectory must be a list when present")
    return trajectory


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _default_pose_targets() -> dict[str, float]:
    return {
        "lift": float(os.getenv("ASK2ACT_STRETCH_HOME_LIFT_M", "0.60")),
        "arm": float(os.getenv("ASK2ACT_STRETCH_HOME_ARM_M", "0.0")),
        "wrist_yaw": float(os.getenv("ASK2ACT_STRETCH_HOME_WRIST_YAW_RAD", "0.0")),
        "wrist_pitch": float(os.getenv("ASK2ACT_STRETCH_HOME_WRIST_PITCH_RAD", "-1.57")),
        "wrist_roll": float(os.getenv("ASK2ACT_STRETCH_HOME_WRIST_ROLL_RAD", "0.0")),
        "stretch_gripper": float(os.getenv("ASK2ACT_STRETCH_HOME_GRIPPER_CMD", "0.56")),
    }


def _status_snapshot(robot: Any) -> dict[str, Any]:
    try:
        robot.pull_status()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    def status_of(component_name: str) -> Any:
        component = getattr(robot, component_name, None)
        if component is None:
            return None
        return getattr(component, "status", None)

    return {
        "ok": True,
        "timestamp_epoch_s": time.time(),
        "base": status_of("base"),
        "lift": status_of("lift"),
        "arm": status_of("arm"),
        "head": status_of("head"),
        "end_of_arm": status_of("end_of_arm"),
    }


def _command_default_pose(robot: Any, *, include_gripper: bool, reason: str) -> dict[str, Any]:
    targets = _default_pose_targets()
    status_before = _status_snapshot(robot)
    if include_gripper:
        robot.end_of_arm.move_to("stretch_gripper", targets["stretch_gripper"])
    robot.arm.move_to(targets["arm"])
    robot.end_of_arm.move_to("wrist_yaw", targets["wrist_yaw"])
    robot.end_of_arm.move_to("wrist_pitch", targets["wrist_pitch"])
    robot.end_of_arm.move_to("wrist_roll", targets["wrist_roll"])
    robot.lift.move_to(targets["lift"])
    robot.push_command()
    settle_s = max(0.0, float(os.getenv("ASK2ACT_STRETCH_HOME_SETTLE_S", "2.0")))
    if settle_s > 0.0:
        time.sleep(settle_s)
    try:
        robot.pull_status()
    except Exception:
        pass
    status_after = _status_snapshot(robot)
    return {
        "name": f"default_pose_{reason}",
        "joint_targets": targets if include_gripper else {key: value for key, value in targets.items() if key != "stretch_gripper"},
        "include_gripper": include_gripper,
        "ok": True,
        "settle_s": settle_s,
        "status_before": status_before,
        "status_after": status_after,
    }


def _current_base_theta(robot: Any) -> float:
    try:
        robot.pull_status()
    except Exception:
        pass
    try:
        return float(robot.base.status["theta"])
    except Exception as exc:
        raise RuntimeError("Unable to read current Stretch base theta") from exc


def _angle_diff_rad(target: float, current: float) -> float:
    return float((target - current + math.pi) % (2.0 * math.pi) - math.pi)


def _move_component(robot: Any, joint_name: str, target: float, *, base_reference_theta: float) -> dict[str, Any] | None:
    if joint_name == "lift":
        robot.lift.move_to(target)
    elif joint_name == "arm":
        robot.arm.move_to(target)
    elif joint_name in {"wrist_yaw", "wrist_pitch", "wrist_roll", "stretch_gripper"}:
        robot.end_of_arm.move_to(joint_name, target)
    elif joint_name in {"head_pan", "head_tilt"}:
        robot.head.move_to(joint_name, target)
    elif joint_name == "base_rotate":
        current_theta = _current_base_theta(robot)
        target_theta = float(base_reference_theta + target)
        delta = _angle_diff_rad(target_theta, current_theta)
        eps = float(os.getenv("ASK2ACT_STRETCH_BASE_ROTATE_EPS_RAD", "0.02"))
        if abs(delta) <= eps:
            return {
                "joint_name": joint_name,
                "target_relative_rad": float(target),
                "target_theta_rad": target_theta,
                "current_theta_rad": current_theta,
                "delta_rad": delta,
                "skipped": True,
                "skip_reason": f"abs(delta) <= {eps}",
            }
        max_delta = float(os.getenv("ASK2ACT_STRETCH_BASE_ROTATE_MAX_DELTA_RAD", "0.40"))
        if abs(delta) > max_delta:
            raise RuntimeError(
                f"Refusing large base rotation delta {delta:.3f} rad for target {target:.3f} rad. "
                "Planner base_rotate is expected to be relative to the execution start pose."
            )
        robot.base.rotate_by(delta)
        return {
            "joint_name": joint_name,
            "target_relative_rad": float(target),
            "target_theta_rad": target_theta,
            "current_theta_rad": current_theta,
            "delta_rad": delta,
            "skipped": False,
        }
    else:
        raise RuntimeError(f"Unsupported real-robot joint target: {joint_name}")
    return None


def _execute_trajectory(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import stretch_body.robot

    robot = stretch_body.robot.Robot()
    if not robot.startup():
        raise RuntimeError("Failed to startup Stretch robot")

    trace: list[dict[str, Any]] = []
    default_settle_s = float(os.getenv("ASK2ACT_STRETCH_WAYPOINT_SETTLE_S", "2.0"))
    try:
        try:
            if _truthy("ASK2ACT_STRETCH_HOME_POSE_ON_EXECUTE_START", "1"):
                trace.append(
                    _command_default_pose(
                        robot,
                        include_gripper=_truthy("ASK2ACT_STRETCH_HOME_GRIPPER_ON_EXECUTE_START", "1"),
                        reason="execute_start",
                    )
                )
            base_reference_theta = _current_base_theta(robot)
            for waypoint in trajectory:
                name = str(waypoint.get("name") or "unnamed_waypoint")
                joint_targets = waypoint.get("joint_targets") or {}
                if not isinstance(joint_targets, dict):
                    raise RuntimeError(f"{name}: joint_targets must be an object")
                waypoint_trace: dict[str, Any] = {
                    "name": name,
                    "joint_targets": joint_targets,
                    "status_before": _status_snapshot(robot),
                    "command_trace": [],
                    "ok": False,
                }
                try:
                    for joint_name, target in joint_targets.items():
                        command_result = _move_component(
                            robot,
                            str(joint_name),
                            float(target),
                            base_reference_theta=base_reference_theta,
                        )
                        if command_result is not None:
                            waypoint_trace["command_trace"].append(command_result)
                    robot.push_command()
                    settle_s = max(default_settle_s, float(waypoint.get("settle_s") or 0.0))
                    if settle_s > 0.0:
                        time.sleep(settle_s)
                    waypoint_trace.update(
                        {
                            "ok": True,
                            "settle_s": settle_s,
                            "status_after": _status_snapshot(robot),
                        }
                    )
                    trace.append(waypoint_trace)
                except Exception as waypoint_exc:
                    waypoint_trace.update(
                        {
                            "ok": False,
                            "error": str(waypoint_exc),
                            "traceback": traceback.format_exc(limit=4),
                            "status_after": _status_snapshot(robot),
                        }
                    )
                    trace.append(waypoint_trace)
                    raise
            if _truthy("ASK2ACT_STRETCH_HOME_POSE_ON_EXECUTE_END", "1"):
                trace.append(
                    _command_default_pose(
                        robot,
                        include_gripper=_truthy("ASK2ACT_STRETCH_HOME_GRIPPER_ON_EXECUTE_END", "0"),
                        reason="execute_end",
                    )
                )
        except Exception as exc:
            if _truthy("ASK2ACT_STRETCH_HOME_POSE_ON_EXECUTE_FAILURE", "1"):
                try:
                    trace.append(
                        _command_default_pose(
                            robot,
                            include_gripper=_truthy("ASK2ACT_STRETCH_HOME_GRIPPER_ON_EXECUTE_FAILURE", "0"),
                            reason="execute_failure",
                        )
                    )
                except Exception as home_exc:
                    trace.append(
                        {
                            "name": "default_pose_execute_failure",
                            "ok": False,
                            "error": str(home_exc),
                            "traceback": traceback.format_exc(limit=4),
                        }
                    )
            raise GraspExecutionError(str(exc), trace) from exc
        return trace
    finally:
        try:
            robot.stop()
        except Exception:
            pass


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: dispatch_grasp.py REQUEST_JSON RESPONSE_JSON")

    request_path = Path(argv[1]).expanduser()
    response_path = Path(argv[2]).expanduser()
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    artifact_path = _write_artifact(payload)

    try:
        if payload.get("dry_run"):
            response = {
                "ok": True,
                "execution_status": "dry_run_accepted",
                "artifact_path": str(artifact_path),
                "note": "Dry-run accepted on Stretch; no live motion executed.",
            }
        else:
            trajectory = _trajectory_from_payload(payload)
            if not trajectory:
                raise RuntimeError(
                    "No grasp_plan.trajectory was provided by the A6000. "
                    "The current real-robot executor only runs waypoint trajectories."
                )
            trace = _execute_trajectory(trajectory)
            response = {
                "ok": True,
                "execution_status": "completed",
                "artifact_path": str(artifact_path),
                "waypoint_count": len(trace),
                "trajectory_trace": trace,
            }
    except Exception as exc:
        response = {
            "ok": False,
            "execution_status": "failed",
            "artifact_path": str(artifact_path),
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }
        if isinstance(exc, GraspExecutionError):
            response["trajectory_trace"] = exc.trace

    try:
        _write_response_artifact(artifact_path, response)
    except Exception as artifact_exc:
        response["response_artifact_error"] = str(artifact_exc)
    response_path.write_text(json.dumps(response, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
