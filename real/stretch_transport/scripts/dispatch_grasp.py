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


def _map_stretch_gripper_target(target: float) -> tuple[float, str]:
    """Map planner gripper commands to the real Stretch Body gripper units."""
    mode = os.getenv("ASK2ACT_STRETCH_GRIPPER_COMMAND_MODE", "real_pct").strip().lower()
    target = float(target)
    if mode in {"raw", "passthrough", "planner"}:
        return target, mode

    open_threshold = float(os.getenv("ASK2ACT_STRETCH_GRIPPER_PLANNER_OPEN_THRESHOLD", "0.5"))
    close_threshold = float(os.getenv("ASK2ACT_STRETCH_GRIPPER_PLANNER_CLOSE_THRESHOLD", "-0.3"))
    open_cmd = float(os.getenv("ASK2ACT_STRETCH_GRIPPER_REAL_OPEN_CMD", "100.0"))
    close_cmd = float(os.getenv("ASK2ACT_STRETCH_GRIPPER_REAL_CLOSE_CMD", "-50.0"))
    if target >= open_threshold:
        return open_cmd, mode
    if target <= close_threshold:
        return close_cmd, mode
    return target, mode


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
    command_targets = dict(targets)
    status_before = _status_snapshot(robot)
    if include_gripper:
        command_targets["stretch_gripper"], gripper_command_mode = _map_stretch_gripper_target(targets["stretch_gripper"])
        robot.end_of_arm.move_to("stretch_gripper", command_targets["stretch_gripper"])
    else:
        gripper_command_mode = None
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
        "command_targets": command_targets if include_gripper else {key: value for key, value in command_targets.items() if key != "stretch_gripper"},
        "gripper_command_mode": gripper_command_mode,
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


def _current_base_xy(robot: Any) -> tuple[float, float] | None:
    try:
        robot.pull_status()
    except Exception:
        pass
    status = getattr(getattr(robot, "base", None), "status", None)
    if not isinstance(status, dict):
        return None
    for x_key, y_key in (("x", "y"), ("x_m", "y_m")):
        if x_key in status and y_key in status:
            try:
                return float(status[x_key]), float(status[y_key])
            except Exception:
                return None
    return None


def _angle_diff_rad(target: float, current: float) -> float:
    return float((target - current + math.pi) % (2.0 * math.pi) - math.pi)


def _read_joint_position(robot: Any, joint_name: str) -> float | None:
    try:
        robot.pull_status()
    except Exception:
        pass

    if joint_name == "lift":
        status = getattr(getattr(robot, "lift", None), "status", None)
        if isinstance(status, dict) and "pos" in status:
            return float(status["pos"])
    if joint_name == "arm":
        status = getattr(getattr(robot, "arm", None), "status", None)
        if isinstance(status, dict) and "pos" in status:
            return float(status["pos"])
    if joint_name in {"wrist_yaw", "wrist_pitch", "wrist_roll", "stretch_gripper"}:
        status = getattr(getattr(robot, "end_of_arm", None), "status", None)
        if isinstance(status, dict):
            candidates = [joint_name]
            if joint_name == "stretch_gripper":
                candidates.extend(["gripper", "stretch_gripper"])
            for key in candidates:
                entry = status.get(key)
                if isinstance(entry, dict):
                    for pos_key in ("pos", "pos_rad", "pos_m", "pos_pct"):
                        if pos_key in entry:
                            return float(entry[pos_key])
                elif isinstance(entry, (int, float)):
                    return float(entry)
    if joint_name in {"head_pan", "head_tilt"}:
        status = getattr(getattr(robot, "head", None), "status", None)
        if isinstance(status, dict):
            entry = status.get(joint_name)
            if isinstance(entry, dict) and "pos" in entry:
                return float(entry["pos"])
            if isinstance(entry, (int, float)):
                return float(entry)
    return None


def _joint_tolerance(joint_name: str, waypoint_name: str) -> float:
    defaults = {
        "lift": "0.025",
        "arm": "0.025",
        "wrist_yaw": "0.08",
        "wrist_pitch": "0.08",
        "wrist_roll": "0.08",
        "stretch_gripper": "0.08",
        "head_pan": "0.08",
        "head_tilt": "0.08",
    }
    specific = f"ASK2ACT_STRETCH_WAIT_TOLERANCE_{joint_name.upper()}".replace("STRETCH_GRIPPER", "GRIPPER")
    raw = os.getenv(specific, os.getenv("ASK2ACT_STRETCH_WAIT_TOLERANCE_DEFAULT", defaults.get(joint_name, "0.05")))
    tolerance = max(0.0, float(raw))
    if waypoint_name == "descend_to_grasp" and joint_name == "lift":
        tolerance = max(0.0, float(os.getenv("ASK2ACT_STRETCH_DESCEND_LIFT_TOLERANCE_M", str(tolerance))))
    return tolerance


def _joint_timeout_s(joint_name: str, waypoint_name: str) -> float:
    default = float(os.getenv("ASK2ACT_STRETCH_JOINT_WAIT_TIMEOUT_S", "20.0"))
    if joint_name == "lift":
        default = float(os.getenv("ASK2ACT_STRETCH_LIFT_WAIT_TIMEOUT_S", str(default)))
    elif joint_name == "arm":
        default = float(os.getenv("ASK2ACT_STRETCH_ARM_WAIT_TIMEOUT_S", str(default)))
    elif joint_name == "stretch_gripper":
        default = float(os.getenv("ASK2ACT_STRETCH_GRIPPER_WAIT_TIMEOUT_S", "8.0"))
    elif joint_name.startswith("wrist_"):
        default = float(os.getenv("ASK2ACT_STRETCH_WRIST_WAIT_TIMEOUT_S", str(default)))
    if waypoint_name == "descend_to_grasp":
        default = float(os.getenv("ASK2ACT_STRETCH_DESCEND_WAIT_TIMEOUT_S", str(default)))
    return max(0.0, default)


def _wait_required_for_joint(joint_name: str) -> bool:
    raw = os.getenv("ASK2ACT_STRETCH_REQUIRED_WAIT_JOINTS", "lift,arm,base_rotate,base_translate_arm_axis")
    required = {item.strip() for item in raw.split(",") if item.strip()}
    return joint_name in required


def _wait_for_joint_target(
    robot: Any,
    *,
    joint_name: str,
    target: float,
    waypoint_name: str,
) -> dict[str, Any]:
    if joint_name == "base_translate_arm_axis":
        return {"joint_name": joint_name, "ok": True, "skipped": True, "reason": "waited_inside_command"}
    if joint_name == "base_rotate":
        return {"joint_name": joint_name, "ok": True, "skipped": True, "reason": "handled_by_base_theta_wait"}

    required = _wait_required_for_joint(joint_name)
    timeout_s = _joint_timeout_s(joint_name, waypoint_name)
    tolerance = _joint_tolerance(joint_name, waypoint_name)
    started_at = time.monotonic()
    samples: list[dict[str, float]] = []
    actual = _read_joint_position(robot, joint_name)
    ok = False
    while time.monotonic() - started_at <= timeout_s:
        actual = _read_joint_position(robot, joint_name)
        if actual is None:
            if not required and time.monotonic() - started_at >= min(0.5, timeout_s):
                return {
                    "joint_name": joint_name,
                    "target": float(target),
                    "actual": None,
                    "ok": True,
                    "required": False,
                    "warning": "joint_status_unavailable; command was sent but wait is non-blocking for this joint",
                    "timeout_s": timeout_s,
                    "tolerance": tolerance,
                    "samples": samples[-6:],
                }
            time.sleep(0.05)
            continue
        error = float(target - actual)
        if len(samples) < 5 or abs(error) <= tolerance:
            samples.append(
                {
                    "elapsed_s": round(time.monotonic() - started_at, 3),
                    "actual": float(actual),
                    "error": error,
                }
            )
        if abs(error) <= tolerance:
            ok = True
            break
        if joint_name == "stretch_gripper" and target >= 50.0:
            open_threshold = float(os.getenv("ASK2ACT_STRETCH_GRIPPER_OPEN_ACCEPT_POS", "80.0"))
            planner_open_threshold = float(os.getenv("ASK2ACT_STRETCH_GRIPPER_OPEN_ACCEPT_PLANNER_POS", "0.45"))
            if actual >= open_threshold or (actual <= 1.5 and actual >= planner_open_threshold):
                ok = True
                break
        if joint_name == "stretch_gripper" and target <= -0.35:
            close_threshold = float(os.getenv("ASK2ACT_STRETCH_GRIPPER_CLOSE_ACCEPT_POS", "0.05"))
            if actual <= close_threshold:
                ok = True
                break
        if not required and time.monotonic() - started_at >= min(0.75, timeout_s):
            return {
                "joint_name": joint_name,
                "target": float(target),
                "actual": float(actual),
                "ok": True,
                "required": False,
                "warning": "non_required_joint_wait_recorded_only",
                "timeout_s": timeout_s,
                "tolerance": tolerance,
                "samples": samples[-6:],
            }
        time.sleep(0.05)
    return {
        "joint_name": joint_name,
        "target": float(target),
        "actual": None if actual is None else float(actual),
        "ok": ok,
        "required": required,
        "timeout_s": timeout_s,
        "tolerance": tolerance,
        "samples": samples[-6:],
    }


def _wait_for_base_theta(
    robot: Any,
    *,
    target_theta: float,
    timeout_s: float,
    tolerance_rad: float,
) -> dict[str, Any]:
    started_at = time.monotonic()
    samples: list[dict[str, float]] = []
    final_theta = _current_base_theta(robot)
    ok = False
    while time.monotonic() - started_at <= max(0.0, timeout_s):
        final_theta = _current_base_theta(robot)
        error = _angle_diff_rad(target_theta, final_theta)
        if len(samples) < 5 or abs(error) <= tolerance_rad:
            samples.append(
                {
                    "elapsed_s": round(time.monotonic() - started_at, 3),
                    "theta_rad": final_theta,
                    "error_rad": error,
                }
            )
        if abs(error) <= tolerance_rad:
            ok = True
            break
        time.sleep(0.05)
    final_error = _angle_diff_rad(target_theta, final_theta)
    return {
        "ok": ok,
        "target_theta_rad": float(target_theta),
        "final_theta_rad": float(final_theta),
        "final_error_rad": float(final_error),
        "timeout_s": float(timeout_s),
        "tolerance_rad": float(tolerance_rad),
        "samples": samples[-6:],
    }


def _wait_for_base_translation_delta(
    robot: Any,
    *,
    start_xy: tuple[float, float] | None,
    start_theta: float,
    target_distance_m: float,
    timeout_s: float,
    tolerance_m: float,
) -> dict[str, Any]:
    if start_xy is None:
        time.sleep(max(0.0, timeout_s))
        return {
            "ok": True,
            "used_odometry": False,
            "reason": "base_status_xy_unavailable",
            "target_distance_m": float(target_distance_m),
            "timeout_s": float(timeout_s),
        }

    started_at = time.monotonic()
    axis = (math.cos(start_theta), math.sin(start_theta))
    start_x, start_y = start_xy
    final_xy = start_xy
    actual_distance = 0.0
    ok = False
    samples: list[dict[str, float]] = []
    while time.monotonic() - started_at <= max(0.0, timeout_s):
        current_xy = _current_base_xy(robot)
        if current_xy is None:
            break
        final_xy = current_xy
        actual_distance = (current_xy[0] - start_x) * axis[0] + (current_xy[1] - start_y) * axis[1]
        error = float(target_distance_m - actual_distance)
        if len(samples) < 5 or abs(error) <= tolerance_m:
            samples.append(
                {
                    "elapsed_s": round(time.monotonic() - started_at, 3),
                    "actual_distance_m": float(actual_distance),
                    "error_m": error,
                }
            )
        if abs(error) <= tolerance_m:
            ok = True
            break
        time.sleep(0.05)
    return {
        "ok": ok,
        "used_odometry": True,
        "target_distance_m": float(target_distance_m),
        "actual_distance_m": float(actual_distance),
        "final_error_m": float(target_distance_m - actual_distance),
        "start_xy": [float(start_xy[0]), float(start_xy[1])],
        "final_xy": [float(final_xy[0]), float(final_xy[1])],
        "drive_theta_rad": float(start_theta),
        "timeout_s": float(timeout_s),
        "tolerance_m": float(tolerance_m),
        "samples": samples[-6:],
    }


def _move_base_translate_arm_axis(robot: Any, distance_m: float) -> dict[str, Any]:
    if not _truthy("ASK2ACT_STRETCH_BASE_TRANSLATE_ARM_AXIS_ENABLED", "1"):
        raise RuntimeError("base_translate_arm_axis requested but ASK2ACT_STRETCH_BASE_TRANSLATE_ARM_AXIS_ENABLED=0")
    requested = float(distance_m)
    eps = float(os.getenv("ASK2ACT_STRETCH_BASE_TRANSLATE_EPS_M", "0.01"))
    if abs(requested) <= eps:
        return {
            "joint_name": "base_translate_arm_axis",
            "requested_distance_m": requested,
            "skipped": True,
            "skip_reason": f"abs(distance) <= {eps}",
        }
    max_distance = float(os.getenv("ASK2ACT_STRETCH_BASE_TRANSLATE_ARM_AXIS_MAX_M", "0.18"))
    if abs(requested) > max_distance:
        raise RuntimeError(
            f"Refusing base_translate_arm_axis distance {requested:.3f} m; "
            f"max is {max_distance:.3f} m"
        )

    translate_timeout_s = max(0.0, float(os.getenv("ASK2ACT_STRETCH_BASE_TRANSLATE_ARM_AXIS_TIMEOUT_S", "8.0")))
    rotate_timeout_s = max(0.0, float(os.getenv("ASK2ACT_STRETCH_BASE_TRANSLATE_ARM_AXIS_ROTATE_TIMEOUT_S", "8.0")))
    translate_tolerance_m = max(0.001, float(os.getenv("ASK2ACT_STRETCH_BASE_TRANSLATE_TOLERANCE_M", "0.02")))
    rotate_tolerance_rad = max(0.001, float(os.getenv("ASK2ACT_STRETCH_BASE_TRANSLATE_ROTATE_TOLERANCE_RAD", "0.035")))
    start_theta = _current_base_theta(robot)
    clockwise_quarter_turn = -math.pi / 2.0
    side_theta = start_theta + clockwise_quarter_turn
    status_before = _status_snapshot(robot)

    robot.base.rotate_by(clockwise_quarter_turn)
    robot.push_command()
    rotate_to_side = _wait_for_base_theta(
        robot,
        target_theta=side_theta,
        timeout_s=rotate_timeout_s,
        tolerance_rad=rotate_tolerance_rad,
    )
    if not rotate_to_side["ok"]:
        raise RuntimeError(
            "base_translate_arm_axis failed while rotating toward table: "
            f"theta_error={float(rotate_to_side['final_error_rad']):.3f} rad"
        )

    drive_start_theta = _current_base_theta(robot)
    drive_start_xy = _current_base_xy(robot)
    robot.base.translate_by(requested)
    robot.push_command()
    translate_wait = _wait_for_base_translation_delta(
        robot,
        start_xy=drive_start_xy,
        start_theta=drive_start_theta,
        target_distance_m=requested,
        timeout_s=translate_timeout_s,
        tolerance_m=translate_tolerance_m,
    )
    if not translate_wait["ok"]:
        raise RuntimeError(
            "base_translate_arm_axis failed while translating for reach: "
            f"distance_error={float(translate_wait.get('final_error_m', 0.0)):.3f} m"
        )

    before_return_theta = _current_base_theta(robot)
    robot.base.rotate_by(_angle_diff_rad(start_theta, before_return_theta))
    robot.push_command()
    rotate_back = _wait_for_base_theta(
        robot,
        target_theta=start_theta,
        timeout_s=rotate_timeout_s,
        tolerance_rad=rotate_tolerance_rad,
    )
    if not rotate_back["ok"]:
        raise RuntimeError(
            "base_translate_arm_axis failed while rotating back to grasp heading: "
            f"theta_error={float(rotate_back['final_error_rad']):.3f} rad"
        )

    final_theta = _current_base_theta(robot)
    return {
        "joint_name": "base_translate_arm_axis",
        "requested_distance_m": requested,
        "side_turn_rad": clockwise_quarter_turn,
        "start_theta_rad": start_theta,
        "side_theta_rad": side_theta,
        "rotate_to_side": rotate_to_side,
        "translate_wait": translate_wait,
        "rotate_back": rotate_back,
        "final_theta_rad": final_theta,
        "final_theta_error_rad": _angle_diff_rad(start_theta, final_theta),
        "translate_timeout_s": translate_timeout_s,
        "rotate_timeout_s": rotate_timeout_s,
        "translate_tolerance_m": translate_tolerance_m,
        "rotate_tolerance_rad": rotate_tolerance_rad,
        "skipped": False,
        "status_before": status_before,
        "status_after": _status_snapshot(robot),
    }


def _move_component(robot: Any, joint_name: str, target: float, *, base_reference_theta: float) -> dict[str, Any] | None:
    if joint_name == "lift":
        robot.lift.move_to(target)
        return {"joint_name": joint_name, "target": float(target), "component": "lift"}
    elif joint_name == "arm":
        robot.arm.move_to(target)
        return {"joint_name": joint_name, "target": float(target), "component": "arm"}
    elif joint_name in {"wrist_yaw", "wrist_pitch", "wrist_roll", "stretch_gripper"}:
        command_target = float(target)
        command_mode = None
        if joint_name == "stretch_gripper":
            command_target, command_mode = _map_stretch_gripper_target(target)
        robot.end_of_arm.move_to(joint_name, command_target)
        result = {
            "joint_name": joint_name,
            "target": float(target),
            "command_target": float(command_target),
            "component": "end_of_arm",
        }
        if command_mode is not None:
            result["command_mode"] = command_mode
        return result
    elif joint_name in {"head_pan", "head_tilt"}:
        robot.head.move_to(joint_name, target)
        return {"joint_name": joint_name, "target": float(target), "component": "head"}
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
    elif joint_name == "base_translate_arm_axis":
        return _move_base_translate_arm_axis(robot, target)
    else:
        raise RuntimeError(f"Unsupported real-robot joint target: {joint_name}")


def _execute_trajectory(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import stretch_body.robot

    robot = stretch_body.robot.Robot()
    if not robot.startup():
        raise RuntimeError("Failed to startup Stretch robot")

    trace: list[dict[str, Any]] = []
    default_settle_s = float(os.getenv("ASK2ACT_STRETCH_WAYPOINT_SETTLE_S", "2.0"))
    deadline_s = float(os.getenv("ASK2ACT_STRETCH_EXECUTE_DEADLINE_S", "90.0"))
    started_at = time.monotonic()

    def check_deadline(stage: str) -> None:
        if deadline_s <= 0.0:
            return
        elapsed_s = time.monotonic() - started_at
        if elapsed_s > deadline_s:
            raise RuntimeError(f"Stretch execution deadline exceeded during {stage}: {elapsed_s:.1f}s > {deadline_s:.1f}s")

    try:
        try:
            check_deadline("execute_start")
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
                check_deadline(f"{waypoint.get('name') or 'unnamed_waypoint'} before_command")
                name = str(waypoint.get("name") or "unnamed_waypoint")
                joint_targets = waypoint.get("joint_targets") or {}
                if not isinstance(joint_targets, dict):
                    raise RuntimeError(f"{name}: joint_targets must be an object")
                waypoint_trace: dict[str, Any] = {
                    "name": name,
                    "joint_targets": joint_targets,
                    "status_before": _status_snapshot(robot),
                    "command_trace": [],
                    "wait_trace": [],
                    "ok": False,
                }
                try:
                    wait_targets: dict[str, float] = {}
                    for joint_name, target in joint_targets.items():
                        joint_name_str = str(joint_name)
                        command_result = _move_component(
                            robot,
                            joint_name_str,
                            float(target),
                            base_reference_theta=base_reference_theta,
                        )
                        if command_result is not None:
                            waypoint_trace["command_trace"].append(command_result)
                            wait_targets[joint_name_str] = float(command_result.get("command_target", target))
                        else:
                            wait_targets[joint_name_str] = float(target)
                    robot.push_command()
                    waypoint_ok = True
                    for joint_name, target in joint_targets.items():
                        joint_name_str = str(joint_name)
                        wait_target = wait_targets.get(joint_name_str, float(target))
                        if joint_name_str == "base_rotate":
                            target_theta = base_reference_theta + float(wait_target)
                            wait_result = _wait_for_base_theta(
                                robot,
                                target_theta=target_theta,
                                timeout_s=float(os.getenv("ASK2ACT_STRETCH_BASE_ROTATE_WAIT_TIMEOUT_S", "12.0")),
                                tolerance_rad=float(os.getenv("ASK2ACT_STRETCH_BASE_ROTATE_WAIT_TOLERANCE_RAD", "0.035")),
                            )
                            wait_result["required"] = _wait_required_for_joint(joint_name_str)
                        else:
                            wait_result = _wait_for_joint_target(
                                robot,
                                joint_name=joint_name_str,
                                target=float(wait_target),
                                waypoint_name=name,
                            )
                        waypoint_trace["wait_trace"].append(wait_result)
                        waypoint_ok = waypoint_ok and (
                            bool(wait_result.get("ok", False)) or not bool(wait_result.get("required", True))
                        )
                    if not waypoint_ok:
                        failed_waits = [
                            item
                            for item in waypoint_trace["wait_trace"]
                            if isinstance(item, dict) and not bool(item.get("ok", False))
                        ]
                        raise RuntimeError(f"{name}: failed to reach waypoint targets: {failed_waits}")
                    settle_s = max(default_settle_s, float(waypoint.get("settle_s") or 0.0))
                    if settle_s > 0.0:
                        time.sleep(settle_s)
                    check_deadline(f"{name} after_settle")
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
