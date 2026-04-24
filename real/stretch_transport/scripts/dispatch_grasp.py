from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


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


def _trajectory_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    grasp_plan = payload.get("grasp_plan") or {}
    trajectory = grasp_plan.get("trajectory") or []
    if not isinstance(trajectory, list):
        raise RuntimeError("grasp_plan.trajectory must be a list when present")
    return trajectory


def _current_base_theta(robot: Any) -> float:
    try:
        robot.pull_status()
    except Exception:
        pass
    try:
        return float(robot.base.status["theta"])
    except Exception as exc:
        raise RuntimeError("Unable to read current Stretch base theta") from exc


def _move_component(robot: Any, joint_name: str, target: float) -> None:
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
        robot.base.rotate_by(float(target - current_theta))
    else:
        raise RuntimeError(f"Unsupported real-robot joint target: {joint_name}")


def _execute_trajectory(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import stretch_body.robot

    robot = stretch_body.robot.Robot()
    if not robot.startup():
        raise RuntimeError("Failed to startup Stretch robot")

    trace: list[dict[str, Any]] = []
    default_settle_s = float(os.getenv("ASK2ACT_STRETCH_WAYPOINT_SETTLE_S", "2.0"))
    try:
        for waypoint in trajectory:
            name = str(waypoint.get("name") or "unnamed_waypoint")
            joint_targets = waypoint.get("joint_targets") or {}
            if not isinstance(joint_targets, dict):
                raise RuntimeError(f"{name}: joint_targets must be an object")
            for joint_name, target in joint_targets.items():
                _move_component(robot, str(joint_name), float(target))
            robot.push_command()
            settle_s = max(default_settle_s, float(waypoint.get("settle_s") or 0.0))
            if settle_s > 0.0:
                time.sleep(settle_s)
            try:
                robot.pull_status()
            except Exception:
                pass
            trace.append(
                {
                    "name": name,
                    "joint_targets": joint_targets,
                    "ok": True,
                    "settle_s": settle_s,
                }
            )
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

    response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
