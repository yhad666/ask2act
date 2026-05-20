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
        "ASK2ACT_STRETCH_CAPTURE_ARTIFACT_ROOT",
        str(Path(__file__).resolve().parents[1] / "artifacts" / "observations"),
    )
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _head_pose_stamp_path() -> Path:
    instance_id = os.getenv("ASK2ACT_STRETCH_SERVER_INSTANCE_ID", "").strip()
    raw = os.getenv(
        "ASK2ACT_STRETCH_HEAD_POSE_STAMP_PATH",
        str(
            _artifact_root().parent
            / (
                f"head_pose_initialized_{instance_id}.json"
                if instance_id
                else "head_pose_initialized.json"
            )
        ),
    )
    return Path(raw).expanduser()


def _head_pose_override_path() -> Path:
    raw = os.getenv(
        "ASK2ACT_STRETCH_HEAD_POSE_OVERRIDE_PATH",
        str(_artifact_root().parent / "head_pose_override.json"),
    )
    return Path(raw).expanduser()


def _load_head_pose_override() -> dict | None:
    if not _truthy("ASK2ACT_STRETCH_HEAD_POSE_OVERRIDE_ENABLED", "1"):
        return None
    path = _head_pose_override_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "head_pan": float(payload["head_pan_rad"]),
            "head_tilt": float(payload["head_tilt_rad"]),
            "source": "manual_override",
            "path": str(path),
            "saved_at_epoch_s": payload.get("saved_at_epoch_s"),
        }
    except Exception:
        return None


def _save_head_pose_override(head_pan: float, head_tilt: float, *, source: str = "manual_ui") -> dict:
    path = _head_pose_override_path()
    payload = {
        "head_pan_rad": float(head_pan),
        "head_tilt_rad": float(head_tilt),
        "source": source,
        "saved_at_epoch_s": time.time(),
        "note": "Manual head pose override used by future Ask2Act observations and video starts.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"path": str(path), **payload}


def _head_pose_mode() -> str:
    raw = os.getenv("ASK2ACT_STRETCH_INIT_HEAD_POSE_MODE", "once_per_server").strip().lower()
    if raw in {"", "once", "once_per_server"}:
        return "once_per_server"
    if raw in {"every", "every_observe", "per_observe"}:
        return "every_observe"
    if raw in {"0", "false", "no", "off", "disabled"}:
        return "disabled"
    return "once_per_server"


def _head_pose_failure_payload(error: str, note: str, **extra: object) -> dict:
    payload = {
        "ok": False,
        "status": "failed",
        "error": error,
        "note": note,
        "timestamp_epoch_s": time.time(),
    }
    payload.update(extra)
    return payload


def _camera_extrinsics_payload(head_pan: float | None, head_tilt: float | None) -> dict | None:
    if head_pan is None or head_tilt is None:
        return None
    try:
        from .generate_head_camera_extrinsics import compute_camera_color_optical_transform
    except Exception:
        try:
            from generate_head_camera_extrinsics import compute_camera_color_optical_transform
        except Exception:
            return None
    transform = compute_camera_color_optical_transform(float(head_pan), float(head_tilt))
    return {
        "source": "stretch_se3_urdf_chain",
        "parent_frame": "base_link",
        "child_frame": "camera_color_optical_frame",
        "head_pan_rad": float(head_pan),
        "head_tilt_rad": float(head_tilt),
        "camera_to_world": transform.tolist(),
    }


def _reason_key(reason: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in reason.upper()).strip("_")


def _reason_float_env(name: str, reason: str, default: str, *, unit_suffix: str = "") -> float:
    reason_key = _reason_key(reason)
    specific = f"{name}_{reason_key}{unit_suffix}" if reason_key else ""
    return float(os.getenv(specific, os.getenv(f"{name}{unit_suffix}", default)))


def _default_pose_targets(*, reason: str = "") -> dict[str, float]:
    return {
        "lift": _reason_float_env("ASK2ACT_STRETCH_HOME_LIFT", reason, "0.60", unit_suffix="_M"),
        "arm": _reason_float_env("ASK2ACT_STRETCH_HOME_ARM", reason, "0.0", unit_suffix="_M"),
        "wrist_yaw": _reason_float_env("ASK2ACT_STRETCH_HOME_WRIST_YAW", reason, "0.0", unit_suffix="_RAD"),
        "wrist_pitch": _reason_float_env("ASK2ACT_STRETCH_HOME_WRIST_PITCH", reason, "-1.57", unit_suffix="_RAD"),
        "wrist_roll": _reason_float_env("ASK2ACT_STRETCH_HOME_WRIST_ROLL", reason, "0.0", unit_suffix="_RAD"),
        "stretch_gripper": _reason_float_env("ASK2ACT_STRETCH_HOME_GRIPPER", reason, "0.56", unit_suffix="_CMD"),
    }


def _home_settle_s(reason: str) -> float:
    reason_key = _reason_key(reason)
    specific = f"ASK2ACT_STRETCH_HOME_SETTLE_{reason_key}_S" if reason_key else ""
    raw = os.getenv(specific, os.getenv("ASK2ACT_STRETCH_HOME_SETTLE_S", "2.0"))
    return max(0.0, float(raw))


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


def _csv_env(name: str, default: str) -> set[str]:
    raw = os.getenv(name, default)
    return {item.strip() for item in raw.split(",") if item.strip()}


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
    if joint_name in {"wrist_yaw", "wrist_pitch", "wrist_roll"}:
        status = getattr(getattr(robot, "end_of_arm", None), "status", None)
        if isinstance(status, dict):
            entry = status.get(joint_name)
            if isinstance(entry, dict):
                for key in ("pos", "pos_rad", "pos_m"):
                    if key in entry:
                        return float(entry[key])
            if isinstance(entry, (int, float)):
                return float(entry)
    return None


def _home_verify_tolerance(joint_name: str) -> float:
    defaults = {
        "lift": "0.035",
        "arm": "0.025",
        "wrist_yaw": "0.10",
        "wrist_pitch": "0.10",
        "wrist_roll": "0.10",
    }
    key = f"ASK2ACT_STRETCH_HOME_VERIFY_TOLERANCE_{joint_name.upper()}"
    return max(0.0, float(os.getenv(key, defaults.get(joint_name, "0.05"))))


def _verify_default_pose_for_camera(robot: Any, targets: dict[str, float]) -> dict[str, Any]:
    joints = _csv_env("ASK2ACT_STRETCH_HOME_OBSERVE_VERIFY_JOINTS", "lift,arm")
    timeout_s = max(0.0, float(os.getenv("ASK2ACT_STRETCH_HOME_OBSERVE_VERIFY_TIMEOUT_S", "10.0")))
    started_at = time.monotonic()
    results: dict[str, dict[str, Any]] = {}
    pending = set(joints)
    while pending and time.monotonic() - started_at <= timeout_s:
        for joint_name in list(pending):
            if joint_name not in targets:
                pending.remove(joint_name)
                continue
            actual = _read_joint_position(robot, joint_name)
            tolerance = _home_verify_tolerance(joint_name)
            target = float(targets[joint_name])
            error = None if actual is None else float(target - actual)
            results[joint_name] = {
                "target": target,
                "actual": None if actual is None else float(actual),
                "error": error,
                "tolerance": tolerance,
            }
            if actual is not None and abs(error or 0.0) <= tolerance:
                pending.remove(joint_name)
        if pending:
            time.sleep(0.05)
    return {
        "ok": not pending,
        "timeout_s": timeout_s,
        "elapsed_s": round(time.monotonic() - started_at, 3),
        "pending_joints": sorted(pending),
        "joints": results,
    }


def _command_default_pose(robot: Any, *, include_gripper: bool, reason: str) -> dict:
    targets = _default_pose_targets(reason=reason)
    status_before = _status_snapshot(robot)
    if include_gripper:
        robot.end_of_arm.move_to("stretch_gripper", targets["stretch_gripper"])
    robot.arm.move_to(targets["arm"])
    robot.end_of_arm.move_to("wrist_yaw", targets["wrist_yaw"])
    robot.end_of_arm.move_to("wrist_pitch", targets["wrist_pitch"])
    robot.end_of_arm.move_to("wrist_roll", targets["wrist_roll"])
    robot.lift.move_to(targets["lift"])
    robot.push_command()
    settle_s = _home_settle_s(reason)
    if settle_s > 0.0:
        time.sleep(settle_s)
    try:
        robot.pull_status()
    except Exception:
        pass
    status_after = _status_snapshot(robot)
    verify_result = None
    if reason == "observe_start" and _truthy("ASK2ACT_STRETCH_HOME_VERIFY_ON_OBSERVE", "1"):
        verify_result = _verify_default_pose_for_camera(robot, targets)
    return {
        "ok": True if verify_result is None else bool(verify_result.get("ok", False)),
        "status": "default_pose_commanded",
        "error": None
        if verify_result is None or bool(verify_result.get("ok", False))
        else "Camera-safe observe-start pose was not reached before capture",
        "note": None
        if verify_result is None or bool(verify_result.get("ok", False))
        else "The D435i capture was blocked before the arm/lift reached the non-occluding observe pose.",
        "reason": reason,
        "include_gripper": include_gripper,
        "targets": targets if include_gripper else {key: value for key, value in targets.items() if key != "stretch_gripper"},
        "settle_s": settle_s,
        "verify": verify_result,
        "status_before": status_before,
        "status_after": status_after,
        "timestamp_epoch_s": time.time(),
    }


def _ensure_default_pose_before_observe() -> dict | None:
    if not _truthy("ASK2ACT_STRETCH_HOME_POSE_ON_OBSERVE", "0"):
        return None
    try:
        import stretch_body.robot
    except Exception as exc:
        return _head_pose_failure_payload(
            error=str(exc),
            note="stretch_body is unavailable, so the default observe-start pose could not be commanded.",
        )

    robot = stretch_body.robot.Robot()
    if not robot.startup():
        return _head_pose_failure_payload(
            error="Failed to startup Stretch robot for observe-start default pose",
            note="Another process may already be using Stretch.",
        )
    try:
        return _command_default_pose(
            robot,
            include_gripper=_truthy("ASK2ACT_STRETCH_HOME_GRIPPER_ON_OBSERVE", "0"),
            reason="observe_start",
        )
    except Exception as exc:
        return _head_pose_failure_payload(
            error=str(exc),
            note="Failed to command Stretch default pose before observation.",
        )
    finally:
        try:
            robot.stop()
        except Exception:
            pass


def _ensure_initial_head_pose() -> dict | None:
    if not _truthy("ASK2ACT_STRETCH_INIT_HEAD_POSE_ON_START", "1"):
        return None

    mode = _head_pose_mode()
    if mode == "disabled":
        return None

    override = _load_head_pose_override()
    if override is not None:
        return _command_head_pose(
            head_pan=float(override["head_pan"]),
            head_tilt=float(override["head_tilt"]),
            mode="manual_override",
            write_stamp=False,
            pose_source=override,
        )

    stamp_path = _head_pose_stamp_path()
    if mode == "once_per_server" and stamp_path.exists():
        return None

    head_pan = float(os.getenv("ASK2ACT_STRETCH_INIT_HEAD_PAN_RAD", "-1.57"))
    head_tilt = float(os.getenv("ASK2ACT_STRETCH_INIT_HEAD_TILT_RAD", "-0.55"))
    return _command_head_pose(head_pan=head_pan, head_tilt=head_tilt, mode=mode, write_stamp=mode == "once_per_server")


def _command_head_pose(
    *,
    head_pan: float,
    head_tilt: float,
    mode: str = "manual",
    write_stamp: bool = False,
    persist_override: bool = False,
    pose_source: dict | None = None,
) -> dict:
    try:
        import stretch_body.robot
    except Exception as exc:
        return _head_pose_failure_payload(
            error=str(exc),
            note="stretch_body is unavailable, so the head pose could not be commanded.",
            mode=mode,
        )

    settle_s = max(0.0, float(os.getenv("ASK2ACT_STRETCH_INIT_HEAD_SETTLE_S", "2.0")))
    tolerance_rad = max(0.0, float(os.getenv("ASK2ACT_STRETCH_INIT_HEAD_TOLERANCE_RAD", "0.15")))
    stamp_path = _head_pose_stamp_path()

    robot = stretch_body.robot.Robot()
    if not robot.startup():
        return _head_pose_failure_payload(
            error="Failed to startup Stretch robot for head positioning",
            note=(
                "Another process may already be using Stretch. Free the robot process, "
                "or temporarily disable ASK2ACT_STRETCH_INIT_HEAD_POSE_ON_START if you only "
                "want to capture the current camera view."
            ),
            mode=mode,
        )

    try:
        robot.head.move_to("head_pan", head_pan)
        robot.head.move_to("head_tilt", head_tilt)
        robot.push_command()
        if settle_s > 0.0:
            time.sleep(settle_s)
        try:
            robot.pull_status()
            actual_pan = float(robot.head.status["head_pan"]["pos"])
            actual_tilt = float(robot.head.status["head_tilt"]["pos"])
        except Exception:
            actual_pan = None
            actual_tilt = None

        if actual_pan is None or actual_tilt is None:
            return _head_pose_failure_payload(
                error="Unable to verify actual Stretch head pose after commanding it",
                note="The head command was sent, but the final pan/tilt positions could not be read back.",
                mode=mode,
                commanded_head_pan_rad=head_pan,
                commanded_head_tilt_rad=head_tilt,
                actual_head_pan_rad=actual_pan,
                actual_head_tilt_rad=actual_tilt,
                settle_s=settle_s,
                tolerance_rad=tolerance_rad,
                camera_extrinsics=_camera_extrinsics_payload(actual_pan, actual_tilt),
            )

        pan_error = abs(actual_pan - head_pan)
        tilt_error = abs(actual_tilt - head_tilt)
        if pan_error > tolerance_rad or tilt_error > tolerance_rad:
            return _head_pose_failure_payload(
                error="Stretch head pose did not settle near the required tabletop pose",
                note=(
                    "Observation capture should not proceed to detection with the wrong head view. "
                    "Check for robot contention or increase settle time if the motion is simply slow."
                ),
                mode=mode,
                commanded_head_pan_rad=head_pan,
                commanded_head_tilt_rad=head_tilt,
                actual_head_pan_rad=actual_pan,
                actual_head_tilt_rad=actual_tilt,
                pan_error_rad=pan_error,
                tilt_error_rad=tilt_error,
                settle_s=settle_s,
                tolerance_rad=tolerance_rad,
                camera_extrinsics=_camera_extrinsics_payload(actual_pan, actual_tilt),
            )

        payload = {
            "ok": True,
            "status": "initialized",
            "initialized_at_epoch_s": time.time(),
            "mode": mode,
            "commanded_head_pan_rad": head_pan,
            "commanded_head_tilt_rad": head_tilt,
            "actual_head_pan_rad": actual_pan,
            "actual_head_tilt_rad": actual_tilt,
            "pan_error_rad": pan_error,
            "tilt_error_rad": tilt_error,
            "settle_s": settle_s,
            "tolerance_rad": tolerance_rad,
            "pose_source": pose_source or {"source": mode},
            "camera_extrinsics": _camera_extrinsics_payload(actual_pan, actual_tilt),
        }
        if persist_override:
            payload["manual_override"] = _save_head_pose_override(actual_pan, actual_tilt, source=mode)
        if write_stamp:
            stamp_path.parent.mkdir(parents=True, exist_ok=True)
            stamp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    except Exception as exc:
        return _head_pose_failure_payload(
            error=str(exc),
            note="The head pose command failed, but observation capture will continue.",
            mode=mode,
        )
    finally:
        try:
            robot.stop()
        except Exception:
            pass


def _capture_with_realsense() -> dict:
    import numpy as np
    import pyrealsense2 as rs
    from PIL import Image

    default_pose_result = _ensure_default_pose_before_observe()
    if default_pose_result and not bool(default_pose_result.get("ok", False)):
        if _truthy("ASK2ACT_STRETCH_HOME_POSE_REQUIRED", "1"):
            error = str(default_pose_result.get("error") or "Default observe-start pose failed")
            note = str(default_pose_result.get("note") or "")
            raise RuntimeError(error if not note else f"{error}. {note}")

    head_pose_result = _ensure_initial_head_pose()
    if head_pose_result and not bool(head_pose_result.get("ok", False)):
        if _truthy("ASK2ACT_STRETCH_INIT_HEAD_POSE_REQUIRED", "0"):
            error = str(head_pose_result.get("error") or "Initial head pose failed")
            note = str(head_pose_result.get("note") or "")
            raise RuntimeError(error if not note else f"{error}. {note}")

    serial = (
        os.getenv("ASK2ACT_STRETCH_D435I_SERIAL", "").strip()
        or os.getenv("ASK2ACT_STRETCH_CAMERA_SERIAL", "").strip()
    )
    width = int(os.getenv("ASK2ACT_STRETCH_CAMERA_WIDTH", "1280"))
    height = int(os.getenv("ASK2ACT_STRETCH_CAMERA_HEIGHT", "720"))
    fps = int(os.getenv("ASK2ACT_STRETCH_CAMERA_FPS", "15"))
    warmup_frames = int(os.getenv("ASK2ACT_STRETCH_CAMERA_WARMUP_FRAMES", "20"))
    timeout_ms = int(os.getenv("ASK2ACT_STRETCH_CAMERA_TIMEOUT_MS", "5000"))
    capture_depth = _truthy("ASK2ACT_STRETCH_CAPTURE_DEPTH", "1")

    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    if capture_depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = None
    try:
        profile = pipeline.start(config)
        for _ in range(max(1, warmup_frames)):
            pipeline.wait_for_frames(timeout_ms=timeout_ms)
        frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
        if capture_depth:
            frames = rs.align(rs.stream.color).process(frames)
        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError("No color frame returned from D435i")
        depth_frame = frames.get_depth_frame() if capture_depth else None
        if capture_depth and depth_frame is None:
            raise RuntimeError("No depth frame returned from D435i")

        bgr = np.asanyarray(color_frame.get_data())
        rgb = bgr[..., ::-1]

        stamp = time.strftime("%Y%m%d_%H%M%S")
        image_path = _artifact_root() / f"head_d435i_{stamp}.jpg"
        Image.fromarray(rgb).save(image_path, quality=95)

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_stream.get_intrinsics()
        intrinsics_payload = {
            "width": int(intrinsics.width),
            "height": int(intrinsics.height),
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "cx": float(intrinsics.ppx),
            "cy": float(intrinsics.ppy),
            "coeffs": [float(v) for v in intrinsics.coeffs],
            "model": str(intrinsics.model),
        }
        intrinsics_path = _artifact_root() / f"head_d435i_intrinsics_{stamp}.json"
        intrinsics_path.write_text(json.dumps(intrinsics_payload, indent=2), encoding="utf-8")

        depth_path = None
        depth_scale = None
        depth_shape = None
        if depth_frame is not None:
            depth = np.asanyarray(depth_frame.get_data())
            depth_shape = [int(depth.shape[0]), int(depth.shape[1])]
            depth_path = _artifact_root() / f"head_d435i_depth_{stamp}.npy"
            np.save(depth_path, depth)
            try:
                depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            except Exception:
                depth_scale = None

        device = profile.get_device()
        serial_out = device.get_info(rs.camera_info.serial_number)
        return {
            "ok": True,
            "observation_id": f"d435i-{stamp}",
            "mime_type": "image/jpeg",
            "image_path": str(image_path),
            "source": "realsense_d435i",
            "camera_serial": serial_out,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "rgb_shape_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
            "depth_shape_hw": depth_shape,
            "depth_npy_path": str(depth_path) if depth_path is not None else None,
            "depth_scale_m_per_unit": depth_scale,
            "depth_aligned_to_color": bool(depth_frame is not None),
            "camera_intrinsics_path": str(intrinsics_path),
            "camera_intrinsics": intrinsics_payload,
            "camera_extrinsics": (head_pose_result or {}).get("camera_extrinsics"),
            "default_pose_init": default_pose_result,
            "head_pose_init": head_pose_result,
        }
    finally:
        if profile is not None:
            pipeline.stop()


def _sample_fallback(request: dict) -> dict:
    image_path = (
        Path(__file__).resolve().parents[3]
        / "simulation"
        / "logs"
        / "sim_validation"
        / "data_samples"
        / "tabletop_scene"
        / "head_d435i_rgb.png"
    )
    return {
        "ok": True,
        "observation_id": f"sample-{request.get('session_id', 'session')}-{int(time.time())}",
        "mime_type": "image/png",
        "image_path": str(image_path),
        "source": "sample_fallback",
        "note": "RealSense capture failed; sample fallback was explicitly enabled.",
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: capture_observation.py REQUEST_JSON RESPONSE_JSON")

    request_path = Path(argv[1]).expanduser()
    response_path = Path(argv[2]).expanduser()
    request = json.loads(request_path.read_text(encoding="utf-8"))

    try:
        response = _capture_with_realsense()
    except Exception as exc:
        if _truthy("ASK2ACT_STRETCH_ALLOW_SAMPLE_FALLBACK", "0"):
            response = _sample_fallback(request)
        else:
            response = {
                "ok": False,
                "error": str(exc),
                "source": "realsense_d435i",
                "note": (
                    "Failed to capture from the real D435i. Install/configure pyrealsense2 on Stretch, "
                    "or set ASK2ACT_STRETCH_ALLOW_SAMPLE_FALLBACK=1 only for debugging."
                ),
                "traceback": traceback.format_exc(limit=6),
            }

    response_path.write_text(json.dumps(response, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
