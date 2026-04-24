from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


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


def _head_pose_failure_payload(error: str, note: str) -> dict:
    return {
        "ok": False,
        "status": "failed",
        "error": error,
        "note": note,
        "timestamp_epoch_s": time.time(),
    }


def _ensure_initial_head_pose() -> dict | None:
    if not _truthy("ASK2ACT_STRETCH_INIT_HEAD_POSE_ON_START", "1"):
        return None

    stamp_path = _head_pose_stamp_path()
    if stamp_path.exists():
        return None

    try:
        import stretch_body.robot
    except Exception as exc:
        return _head_pose_failure_payload(
            error=str(exc),
            note="stretch_body is unavailable, so the initial head pose could not be commanded.",
        )

    head_pan = float(os.getenv("ASK2ACT_STRETCH_INIT_HEAD_PAN_RAD", "-1.57"))
    head_tilt = float(os.getenv("ASK2ACT_STRETCH_INIT_HEAD_TILT_RAD", "-0.55"))
    settle_s = max(0.0, float(os.getenv("ASK2ACT_STRETCH_INIT_HEAD_SETTLE_S", "2.0")))

    robot = stretch_body.robot.Robot()
    if not robot.startup():
        return _head_pose_failure_payload(
            error="Failed to startup Stretch robot for initial head positioning",
            note=(
                "Another process may already be using Stretch. Free the robot process, "
                "or temporarily disable ASK2ACT_STRETCH_INIT_HEAD_POSE_ON_START if you only "
                "want to capture the current camera view."
            ),
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

        payload = {
            "ok": True,
            "status": "initialized",
            "initialized_at_epoch_s": time.time(),
            "commanded_head_pan_rad": head_pan,
            "commanded_head_tilt_rad": head_tilt,
            "actual_head_pan_rad": actual_pan,
            "actual_head_tilt_rad": actual_tilt,
            "settle_s": settle_s,
        }
        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        stamp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    except Exception as exc:
        return _head_pose_failure_payload(
            error=str(exc),
            note="The initial head pose command failed, but observation capture will continue.",
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

    head_pose_result = _ensure_initial_head_pose()

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
        if depth_frame is not None:
            depth = np.asanyarray(depth_frame.get_data())
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
            "depth_npy_path": str(depth_path) if depth_path is not None else None,
            "depth_scale_m_per_unit": depth_scale,
            "camera_intrinsics_path": str(intrinsics_path),
            "camera_intrinsics": intrinsics_payload,
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

    response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
