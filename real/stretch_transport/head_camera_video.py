from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _artifact_root() -> Path:
    raw = os.getenv(
        "ASK2ACT_STRETCH_CAPTURE_ARTIFACT_ROOT",
        str(Path(__file__).resolve().parent / "artifacts" / "observations"),
    )
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _video_root(default_root: Path) -> Path:
    raw = os.getenv("ASK2ACT_STRETCH_VIDEO_ARTIFACT_ROOT", str(default_root / "head_camera_video"))
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _load_capture_module():
    try:
        from .scripts import capture_observation as capture_module

        return capture_module
    except Exception:
        return None


class _VideoWriter:
    def __init__(self, *, output_path: Path, source_width: int, source_height: int, fps: int, out_width: int, out_height: int):
        self.output_path = output_path
        self.source_width = int(source_width)
        self.source_height = int(source_height)
        self.fps = max(1, int(fps))
        self.out_width = int(out_width)
        self.out_height = int(out_height)
        self.backend = ""
        self._proc: subprocess.Popen | None = None
        self._cv2 = None
        self._cv2_writer = None

    def open(self) -> None:
        ffmpeg_bin = os.getenv("ASK2ACT_STRETCH_VIDEO_FFMPEG_BIN", "ffmpeg").strip() or "ffmpeg"
        if shutil.which(ffmpeg_bin):
            command = [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s:v",
                f"{self.source_width}x{self.source_height}",
                "-r",
                str(self.fps),
                "-i",
                "pipe:0",
                "-vf",
                f"scale={self.out_width}:{self.out_height}",
                "-an",
                "-vcodec",
                "libx264",
                "-preset",
                os.getenv("ASK2ACT_STRETCH_VIDEO_FFMPEG_PRESET", "ultrafast"),
                "-crf",
                os.getenv("ASK2ACT_STRETCH_VIDEO_FFMPEG_CRF", "28"),
                "-pix_fmt",
                "yuv420p",
                str(self.output_path),
            ]
            self._proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.backend = "ffmpeg"
            return

        try:
            import cv2  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Head-camera video needs either ffmpeg on PATH or python package cv2/opencv. "
                "Install ffmpeg on Stretch or install opencv-python-headless in the robot Python env."
            ) from exc
        fourcc = cv2.VideoWriter_fourcc(*os.getenv("ASK2ACT_STRETCH_VIDEO_CV2_FOURCC", "mp4v"))
        writer = cv2.VideoWriter(str(self.output_path), fourcc, float(self.fps), (self.out_width, self.out_height))
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV VideoWriter could not open {self.output_path}")
        self._cv2 = cv2
        self._cv2_writer = writer
        self.backend = "opencv"

    def write(self, bgr_frame: Any) -> None:
        if self._proc is not None:
            if self._proc.stdin is None:
                raise RuntimeError("ffmpeg stdin is closed")
            self._proc.stdin.write(bgr_frame.tobytes())
            return
        if self._cv2_writer is None:
            raise RuntimeError("video writer is not open")
        frame = bgr_frame
        if int(frame.shape[1]) != self.out_width or int(frame.shape[0]) != self.out_height:
            frame = self._cv2.resize(frame, (self.out_width, self.out_height), interpolation=self._cv2.INTER_AREA)
        self._cv2_writer.write(frame)

    def close(self) -> Dict[str, Any]:
        if self._proc is not None:
            stderr = ""
            if self._proc.stdin is not None:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            try:
                self._proc.wait(timeout=30)
                stderr = (self._proc.stderr.read() if self._proc.stderr else b"").decode("utf-8", errors="replace").strip()
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
                stderr = (self._proc.stderr.read() if self._proc.stderr else b"").decode("utf-8", errors="replace").strip()
            return {"backend": self.backend, "returncode": self._proc.returncode, "stderr": stderr}
        if self._cv2_writer is not None:
            self._cv2_writer.release()
            return {"backend": self.backend, "returncode": 0, "stderr": ""}
        return {"backend": self.backend or "none", "returncode": None, "stderr": ""}


class HeadCameraVideoManager:
    """Owns the D435i while recording so online observe can reuse the same high-res stream."""

    def __init__(self, *, artifact_root: Path) -> None:
        self.artifact_root = Path(artifact_root).expanduser()
        self.video_root = _video_root(self.artifact_root)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._recording: Dict[str, Any] | None = None
        self._latest: Dict[str, Any] | None = None
        self._pending_transfer: Dict[str, Any] | None = None
        self._frame_no = 0
        self._startup_error: str | None = None
        self._final_error: str | None = None

    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    def _delete_paths(self, *paths: Path) -> None:
        for path in paths:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

    def _cleanup_pending_locked(self) -> None:
        pending = self._pending_transfer or {}
        for key in ("output_path", "metadata_path"):
            raw = pending.get(key)
            if raw:
                self._delete_paths(Path(str(raw)))
        self._pending_transfer = None

    def start(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self.is_running():
                status = self.status(include_internal=False)
                status["already_running"] = True
                return status
            self._cleanup_pending_locked()
            stamp = time.strftime("%Y%m%d_%H%M%S")
            video_id = str(request_payload.get("video_id") or f"head_camera_{stamp}")
            output_path = self.video_root / f"{video_id}.mp4"
            metadata_path = self.video_root / f"{video_id}.json"
            self._stop_event.clear()
            self._ready_event.clear()
            self._startup_error = None
            self._final_error = None
            self._latest = None
            self._frame_no = 0
            self._recording = {
                "video_id": video_id,
                "started_at_epoch_s": time.time(),
                "output_path": str(output_path),
                "metadata_path": str(metadata_path),
                "experiment_id": request_payload.get("experiment_id"),
                "source": "stretch_head_d435i",
            }
            self._thread = threading.Thread(target=self._run, args=(output_path, metadata_path), daemon=True)
            self._thread.start()
        if not self._ready_event.wait(timeout=float(os.getenv("ASK2ACT_STRETCH_VIDEO_START_TIMEOUT_S", "8.0"))):
            return {
                "ok": False,
                "status": "start_timeout",
                "error": self._startup_error or "Head-camera video did not become ready before timeout.",
            }
        if self._startup_error:
            return {"ok": False, "status": "failed", "error": self._startup_error}
        return {"ok": True, "status": "recording", **self.status(include_internal=False)}

    def stop(self, request_payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        request_payload = request_payload or {}
        with self._lock:
            recording = dict(self._recording or {})
            thread = self._thread
            if thread is None:
                return {"ok": True, "status": "not_running", "video": None}
            self._stop_event.set()
        thread.join(timeout=float(os.getenv("ASK2ACT_STRETCH_VIDEO_STOP_TIMEOUT_S", "45.0")))
        if thread.is_alive():
            return {"ok": False, "status": "stop_timeout", "error": "Head-camera recorder did not stop before timeout."}

        output_path = Path(str(recording.get("output_path") or ""))
        metadata_path = Path(str(recording.get("metadata_path") or ""))
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
        ok = self._final_error is None and output_path.exists() and output_path.stat().st_size > 0
        response: Dict[str, Any] = {
            "ok": bool(ok),
            "status": "stopped" if ok else "failed",
            "video_id": recording.get("video_id"),
            "metadata": metadata,
            "error": self._final_error,
        }
        if ok:
            mime_type = mimetypes.guess_type(output_path.name)[0] or "video/mp4"
            response.update(
                {
                    "filename": output_path.name,
                    "mime_type": mime_type,
                    "size_bytes": output_path.stat().st_size,
                }
            )
            if _truthy("ASK2ACT_STRETCH_VIDEO_INLINE_ON_STOP", "0") or bool(request_payload.get("return_video_base64")):
                response["video_base64"] = _encode_file(output_path)
                self._delete_paths(output_path, metadata_path)
            else:
                pending_id = str(recording.get("video_id") or output_path.stem)
                response["pending_transfer_id"] = pending_id
                response["transfer_mode"] = "chunked_base64"
                with self._lock:
                    self._pending_transfer = {
                        "video_id": pending_id,
                        "output_path": str(output_path),
                        "metadata_path": str(metadata_path),
                        "filename": output_path.name,
                        "mime_type": mime_type,
                        "size_bytes": output_path.stat().st_size,
                    }
        else:
            self._delete_paths(output_path, metadata_path)
        with self._lock:
            self._thread = None
            self._recording = None
            self._latest = None
            self._frame_no = 0
        return response

    def fetch(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        transfer_id = str(request_payload.get("transfer_id") or request_payload.get("video_id") or "").strip()
        offset = max(0, int(request_payload.get("offset") or 0))
        max_bytes = int(request_payload.get("max_bytes") or os.getenv("ASK2ACT_STRETCH_VIDEO_CHUNK_BYTES", "4194304"))
        max_bytes = min(max(65536, max_bytes), int(os.getenv("ASK2ACT_STRETCH_VIDEO_MAX_CHUNK_BYTES", "16777216")))
        with self._lock:
            pending = dict(self._pending_transfer or {})
        if not pending:
            return {"ok": False, "status": "not_found", "error": "No pending head-camera video transfer."}
        pending_id = str(pending.get("video_id") or "")
        if transfer_id and transfer_id != pending_id:
            return {"ok": False, "status": "not_found", "error": f"Unknown head-camera video transfer: {transfer_id}"}
        output_path = Path(str(pending.get("output_path") or ""))
        metadata_path = Path(str(pending.get("metadata_path") or ""))
        if not output_path.exists():
            with self._lock:
                self._pending_transfer = None
            return {"ok": False, "status": "not_found", "error": "Pending head-camera video file is missing."}
        size = output_path.stat().st_size
        with output_path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read(max_bytes)
        next_offset = offset + len(chunk)
        done = next_offset >= size
        response = {
            "ok": True,
            "status": "done" if done else "chunk",
            "transfer_id": pending_id,
            "filename": pending.get("filename") or output_path.name,
            "mime_type": pending.get("mime_type") or "video/mp4",
            "size_bytes": size,
            "offset": offset,
            "next_offset": next_offset,
            "done": done,
            "chunk_base64": base64.b64encode(chunk).decode("ascii"),
        }
        if done:
            self._delete_paths(output_path, metadata_path)
            with self._lock:
                self._pending_transfer = None
        return response

    def status(self, *, include_internal: bool = True) -> Dict[str, Any]:
        with self._lock:
            recording = dict(self._recording or {})
            running = self.is_running()
            out = {
                "ok": True,
                "running": running,
                "status": "recording" if running else "idle",
                "video_id": recording.get("video_id"),
                "started_at_epoch_s": recording.get("started_at_epoch_s"),
                "frame_count": self._frame_no,
                "source_width": recording.get("source_width"),
                "source_height": recording.get("source_height"),
                "video_width": recording.get("video_width"),
                "video_height": recording.get("video_height"),
                "fps": recording.get("fps"),
                "writer_backend": recording.get("writer_backend"),
                "pending_transfer_id": (self._pending_transfer or {}).get("video_id"),
                "error": self._startup_error or self._final_error,
            }
            if include_internal:
                out["output_path"] = recording.get("output_path")
            return out

    def capture_observation(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_running():
            raise RuntimeError("Head-camera video recorder is not running")
        capture_module = _load_capture_module()
        default_pose_result = None
        head_pose_result = None
        if capture_module is not None:
            default_pose_result = capture_module._ensure_default_pose_before_observe()
            if default_pose_result and not bool(default_pose_result.get("ok", False)):
                if _truthy("ASK2ACT_STRETCH_HOME_POSE_REQUIRED", "1"):
                    error = str(default_pose_result.get("error") or "Default observe-start pose failed")
                    note = str(default_pose_result.get("note") or "")
                    raise RuntimeError(error if not note else f"{error}. {note}")
            head_pose_result = capture_module._ensure_initial_head_pose()
            if head_pose_result and not bool(head_pose_result.get("ok", False)):
                if _truthy("ASK2ACT_STRETCH_INIT_HEAD_POSE_REQUIRED", "0"):
                    error = str(head_pose_result.get("error") or "Initial head pose failed")
                    note = str(head_pose_result.get("note") or "")
                    raise RuntimeError(error if not note else f"{error}. {note}")

        frame = self._wait_for_new_frame(after_frame_no=self._frame_no, timeout_s=float(os.getenv("ASK2ACT_STRETCH_VIDEO_OBSERVE_TIMEOUT_S", "8.0")))
        return self._write_observation_artifacts(frame, default_pose_result, head_pose_result, request_payload)

    def _wait_for_new_frame(self, *, after_frame_no: int, timeout_s: float) -> Dict[str, Any]:
        deadline = time.time() + max(0.1, timeout_s)
        with self._condition:
            while time.time() < deadline:
                latest = self._latest
                if latest is not None and int(latest.get("frame_no") or 0) > int(after_frame_no):
                    return dict(latest)
                remaining = max(0.05, deadline - time.time())
                self._condition.wait(timeout=remaining)
            if self._latest is not None:
                return dict(self._latest)
        raise RuntimeError("Timed out waiting for a fresh D435i frame from the video recorder.")

    def _write_observation_artifacts(
        self,
        frame: Dict[str, Any],
        default_pose_result: Dict[str, Any] | None,
        head_pose_result: Dict[str, Any] | None,
        request_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        import numpy as np
        from PIL import Image

        rgb = frame["bgr"][..., ::-1]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"{stamp}_{int(frame.get('frame_no') or 0)}"
        root = _artifact_root()
        image_path = root / f"head_d435i_{suffix}.jpg"
        Image.fromarray(rgb).save(image_path, quality=95)

        intrinsics_payload = dict(frame.get("intrinsics") or {})
        intrinsics_path = root / f"head_d435i_intrinsics_{suffix}.json"
        intrinsics_path.write_text(json.dumps(intrinsics_payload, indent=2), encoding="utf-8")

        depth = frame.get("depth")
        depth_path = None
        depth_shape = None
        if depth is not None:
            depth_shape = [int(depth.shape[0]), int(depth.shape[1])]
            depth_path = root / f"head_d435i_depth_{suffix}.npy"
            np.save(depth_path, depth)

        return {
            "ok": True,
            "observation_id": f"d435i-video-{suffix}",
            "mime_type": "image/jpeg",
            "image_path": str(image_path),
            "source": "realsense_d435i_shared_video",
            "camera_serial": frame.get("camera_serial"),
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "rgb_shape_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
            "depth_shape_hw": depth_shape,
            "depth_npy_path": str(depth_path) if depth_path is not None else None,
            "depth_scale_m_per_unit": frame.get("depth_scale_m_per_unit"),
            "depth_aligned_to_color": bool(depth is not None),
            "camera_intrinsics_path": str(intrinsics_path),
            "camera_intrinsics": intrinsics_payload,
            "default_pose_init": default_pose_result,
            "head_pose_init": head_pose_result,
            "video_recording": self.status(include_internal=False),
            "request_session_id": request_payload.get("session_id"),
        }

    def _run(self, output_path: Path, metadata_path: Path) -> None:
        import numpy as np
        import pyrealsense2 as rs

        writer: _VideoWriter | None = None
        pipeline = rs.pipeline()
        profile = None
        started_at = time.time()
        frame_count = 0
        video_head_pose_result = None
        try:
            capture_module = _load_capture_module()
            if capture_module is not None and _truthy("ASK2ACT_STRETCH_VIDEO_INIT_HEAD_POSE_ON_START", "1"):
                video_head_pose_result = capture_module._ensure_initial_head_pose()
                if video_head_pose_result and not bool(video_head_pose_result.get("ok", False)):
                    if _truthy("ASK2ACT_STRETCH_INIT_HEAD_POSE_REQUIRED", "0"):
                        error = str(video_head_pose_result.get("error") or "Initial head pose failed")
                        note = str(video_head_pose_result.get("note") or "")
                        raise RuntimeError(error if not note else f"{error}. {note}")

            serial = (
                os.getenv("ASK2ACT_STRETCH_D435I_SERIAL", "").strip()
                or os.getenv("ASK2ACT_STRETCH_CAMERA_SERIAL", "").strip()
            )
            width = int(os.getenv("ASK2ACT_STRETCH_CAMERA_WIDTH", "1280"))
            height = int(os.getenv("ASK2ACT_STRETCH_CAMERA_HEIGHT", "720"))
            fps = int(os.getenv("ASK2ACT_STRETCH_CAMERA_FPS", "15"))
            video_width = int(os.getenv("ASK2ACT_STRETCH_VIDEO_WIDTH", str(width)))
            video_height = int(os.getenv("ASK2ACT_STRETCH_VIDEO_HEIGHT", str(height)))
            video_fps = int(os.getenv("ASK2ACT_STRETCH_VIDEO_FPS", str(min(fps, 10))))
            capture_depth = _truthy("ASK2ACT_STRETCH_CAPTURE_DEPTH", "1")
            timeout_ms = int(os.getenv("ASK2ACT_STRETCH_CAMERA_TIMEOUT_MS", "5000"))

            config = rs.config()
            if serial:
                config.enable_device(serial)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            if capture_depth:
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            profile = pipeline.start(config)
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
            device = profile.get_device()
            camera_serial = device.get_info(rs.camera_info.serial_number)
            depth_scale = None
            if capture_depth:
                try:
                    depth_scale = float(device.first_depth_sensor().get_depth_scale())
                except Exception:
                    depth_scale = None
            writer = _VideoWriter(
                output_path=output_path,
                source_width=width,
                source_height=height,
                fps=video_fps,
                out_width=video_width,
                out_height=video_height,
            )
            writer.open()
            with self._lock:
                if self._recording is not None:
                    self._recording.update(
                        {
                            "source_width": width,
                            "source_height": height,
                            "fps": video_fps,
                            "video_width": video_width,
                            "video_height": video_height,
                            "writer_backend": writer.backend,
                            "camera_serial": camera_serial,
                            "head_pose_init": video_head_pose_result,
                        }
                    )
            self._ready_event.set()

            warmup_frames = int(os.getenv("ASK2ACT_STRETCH_CAMERA_WARMUP_FRAMES", "20"))
            for _ in range(max(1, warmup_frames)):
                if self._stop_event.is_set():
                    break
                pipeline.wait_for_frames(timeout_ms=timeout_ms)

            align = rs.align(rs.stream.color) if capture_depth else None
            min_period = 1.0 / max(1, video_fps)
            next_video_time = 0.0
            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
                if align is not None:
                    frames = align.process(frames)
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue
                depth_frame = frames.get_depth_frame() if capture_depth else None
                bgr = np.asanyarray(color_frame.get_data()).copy()
                depth = np.asanyarray(depth_frame.get_data()).copy() if depth_frame is not None else None
                frame_count += 1
                with self._condition:
                    self._frame_no = frame_count
                    self._latest = {
                        "frame_no": frame_count,
                        "bgr": bgr,
                        "depth": depth,
                        "intrinsics": intrinsics_payload,
                        "camera_serial": camera_serial,
                        "depth_scale_m_per_unit": depth_scale,
                        "captured_at_epoch_s": time.time(),
                    }
                    self._condition.notify_all()
                now = time.time()
                if now >= next_video_time:
                    writer.write(bgr)
                    next_video_time = now + min_period
        except Exception as exc:
            self._startup_error = self._startup_error or str(exc)
            self._final_error = str(exc)
            self._ready_event.set()
            metadata_path.write_text(
                json.dumps(
                    {
                        "ok": False,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=8),
                        "started_at_epoch_s": started_at,
                        "stopped_at_epoch_s": time.time(),
                        "frame_count": frame_count,
                        "head_pose_init": video_head_pose_result,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        finally:
            writer_result = {}
            if writer is not None:
                try:
                    writer_result = writer.close()
                    if writer_result.get("returncode") not in {0, None}:
                        self._final_error = self._final_error or str(writer_result.get("stderr") or "video writer failed")
                except Exception as exc:
                    self._final_error = self._final_error or str(exc)
                    writer_result = {"backend": getattr(writer, "backend", "unknown"), "error": str(exc)}
            if profile is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            stopped_at = time.time()
            if not metadata_path.exists():
                metadata_path.write_text(
                    json.dumps(
                        {
                            "ok": self._final_error is None,
                            "video_id": (self._recording or {}).get("video_id"),
                            "started_at_epoch_s": started_at,
                            "stopped_at_epoch_s": stopped_at,
                            "duration_s": max(0.0, stopped_at - started_at),
                            "frame_count": frame_count,
                            "output_path": str(output_path),
                            "writer": writer_result,
                            "head_pose_init": video_head_pose_result,
                            "note": "Robot-side video file is deleted after it is transferred to A6000.",
                        },
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
