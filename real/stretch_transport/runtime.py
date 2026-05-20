from __future__ import annotations

import base64
import json
import mimetypes
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

from .head_camera_video import HeadCameraVideoManager


class StretchRobotRuntime:
    def __init__(
        self,
        *,
        observe_mode: str = "command",
        observe_command: str = "",
        execute_mode: str = "command",
        execute_command: str = "",
        observation_image_path: str = "",
        artifact_root: str = "",
        command_timeout_s: int = 60,
        observe_timeout_s: int | None = None,
        execute_timeout_s: int | None = None,
    ) -> None:
        self.observe_mode = (observe_mode or "command").strip().lower()
        self.observe_command = observe_command.strip()
        self.execute_mode = (execute_mode or "command").strip().lower()
        self.execute_command = execute_command.strip()
        self.observation_image_path = observation_image_path.strip()
        self.command_timeout_s = int(command_timeout_s)
        self.observe_timeout_s = int(observe_timeout_s or command_timeout_s)
        self.execute_timeout_s = int(execute_timeout_s or command_timeout_s)
        self.repo_root = Path(__file__).resolve().parents[2]
        self.hooks_root = Path(__file__).resolve().parent / "hooks"
        self.artifact_root = (
            Path(artifact_root).expanduser()
            if artifact_root
            else Path(__file__).resolve().parent / "artifacts"
        )
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.video_manager = HeadCameraVideoManager(artifact_root=self.artifact_root)

    def _default_sample_image_path(self) -> Path:
        return (
            self.repo_root
            / "simulation"
            / "logs"
            / "sim_validation"
            / "data_samples"
            / "tabletop_scene"
            / "head_d435i_rgb.png"
        )

    def _default_observe_command(self) -> str:
        hook_path = self.hooks_root / "observe_hook.py"
        return f"{shlex.quote(sys.executable)} {shlex.quote(str(hook_path))}"

    def _default_execute_command(self) -> str:
        hook_path = self.hooks_root / "execute_hook.py"
        return f"{shlex.quote(sys.executable)} {shlex.quote(str(hook_path))}"

    def _run_json_command(
        self,
        *,
        command: str,
        request_payload: Dict[str, Any],
        op_name: str,
        timeout_s: int,
    ) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix=f"ask2act_{op_name}_") as temp_dir:
            request_path = Path(temp_dir) / "request.json"
            response_path = Path(temp_dir) / "response.json"
            request_path.write_text(json.dumps(request_payload, indent=2), encoding="utf-8")

            env = os.environ.copy()
            env["ASK2ACT_REQUEST_JSON"] = str(request_path)
            env["ASK2ACT_RESPONSE_JSON"] = str(response_path)

            shell_command = f"{command} {shlex.quote(str(request_path))} {shlex.quote(str(response_path))}"
            try:
                completed = subprocess.run(
                    ["bash", "-lc", shell_command],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
                stderr = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
                detail = stderr or stdout or "no stderr/stdout available"
                raise RuntimeError(f"{op_name} command timed out after {timeout_s}s: {detail}") from exc
            if completed.returncode != 0:
                stderr = completed.stderr.strip()
                stdout = completed.stdout.strip()
                detail = stderr or stdout or "no stderr/stdout available"
                raise RuntimeError(f"{op_name} command failed: {detail}")
            if not response_path.exists():
                raise RuntimeError(f"{op_name} command did not create response JSON: {response_path}")
            reply = json.loads(response_path.read_text(encoding="utf-8"))
            if not isinstance(reply, dict):
                raise RuntimeError(f"{op_name} command reply must be a JSON object")
            return reply

    def _load_image_bytes(self, image_path: str) -> tuple[bytes, str]:
        path = Path(image_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Observation image not found: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return path.read_bytes(), mime_type

    def _normalize_observe_reply(self, reply: Dict[str, Any], request_payload: Dict[str, Any]) -> Dict[str, Any]:
        ok = bool(reply.get("ok", True))
        if not ok:
            return {
                "ok": False,
                "error": reply.get("error") or "observe hook returned ok=false",
                "detail": reply,
            }

        image_bytes: bytes
        mime_type = str(reply.get("mime_type") or "image/jpeg")
        if reply.get("image_path"):
            image_bytes, mime_type = self._load_image_bytes(str(reply["image_path"]))
        elif reply.get("image_base64"):
            image_bytes = base64.b64decode(str(reply["image_base64"]))
        elif reply.get("image_data_url"):
            _, payload = str(reply["image_data_url"]).split(",", 1)
            image_bytes = base64.b64decode(payload)
        else:
            raise RuntimeError("observe hook reply must contain image_path, image_base64, or image_data_url")

        observation_id = str(
            reply.get("observation_id")
            or f"stretch-{request_payload.get('session_id', 'session')}-{int(time.time())}"
        )
        normalized = {
            "ok": True,
            "observation_id": observation_id,
            "mime_type": mime_type,
            "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
            "source": reply.get("source") or self.observe_mode,
            "detail": reply,
        }
        depth_path = reply.get("depth_npy_path")
        if depth_path:
            depth_bytes, _depth_mime_type = self._load_image_bytes(str(depth_path))
            normalized["depth_npy_base64"] = base64.b64encode(depth_bytes).decode("utf-8")
            normalized["depth_npy_mime_type"] = "application/octet-stream"
            normalized["depth_npy_path"] = str(depth_path)
        for key in (
            "depth_scale_m_per_unit",
            "camera_intrinsics",
            "camera_intrinsics_path",
            "camera_extrinsics",
            "camera_extrinsics_path",
            "camera_serial",
            "width",
            "height",
            "rgb_shape_hw",
            "depth_shape_hw",
            "depth_aligned_to_color",
        ):
            if key in reply:
                normalized[key] = reply[key]
        return normalized

    def observe(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        if (
            self.video_manager.is_running()
            and os.getenv("ASK2ACT_STRETCH_OBSERVE_FROM_VIDEO_STREAM", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        ):
            reply = self.video_manager.capture_observation(request_payload)
            return self._normalize_observe_reply(reply, request_payload)

        if self.observe_mode == "file":
            image_path = self.observation_image_path or str(self._default_sample_image_path())
            reply = {
                "ok": True,
                "observation_id": f"stretch-file-{int(time.time())}",
                "mime_type": mimetypes.guess_type(image_path)[0] or "image/jpeg",
                "image_path": image_path,
                "source": "file",
            }
            return self._normalize_observe_reply(reply, request_payload)

        if self.observe_mode == "command":
            reply = self._run_json_command(
                command=self.observe_command or self._default_observe_command(),
                request_payload=request_payload,
                op_name="observe",
                timeout_s=self.observe_timeout_s,
            )
            return self._normalize_observe_reply(reply, request_payload)

        raise RuntimeError(f"Unsupported observe mode: {self.observe_mode}")

    def start_video(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.video_manager.start(request_payload)

    def stop_video(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.video_manager.stop(request_payload)

    def video_status(self, request_payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self.video_manager.status(include_internal=False)

    def fetch_video(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.video_manager.fetch(request_payload)

    def set_head_pose(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from .scripts import capture_observation as capture_module
        except Exception as exc:
            return {"ok": False, "error": f"Unable to load head pose helper: {exc}"}

        head_pan = float(
            request_payload.get(
                "head_pan_rad",
                os.getenv("ASK2ACT_STRETCH_INIT_HEAD_PAN_RAD", "-1.57"),
            )
        )
        head_tilt = float(
            request_payload.get(
                "head_tilt_rad",
                os.getenv("ASK2ACT_STRETCH_INIT_HEAD_TILT_RAD", "-0.68"),
            )
        )
        persist = str(request_payload.get("persist", "1")).strip().lower() in {"1", "true", "yes", "on"}
        if persist:
            os.environ["ASK2ACT_STRETCH_INIT_HEAD_PAN_RAD"] = str(head_pan)
            os.environ["ASK2ACT_STRETCH_INIT_HEAD_TILT_RAD"] = str(head_tilt)
            os.environ["ASK2ACT_STRETCH_INIT_HEAD_POSE_MODE"] = "every_observe"

        result = capture_module._command_head_pose(
            head_pan=head_pan,
            head_tilt=head_tilt,
            mode="manual_ui",
            write_stamp=False,
            persist_override=persist,
            pose_source={"source": "manual_ui_request"},
        )
        result["persisted_for_future_observations"] = persist
        result["video_running"] = self.video_manager.is_running()
        return result

    def set_runtime_config(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        allowed_env = {
            "ASK2ACT_STRETCH_GRIPPER_REAL_OPEN_CMD",
            "ASK2ACT_STRETCH_GRIPPER_REAL_CLOSE_CMD",
            "ASK2ACT_STRETCH_RELEASE_GRIPPER_CMD",
            "ASK2ACT_STRETCH_GRIPPER_PLANNER_OPEN_THRESHOLD",
            "ASK2ACT_STRETCH_GRIPPER_OPEN_ACCEPT_PCT",
            "ASK2ACT_STRETCH_GRIPPER_OPEN_ACCEPT_POS",
            "ASK2ACT_STRETCH_GRIPPER_OPEN_ACCEPT_PLANNER_POS",
        }
        updates = request_payload.get("env") or {}
        if not isinstance(updates, dict):
            raise RuntimeError("runtime config env payload must be an object")
        applied: Dict[str, str] = {}
        rejected: Dict[str, str] = {}
        for key, value in updates.items():
            key_str = str(key)
            if key_str not in allowed_env:
                rejected[key_str] = "not_allowed"
                continue
            os.environ[key_str] = str(value)
            applied[key_str] = str(value)
        return {
            "ok": True,
            "applied_env": applied,
            "rejected_env": rejected,
        }

    def execute(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.execute_mode == "ack":
            return {
                "ok": True,
                "execution_status": "dry_run_ack" if request_payload.get("dry_run") else "ack_only",
                "mode": "ack",
                "note": "Stretch execution is running in ACK mode only.",
            }

        if self.execute_mode == "command":
            reply = self._run_json_command(
                command=self.execute_command or self._default_execute_command(),
                request_payload=request_payload,
                op_name="execute_grasp",
                timeout_s=self.execute_timeout_s,
            )
            if not isinstance(reply, dict):
                raise RuntimeError("execute hook reply must be a JSON object")
            return {
                "ok": bool(reply.get("ok", False)),
                "execution_status": reply.get("execution_status") or "unknown",
                "mode": "command",
                "detail": reply,
                "error": reply.get("error"),
            }

        raise RuntimeError(f"Unsupported execute mode: {self.execute_mode}")
