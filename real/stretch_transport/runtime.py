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
    ) -> None:
        self.observe_mode = (observe_mode or "command").strip().lower()
        self.observe_command = observe_command.strip()
        self.execute_mode = (execute_mode or "command").strip().lower()
        self.execute_command = execute_command.strip()
        self.observation_image_path = observation_image_path.strip()
        self.command_timeout_s = int(command_timeout_s)
        self.repo_root = Path(__file__).resolve().parents[2]
        self.hooks_root = Path(__file__).resolve().parent / "hooks"
        self.artifact_root = (
            Path(artifact_root).expanduser()
            if artifact_root
            else Path(__file__).resolve().parent / "artifacts"
        )
        self.artifact_root.mkdir(parents=True, exist_ok=True)

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

    def _run_json_command(self, *, command: str, request_payload: Dict[str, Any], op_name: str) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix=f"ask2act_{op_name}_") as temp_dir:
            request_path = Path(temp_dir) / "request.json"
            response_path = Path(temp_dir) / "response.json"
            request_path.write_text(json.dumps(request_payload, indent=2), encoding="utf-8")

            env = os.environ.copy()
            env["ASK2ACT_REQUEST_JSON"] = str(request_path)
            env["ASK2ACT_RESPONSE_JSON"] = str(response_path)

            shell_command = f"{command} {shlex.quote(str(request_path))} {shlex.quote(str(response_path))}"
            completed = subprocess.run(
                ["bash", "-lc", shell_command],
                capture_output=True,
                text=True,
                timeout=self.command_timeout_s,
                env=env,
            )
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
        return {
            "ok": True,
            "observation_id": observation_id,
            "mime_type": mime_type,
            "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
            "source": reply.get("source") or self.observe_mode,
            "detail": reply,
        }

    def observe(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
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
            )
            return self._normalize_observe_reply(reply, request_payload)

        raise RuntimeError(f"Unsupported observe mode: {self.observe_mode}")

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
