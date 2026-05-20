from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class StretchObservationResult:
    observation_id: Optional[str]
    image_bytes: bytes
    mime_type: str
    raw_response: Dict[str, Any]


class StretchTransportClient:
    def __init__(
        self,
        mode: str = "mock",
        zmq_endpoint: str = "tcp://127.0.0.1:5557",
        timeout_ms: int = 30000,
        observe_timeout_ms: Optional[int] = None,
        execute_timeout_ms: Optional[int] = None,
        mock_image_path: str = "",
    ) -> None:
        self.mode = (mode or "mock").strip().lower()
        self.zmq_endpoint = zmq_endpoint
        self.timeout_ms = int(timeout_ms)
        self.observe_timeout_ms = int(observe_timeout_ms or timeout_ms)
        self.execute_timeout_ms = int(execute_timeout_ms or timeout_ms)
        self.mock_image_path = mock_image_path.strip()

    @property
    def enabled(self) -> bool:
        return self.mode in {"mock", "zmq"}

    def _default_mock_image_path(self) -> Path:
        return Path(__file__).resolve().parents[3] / "workspace" / "project" / "mimo" / "test" / "demo.jpg"

    def _load_mock_image(self) -> tuple[Path, bytes, str]:
        path = Path(self.mock_image_path).expanduser() if self.mock_image_path else self._default_mock_image_path()
        if not path.exists():
            raise FileNotFoundError(f"Mock observation image not found: {path}")
        mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return path, path.read_bytes(), mime_type

    def _zmq_roundtrip(self, payload: Dict[str, Any], *, timeout_ms: Optional[int] = None) -> Dict[str, Any]:
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("pyzmq is required for ASK2ACT_STRETCH_TRANSPORT=zmq") from exc

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        effective_timeout_ms = int(timeout_ms or self.timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, effective_timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, effective_timeout_ms)
        try:
            sock.connect(self.zmq_endpoint)
            sock.send_json(payload)
            reply = sock.recv_json()
        except Exception as exc:
            raise RuntimeError(
                f"Stretch transport request failed via {self.zmq_endpoint} "
                f"after {effective_timeout_ms} ms: {exc}"
            ) from exc
        finally:
            sock.close()
        if not isinstance(reply, dict):
            raise RuntimeError("Stretch transport reply must be a JSON object")
        return reply

    def fetch_observation(self, session_id: str, instruction: str) -> StretchObservationResult:
        if self.mode == "mock":
            path, image_bytes, mime_type = self._load_mock_image()
            observation_id = f"mock-{int(time.time())}"
            return StretchObservationResult(
                observation_id=observation_id,
                image_bytes=image_bytes,
                mime_type=mime_type,
                raw_response={
                    "ok": True,
                    "mode": "mock",
                    "observation_id": observation_id,
                    "mock_image_path": str(path),
                },
            )

        if self.mode != "zmq":
            raise RuntimeError(f"Unsupported stretch transport mode: {self.mode}")

        reply = self._zmq_roundtrip(
            {
                "op": "observe",
                "session_id": session_id,
                "instruction": instruction,
            },
            timeout_ms=self.observe_timeout_ms,
        )
        if reply.get("ok") is False:
            detail = reply.get("error") or reply.get("note") or reply
            raise RuntimeError(f"Stretch observation failed: {detail}")
        image_bytes: bytes
        if reply.get("image_data_url"):
            _, encoded = str(reply["image_data_url"]).split(",", 1)
            image_bytes = base64.b64decode(encoded)
        elif reply.get("image_base64"):
            image_bytes = base64.b64decode(reply["image_base64"])
        else:
            raise RuntimeError("Stretch observation reply is missing image_data_url/image_base64")

        return StretchObservationResult(
            observation_id=reply.get("observation_id"),
            image_bytes=image_bytes,
            mime_type=reply.get("mime_type") or "image/jpeg",
            raw_response=reply,
        )

    def start_head_camera_video(self, *, experiment_id: str, video_id: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "mock":
            started_at = time.time()
            return {
                "ok": True,
                "status": "recording",
                "mode": "mock",
                "video_id": video_id or f"mock_head_camera_{int(started_at)}",
                "started_at_epoch_s": started_at,
                "source_width": 1280,
                "source_height": 720,
                "video_width": 1280,
                "video_height": 720,
                "fps": 10,
                "note": "Mock transport does not record real video.",
            }
        if self.mode != "zmq":
            raise RuntimeError(f"Unsupported stretch transport mode: {self.mode}")
        reply = self._zmq_roundtrip(
            {
                "op": "start_head_camera_video",
                "experiment_id": experiment_id,
                "video_id": video_id,
            },
            timeout_ms=self.execute_timeout_ms,
        )
        if reply.get("ok") is False:
            raise RuntimeError(reply.get("error") or "Stretch head-camera video start failed")
        reply["mode"] = "zmq"
        return reply

    def stop_head_camera_video(self, *, experiment_id: str) -> Dict[str, Any]:
        if self.mode == "mock":
            payload = b"mock head camera video"
            return {
                "ok": True,
                "status": "stopped",
                "mode": "mock",
                "video_id": f"mock_head_camera_{int(time.time())}",
                "filename": "mock_head_camera.mp4",
                "mime_type": "video/mp4",
                "size_bytes": len(payload),
                "video_base64": base64.b64encode(payload).decode("ascii"),
                "metadata": {"ok": True, "mock": True},
            }
        if self.mode != "zmq":
            raise RuntimeError(f"Unsupported stretch transport mode: {self.mode}")
        reply = self._zmq_roundtrip(
            {"op": "stop_head_camera_video", "experiment_id": experiment_id},
            timeout_ms=self.execute_timeout_ms,
        )
        if reply.get("ok") is False:
            raise RuntimeError(reply.get("error") or "Stretch head-camera video stop failed")
        if reply.get("pending_transfer_id"):
            video_chunks: list[bytes] = []
            offset = 0
            size_bytes = int(reply.get("size_bytes") or 0)
            transfer_id = str(reply["pending_transfer_id"])
            while True:
                chunk_reply = self._zmq_roundtrip(
                    {
                        "op": "fetch_head_camera_video",
                        "experiment_id": experiment_id,
                        "transfer_id": transfer_id,
                        "offset": offset,
                    },
                    timeout_ms=self.execute_timeout_ms,
                )
                if chunk_reply.get("ok") is False:
                    raise RuntimeError(chunk_reply.get("error") or "Stretch head-camera video transfer failed")
                video_chunks.append(base64.b64decode(chunk_reply.get("chunk_base64") or ""))
                offset = int(chunk_reply.get("next_offset") or offset)
                if chunk_reply.get("done"):
                    reply["video_bytes"] = b"".join(video_chunks)
                    reply["size_bytes"] = size_bytes or len(reply["video_bytes"])
                    reply["filename"] = chunk_reply.get("filename") or reply.get("filename")
                    reply["mime_type"] = chunk_reply.get("mime_type") or reply.get("mime_type") or "video/mp4"
                    break
        reply["mode"] = "zmq"
        return reply

    def head_camera_video_status(self) -> Dict[str, Any]:
        if self.mode == "mock":
            return {"ok": True, "running": False, "status": "idle", "mode": "mock"}
        if self.mode != "zmq":
            raise RuntimeError(f"Unsupported stretch transport mode: {self.mode}")
        reply = self._zmq_roundtrip({"op": "head_camera_video_status"}, timeout_ms=self.timeout_ms)
        if reply.get("ok") is False:
            raise RuntimeError(reply.get("error") or "Stretch head-camera video status failed")
        reply["mode"] = "zmq"
        return reply

    def set_head_camera_pose(self, *, head_pan_rad: float, head_tilt_rad: float, persist: bool = True) -> Dict[str, Any]:
        if self.mode == "mock":
            return {
                "ok": True,
                "mode": "mock",
                "status": "initialized",
                "commanded_head_pan_rad": float(head_pan_rad),
                "commanded_head_tilt_rad": float(head_tilt_rad),
                "actual_head_pan_rad": float(head_pan_rad),
                "actual_head_tilt_rad": float(head_tilt_rad),
                "persisted_for_future_observations": bool(persist),
            }
        if self.mode != "zmq":
            raise RuntimeError(f"Unsupported stretch transport mode: {self.mode}")
        reply = self._zmq_roundtrip(
            {
                "op": "set_head_camera_pose",
                "head_pan_rad": float(head_pan_rad),
                "head_tilt_rad": float(head_tilt_rad),
                "persist": bool(persist),
            },
            timeout_ms=self.execute_timeout_ms,
        )
        if reply.get("ok") is False:
            raise RuntimeError(reply.get("error") or "Stretch head-camera pose command failed")
        reply["mode"] = "zmq"
        return reply

    def dispatch_grasp(
        self,
        *,
        session_id: str,
        instruction: str,
        observation_id: Optional[str],
        resolved_target: Dict[str, Any],
        grasp_plan: Dict[str, Any],
        dry_run: bool,
    ) -> Dict[str, Any]:
        payload = {
            "session_id": session_id,
            "instruction": instruction,
            "observation_id": observation_id,
            "resolved_target": resolved_target,
            "grasp_plan": grasp_plan,
            "dry_run": dry_run,
        }

        if self.mode == "mock":
            return {
                "ok": True,
                "mode": "mock",
                "dry_run": dry_run,
                "payload": payload,
                "message": "Mock Stretch transport accepted the request.",
            }

        if self.mode != "zmq":
            raise RuntimeError(f"Unsupported stretch transport mode: {self.mode}")

        reply = self._zmq_roundtrip({"op": "execute_grasp", **payload}, timeout_ms=self.execute_timeout_ms)
        return {
            "ok": bool(reply.get("ok", True)),
            "mode": "zmq",
            "dry_run": dry_run,
            "payload": payload,
            "reply": reply,
        }
