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
