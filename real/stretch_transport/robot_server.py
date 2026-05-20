from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict

import zmq

from .runtime import StretchRobotRuntime


LOG = logging.getLogger("ask2act.stretch_transport")


def build_runtime_from_env() -> StretchRobotRuntime:
    return StretchRobotRuntime(
        observe_mode=os.getenv("ASK2ACT_STRETCH_SERVER_OBSERVE_MODE", "command"),
        observe_command=os.getenv("ASK2ACT_STRETCH_SERVER_OBSERVE_COMMAND", ""),
        execute_mode=os.getenv("ASK2ACT_STRETCH_SERVER_EXECUTE_MODE", "command"),
        execute_command=os.getenv("ASK2ACT_STRETCH_SERVER_EXECUTE_COMMAND", ""),
        observation_image_path=os.getenv("ASK2ACT_STRETCH_OBSERVATION_IMAGE_PATH", ""),
        artifact_root=os.getenv("ASK2ACT_STRETCH_SERVER_ARTIFACT_ROOT", ""),
        command_timeout_s=int(os.getenv("ASK2ACT_STRETCH_SERVER_COMMAND_TIMEOUT_S", "180")),
        observe_timeout_s=int(
            os.getenv(
                "ASK2ACT_STRETCH_SERVER_OBSERVE_TIMEOUT_S",
                os.getenv("ASK2ACT_STRETCH_SERVER_COMMAND_TIMEOUT_S", "180"),
            )
        ),
        execute_timeout_s=int(
            os.getenv(
                "ASK2ACT_STRETCH_SERVER_EXECUTE_TIMEOUT_S",
                os.getenv("ASK2ACT_STRETCH_SERVER_COMMAND_TIMEOUT_S", "180"),
            )
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stretch-side ZMQ REP server for Ask2Act")
    parser.add_argument(
        "--bind",
        default=os.getenv("ASK2ACT_STRETCH_SERVER_BIND", "tcp://0.0.0.0:5557"),
        help="ZMQ REP bind endpoint, e.g. tcp://0.0.0.0:5557",
    )
    return parser.parse_args()


def _validate_request(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    if not payload.get("op"):
        raise ValueError("request is missing op")


def serve(bind_endpoint: str) -> None:
    runtime = build_runtime_from_env()
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(bind_endpoint)
    LOG.info("Stretch transport server listening on %s", bind_endpoint)

    try:
        while True:
            payload = sock.recv_json()
            try:
                _validate_request(payload)
                op = str(payload["op"])
                if op == "observe":
                    reply = runtime.observe(payload)
                elif op == "execute_grasp":
                    reply = runtime.execute(payload)
                elif op == "start_head_camera_video":
                    reply = runtime.start_video(payload)
                elif op == "stop_head_camera_video":
                    reply = runtime.stop_video(payload)
                elif op == "head_camera_video_status":
                    reply = runtime.video_status(payload)
                elif op == "fetch_head_camera_video":
                    reply = runtime.fetch_video(payload)
                elif op == "set_head_camera_pose":
                    reply = runtime.set_head_pose(payload)
                else:
                    raise ValueError(f"unsupported op: {op}")
            except Exception as exc:
                LOG.exception("Stretch transport request failed")
                reply = {"ok": False, "error": str(exc)}
            sock.send_json(reply)
    except KeyboardInterrupt:
        LOG.info("Stretch transport server stopped")
    finally:
        sock.close()


def main() -> int:
    logging.basicConfig(
        level=os.getenv("ASK2ACT_STRETCH_SERVER_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    LOG.info(
        "observe_mode=%s execute_mode=%s",
        os.getenv("ASK2ACT_STRETCH_SERVER_OBSERVE_MODE", "command"),
        os.getenv("ASK2ACT_STRETCH_SERVER_EXECUTE_MODE", "command"),
    )
    LOG.info("bind_endpoint=%s", args.bind)
    serve(args.bind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
