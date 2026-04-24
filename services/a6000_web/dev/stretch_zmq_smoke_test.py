from __future__ import annotations

import json
import os

from services.a6000_web.stretch_transport import StretchTransportClient


def main() -> int:
    endpoint = os.getenv("ASK2ACT_STRETCH_ZMQ_ENDPOINT", "tcp://127.0.0.1:5557")
    client = StretchTransportClient(
        mode="zmq",
        zmq_endpoint=endpoint,
        timeout_ms=int(os.getenv("ASK2ACT_STRETCH_TIMEOUT_MS", "30000")),
    )

    observation = client.fetch_observation(
        session_id="smoke-test-session",
        instruction="Pick up the target cup.",
    )
    print(
        json.dumps(
            {
                "phase": "observe",
                "endpoint": endpoint,
                "observation_id": observation.observation_id,
                "mime_type": observation.mime_type,
                "image_bytes": len(observation.image_bytes),
                "raw_response": observation.raw_response,
            },
            indent=2,
        )
    )

    execute = client.dispatch_grasp(
        session_id="smoke-test-session",
        instruction="Pick up the target cup.",
        observation_id=observation.observation_id,
        resolved_target={
            "candidate_id": "cand_smoke",
            "bbox_xyxy": [120.0, 240.0, 360.0, 510.0],
            "mask_rle": None,
        },
        grasp_plan={
            "pipeline_mode": "mock",
            "planner_backend": "smoke-test",
            "target_bbox_2d": [120, 240, 360, 510],
        },
        dry_run=True,
    )
    print(json.dumps({"phase": "execute", "result": execute}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
