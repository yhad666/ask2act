from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.a6000_web.stretch_transport import StretchTransportClient


def _decode_depth_npy(raw_response: dict[str, Any]) -> np.ndarray | None:
    encoded = raw_response.get("depth_npy_base64")
    if not encoded:
        return None
    import io

    return np.load(io.BytesIO(base64.b64decode(str(encoded))))


def _save_depth_color(depth: np.ndarray, scale_m_per_unit: float | None, output_path: Path) -> None:
    depth_m = depth.astype(np.float32)
    if scale_m_per_unit is not None:
        depth_m *= float(scale_m_per_unit)
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        Image.fromarray(np.zeros((*depth.shape[:2], 3), dtype=np.uint8)).save(output_path)
        return

    lo, hi = np.percentile(depth_m[valid], [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((depth_m - lo) / (hi - lo), 0.0, 1.0)
    norm[~valid] = 0.0

    # Lightweight blue->cyan->yellow->red depth visualization without matplotlib.
    stops = np.asarray(
        [
            [20, 30, 120],
            [35, 160, 220],
            [245, 220, 70],
            [210, 45, 35],
        ],
        dtype=np.float32,
    )
    x = norm * (len(stops) - 1)
    idx = np.floor(x).astype(np.int32)
    idx = np.clip(idx, 0, len(stops) - 2)
    frac = (x - idx)[..., None]
    rgb = stops[idx] * (1.0 - frac) + stops[idx + 1] * frac
    rgb[~valid] = 0
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture current Stretch RGB-D observation via Ask2Act ZMQ.")
    parser.add_argument(
        "--endpoint",
        default=os.getenv("ASK2ACT_STRETCH_ZMQ_ENDPOINT", "tcp://192.168.1.145:5557"),
        help="Stretch ZMQ endpoint. Use IP if .local mDNS resolution is unavailable.",
    )
    parser.add_argument("--session-id", default=f"manual-current-scene-{int(time.time())}")
    parser.add_argument("--instruction", default="Capture the current tabletop scene.")
    parser.add_argument(
        "--out-root",
        default="services/a6000_web/artifacts/current_scene_captures",
        help="Directory on this server where RGB-D artifacts are written.",
    )
    parser.add_argument("--timeout-ms", type=int, default=120000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root).expanduser() / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    client = StretchTransportClient(
        mode="zmq",
        zmq_endpoint=args.endpoint,
        timeout_ms=args.timeout_ms,
        observe_timeout_ms=args.timeout_ms,
    )
    observation = client.fetch_observation(session_id=args.session_id, instruction=args.instruction)

    suffix = ".jpg" if "jpeg" in observation.mime_type.lower() else ".png"
    rgb_path = out_dir / f"rgb{suffix}"
    rgb_path.write_bytes(observation.image_bytes)

    raw = observation.raw_response
    depth = _decode_depth_npy(raw)
    depth_path = None
    depth_color_path = None
    if depth is not None:
        depth_path = out_dir / "depth_raw.npy"
        np.save(depth_path, depth)
        depth_color_path = out_dir / "depth_color.png"
        _save_depth_color(depth, raw.get("depth_scale_m_per_unit"), depth_color_path)

    metadata = {
        "captured_at_epoch_s": time.time(),
        "endpoint": args.endpoint,
        "session_id": args.session_id,
        "instruction": args.instruction,
        "observation_id": observation.observation_id,
        "mime_type": observation.mime_type,
        "rgb_path": str(rgb_path),
        "depth_raw_path": str(depth_path) if depth_path else None,
        "depth_color_path": str(depth_color_path) if depth_color_path else None,
        "raw_response": raw,
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "out_dir": str(out_dir),
                "observation_id": observation.observation_id,
                "rgb_path": str(rgb_path),
                "depth_raw_path": str(depth_path) if depth_path else None,
                "depth_color_path": str(depth_color_path) if depth_color_path else None,
                "metadata_path": str(metadata_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
