from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Contact-GraspNet inference service.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--server-dir", required=True)
    parser.add_argument("--ckpt_dir", default="")
    parser.add_argument("--forward_passes", type=int, default=5)
    parser.add_argument("--z-range", default="[0.2,1.1]")
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    return parser.parse_args()


def _load_model(repo_root: Path, checkpoint_dir: str, forward_passes: int):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import torch
    from contact_graspnet_pytorch import config_utils
    from contact_graspnet_pytorch.checkpoints import CheckpointIO
    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator

    global_config = config_utils.load_config(
        checkpoint_dir,
        batch_size=forward_passes,
        arg_configs=[],
    )
    grasp_estimator = GraspEstimator(global_config)

    model_checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")
    checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=grasp_estimator.model)
    if not torch.cuda.is_available():
        original_torch_load = torch.load

        def cpu_safe_torch_load(*load_args, **load_kwargs):
            load_kwargs.setdefault("map_location", torch.device("cpu"))
            return original_torch_load(*load_args, **load_kwargs)

        torch.load = cpu_safe_torch_load
    checkpoint_io.load("model.pt")
    return grasp_estimator


def _predict(grasp_estimator, scene_points_xyz: np.ndarray, z_range: tuple[float, float], forward_passes: int):
    valid_mask = (
        np.isfinite(scene_points_xyz).all(axis=1)
        & (scene_points_xyz[:, 2] >= float(z_range[0]))
        & (scene_points_xyz[:, 2] <= float(z_range[1]))
    )
    filtered_xyz = np.asarray(scene_points_xyz[valid_mask], dtype=np.float32).reshape(-1, 3)
    pred_grasps_cam, scores, _contact_pts, gripper_openings = grasp_estimator.predict_scene_grasps(
        filtered_xyz,
        pc_segments={},
        local_regions=False,
        filter_grasps=False,
        forward_passes=forward_passes,
    )
    grasp_array = np.asarray(pred_grasps_cam.get(-1, np.empty((0, 4, 4), dtype=np.float32)), dtype=np.float32)
    score_array = np.asarray(scores.get(-1, np.empty((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    width_array = np.asarray(gripper_openings.get(-1, np.empty((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    return grasp_array, score_array, width_array


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    server_dir = Path(args.server_dir).expanduser().resolve()
    request_dir = server_dir / "requests"
    response_dir = server_dir / "responses"
    stop_file = server_dir / "STOP"
    server_dir.mkdir(parents=True, exist_ok=True)
    request_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = args.ckpt_dir or str(repo_root / "checkpoints" / "contact_graspnet")
    z_range = eval(str(args.z_range))
    grasp_estimator = _load_model(repo_root, checkpoint_dir, args.forward_passes)
    ready_payload = {
        "repo_root": str(repo_root),
        "checkpoint_dir": str(checkpoint_dir),
        "forward_passes": int(args.forward_passes),
        "z_range": [float(z_range[0]), float(z_range[1])],
        "pid": os.getpid(),
    }
    (server_dir / "ready.json").write_text(json.dumps(ready_payload, indent=2))
    print(f"CGN inference server ready at {server_dir}", flush=True)

    while not stop_file.exists():
        request_paths = sorted(request_dir.glob("*.npz"))
        if not request_paths:
            time.sleep(args.poll_interval_s)
            continue

        for request_path in request_paths:
            response_path = response_dir / request_path.name
            error_path = response_dir / f"{request_path.stem}.error.txt"
            try:
                with np.load(request_path, allow_pickle=True) as payload:
                    scene_points_xyz = np.asarray(payload["xyz"], dtype=np.float32).reshape(-1, 3)
                pred_grasps, pred_scores, pred_width = _predict(
                    grasp_estimator,
                    scene_points_xyz,
                    z_range=(float(z_range[0]), float(z_range[1])),
                    forward_passes=args.forward_passes,
                )
                tmp_response = response_dir / f"{request_path.stem}.tmp"
                np.savez(
                    tmp_response,
                    pred_grasps=pred_grasps,
                    pred_scores=pred_scores,
                    pred_width=pred_width,
                    z_range=np.asarray(z_range, dtype=np.float32),
                )
                Path(f"{tmp_response}.npz").replace(response_path)
            except Exception as exc:
                error_path.write_text(str(exc))
            finally:
                request_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
