from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Contact-GraspNet inference in an isolated environment.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--np-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--ckpt_dir", default="")
    parser.add_argument("--forward_passes", type=int, default=5)
    parser.add_argument("--z_range", default="[0.2,1.1]")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import torch

    from contact_graspnet_pytorch import config_utils
    from contact_graspnet_pytorch.checkpoints import CheckpointIO
    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator

    checkpoint_dir = args.ckpt_dir or str(repo_root / "checkpoints" / "contact_graspnet")
    z_range = eval(str(args.z_range))

    with np.load(args.np_path, allow_pickle=True) as payload:
        scene_points_xyz = np.asarray(payload["xyz"], dtype=np.float32).reshape(-1, 3)
    valid_mask = (
        np.isfinite(scene_points_xyz).all(axis=1)
        & (scene_points_xyz[:, 2] >= float(z_range[0]))
        & (scene_points_xyz[:, 2] <= float(z_range[1]))
    )
    scene_points_xyz = scene_points_xyz[valid_mask]

    global_config = config_utils.load_config(
        checkpoint_dir,
        batch_size=args.forward_passes,
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

    pred_grasps_cam, scores, _contact_pts, gripper_openings = grasp_estimator.predict_scene_grasps(
        scene_points_xyz,
        pc_segments={},
        local_regions=False,
        filter_grasps=False,
        forward_passes=args.forward_passes,
    )

    grasp_array = np.asarray(pred_grasps_cam.get(-1, np.empty((0, 4, 4), dtype=np.float32)), dtype=np.float32)
    score_array = np.asarray(scores.get(-1, np.empty((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    width_array = np.asarray(gripper_openings.get(-1, np.empty((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)

    # Keep the same simple array layout regardless of how CGN internally groups results.
    np.savez(
        args.output_path,
        pred_grasps=grasp_array,
        pred_scores=score_array,
        pred_width=width_array,
        z_range=np.asarray(z_range, dtype=np.float32),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
