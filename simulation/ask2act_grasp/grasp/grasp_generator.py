from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ask2act_grasp.types import GraspCandidate, GraspConfig
from ask2act_grasp.utils.tf_utils import normalize, pose_from_axes


class ContactGraspNetWrapper:
    def __init__(self, grasp_config: GraspConfig) -> None:
        self.grasp_config = grasp_config

    def generate(self, scene_points_xyz: np.ndarray) -> list[GraspCandidate]:
        repo_root = Path(self.grasp_config.contact_graspnet_repo).expanduser()
        if self.grasp_config.enable_contact_graspnet and repo_root.exists():
            return self._run_contact_graspnet(repo_root, scene_points_xyz)
        if self.grasp_config.enable_fallback_generator:
            return self._generate_fallback(scene_points_xyz)
        raise FileNotFoundError(
            f"Contact-GraspNet repo not found at {repo_root} and fallback generator is disabled."
        )

    def _run_contact_graspnet(self, repo_root: Path, scene_points_xyz: np.ndarray) -> list[GraspCandidate]:
        inference_script = repo_root / "inference.py"
        if not inference_script.exists():
            return self._generate_fallback(scene_points_xyz)

        with tempfile.TemporaryDirectory(prefix="ask2act_cgn_") as temp_dir:
            temp_root = Path(temp_dir)
            input_path = temp_root / "scene.npz"
            output_path = temp_root / "grasp_preds.npz"
            np.savez(input_path, xyz=scene_points_xyz.astype(np.float32))
            command = [
                "python",
                str(inference_script),
                f"--np_path={input_path}",
                f"--forward_passes={self.grasp_config.forward_passes}",
                f"--z_range=[{self.grasp_config.z_min_m},{self.grasp_config.z_max_m}]",
                f"--output_path={output_path}",
            ]
            if self.grasp_config.contact_graspnet_checkpoint:
                command.append(f"--ckpt_dir={self.grasp_config.contact_graspnet_checkpoint}")
            try:
                subprocess.run(
                    command,
                    cwd=str(repo_root),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if output_path.exists():
                    predictions = np.load(output_path, allow_pickle=True)
                    return self._parse_prediction_file(predictions)
            except Exception:
                pass
        return self._generate_fallback(scene_points_xyz)

    def _parse_prediction_file(self, predictions: np.lib.npyio.NpzFile) -> list[GraspCandidate]:
        poses = predictions.get("grasp_poses") or predictions.get("pred_grasps")
        scores = predictions.get("scores") or predictions.get("pred_scores")
        widths = predictions.get("widths") or predictions.get("pred_width")
        if poses is None or scores is None or widths is None:
            return []
        candidates: list[GraspCandidate] = []
        for pose, score, width in zip(poses, scores, widths):
            candidates.append(
                GraspCandidate(
                    pose_4x4=np.asarray(pose, dtype=float),
                    score=float(score),
                    width_m=float(width),
                    source="contact_graspnet",
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def _generate_fallback(self, scene_points_xyz: np.ndarray) -> list[GraspCandidate]:
        if scene_points_xyz.size == 0:
            return []
        centroid = np.median(scene_points_xyz, axis=0)
        extents = np.ptp(scene_points_xyz, axis=0)
        major_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        if extents[1] > extents[0]:
            major_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        z_axis = normalize(np.array([0.0, 0.0, -1.0], dtype=float))
        x_axis = normalize(major_axis)
        y_axis = normalize(np.cross(z_axis, x_axis))
        pose = pose_from_axes(centroid, x_axis, y_axis, z_axis)
        width = float(min(max(extents[0], extents[1]) * 0.9, self.grasp_config.max_gripper_width_m))
        variants = [
            GraspCandidate(pose_4x4=pose, score=0.65, width_m=width, source="fallback_pca", metadata={"extents": extents.tolist()}),
            GraspCandidate(
                pose_4x4=pose_from_axes(centroid + np.array([0.0, 0.0, 0.01]), y_axis, -x_axis, z_axis),
                score=0.58,
                width_m=width,
                source="fallback_pca",
                metadata={"extents": extents.tolist()},
            ),
        ]
        variants.sort(key=lambda item: item.score, reverse=True)
        return variants

