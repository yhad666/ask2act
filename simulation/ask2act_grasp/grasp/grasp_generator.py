from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

from ask2act_grasp.types import GraspCandidate, GraspConfig
from ask2act_grasp.utils.tf_utils import normalize, pose_from_axes


DEFAULT_CGN_REPO_CANDIDATES = (
    Path("/home/yhad/robot/contact_graspnet_pytorch"),
    Path("/home/yhad/robot/ask2act/simulation/third_party/contact_graspnet_pytorch"),
)
DEFAULT_CGN_PYTHON_CANDIDATES = (
    Path("/home/yhad/robot/ask2act/simulation/third_party/contact_graspnet_pytorch/.venv/bin/python"),
    Path("/home/yhad/robot/contact_graspnet_pytorch/.venv/bin/python"),
)
DEFAULT_CGN_CACHE_PATH = Path("/tmp/ask2act_cgn_cache.npz")
CGN_RUNNER_PATH = Path(__file__).resolve().with_name("cgn_inference_runner.py")
CGN_SERVER_DIR_ENV = "ASK2ACT_CGN_SERVER_DIR"


class ContactGraspNetGenerator:
    """Generate grasp candidates from a scene point cloud using Contact-GraspNet."""

    def __init__(self, grasp_config: GraspConfig) -> None:
        self.grasp_config = grasp_config
        self.cache_path = DEFAULT_CGN_CACHE_PATH

    def generate(
        self,
        scene_pcd: np.ndarray,
        frame_transform: np.ndarray | None = None,
    ) -> list[GraspCandidate]:
        """Return a descending-score list of grasp candidates for the given scene."""
        scene_points_xyz = np.asarray(scene_pcd, dtype=np.float32)
        if scene_points_xyz.size == 0:
            return []

        repo_root = self._resolve_contact_graspnet_repo()
        if (
            self.grasp_config.enable_contact_graspnet
            and repo_root is not None
            and self._has_contact_graspnet_runtime(repo_root)
        ):
            cache_key = self._hash_scene(scene_points_xyz, config_signature=self._cache_config_signature(repo_root))
            cached = self._load_cache(cache_key)
            if cached is not None:
                return self._transform_candidates(cached, frame_transform)
            generated = self._run_contact_graspnet(repo_root, scene_points_xyz)
            if generated:
                self._save_cache(cache_key, generated)
                return self._transform_candidates(generated, frame_transform)

        if self.grasp_config.enable_fallback_generator:
            return self._transform_candidates(self._generate_fallback(scene_points_xyz), frame_transform)

        raise FileNotFoundError("Contact-GraspNet repo/inference script not available and fallback generator is disabled.")

    def _resolve_contact_graspnet_repo(self) -> Path | None:
        """Find the configured Contact-GraspNet repo or a known fallback path."""
        configured = Path(self.grasp_config.contact_graspnet_repo).expanduser()
        candidates = [configured, *DEFAULT_CGN_REPO_CANDIDATES]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _resolve_contact_graspnet_python(self, repo_root: Path) -> str:
        """Resolve the python executable used for isolated CGN inference."""
        configured = self.grasp_config.contact_graspnet_python
        candidates: list[Path] = []
        if configured:
            candidates.append(Path(configured).expanduser())
        candidates.extend(
            [
                repo_root / ".venv" / "bin" / "python",
                *DEFAULT_CGN_PYTHON_CANDIDATES,
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return sys.executable

    @staticmethod
    def _has_contact_graspnet_runtime(repo_root: Path) -> bool:
        """Check for the CGN package layout expected by the subprocess runner."""
        return (
            (repo_root / "contact_graspnet_pytorch" / "contact_grasp_estimator.py").exists()
            and (repo_root / "contact_graspnet_pytorch" / "checkpoints.py").exists()
        )

    def _run_contact_graspnet(self, repo_root: Path, scene_points_xyz: np.ndarray) -> list[GraspCandidate]:
        """Run the isolated CGN inference runner in a temp directory and parse its output."""
        persistent_predictions = self._run_contact_graspnet_via_server(scene_points_xyz)
        if persistent_predictions is not None:
            return persistent_predictions

        checkpoint_dir = self._resolve_checkpoint_dir(repo_root)
        python_executable = self._resolve_contact_graspnet_python(repo_root)
        with tempfile.TemporaryDirectory(prefix="ask2act_cgn_") as temp_dir:
            temp_root = Path(temp_dir)
            input_path = temp_root / "scene.npz"
            output_path = temp_root / "grasp_preds.npz"
            np.savez(input_path, xyz=scene_points_xyz.astype(np.float32))

            command = [
                python_executable,
                str(CGN_RUNNER_PATH),
                f"--repo-root={repo_root}",
                f"--np-path={input_path}",
                f"--output-path={output_path}",
                f"--forward_passes={self.grasp_config.forward_passes}",
                f"--z_range=[{self.grasp_config.z_min_m},{self.grasp_config.z_max_m}]",
            ]
            if checkpoint_dir is not None:
                command.append(f"--ckpt_dir={checkpoint_dir}")

            env = os.environ.copy()
            if self.grasp_config.cgn_device.lower() == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

            try:
                completed = self._run_cgn_command(command, repo_root, env)
            except subprocess.CalledProcessError as exc:
                self._log_subprocess_failure(exc)
                return []
            except Exception as exc:
                print(f"CGN subprocess failed unexpectedly: {exc}", file=sys.stderr)
                return []

            if completed.stderr.strip():
                print(completed.stderr.strip(), file=sys.stderr)

            if not output_path.exists():
                return []

            with np.load(output_path, allow_pickle=True) as predictions:
                return self._parse_prediction_file(predictions)

    def _run_contact_graspnet_via_server(self, scene_points_xyz: np.ndarray) -> list[GraspCandidate] | None:
        """Use a batch-scoped persistent CGN server when available."""
        server_dir_raw = os.environ.get(CGN_SERVER_DIR_ENV, "").strip()
        if not server_dir_raw:
            return None

        server_dir = Path(server_dir_raw).expanduser()
        ready_path = server_dir / "ready.json"
        request_dir = server_dir / "requests"
        response_dir = server_dir / "responses"
        if not ready_path.exists() or not request_dir.exists() or not response_dir.exists():
            return None

        request_id = uuid.uuid4().hex
        request_path = request_dir / f"{request_id}.npz"
        response_path = response_dir / f"{request_id}.npz"
        error_path = response_dir / f"{request_id}.error.txt"
        np.savez(request_path, xyz=np.asarray(scene_points_xyz, dtype=np.float32))

        timeout_s = float(os.environ.get("ASK2ACT_CGN_SERVER_TIMEOUT_S", "180.0"))
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if response_path.exists():
                with np.load(response_path, allow_pickle=True) as predictions:
                    parsed = self._parse_prediction_file(predictions)
                response_path.unlink(missing_ok=True)
                error_path.unlink(missing_ok=True)
                return parsed
            if error_path.exists():
                error_message = error_path.read_text().strip()
                error_path.unlink(missing_ok=True)
                print(f"CGN persistent server failed: {error_message}", file=sys.stderr)
                return []
            time.sleep(0.05)

        request_path.unlink(missing_ok=True)
        print(f"Timed out waiting for persistent CGN server after {timeout_s:.1f}s.", file=sys.stderr)
        return []

    def _run_cgn_command(
        self,
        command: list[str],
        repo_root: Path,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        """Run CGN inference and retry on CPU if CUDA runs out of memory."""
        try:
            return subprocess.run(
                command,
                cwd=str(repo_root),
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            combined_output = "\n".join(
                part for part in [exc.stdout.strip(), exc.stderr.strip()] if part
            )
            if self._is_cuda_oom(combined_output) and self.grasp_config.cgn_device.lower() != "cpu":
                print(
                    "CGN CUDA run hit OOM; retrying inference on CPU with the same point cloud.",
                    file=sys.stderr,
                )
                cpu_env = dict(env)
                cpu_env["CUDA_VISIBLE_DEVICES"] = ""
                return subprocess.run(
                    command,
                    cwd=str(repo_root),
                    check=True,
                    capture_output=True,
                    text=True,
                    env=cpu_env,
                )
            raise

    @staticmethod
    def _is_cuda_oom(output_text: str) -> bool:
        """Detect CUDA OOM failures from subprocess output."""
        lowered = output_text.lower()
        return "cuda out of memory" in lowered or "outofmemoryerror" in lowered

    @staticmethod
    def _log_subprocess_failure(exc: subprocess.CalledProcessError) -> None:
        """Print useful subprocess logs instead of silently swallowing them."""
        print(f"CGN subprocess exited with code {exc.returncode}.", file=sys.stderr)
        if exc.stdout.strip():
            print(exc.stdout.strip(), file=sys.stderr)
        if exc.stderr.strip():
            print(exc.stderr.strip(), file=sys.stderr)

    def _resolve_checkpoint_dir(self, repo_root: Path) -> str | None:
        """Resolve the configured checkpoint path or a reasonable default under the repo."""
        if self.grasp_config.contact_graspnet_checkpoint:
            return str(Path(self.grasp_config.contact_graspnet_checkpoint).expanduser())
        pytorch_default_dir = repo_root / "checkpoints" / "contact_graspnet"
        if pytorch_default_dir.exists():
            return str(pytorch_default_dir)
        default_dir = repo_root / "checkpoints" / "scene_test_2048_bs3_hor_sigma_001"
        return str(default_dir) if default_dir.exists() else None

    def _parse_prediction_file(self, predictions: np.lib.npyio.NpzFile) -> list[GraspCandidate]:
        """Parse common CGN output layouts into a normalized candidate list."""
        if "pred_grasps_cam" in predictions:
            pred_grasps_cam = predictions["pred_grasps_cam"].item()
            scores = predictions["scores"].item() if "scores" in predictions else {}
            widths = predictions["contact_width"].item() if "contact_width" in predictions else {}
            return self._parse_grasp_dicts(pred_grasps_cam, scores, widths)

        if "pred_grasps" in predictions and "pred_scores" in predictions:
            return self._parse_grasp_arrays(
                predictions["pred_grasps"],
                predictions["pred_scores"],
                predictions["pred_width"] if "pred_width" in predictions else None,
            )

        if "grasp_poses" in predictions and "scores" in predictions:
            return self._parse_grasp_arrays(
                predictions["grasp_poses"],
                predictions["scores"],
                predictions["widths"] if "widths" in predictions else None,
            )

        return []

    def _parse_grasp_dicts(
        self,
        grasp_dict: dict,
        score_dict: dict,
        width_dict: dict,
    ) -> list[GraspCandidate]:
        """Parse dictionary-form CGN outputs keyed by segmap or -1."""
        candidates: list[GraspCandidate] = []
        for key, poses in grasp_dict.items():
            key_scores = np.asarray(score_dict.get(key, []), dtype=float)
            key_widths = np.asarray(width_dict.get(key, []), dtype=float)
            candidates.extend(self._parse_grasp_arrays(poses, key_scores, key_widths))
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def _parse_grasp_arrays(
        self,
        poses: np.ndarray,
        scores: np.ndarray,
        widths: np.ndarray | None,
    ) -> list[GraspCandidate]:
        """Parse array-form CGN outputs."""
        pose_array = np.asarray(poses, dtype=float)
        score_array = np.asarray(scores, dtype=float).reshape(-1)
        if widths is None:
            width_array = np.full(score_array.shape, self.grasp_config.max_gripper_width_m, dtype=float)
        else:
            width_array = np.asarray(widths, dtype=float).reshape(-1)

        candidates: list[GraspCandidate] = []
        for pose, score, width in zip(pose_array, score_array, width_array):
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

    def _load_cache(self, cache_key: str) -> list[GraspCandidate] | None:
        """Load cached grasp predictions if the scene hash matches."""
        if not self.cache_path.exists():
            return None
        try:
            with np.load(self.cache_path, allow_pickle=True) as cache:
                stored_key = str(cache["scene_hash"].item())
                if stored_key != cache_key:
                    return None
                return self._parse_grasp_arrays(cache["poses"], cache["scores"], cache["widths"])
        except Exception:
            return None

    def _save_cache(self, cache_key: str, candidates: list[GraspCandidate]) -> None:
        """Persist a small cache of the latest CGN prediction set."""
        poses = np.stack([candidate.pose_4x4 for candidate in candidates], axis=0)
        scores = np.asarray([candidate.score for candidate in candidates], dtype=float)
        widths = np.asarray([candidate.width_m for candidate in candidates], dtype=float)
        np.savez(self.cache_path, scene_hash=cache_key, poses=poses, scores=scores, widths=widths)

    @staticmethod
    def _hash_scene(scene_points_xyz: np.ndarray, config_signature: str = "") -> str:
        """Build a stable cache key from the scene point cloud."""
        rounded = np.round(np.asarray(scene_points_xyz, dtype=np.float32), 4)
        payload = rounded.tobytes() + config_signature.encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    def _cache_config_signature(self, repo_root: Path) -> str:
        """Capture the CGN settings that materially affect inference output."""
        checkpoint_dir = self._resolve_checkpoint_dir(repo_root) or ""
        return "|".join(
            [
                str(repo_root),
                checkpoint_dir,
                str(self.grasp_config.forward_passes),
                f"{self.grasp_config.z_min_m:.4f}",
                f"{self.grasp_config.z_max_m:.4f}",
                str(self.grasp_config.cgn_device).lower(),
            ]
        )

    def _generate_fallback(self, scene_points_xyz: np.ndarray) -> list[GraspCandidate]:
        """Generate a simple fallback grasp set when CGN is unavailable."""
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
            GraspCandidate(
                pose_4x4=pose,
                score=0.65,
                width_m=width,
                source="fallback_pca",
                metadata={"extents": extents.tolist()},
            ),
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

    @staticmethod
    def _transform_candidates(
        candidates: list[GraspCandidate],
        frame_transform: np.ndarray | None,
    ) -> list[GraspCandidate]:
        """Transform raw CGN-frame poses into world frame when a transform is provided."""
        if frame_transform is None:
            return candidates

        transform = np.asarray(frame_transform, dtype=float)
        transformed: list[GraspCandidate] = []
        for candidate in candidates:
            pose_world = transform @ np.asarray(candidate.pose_4x4, dtype=float)
            transformed.append(
                GraspCandidate(
                    pose_4x4=pose_world,
                    score=float(candidate.score),
                    width_m=float(candidate.width_m),
                    source=candidate.source,
                    metadata=dict(candidate.metadata),
                    preferred_approach=candidate.preferred_approach,
                    needs_wrist_refinement=bool(candidate.needs_wrist_refinement),
                    approach_type=candidate.approach_type,
                )
            )
        transformed.sort(key=lambda item: item.score, reverse=True)
        return transformed

    @staticmethod
    def candidates_to_jsonable(candidates: list[GraspCandidate]) -> list[dict[str, object]]:
        """Convert candidates to a JSON-serializable structure for artifacts."""
        return [
            {
                "score": float(candidate.score),
                "width_m": float(candidate.width_m),
                "source": candidate.source,
                "preferred_approach": candidate.preferred_approach,
                "needs_wrist_refinement": bool(candidate.needs_wrist_refinement),
                "approach_type": candidate.approach_type,
                "pose_4x4": np.asarray(candidate.pose_4x4, dtype=float).tolist(),
                "metadata": json.loads(json.dumps(candidate.metadata, default=_json_default)),
            }
            for candidate in candidates
        ]


ContactGraspNetWrapper = ContactGraspNetGenerator


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
