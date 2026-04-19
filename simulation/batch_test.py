from __future__ import annotations

"""Batch test the grasp pipeline across randomized cup positions."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ask2act_grasp.planning.motion_planner import MotionPlanner
from ask2act_grasp.utils.config_loader import load_grasp_config
from ask2act_grasp.utils.config_loader import load_scene_config


SIMULATION_ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE_CONFIG = SIMULATION_ROOT / "ask2act_grasp" / "config" / "scene_config.yaml"
DEFAULT_OUTPUT_DIR = Path("/tmp/ask2act_batch_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run randomized grasp pipeline batch tests.")
    parser.add_argument("--n_trials", type=int, default=10, help="Trials per method.")
    parser.add_argument("--noise", type=float, default=0.05, help="XY position randomization in meters.")
    parser.add_argument("--methods", nargs="+", default=["geometric", "oracle"], help="Methods to test.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for all outputs.")
    parser.add_argument("--random-seed", type=int, default=7, help="Seed for reproducible randomized positions.")
    parser.add_argument("--timeout-s", type=float, default=420.0, help="Per-trial subprocess timeout in seconds.")
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python interpreter used for trial subprocesses.",
    )
    parser.add_argument("--multi_cup", action="store_true", help="Run the multi-cup target-isolation scenario.")
    parser.add_argument("--multi-cup-count", type=int, default=3, help="Number of cups to place in multi-cup mode.")
    parser.add_argument("--gui", action="store_true", help="Run with the MuJoCo viewer instead of headless mode.")
    parser.add_argument("--show-viewer-ui", action="store_true", help="Show the full MuJoCo viewer UI panels.")
    parser.add_argument(
        "--full-safe-zone",
        action="store_true",
        help="Sample cup positions from the full table safe zone instead of only within the noise radius around default.",
    )
    parser.add_argument(
        "--gui-timeout-scale",
        type=float,
        default=2.5,
        help="Multiply waypoint timeouts by this factor when running with --gui.",
    )
    parser.add_argument(
        "--final-pause-s",
        type=float,
        default=4.0,
        help="How long to keep the GUI scene open after each trial for screenshots.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _compute_distance_to_center(
    diagnostics: dict[str, Any],
    pipeline_result: dict[str, Any],
) -> float | None:
    if diagnostics.get("distance_to_center") is not None:
        return float(diagnostics["distance_to_center"])

    object_center_xy = diagnostics.get("object_center_xy")
    contact_point = diagnostics.get("contact_point_xyz")
    if object_center_xy is not None and contact_point is not None:
        return float(np.linalg.norm(np.asarray(contact_point[:2], dtype=float) - np.asarray(object_center_xy, dtype=float)))

    best_distance = diagnostics.get("best_distance_to_center")
    if best_distance is not None:
        return float(best_distance)

    intermediate = pipeline_result.get("intermediate") or {}
    selected_position = intermediate.get("selected_grasp_position_m")
    if object_center_xy is not None and selected_position is not None:
        return float(np.linalg.norm(np.asarray(selected_position[:2], dtype=float) - np.asarray(object_center_xy, dtype=float)))

    return None


def _infer_method_used(requested_method: str, pipeline_result: dict[str, Any]) -> str:
    intermediate = pipeline_result.get("intermediate") or {}
    selected_source = str(intermediate.get("selected_grasp_source") or "")
    if requested_method != "cgn":
        return requested_method
    if selected_source == "contact_graspnet":
        return "cgn"
    if selected_source:
        return "oracle_fallback"
    return requested_method


def _extract_stage_times(
    diagnostics: dict[str, Any],
    pipeline_result: dict[str, Any],
) -> dict[str, float]:
    extracted: dict[str, float] = {}
    diagnostics_stage_times = diagnostics.get("stage_times_s")
    if isinstance(diagnostics_stage_times, dict):
        for key, value in diagnostics_stage_times.items():
            try:
                extracted[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

    intermediate = pipeline_result.get("intermediate") or {}
    pipeline_stage_times = intermediate.get("pipeline_stage_times_s")
    if isinstance(pipeline_stage_times, dict):
        for key, value in pipeline_stage_times.items():
            try:
                extracted[f"pipeline_{key}"] = float(value)
            except (TypeError, ValueError):
                continue
    return extracted


def _copy_trial_artifacts(trial_dir: Path) -> None:
    artifacts_dir = trial_dir / "artifacts"
    diagnostics_path = artifacts_dir / "diagnostics.json"
    if diagnostics_path.exists():
        shutil.copy2(diagnostics_path, trial_dir / "diagnostics.json")
    for name in ("pre_close_diagnostic.json", "geometric_grasp_debug.png"):
        path = artifacts_dir / name
        if path.exists():
            shutil.copy2(path, trial_dir / name)


def _load_pre_close_diagnostic(trial_dir: Path) -> dict[str, Any] | None:
    return _load_json(trial_dir / "artifacts" / "pre_close_diagnostic.json")


def _compute_success_level(
    *,
    pipeline_result: dict[str, Any],
    pre_close: dict[str, Any] | None,
) -> str:
    if bool(pipeline_result.get("success")):
        return "full"
    if not pre_close:
        return "fail"
    delta = pre_close.get("rubber_contact_center_delta_to_target")
    if not isinstance(delta, list) or len(delta) != 3:
        return "fail"
    if max(abs(float(v)) for v in delta) <= 0.015:
        return "partial"
    return "fail"


def run_single_trial(
    *,
    method: str,
    cup_x: float,
    cup_y: float,
    output_dir: str | Path,
    python_executable: str,
    timeout_s: float,
    extra_env: dict[str, str] | None = None,
    multi_cup_config_path: str | Path | None = None,
    headless: bool = True,
    show_viewer_ui: bool = False,
    gui_timeout_scale: float = 2.5,
    final_pause_s: float = 4.0,
) -> dict[str, Any]:
    trial_dir = Path(output_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    command = [
        python_executable,
        "-m",
        "ask2act_grasp.pipeline",
        "--grasp-method",
        method,
        "--cup-x",
        f"{cup_x:.6f}",
        "--cup-y",
        f"{cup_y:.6f}",
        "--run-dir",
        str(trial_dir),
    ]
    if headless:
        command.append("--headless")
    elif show_viewer_ui:
        command.append("--show-viewer-ui")
    if multi_cup_config_path is not None:
        command.extend(["--multi-cup", "--multi-cup-config", str(multi_cup_config_path)])
    (trial_dir / "command.json").write_text(json.dumps({"command": command}, indent=2))

    trial_env = {**os.environ, **(extra_env or {})}
    if not headless:
        trial_env["ASK2ACT_WAYPOINT_TIMEOUT_SCALE"] = str(max(1.0, float(gui_timeout_scale)))
        trial_env["ASK2ACT_FINAL_PAUSE_S"] = str(max(0.0, float(final_pause_s)))
        trial_env["ASK2ACT_USE_PASSIVE_VIEWER"] = "0"

    started_at = time.time()
    process_error: str | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=str(SIMULATION_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=trial_env,
        )
        stdout_text = completed.stdout
        stderr_text = completed.stderr
        return_code = completed.returncode
        if return_code != 0:
            process_error = f"pipeline exited with code {return_code}"
    except subprocess.TimeoutExpired as exc:
        stdout_text = _coerce_text(exc.stdout)
        stderr_text = _coerce_text(exc.stderr)
        return_code = -1
        process_error = f"pipeline timed out after {timeout_s:.1f}s"
    elapsed_s = time.time() - started_at

    (trial_dir / "stdout.log").write_text(stdout_text or "")
    (trial_dir / "stderr.log").write_text(stderr_text or "")
    pipeline_result = _load_json(trial_dir / "pipeline_result.json") or {}
    diagnostics = _load_json(trial_dir / "artifacts" / "diagnostics.json") or {}
    pre_close = _load_pre_close_diagnostic(trial_dir)
    _copy_trial_artifacts(trial_dir)

    if not pipeline_result:
        return {
            "success": False,
            "elapsed_s": elapsed_s,
            "error": process_error or "missing pipeline_result.json",
        }

    intermediate = pipeline_result.get("intermediate") or {}
    selected_score = diagnostics.get("selected_score", pipeline_result.get("selected_grasp_score"))
    approach_type = diagnostics.get("selected_approach_type", intermediate.get("selected_grasp_approach_type"))
    distance_to_center = _compute_distance_to_center(diagnostics, pipeline_result)
    result = {
        "success": bool(pipeline_result.get("success")),
        "success_level": _compute_success_level(pipeline_result=pipeline_result, pre_close=pre_close),
        "elapsed_s": elapsed_s,
        "error": pipeline_result.get("error") or process_error,
        "grasp_method_used": _infer_method_used(method, pipeline_result),
        "grasp_verified": intermediate.get("grasp_verified"),
        "cup_z_after_lift": intermediate.get("cup_z_after_lift"),
        "verification_method": intermediate.get("verification_method"),
        "selected_score": float(selected_score) if selected_score is not None else None,
        "approach_type": str(approach_type) if approach_type is not None else None,
        "distance_to_center": distance_to_center,
        "contact_point": diagnostics.get("contact_point_xyz"),
        "wrist_target": diagnostics.get("wrist_target_xyz"),
        "cgn_candidates_total": diagnostics.get("total_cgn_candidates"),
        "cgn_after_strategy": diagnostics.get("after_cup_strategy"),
        "selected_grasp_source": intermediate.get("selected_grasp_source"),
        "fk_error_m": ((intermediate.get("planner_metadata") or {}).get("numeric_targets") or {}).get("fk_error_m"),
        "rubber_delta_xyz": pre_close.get("rubber_contact_center_delta_to_target") if isinstance(pre_close, dict) else None,
        "rigid_delta_xyz": pre_close.get("rigid_grasp_center_delta_to_target") if isinstance(pre_close, dict) else None,
        "other_cups_stationary": (intermediate.get("verification_details") or {}).get("other_cups_stationary"),
        "pipeline_result_path": str(trial_dir / "pipeline_result.json"),
        "diagnostics_path": str(trial_dir / "diagnostics.json"),
        "pre_close_path": str(trial_dir / "pre_close_diagnostic.json"),
        "stdout_log_path": str(trial_dir / "stdout.log"),
        "stderr_log_path": str(trial_dir / "stderr.log"),
        "stage_times_s": _extract_stage_times(diagnostics, pipeline_result),
    }
    if not result["success"] and result["error"] is None:
        result["error"] = process_error or "pipeline reported failure"
    return result


@contextmanager
def maybe_start_cgn_server(
    *,
    methods: tuple[str, ...],
    python_executable: str,
    timeout_s: float,
    output_root: Path,
) -> dict[str, str]:
    if "cgn" not in methods:
        yield {}
        return

    grasp_config = load_grasp_config(SIMULATION_ROOT / "ask2act_grasp" / "config" / "grasp_config.yaml")
    repo_root = Path(grasp_config.contact_graspnet_repo).expanduser()
    checkpoint_dir = grasp_config.contact_graspnet_checkpoint or ""
    cgn_python = grasp_config.contact_graspnet_python or python_executable
    server_dir = output_root / "_cgn_server"
    server_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = server_dir / "server_stdout.log"
    stderr_log = server_dir / "server_stderr.log"
    stdout_handle = stdout_log.open("w")
    stderr_handle = stderr_log.open("w")

    command = [
        cgn_python,
        str(SIMULATION_ROOT / "ask2act_grasp" / "grasp" / "cgn_inference_server.py"),
        f"--repo-root={repo_root}",
        f"--server-dir={server_dir}",
        f"--forward_passes={grasp_config.forward_passes}",
        f"--z-range=[{grasp_config.z_min_m},{grasp_config.z_max_m}]",
    ]
    if checkpoint_dir:
        command.append(f"--ckpt_dir={checkpoint_dir}")

    env = dict(os.environ)
    if grasp_config.cgn_device.lower() == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    process = subprocess.Popen(
        command,
        cwd=str(repo_root),
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        env=env,
    )
    ready_path = server_dir / "ready.json"
    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline:
            if ready_path.exists():
                yield {
                    "ASK2ACT_CGN_SERVER_DIR": str(server_dir),
                    "ASK2ACT_CGN_SERVER_TIMEOUT_S": str(timeout_s),
                }
                break
            if process.poll() is not None:
                raise RuntimeError(
                    "CGN server exited before becoming ready. "
                    f"See {stdout_log} and {stderr_log}."
                )
            time.sleep(0.1)
        else:
            raise RuntimeError(f"Timed out waiting for CGN server readiness at {ready_path}")
    finally:
        stop_file = server_dir / "STOP"
        stop_file.write_text("stop\n")
        if process.poll() is None:
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
        stdout_handle.close()
        stderr_handle.close()


def print_summary(results: list[dict[str, Any]], methods: tuple[str, ...], output_dir: str | Path) -> str:
    lines = ["", "=" * 60, "BATCH TEST SUMMARY", "=" * 60]

    for method in methods:
        method_results = [entry for entry in results if entry["method"] == method]
        if not method_results:
            continue
        n_trials = len(method_results)
        successes = sum(1 for entry in method_results if entry["success"])
        full_successes = sum(1 for entry in method_results if entry.get("success_level") == "full")
        partial_successes = sum(1 for entry in method_results if entry.get("success_level") == "partial")
        errors = sum(1 for entry in method_results if entry.get("error"))
        times = np.asarray([float(entry["elapsed_s"]) for entry in method_results], dtype=float)

        lines.extend(
            [
                "",
                f"--- {method.upper()} ---",
                f"  Trials:       {n_trials}",
                f"  Successes:    {successes}/{n_trials} ({100.0 * successes / n_trials:.0f}%)",
                f"  Full success: {full_successes}/{n_trials}",
                f"  Half success: {partial_successes}/{n_trials}",
                f"  Failures:     {n_trials - successes}",
                f"  Errors:       {errors}",
                f"  Avg time:     {times.mean():.1f}s",
                f"  Median time:  {np.median(times):.1f}s",
                "  Per trial:",
            ]
        )

        for entry in method_results:
            status = "OK" if entry["success"] else "FAIL"
            error_suffix = f" [{str(entry['error'])[:60]}]" if entry.get("error") else ""
            lines.append(
                f"    #{int(entry['trial']):02d} ({float(entry['cup_x']):+.3f},{float(entry['cup_y']):+.3f}) "
                f"-> {status}/{entry.get('success_level','fail')} "
                f"fk={entry.get('fk_error_m')} rubber={entry.get('rubber_delta_xyz')} "
                f"{float(entry['elapsed_s']):.1f}s{error_suffix}"
            )

        if method == "cgn":
            scores = [float(entry["selected_score"]) for entry in method_results if entry.get("selected_score") is not None]
            distances = [float(entry["distance_to_center"]) for entry in method_results if entry.get("distance_to_center") is not None]
            approach_types = [str(entry["approach_type"]) for entry in method_results if entry.get("approach_type")]
            method_used = Counter(str(entry["grasp_method_used"]) for entry in method_results)

            if scores:
                score_array = np.asarray(scores, dtype=float)
                lines.append(
                    f"  CGN scores:   mean={score_array.mean():.3f}, min={score_array.min():.3f}, max={score_array.max():.3f}"
                )
            if distances:
                distance_array = np.asarray(distances, dtype=float)
                lines.append(
                    f"  Dist to center: mean={distance_array.mean():.3f}m, max={distance_array.max():.3f}m"
                )
            if approach_types:
                lines.append(f"  Approach types: {dict(Counter(approach_types))}")
            if method_used:
                lines.append(f"  Method used: {dict(method_used)}")
            stage_keys = sorted(
                {
                    stage_name
                    for entry in method_results
                    for stage_name in (entry.get("stage_times_s") or {}).keys()
                }
            )
            if stage_keys:
                lines.append("  Stage means (successful trials):")
                successful_results = [entry for entry in method_results if entry["success"]]
                for stage_name in stage_keys:
                    values = [
                        float(entry["stage_times_s"][stage_name])
                        for entry in successful_results
                        if stage_name in (entry.get("stage_times_s") or {})
                    ]
                    if values:
                        lines.append(f"    {stage_name}: {np.mean(values):.2f}s")

    summary_text = "\n".join(lines)
    print(summary_text, flush=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "summary.txt").write_text(summary_text)
    return summary_text


def run_batch_test(
    *,
    n_trials: int = 20,
    position_noise_m: float = 0.05,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    methods: tuple[str, ...] = ("geometric", "oracle"),
    random_seed: int = 7,
    timeout_s: float = 420.0,
    python_executable: str = sys.executable,
    multi_cup: bool = False,
    multi_cup_count: int = 3,
    headless: bool = True,
    show_viewer_ui: bool = False,
    gui_timeout_scale: float = 2.5,
    final_pause_s: float = 4.0,
    full_safe_zone: bool = False,
) -> list[dict[str, Any]]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    scene_config, _ = load_scene_config(DEFAULT_SCENE_CONFIG)
    grasp_config = load_grasp_config(SIMULATION_ROOT / "ask2act_grasp" / "config" / "grasp_config.yaml")
    default_cup_x, default_cup_y, _ = scene_config.cup_position_m
    sampling_planner = None
    if "geometric" in methods:
        sampling_planner = MotionPlanner(scene_config, grasp_config)

    rng = np.random.default_rng(random_seed)
    offsets = _sample_single_cup_offsets(
        scene_config=scene_config,
        motion_planner=sampling_planner,
        grasp_config=grasp_config,
        rng=rng,
        n_trials=n_trials,
        position_noise_m=position_noise_m,
        full_safe_zone=full_safe_zone,
    )

    results: list[dict[str, Any]] = []
    with maybe_start_cgn_server(
        methods=methods,
        python_executable=python_executable,
        timeout_s=timeout_s,
        output_root=output_root,
    ) as cgn_env:
        for method in methods:
            method_dir = output_root / method
            method_dir.mkdir(parents=True, exist_ok=True)
            for trial, (dx, dy) in enumerate(offsets):
                cup_x = float(default_cup_x + dx)
                cup_y = float(default_cup_y + dy)
                trial_dir = method_dir / f"trial_{trial:03d}"
                trial_dir.mkdir(parents=True, exist_ok=True)
                multi_cup_config_path = None
                if multi_cup:
                    multi_cup_config = _sample_multi_cup_config(
                        scene_config=scene_config,
                        rng=rng,
                        cup_count=multi_cup_count,
                        target_position_xy=(cup_x, cup_y),
                    )
                    multi_cup_config_path = trial_dir / "multi_cup_config.json"
                    multi_cup_config_path.write_text(json.dumps(multi_cup_config, indent=2))

                print(flush=True)
                print("=" * 60, flush=True)
                print(
                    f"[{method}] Trial {trial}/{n_trials - 1}: "
                    f"cup=({cup_x:.3f}, {cup_y:.3f}), offset=({dx:.3f}, {dy:.3f})",
                    flush=True,
                )
                print("=" * 60, flush=True)

                try:
                    trial_result = run_single_trial(
                        method=method,
                        cup_x=cup_x,
                        cup_y=cup_y,
                        output_dir=trial_dir,
                        python_executable=python_executable,
                        timeout_s=timeout_s,
                        extra_env=cgn_env if method == "cgn" else None,
                        multi_cup_config_path=multi_cup_config_path,
                        headless=headless,
                        show_viewer_ui=show_viewer_ui,
                        gui_timeout_scale=gui_timeout_scale,
                        final_pause_s=final_pause_s,
                    )
                except Exception as exc:
                    trial_result = {
                        "success": False,
                        "elapsed_s": 0.0,
                        "error": str(exc),
                    }
                    (trial_dir / "outer_exception.txt").write_text(traceback.format_exc())

                trial_result.update(
                    {
                        "method": method,
                        "trial": trial,
                        "cup_x": cup_x,
                        "cup_y": cup_y,
                        "dx": dx,
                        "dy": dy,
                    }
                )
                error_suffix = f" error={trial_result.get('error')}" if trial_result.get("error") else ""
                results.append(trial_result)
                print(
                    f"  Result: {'SUCCESS' if trial_result['success'] else 'FAIL'} "
                    f"({float(trial_result['elapsed_s']):.1f}s){error_suffix}",
                    flush=True,
                )

    (output_root / "batch_results.json").write_text(json.dumps(results, indent=2, default=str))
    print_summary(results, methods=methods, output_dir=output_root)
    return results


def main() -> int:
    args = parse_args()
    run_batch_test(
        n_trials=args.n_trials,
        position_noise_m=args.noise,
        output_dir=args.output_dir,
        methods=tuple(args.methods),
        random_seed=args.random_seed,
        timeout_s=args.timeout_s,
        python_executable=args.python_executable,
        multi_cup=args.multi_cup,
        multi_cup_count=args.multi_cup_count,
        headless=not args.gui,
        show_viewer_ui=args.show_viewer_ui,
        gui_timeout_scale=args.gui_timeout_scale,
        final_pause_s=args.final_pause_s,
        full_safe_zone=args.full_safe_zone,
    )
    return 0


def _single_cup_safe_bounds(
    scene_config,
    position_noise_m: float,
    *,
    full_safe_zone: bool = False,
) -> tuple[tuple[float, float], tuple[float, float]]:
    table_x, table_y, _ = scene_config.table_position_m
    table_sx, table_sy, _ = scene_config.table_size_m
    default_x, default_y, _ = scene_config.cup_position_m
    margin = 0.05
    if full_safe_zone:
        min_x = table_x - table_sx + margin
        max_x = table_x + table_sx - margin
        min_y = table_y - table_sy + margin
        max_y = table_y + table_sy - margin
    else:
        min_x = max(table_x - table_sx + margin, default_x - position_noise_m)
        max_x = min(table_x + table_sx - margin, default_x + position_noise_m)
        min_y = max(table_y - table_sy + margin, default_y - position_noise_m)
        max_y = min(table_y + table_sy - margin, default_y + position_noise_m)
    return (min_x, max_x), (min_y, max_y)


def _sample_single_cup_offsets(
    *,
    scene_config,
    motion_planner: MotionPlanner | None,
    grasp_config,
    rng: np.random.Generator,
    n_trials: int,
    position_noise_m: float,
    full_safe_zone: bool = False,
) -> list[tuple[float, float]]:
    (min_x, max_x), (min_y, max_y) = _single_cup_safe_bounds(
        scene_config,
        position_noise_m,
        full_safe_zone=full_safe_zone,
    )
    default_x, default_y, _ = scene_config.cup_position_m
    offsets: list[tuple[float, float]] = []
    attempts = 0
    max_attempts = max(100, n_trials * 50)
    while len(offsets) < n_trials and attempts < max_attempts:
        attempts += 1
        cup_x = float(rng.uniform(min_x, max_x))
        cup_y = float(rng.uniform(min_y, max_y))
        if motion_planner is not None and not grasp_config.enable_base_preposition:
            nominal_geometric_grasp = {
                "grasp_x": cup_x,
                "grasp_y": cup_y,
                "grasp_z": float(scene_config.table_top_z_m + scene_config.cup_height_m * 0.5),
                "gripper_open_width": float(
                    min(scene_config.cup_radius_m * 2.0 + 0.02, grasp_config.max_gripper_width_m)
                ),
                "grip_angle_rad": 0.0,
                "min_cross_section_width": float(scene_config.cup_radius_m * 2.0),
                "object_center": [cup_x, cup_y, float(scene_config.table_top_z_m + scene_config.cup_height_m * 0.5)],
                "object_height": float(scene_config.cup_height_m),
                "object_top_z": float(scene_config.table_top_z_m + scene_config.cup_height_m),
                "object_bottom_z": float(scene_config.table_top_z_m),
                "grasp_point_validated": True,
                "width_near_limit": True,
            }
            try:
                targets = motion_planner.geometric_grasp_targets(
                    nominal_geometric_grasp,
                    current_state={"wrist_yaw": 0.0},
                )
            except RuntimeError:
                continue
            if "ik_base_rotate" not in targets:
                continue
        offsets.append((cup_x - default_x, cup_y - default_y))
    if len(offsets) < n_trials:
        raise RuntimeError(
            f"Could only find {len(offsets)} reachable randomized cup positions after {attempts} attempts."
        )
    return offsets


def _sample_multi_cup_config(
    *,
    scene_config,
    rng: np.random.Generator,
    cup_count: int,
    target_position_xy: tuple[float, float],
) -> list[dict[str, Any]]:
    table_x, table_y, _ = scene_config.table_position_m
    table_sx, table_sy, _ = scene_config.table_size_m
    margin = 0.05
    min_sep = 0.12
    colors = [
        [0.82, 0.20, 0.16, 1.0],
        [0.18, 0.38, 0.82, 1.0],
        [0.18, 0.66, 0.30, 1.0],
        [0.88, 0.72, 0.18, 1.0],
        [0.58, 0.28, 0.78, 1.0],
    ]
    safe_min_x = table_x - table_sx + margin
    safe_max_x = table_x + table_sx - margin
    safe_min_y = table_y - table_sy + margin
    safe_max_y = table_y + table_sy - margin
    positions = [np.array(target_position_xy, dtype=float)]
    while len(positions) < cup_count:
        candidate = np.array(
            [
                float(rng.uniform(safe_min_x, safe_max_x)),
                float(rng.uniform(safe_min_y, safe_max_y)),
            ],
            dtype=float,
        )
        if all(np.linalg.norm(candidate - existing) >= min_sep for existing in positions):
            positions.append(candidate)

    configs: list[dict[str, Any]] = []
    for index, pos in enumerate(positions[:cup_count]):
        is_target = index == 0
        configs.append(
            {
                "name": scene_config.cup_body_name if is_target else f"cup_{index:02d}",
                "position": [float(pos[0]), float(pos[1]), float(scene_config.cup_position_m[2])],
                "color": colors[index % len(colors)],
                "is_target": is_target,
            }
        )
    return configs


if __name__ == "__main__":
    raise SystemExit(main())
