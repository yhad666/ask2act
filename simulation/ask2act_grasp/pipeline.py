from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import time

SIMULATION_ROOT = Path(__file__).resolve().parent.parent
STRETCH_MUJOCO_ROOT = SIMULATION_ROOT / "stretch_mujoco"
for import_root in (SIMULATION_ROOT, STRETCH_MUJOCO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator

from ask2act_grasp.execution.grasp_executor import GraspExecutor
from ask2act_grasp.scene.scene_setup import SceneSetup
from ask2act_grasp.types import GraspConfig, HeadAlignmentConfig, PipelineContext, PipelineResult, SceneConfig
from ask2act_grasp.utils.config_loader import load_grasp_config, load_scene_config
from ask2act_grasp.utils.pipeline_overrides import apply_scene_overrides


def parse_args() -> argparse.Namespace:
    package_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Ask2Act single-cup simulation grasp pipeline")
    parser.add_argument("--scene-config", default=str(package_root / "config" / "scene_config.yaml"))
    parser.add_argument("--grasp-config", default=str(package_root / "config" / "grasp_config.yaml"))
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--randomize-cup", action="store_true")
    parser.add_argument("--cup-x", type=float, default=None)
    parser.add_argument("--cup-y", type=float, default=None)
    parser.add_argument("--grasp-method", choices=("geometric", "cgn", "oracle"), default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--show-viewer-ui", action="store_true")
    parser.add_argument("--target-bbox-2d", default="")
    parser.add_argument("--multi-cup", action="store_true")
    parser.add_argument("--multi-cup-config", default="")
    return parser.parse_args()


def parse_bbox(raw_value: str) -> tuple[int, int, int, int] | None:
    if not raw_value:
        return None
    parts = [int(item.strip()) for item in raw_value.split(",")]
    if len(parts) != 4:
        raise ValueError("Expected target bbox as x0,y0,x1,y1")
    return tuple(parts)  # type: ignore[return-value]


def build_run_dir(simulation_root: Path, explicit_path: str) -> Path:
    if explicit_path:
        run_dir = Path(explicit_path).expanduser()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = simulation_root / "logs" / "sim_validation" / "ask2act_grasp" / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _apply_overrides(
    scene_config: SceneConfig,
    *,
    grasp_method: str | None = None,
    cup_position_xy: tuple[float, float] | None = None,
) -> SceneConfig:
    return apply_scene_overrides(
        scene_config,
        grasp_method=grasp_method,
        cup_position_xy=cup_position_xy,
    )


def _build_context(
    *,
    scene_config_path: str | Path,
    grasp_config_path: str | Path,
    run_dir: Path,
    random_seed: int | None = None,
    randomize_cup: bool = False,
    cup_position_xy: tuple[float, float] | None = None,
    grasp_method: str | None = None,
    multi_cup: bool = False,
    multi_cup_config_path: str | Path | None = None,
) -> tuple[PipelineContext, SceneConfig, HeadAlignmentConfig, GraspConfig]:
    simulation_root = SIMULATION_ROOT
    workspace_root = simulation_root.parent

    scene_config, head_config = load_scene_config(scene_config_path)
    scene_config = _apply_overrides(
        scene_config,
        grasp_method=grasp_method,
        cup_position_xy=cup_position_xy,
    )
    grasp_config = load_grasp_config(grasp_config_path)
    scene_setup = SceneSetup(scene_config)
    if cup_position_xy is not None:
        cup_position = scene_config.cup_position_m
    elif randomize_cup:
        cup_position = scene_setup.sample_cup_position(random_seed)
    else:
        cup_position = scene_config.cup_position_m

    generated_scene_dir = simulation_root / "stretch_mujoco" / "stretch_mujoco" / "models" / "generated"
    generated_scene_dir.mkdir(parents=True, exist_ok=True)
    generated_assets_link = generated_scene_dir / "assets"
    if not generated_assets_link.exists():
        os.symlink("../assets", generated_assets_link)
    runtime_metadata: dict[str, object] = {}
    if multi_cup:
        if not multi_cup_config_path:
            raise ValueError("multi_cup_config_path is required when multi_cup=True")
        cup_configs = json.loads(Path(multi_cup_config_path).read_text())
        if not isinstance(cup_configs, list) or not cup_configs:
            raise ValueError("Expected multi-cup config JSON to contain a non-empty list")
        scene_xml_path = scene_setup.write_multi_cup_scene(
            generated_scene_dir / f"{run_dir.name}_runtime_scene.xml",
            cup_configs=cup_configs,
        )
        target_cfg = next(
            (
                cfg for cfg in cup_configs
                if bool(cfg.get("is_target")) or str(cfg.get("name")) == scene_config.cup_body_name
            ),
            None,
        )
        if target_cfg is None:
            raise ValueError("No target cup found in multi-cup config")
        runtime_metadata = {
            "multi_cup": True,
            "cup_configs": cup_configs,
            "target_center_xy": [float(target_cfg["position"][0]), float(target_cfg["position"][1])],
            "non_target_cup_body_names": [
                str(cfg["name"])
                for cfg in cup_configs
                if str(cfg["name"]) != scene_config.cup_body_name and not bool(cfg.get("is_target", False))
            ],
            "multi_cup_config_path": str(multi_cup_config_path),
        }
    else:
        scene_xml_path = scene_setup.write_scene(
            generated_scene_dir / f"{run_dir.name}_runtime_scene.xml",
            cup_position_m=cup_position,
        )
    (run_dir / "runtime_scene.xml").write_text(scene_xml_path.read_text())

    context = PipelineContext(
        workspace_root=workspace_root,
        simulation_root=simulation_root,
        run_dir=run_dir,
        scene_xml_path=scene_xml_path,
        scene_config=replace(scene_config, cup_position_m=tuple(cup_position)),
        grasp_config=grasp_config,
        head_config=head_config,
        runtime_metadata=runtime_metadata,
    )
    return context, scene_config, head_config, grasp_config


def _apply_d435i_camera_overrides(grasp_config: GraspConfig) -> None:
    render_width, render_height = grasp_config.d435i_render_resolution_px
    sensor_width, sensor_height = grasp_config.d435i_sensor_resolution_px
    os.environ["ASK2ACT_D435I_RENDER_WIDTH"] = str(int(render_width))
    os.environ["ASK2ACT_D435I_RENDER_HEIGHT"] = str(int(render_height))
    os.environ["ASK2ACT_D435I_SENSOR_WIDTH"] = str(int(sensor_width))
    os.environ["ASK2ACT_D435I_SENSOR_HEIGHT"] = str(int(sensor_height))


def run_pipeline(
    *,
    scene_config_path: str | Path | None = None,
    grasp_config_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    random_seed: int | None = None,
    randomize_cup: bool = False,
    cup_position_xy: tuple[float, float] | None = None,
    grasp_method: str | None = None,
    headless: bool = False,
    show_viewer_ui: bool = False,
    target_bbox_2d: tuple[int, int, int, int] | None = None,
    multi_cup: bool = False,
    multi_cup_config_path: str | Path | None = None,
) -> PipelineResult:
    package_root = Path(__file__).resolve().parent
    resolved_scene_config = scene_config_path or (package_root / "config" / "scene_config.yaml")
    resolved_grasp_config = grasp_config_path or (package_root / "config" / "grasp_config.yaml")
    resolved_run_dir = build_run_dir(SIMULATION_ROOT, str(run_dir or ""))

    os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")
    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

    context, scene_config, head_config, _grasp_config = _build_context(
        scene_config_path=resolved_scene_config,
        grasp_config_path=resolved_grasp_config,
        run_dir=resolved_run_dir,
        random_seed=random_seed,
        randomize_cup=randomize_cup,
        cup_position_xy=cup_position_xy,
        grasp_method=grasp_method,
        multi_cup=multi_cup,
        multi_cup_config_path=multi_cup_config_path,
    )
    _apply_d435i_camera_overrides(_grasp_config)

    if headless:
        os.environ.setdefault("MUJOCO_GL", os.environ.get("STRETCH_SIM_HEADLESS_MUJOCO_GL", "egl"))

    sim = StretchMujocoSimulator(
        scene_xml_path=str(context.scene_xml_path),
        camera_hz=head_config.camera_hz,
        cameras_to_use=[
            StretchCameras.cam_d435i_rgb,
            StretchCameras.cam_d435i_depth,
        ],
        start_translation=list(scene_config.robot_start_translation_m),
        start_rotation_quat=list(scene_config.robot_start_rotation_quat_xyzw),
    )

    try:
        startup_started = datetime.now()
        startup_perf = time.perf_counter()
        default_passive_viewer = "1" if headless else "0"
        use_passive_viewer = os.environ.get("ASK2ACT_USE_PASSIVE_VIEWER", default_passive_viewer).strip().lower() not in {
            "0",
            "false",
            "no",
        }
        sim.start(show_viewer_ui=show_viewer_ui, headless=headless, use_passive_viewer=use_passive_viewer)
        sim_startup_s = round(time.perf_counter() - startup_perf, 4)
        result = GraspExecutor(context).run(sim, target_bbox_2d=target_bbox_2d)
        final_pause_s = max(0.0, float(os.environ.get("ASK2ACT_FINAL_PAUSE_S", "0.0")))
        if final_pause_s > 0.0 and not headless:
            print(f"Holding GUI scene for {final_pause_s:.1f}s after the trial for inspection.", flush=True)
            time.sleep(final_pause_s)
        result.intermediate.setdefault("pipeline_stage_times_s", {})
        result.intermediate["pipeline_stage_times_s"]["sim_startup"] = sim_startup_s
        result.intermediate["pipeline_stage_times_s"]["pipeline_total_after_start"] = result.intermediate.get("elapsed_s")
        result.intermediate["sim_started_at"] = startup_started.isoformat()
        payload = json.dumps(result.__dict__, indent=2, default=_json_default)
        (context.run_dir / "pipeline_result.json").write_text(payload)
        (context.run_dir / "pipeline_log.json").write_text(payload)
        return result
    finally:
        sim.stop()


def main() -> int:
    args = parse_args()
    target_bbox = parse_bbox(args.target_bbox_2d)
    cup_position_xy = None
    if args.cup_x is not None or args.cup_y is not None:
        if args.cup_x is None or args.cup_y is None:
            raise ValueError("Provide both --cup-x and --cup-y when overriding the cup position.")
        cup_position_xy = (float(args.cup_x), float(args.cup_y))

    result = run_pipeline(
        scene_config_path=args.scene_config,
        grasp_config_path=args.grasp_config,
        run_dir=args.run_dir,
        random_seed=args.random_seed,
        randomize_cup=args.randomize_cup,
        cup_position_xy=cup_position_xy,
        grasp_method=args.grasp_method,
        headless=args.headless,
        show_viewer_ui=args.show_viewer_ui,
        target_bbox_2d=target_bbox,
        multi_cup=args.multi_cup,
        multi_cup_config_path=args.multi_cup_config or None,
    )
    print(json.dumps(result.__dict__, indent=2, default=_json_default))
    return 0 if result.success else 1


def _json_default(value):
    try:
        return value.tolist()
    except AttributeError:
        return value


if __name__ == "__main__":
    sys.exit(main())
