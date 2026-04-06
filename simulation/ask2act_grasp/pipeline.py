from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

SIMULATION_ROOT = Path(__file__).resolve().parent.parent
STRETCH_MUJOCO_ROOT = SIMULATION_ROOT / "stretch_mujoco"
for import_root in (SIMULATION_ROOT, STRETCH_MUJOCO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator

from ask2act_grasp.execution.grasp_executor import GraspExecutor
from ask2act_grasp.scene.scene_setup import SceneSetup
from ask2act_grasp.types import PipelineContext
from ask2act_grasp.utils.config_loader import load_grasp_config, load_scene_config


def parse_args() -> argparse.Namespace:
    package_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Ask2Act single-cup simulation grasp pipeline")
    parser.add_argument("--scene-config", default=str(package_root / "config" / "scene_config.yaml"))
    parser.add_argument("--grasp-config", default=str(package_root / "config" / "grasp_config.yaml"))
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--randomize-cup", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--show-viewer-ui", action="store_true")
    parser.add_argument("--target-bbox-2d", default="")
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


def main() -> int:
    args = parse_args()
    simulation_root = SIMULATION_ROOT
    workspace_root = simulation_root.parent
    run_dir = build_run_dir(simulation_root, args.run_dir)

    os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")
    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

    scene_config, head_config = load_scene_config(args.scene_config)
    grasp_config = load_grasp_config(args.grasp_config)
    scene_setup = SceneSetup(scene_config)
    cup_position = scene_setup.sample_cup_position(args.random_seed) if args.randomize_cup else scene_config.cup_position_m
    generated_scene_dir = simulation_root / "stretch_mujoco" / "stretch_mujoco" / "models" / "generated"
    generated_scene_dir.mkdir(parents=True, exist_ok=True)
    generated_assets_link = generated_scene_dir / "assets"
    if not generated_assets_link.exists():
        os.symlink("../assets", generated_assets_link)
    scene_xml_path = scene_setup.write_scene(generated_scene_dir / f"{run_dir.name}_runtime_scene.xml", cup_position_m=cup_position)
    (run_dir / "runtime_scene.xml").write_text(scene_xml_path.read_text())

    context = PipelineContext(
        workspace_root=workspace_root,
        simulation_root=simulation_root,
        run_dir=run_dir,
        scene_xml_path=scene_xml_path,
        scene_config=scene_config,
        grasp_config=grasp_config,
        head_config=head_config,
    )

    if args.headless:
        os.environ.setdefault("MUJOCO_GL", os.environ.get("STRETCH_SIM_HEADLESS_MUJOCO_GL", "egl"))

    sim = StretchMujocoSimulator(
        scene_xml_path=str(scene_xml_path),
        camera_hz=head_config.camera_hz,
        cameras_to_use=[
            StretchCameras.cam_d435i_rgb,
            StretchCameras.cam_d435i_depth,
        ],
        start_translation=list(scene_config.robot_start_translation_m),
        start_rotation_quat=list(scene_config.robot_start_rotation_quat_xyzw),
    )

    try:
        sim.start(show_viewer_ui=args.show_viewer_ui, headless=args.headless, use_passive_viewer=True)
        result = GraspExecutor(context).run(sim, target_bbox_2d=parse_bbox(args.target_bbox_2d))
        print(json.dumps(result.__dict__, indent=2, default=_json_default))
        return 0 if result.success else 1
    finally:
        sim.stop()


def _json_default(value):
    try:
        return value.tolist()
    except AttributeError:
        return value


if __name__ == "__main__":
    sys.exit(main())
