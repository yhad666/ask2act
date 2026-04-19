from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

from minimal_grasping_common import (
    add_scene_and_output_args,
    build_pixel_result,
    capture_wrist_observation,
    choose_pixel,
    create_run_dir,
    save_json,
    save_observation_bundle,
    save_pixel_overlay,
    start_wrist_sim,
)
from minimal_primitive_runner import args_to_params, run_primitive_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal tabletop grasping loop in Stretch MuJoCo.")
    add_scene_and_output_args(parser)
    parser.add_argument("--u", type=int, default=None)
    parser.add_argument("--v", type=int, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--pregrasp-lift-m", type=float, default=0.80)
    parser.add_argument("--pregrasp-arm-m", type=float, default=0.24)
    parser.add_argument("--pregrasp-wrist-yaw-rad", type=float, default=0.0)
    parser.add_argument("--pregrasp-wrist-pitch-rad", type=float, default=-0.55)
    parser.add_argument("--pregrasp-wrist-roll-rad", type=float, default=0.0)
    parser.add_argument("--head-pan-rad", type=float, default=0.0)
    parser.add_argument("--head-tilt-rad", type=float, default=-0.55)
    parser.add_argument("--open-gripper-m", type=float, default=0.03)
    parser.add_argument("--close-gripper-m", type=float, default=-0.02)
    parser.add_argument("--descend-delta-m", type=float, default=0.06)
    parser.add_argument("--min-descend-delta-m", type=float, default=0.03)
    parser.add_argument("--max-descend-delta-m", type=float, default=0.09)
    parser.add_argument("--lift-after-grasp-delta-m", type=float, default=0.08)
    parser.add_argument("--retract-arm-target-m", type=float, default=0.08)
    parser.add_argument("--nominal-target-depth-m", type=float, default=0.28)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = create_run_dir(args.output_dir, args.run_tag, prefix="minimal_loop")
    params = args_to_params(args)

    save_json(
        output_dir / "minimal_grasping_loop_context.json",
        {
            "scene_xml_path": str(Path(args.scene_xml_path).resolve()),
            "output_dir": str(output_dir),
            "u": args.u,
            "v": args.v,
            "interactive": bool(args.interactive),
            "primitive_parameters": params,
        },
    )

    sim = start_wrist_sim(
        scene_xml_path=Path(args.scene_xml_path),
        camera_hz=float(args.camera_hz),
        settle_seconds=float(args.settle_seconds),
    )
    loop_summary: dict[str, Any] = {
        "status": "started",
        "steps": [],
        "output_dir": str(output_dir),
        "scene_xml_path": str(Path(args.scene_xml_path).resolve()),
        "primitive_parameters": params,
    }

    try:
        before_observation = capture_wrist_observation(sim)
        before_files = save_observation_bundle(output_dir, "before", before_observation)
        loop_summary["steps"].append({"name": "capture_before", "artifacts": before_files})

        pixel_u, pixel_v = choose_pixel(
            image_rgb=before_observation["rgb"],
            u=args.u,
            v=args.v,
            interactive=bool(args.interactive),
        )
        pixel_result = build_pixel_result(
            observation=before_observation,
            pixel_u=pixel_u,
            pixel_v=pixel_v,
        )
        save_json(output_dir / "target_pixel_result.json", pixel_result)
        save_pixel_overlay(
            before_observation["rgb"],
            pixel_u=pixel_u,
            pixel_v=pixel_v,
            output_path=output_dir / "target_pixel_overlay.png",
            text_lines=[
                f"pixel=({pixel_u}, {pixel_v})",
                f"depth_m={pixel_result['depth_m']:.4f}",
                (
                    "camera_xyz_m="
                    f"({pixel_result['camera_xyz_m'][0]:+.4f}, "
                    f"{pixel_result['camera_xyz_m'][1]:+.4f}, "
                    f"{pixel_result['camera_xyz_m'][2]:+.4f})"
                ),
            ],
        )
        loop_summary["steps"].append({"name": "pixel_to_3d", "result": pixel_result})

        primitive_summary = run_primitive_sequence(
            sim,
            output_dir=output_dir,
            params=params,
            settle_seconds=float(args.settle_seconds),
            target_pixel_result=pixel_result,
            skip_home=False,
        )
        loop_summary["steps"].append({"name": "primitive_runner", "result_path": str(output_dir / "primitive_runner_summary.json")})

        after_observation = capture_wrist_observation(sim)
        after_files = save_observation_bundle(output_dir, "after", after_observation)
        loop_summary["steps"].append({"name": "capture_after", "artifacts": after_files})
        loop_summary["status"] = "completed"
        loop_summary["target_pixel_result"] = pixel_result
        loop_summary["primitive_summary"] = primitive_summary
    except Exception as exc:
        loop_summary["status"] = "failed"
        loop_summary["error"] = {
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_json(output_dir / "minimal_grasping_loop_summary.json", loop_summary)
        raise
    finally:
        sim.stop()

    save_json(output_dir / "minimal_grasping_loop_summary.json", loop_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
