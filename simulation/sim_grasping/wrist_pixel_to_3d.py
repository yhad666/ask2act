from __future__ import annotations

import argparse
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a wrist D405 image pixel into a camera-frame 3D point.")
    add_scene_and_output_args(parser)
    parser.add_argument("--u", type=int, default=None)
    parser.add_argument("--v", type=int, default=None)
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = create_run_dir(args.output_dir, args.run_tag, prefix="pixel_to_3d")

    save_json(
        output_dir / "pixel_to_3d_context.json",
        {
            "scene_xml_path": str(Path(args.scene_xml_path).resolve()),
            "output_dir": str(output_dir),
            "u": args.u,
            "v": args.v,
            "interactive": bool(args.interactive),
        },
    )

    sim = start_wrist_sim(
        scene_xml_path=Path(args.scene_xml_path),
        camera_hz=float(args.camera_hz),
        settle_seconds=float(args.settle_seconds),
    )
    try:
        observation = capture_wrist_observation(sim)
        save_observation_bundle(output_dir, "before", observation)
        pixel_u, pixel_v = choose_pixel(
            image_rgb=observation["rgb"],
            u=args.u,
            v=args.v,
            interactive=bool(args.interactive),
        )
        pixel_result = build_pixel_result(
            observation=observation,
            pixel_u=pixel_u,
            pixel_v=pixel_v,
        )
        save_json(output_dir / "pixel_to_3d_result.json", pixel_result)
        save_pixel_overlay(
            observation["rgb"],
            pixel_u=pixel_u,
            pixel_v=pixel_v,
            output_path=output_dir / "pixel_selection_overlay.png",
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
    finally:
        sim.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
