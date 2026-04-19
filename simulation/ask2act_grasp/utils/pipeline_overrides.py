from __future__ import annotations

from dataclasses import replace

from ask2act_grasp.types import SceneConfig


def apply_scene_overrides(
    scene_config: SceneConfig,
    *,
    grasp_method: str | None = None,
    cup_position_xy: tuple[float, float] | None = None,
) -> SceneConfig:
    updated = scene_config
    if grasp_method is not None:
        updated = replace(updated, grasp_method=grasp_method)
    if cup_position_xy is not None:
        updated = replace(
            updated,
            cup_position_m=(
                float(cup_position_xy[0]),
                float(cup_position_xy[1]),
                float(updated.cup_position_m[2]),
            ),
        )
    return updated
