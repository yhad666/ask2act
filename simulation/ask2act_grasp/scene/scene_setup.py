from __future__ import annotations

import random
from pathlib import Path

from ask2act_grasp.types import SceneConfig


class SceneSetup:
    def __init__(self, scene_config: SceneConfig) -> None:
        self.scene_config = scene_config
        self.simulation_root = Path(__file__).resolve().parents[2]
        self.stretch_models_root = self.simulation_root / "stretch_mujoco" / "stretch_mujoco" / "models"

    def sample_cup_position(self, seed: int | None = None) -> tuple[float, float, float]:
        rng = random.Random(seed)
        dx_range, dy_range = self.scene_config.cup_randomization_range_m
        base_x, base_y, base_z = self.scene_config.cup_position_m
        return (
            base_x + rng.uniform(-dx_range, dx_range),
            base_y + rng.uniform(-dy_range, dy_range),
            base_z,
        )

    def write_scene(self, output_path: str | Path, cup_position_m: tuple[float, float, float] | None = None) -> Path:
        cup_position = cup_position_m or self.scene_config.cup_position_m
        table_x, table_y, table_z = self.scene_config.table_position_m
        table_sx, table_sy, table_sz = self.scene_config.table_size_m
        cup_x, cup_y, cup_z = cup_position
        cup_radius = self.scene_config.cup_radius_m
        cup_half_height = self.scene_config.cup_height_m / 2.0
        friction = " ".join(str(v) for v in self.scene_config.cup_friction)

        xml = f"""<mujoco model="ask2act single cup pipeline">
  <compiler angle="radian" assetdir="../assets" meshdir="../assets" texturedir="../assets"/>
  <include file="../stretch.xml"/>

  <statistic center="0 -0.72 0.82" extent="1.4" meansize="0.06"/>

  <visual>
    <headlight diffuse="0.75 0.75 0.75" ambient="0.30 0.30 0.30" specular="0.15 0.15 0.15"/>
    <rgba haze="0.18 0.22 0.28 1"/>
    <global azimuth="-112" elevation="-20"/>
  </visual>

  <worldbody>
    <light pos="0.0 -0.2 1.9" dir="0 0 -1" directional="true"/>
    <light pos="0.55 -0.95 1.3" dir="-0.2 0.25 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" rgba="0.16 0.16 0.16 1"/>

    <body name="table" pos="{table_x} {table_y} {table_z}">
      <geom name="table_top" type="box" size="{table_sx} {table_sy} {table_sz}" rgba="0.70 0.58 0.42 1"/>
      <geom name="table_leg_fl" type="box" pos="{table_sx - 0.06} {table_sy - 0.06} -0.58" size="0.025 0.025 0.58" rgba="0.38 0.28 0.19 1"/>
      <geom name="table_leg_fr" type="box" pos="{table_sx - 0.06} {-table_sy + 0.06} -0.58" size="0.025 0.025 0.58" rgba="0.38 0.28 0.19 1"/>
      <geom name="table_leg_rl" type="box" pos="{-table_sx + 0.06} {table_sy - 0.06} -0.58" size="0.025 0.025 0.58" rgba="0.38 0.28 0.19 1"/>
      <geom name="table_leg_rr" type="box" pos="{-table_sx + 0.06} {-table_sy + 0.06} -0.58" size="0.025 0.025 0.58" rgba="0.38 0.28 0.19 1"/>
    </body>

    <body name="target_cup" pos="{cup_x} {cup_y} {cup_z}">
      <freejoint name="target_cup_freejoint"/>
      <geom name="target_cup_body" type="cylinder" size="{cup_radius} {cup_half_height}" mass="{self.scene_config.cup_mass_kg}" friction="{friction}" rgba="0.82 0.20 0.16 1"/>
      <geom name="target_cup_lid" type="cylinder" pos="0 0 {cup_half_height - 0.006}" size="{cup_radius * 0.92} 0.006" mass="0.01" friction="{friction}" rgba="0.94 0.92 0.85 1"/>
    </body>
  </worldbody>
</mujoco>
"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(xml)
        return path
