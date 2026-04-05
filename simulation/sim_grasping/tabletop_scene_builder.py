from __future__ import annotations

import re
from pathlib import Path

from stretch_mujoco.utils import get_absolute_path_stretch_xml

from validation_common import LOG_ROOT, TABLETOP_SCENE_PATH, ensure_dir


def build_resolved_scene(scene_template_path: Path, output_path: Path) -> Path:
    template_xml = scene_template_path.read_text(encoding="utf-8")
    absolute_stretch_xml = get_absolute_path_stretch_xml()

    resolved_xml = template_xml.replace(
        '../stretch_mujoco/stretch_mujoco/models/stretch.xml',
        absolute_stretch_xml,
    )

    ensure_dir(output_path.parent)
    output_path.write_text(resolved_xml, encoding="utf-8")
    return output_path


def build_resolved_tabletop_scene(output_path: Path | None = None) -> Path:
    output_path = output_path or (ensure_dir(LOG_ROOT) / "tabletop_scene_resolved.xml")
    return build_resolved_scene(TABLETOP_SCENE_PATH, output_path)


def build_head_locked_scene(
    scene_template_path: Path,
    *,
    head_pan_rad: float,
    head_tilt_rad: float,
    output_path: Path,
) -> Path:
    stretch_xml_path = Path(get_absolute_path_stretch_xml()).resolve()
    stretch_xml_text = stretch_xml_path.read_text(encoding="utf-8")
    match = re.search(r'<key name="home" ctrl="([^"]+)"', stretch_xml_text)
    if match is None:
        raise ValueError("Failed to find home keyframe in stretch.xml")

    ctrl_values = [float(value) for value in match.group(1).split()]
    if len(ctrl_values) < 6:
        raise ValueError("Unexpected home keyframe format in stretch.xml")

    ctrl_values[4] = float(head_pan_rad)
    ctrl_values[5] = float(head_tilt_rad)
    updated_ctrl = " ".join(f"{value:.6f}" for value in ctrl_values)
    stretch_xml_text = stretch_xml_text.replace(match.group(0), f'<key name="home" ctrl="{updated_ctrl}"')

    custom_stretch_xml = ensure_dir(LOG_ROOT) / f"{scene_template_path.stem}_head_locked_stretch.xml"
    custom_stretch_xml.write_text(stretch_xml_text, encoding="utf-8")

    template_xml = scene_template_path.read_text(encoding="utf-8")
    resolved_xml = template_xml.replace(
        '../stretch_mujoco/stretch_mujoco/models/stretch.xml',
        str(custom_stretch_xml),
    )
    ensure_dir(output_path.parent)
    output_path.write_text(resolved_xml, encoding="utf-8")
    return output_path


def resolve_scene_xml_path(scene_xml_path: Path) -> Path:
    scene_xml_path = scene_xml_path.resolve()
    if scene_xml_path.suffix != ".xml":
        return scene_xml_path

    sim_grasping_dir = TABLETOP_SCENE_PATH.parent.resolve()
    try:
        is_sim_grasping_scene = scene_xml_path.parent.resolve() == sim_grasping_dir
    except FileNotFoundError:
        is_sim_grasping_scene = False

    if is_sim_grasping_scene:
        output_name = scene_xml_path.stem + "_resolved.xml"
        return build_resolved_scene(scene_xml_path, ensure_dir(LOG_ROOT) / output_name)
    return scene_xml_path
