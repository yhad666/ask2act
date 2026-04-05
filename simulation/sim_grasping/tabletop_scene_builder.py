from __future__ import annotations

from pathlib import Path

from stretch_mujoco.utils import get_absolute_path_stretch_xml

from validation_common import LOG_ROOT, TABLETOP_SCENE_PATH, ensure_dir


def build_resolved_tabletop_scene(output_path: Path | None = None) -> Path:
    output_path = output_path or (ensure_dir(LOG_ROOT) / "tabletop_scene_resolved.xml")
    template_xml = TABLETOP_SCENE_PATH.read_text(encoding="utf-8")
    absolute_stretch_xml = get_absolute_path_stretch_xml()

    resolved_xml = template_xml.replace(
        '../stretch_mujoco/stretch_mujoco/models/stretch.xml',
        absolute_stretch_xml,
    )

    output_path.write_text(resolved_xml, encoding="utf-8")
    return output_path


def resolve_scene_xml_path(scene_xml_path: Path) -> Path:
    if scene_xml_path.resolve() == TABLETOP_SCENE_PATH.resolve():
        return build_resolved_tabletop_scene()
    return scene_xml_path.resolve()
