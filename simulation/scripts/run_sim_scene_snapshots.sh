#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
UV_BIN="${HOME}/.local/bin/uv"
SCREENSHOT_DIR="${ROOT_DIR}/logs/sim_validation/screenshots"
DEFAULT_SCENE="${ROOT_DIR}/stretch_mujoco/stretch_mujoco/models/scene.xml"
TABLETOP_SCENE="${ROOT_DIR}/sim_grasping/tabletop_minimal_scene.xml"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/source_sim.sh" >/dev/null

export MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR:-/tmp/mesa_shader_cache}"
mkdir -p "${MESA_SHADER_CACHE_DIR}" "${SCREENSHOT_DIR}"

cd "${ROOT_DIR}/stretch_mujoco"

run_snapshot() {
  MUJOCO_GL="${MUJOCO_GL:-${STRETCH_SIM_HEADLESS_MUJOCO_GL:-egl}}" \
    "${UV_BIN}" run python "${ROOT_DIR}/sim_grasping/render_scene_snapshot.py" "$@"
}

run_snapshot \
  --scene-xml-path "${DEFAULT_SCENE}" \
  --output-path "${SCREENSHOT_DIR}/default_scene_overview.png" \
  --azimuth 145 --elevation -22 --distance 2.2 --lookat 0.0 -0.6 0.55

run_snapshot \
  --scene-xml-path "${DEFAULT_SCENE}" \
  --output-path "${SCREENSHOT_DIR}/motion_template_overview.png" \
  --azimuth 138 --elevation -24 --distance 2.0 --lookat 0.1 -0.55 0.72 \
  --ctrl lift=0.82 \
  --ctrl arm=0.24 \
  --ctrl wrist_yaw=0.0 \
  --ctrl wrist_pitch=-0.55 \
  --ctrl wrist_roll=0.0 \
  --ctrl gripper=0.03 \
  --ctrl head_pan=0.0 \
  --ctrl head_tilt=-0.55

run_snapshot \
  --scene-xml-path "${TABLETOP_SCENE}" \
  --output-path "${SCREENSHOT_DIR}/tabletop_scene_overview.png" \
  --azimuth 146 --elevation -20 --distance 2.25 --lookat 0.0 -0.72 0.56
